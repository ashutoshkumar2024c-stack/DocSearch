"""
The full pipeline in one object:

    ds = DocSearch(index_dir="index")
    ds.ingest(["report.pdf", "scan.jpg"])   # classify -> route tools -> extract -> chunk -> embed -> index
    ans = ds.ask("What was recovered by September 30?")
    print(ans.format())                      # answer + [n] file, p. X citations
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .ocr import Extractor
from .ocr.extract import summarize
from .ocr.types import PageText
from .rag.answer import Answer, condense_question, generate_answer
from .rag.chunking import chunk_pages
from .rag.embeddings import Embedder, get_embedder
from .rag.llm import LLM, get_llm
from .rag.store import Hit, VectorStore
from .rag.visual import get_visual_retriever, needs_visual_index, page_key

log = logging.getLogger(__name__)

_OVERVIEW = re.compile(
    r"\b(what (is|are) (this|these|the) (document|file|pdf|module|deck|slides?|notes?|chapter|unit|report|paper)s?"
    r"( (all|mainly))? about|summar(y|ise|ize)|overview|outline|table of contents|main (topics?|points|ideas)"
    r"|topics? (covered|discussed)|what does (it|this|the \w+) (cover|discuss|contain)|gist)\b", re.I)


def is_overview_question(q: str) -> bool:
    return bool(_OVERVIEW.search(q or ""))


_STRUCTURE = re.compile(r"\b(tables?|figures?|diagrams?|charts?|graphs?|illustrations?|images?|pictures?)\b", re.I)


def is_structure_question(q: str) -> bool:
    """'Are there any tables or figures? What do they show?'"""
    return bool(_STRUCTURE.search(q or "")) and bool(re.search(r"\b(any|are there|is there|which|list|show|how many|what)\b", q or "", re.I))


_LIST = re.compile(r"\b(types?|kinds?|list|categories|levels?|steps?|stages?|phases?|rules?|principles?|"
                   r"advantages?|disadvantages?|benefits?|characteristics?|features?|components?|models?|"
                   r"examples?|what are|which are|how many)\b", re.I)


def is_list_question(q: str) -> bool:
    return bool(_LIST.search(q or "")) and not is_overview_question(q)
Progress = Callable[[str, float], None]          # (message, fraction 0..1)


@dataclass
class IngestReport:
    file: str
    status: str                       # "indexed" | "skipped" | "failed"
    pages: int = 0
    chunks: int = 0
    routes: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)     # doc_type, tools, elements, confidence
    plans: list = field(default_factory=list)       # per-page PagePlan dicts
    seconds: float = 0.0
    error: str | None = None
    page_texts: list = field(default_factory=list)   # [(page number or None, extracted text)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("page_texts", None)
        return d


class DocSearch:
    def __init__(self, index_dir: str | Path | None = "index",
                 embedder: Embedder | str | None = None,
                 llm: LLM | str | None = None,
                 extractor: Extractor | None = None,
                 chunk_size: int = 800, chunk_overlap: int = 120,
                 top_k: int = 4, retrieval: str = "hybrid",
                 visual_retriever=None):
        self.index_dir = Path(index_dir) if index_dir else None
        self.embedder = embedder if not isinstance(embedder, (str, type(None))) else get_embedder(embedder)
        self._llm = llm
        self.extractor = extractor or Extractor()
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.top_k, self.retrieval = top_k, retrieval
        self.history: list[tuple[str, str]] = []
        self._lock = threading.RLock()

        if self.index_dir and VectorStore.exists(self.index_dir):
            self.store = VectorStore.load(self.index_dir)
            if self.store.embedder_name != self.embedder.name:
                raise ValueError(
                    f"Index in {self.index_dir} was built with '{self.store.embedder_name}' but the "
                    f"current embedder is '{self.embedder.name}'. Use the same embedder or a new index folder.")
        else:
            self.store = VectorStore(self.embedder.dim, self.embedder.name)
        self.visual = visual_retriever if visual_retriever is not None else \
            get_visual_retriever(index_dir=self.index_dir)

    @property
    def llm(self) -> LLM:
        if self._llm is None or isinstance(self._llm, str):
            self._llm = get_llm(self._llm)
        return self._llm

    # ---- ingestion ------------------------------------------------------- #
    def ingest_pages(self, pages: list[PageText], path: Path | None = None) -> int:
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        with self._lock:
            added = self.store.add(chunks, vectors)
        if self.visual is not None and path is not None:
            from .ocr.render import render_page
            items = []
            for p in pages:
                if needs_visual_index(p):
                    img = render_page(path, p.page, dpi=100)
                    if img is not None:
                        items.append((page_key(p.source, p.page), img))
            if items:
                self.visual.add_pages(items)
        return added

    def ingest_file(self, path: str | Path, source_name: str | None = None, replace: bool = True,
                    progress: Progress | None = None) -> IngestReport:
        path = Path(path)
        name = source_name or path.name
        say = progress or (lambda msg, frac: None)
        t0 = time.perf_counter()
        try:
            if name in self.store.sources():
                if not replace:
                    return IngestReport(name, "skipped")
                self.remove(name, save=False)
            say("Classifying and extracting", 0.1)
            pages = self.extractor.extract(
                path, source_name=name,
                progress=lambda i, n: say(f"Reading page {i + 1} of {n}", 0.1 + 0.6 * i / max(n, 1)))
            say("Chunking and embedding", 0.7)
            added = self.ingest_pages(pages, path)
            summary = summarize(pages)
            plans = [{**(p.plan.to_dict() if p.plan else {}), "page": p.page, "method": p.method,
                      "ocr_confidence": p.ocr_confidence,
                      "elements": [e.kind for e in p.elements], "chars": len(p.text)} for p in pages]
            say("Indexed", 1.0)
            return IngestReport(name, "indexed", len(pages), added, summary["routes"], summary, plans,
                                time.perf_counter() - t0, page_texts=[(p.page, p.text) for p in pages])
        except Exception as exc:
            log.exception("ingest failed for %s", path)
            return IngestReport(name, "failed", seconds=time.perf_counter() - t0, error=str(exc))

    def ingest(self, paths, replace: bool = True, source_names: list[str] | None = None
               ) -> list[IngestReport]:
        reports = [self.ingest_file(p, source_names[i] if source_names else None, replace)
                   for i, p in enumerate(paths)]
        self.save()
        return reports

    def remove(self, source: str, save: bool = True) -> int:
        with self._lock:
            n = self.store.remove_source(source)
            if self.visual is not None:
                self.visual.remove_source(source)
        if save:
            self.save()
        return n

    def save(self):
        if self.index_dir:
            with self._lock:
                self.store.save(self.index_dir)
                if self.visual is not None:
                    self.visual.save(self.index_dir)

    # ---- querying -------------------------------------------------------- #
    def _visual_ranking(self, query: str, k: int = 10) -> list[str] | None:
        if self.visual is None:
            return None
        pages = self.visual.search(query, k)
        by_page: dict[str, list] = {}
        for c in self.store.chunks:
            key = page_key(c.metadata.get("source", "?"), c.metadata.get("page"))
            by_page.setdefault(key, []).append(c)
        ranking = []
        for key, _ in pages:
            cands = sorted(by_page.get(key, []), key=lambda c: c.metadata.get("element") != "figure")
            ranking += [c.id for c in cands[:2]]
        return ranking

    def retrieve(self, query: str, k: int | None = None, mode: str | None = None,
                 sources: list[str] | None = None) -> list[Hit]:
        mode = mode or self.retrieval
        k = k or self.top_k
        qvec = self.embedder.embed_query(query)
        visual = self._visual_ranking(query) if mode == "hybrid" else None
        with self._lock:
            if is_list_question(query):
                # list answers are often spread over several slides/pages: look wider,
                # and add the outline so section sub-headings can be listed
                hits = self.store.search(qvec, query, max(k, 8), mode, source_filter=sources,
                                         visual_ranking=visual)
                hits = self._add_outline(hits, sources)
            else:
                hits = self.store.search(qvec, query, k, mode, source_filter=sources, visual_ranking=visual)
            if is_overview_question(query):
                hits = self._with_outline(hits, k, sources)
            elif hits and not any(h.chunk.metadata.get("element") == "outline" for h in hits):
                # always give the model/answerer the document title and section list as context
                hits = hits + self._add_outline(hits, sources)[len(hits):][:1]
            if is_structure_question(query):
                hits = self._add_structured(hits, sources)
        return hits

    def _add_structured(self, hits: list[Hit], sources: list[str] | None) -> list[Hit]:
        """Add the document's figure and table passages so 'any tables or figures?' can be answered."""
        docs = sources or list(dict.fromkeys(h.chunk.metadata.get("source") for h in hits))[:2] or self.store.sources()
        have = {h.chunk.id for h in hits}
        extra = [Hit(c, 0.0) for c in self.store.chunks if c.metadata.get("element") in ("figure", "table")
                 and c.metadata.get("source") in docs and c.id not in have]
        return hits + extra[:8]

    def _add_outline(self, hits: list[Hit], sources: list[str] | None) -> list[Hit]:
        docs = sources or list(dict.fromkeys(h.chunk.metadata.get("source") for h in hits))[:1]
        have = {h.chunk.id for h in hits}
        extra = [Hit(c, 0.0) for c in self.store.chunks if c.metadata.get("element") == "outline"
                 and c.metadata.get("source") in docs and c.id not in have]
        return hits + extra[:3]

    def _with_outline(self, hits: list[Hit], k: int, sources: list[str] | None) -> list[Hit]:
        """Overview questions ("what is this about?") are answered from the document outline,
        which lists the title and every section; no single passage can answer them."""
        docs = sources or list(dict.fromkeys(h.chunk.metadata.get("source") for h in hits))
        if not docs:
            docs = self.store.sources()
        outlines = [c for c in self.store.chunks
                    if c.metadata.get("element") == "outline" and c.metadata.get("source") in docs[:2]]
        if not outlines:
            return hits
        ids = {c.id for c in outlines}
        rest = [h for h in hits if h.chunk.id not in ids]
        return [Hit(c, 1.0) for c in outlines[:3]] + rest[: max(1, k - min(len(outlines), 3))]

    def ask(self, question: str, use_history: bool = True, sources: list[str] | None = None,
            history: list[tuple[str, str]] | None = None, k: int | None = None,
            mode: str | None = None) -> Answer:
        hist = self.history if history is None else history
        standalone = condense_question(self.llm, question, hist) if use_history else question
        hits = self.retrieve(standalone, k=k, mode=mode, sources=sources)
        answer = generate_answer(self.llm, question, hits,
                                 standalone if standalone != question else None)
        if use_history and history is None:
            self.history.append((question, answer.answer))
        return answer

    def reset_chat(self):
        self.history = []

    def stats(self) -> dict:
        from collections import Counter
        el = Counter(c.metadata.get("element", "text") for c in self.store.chunks)
        out = {"chunks": len(self.store), "documents": len(self.store.sources()),
               "embedder": self.embedder.name, "dim": self.embedder.dim, "elements": dict(el)}
        if self.visual is not None:
            out["visual"] = self.visual.stats()
        return out