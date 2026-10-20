"""
Stage 3: layout-aware ("semantic") chunking.

RAG-Chatbot cut raw text every 1000 characters, wherever that fell. Here chunks
follow the document's own structure, produced by layout analysis:

* each chunk belongs to one section: the heading path ("Results > Lab values")
  is stored in metadata and prefixed to the chunk text, so a paragraph that
  never repeats its section title can still be found by it;
* tables are never cut mid-row: big tables are split into row groups and every
  piece repeats the header row, so each chunk is a valid table;
* figures (OCR'd chart text / vision descriptions) are their own chunks;
* chunks never cross a page boundary, so every citation names an exact page;
* inside a section, text is split on paragraph -> line -> sentence -> word
  boundaries, 800 characters by default. all-MiniLM-L6-v2 reads at most 256
  word pieces (about 1000 characters of English) and silently drops the rest.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..ocr.types import Element, PageText


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        src = self.metadata.get("source", "?")
        page = self.metadata.get("page")
        return f"{src}, p. {page}" if page is not None else src


_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def _split(text: str, size: int, seps=_SEPARATORS) -> list[str]:
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(seps):
        if sep in text:
            parts = text.split(sep)
            pieces = [p + sep if j < len(parts) - 1 else p for j, p in enumerate(parts)]
            out = []
            for p in pieces:
                out.extend([p] if len(p) <= size else _split(p, size, seps[i + 1:]))
            return out
    return [text[i:i + size] for i in range(0, len(text), size)]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            cur = tail if len(tail) + len(p) <= size else ""
        cur += p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    return [c for c in _merge(_split(text, size), size, overlap) if c]


def chunk_table(markdown: str, size: int = 800) -> list[str]:
    """Split a Markdown table into row groups, repeating the header in each piece."""
    lines = [ln for ln in markdown.splitlines() if ln.strip()]
    if len(lines) <= 2 or len(markdown) <= size:
        return [markdown]
    header, rows = lines[:2], lines[2:]
    head = "\n".join(header)
    pieces, cur = [], []
    for row in rows:
        if cur and len(head) + sum(len(r) + 1 for r in cur) + len(row) > size:
            pieces.append(head + "\n" + "\n".join(cur))
            cur = []
        cur.append(row if len(head) + len(row) < size * 2 else row[: size * 2])
    if cur:
        pieces.append(head + "\n" + "\n".join(cur))
    return pieces


_OCR_TOOLS = ("tesseract", "easyocr", "paddleocr", "vision")


def _is_ocr(el: Element, page: PageText) -> bool:
    """Did this element come from OCR? (Per element: a hybrid page mixes both.)"""
    if el.tool:
        return any(t in el.tool for t in _OCR_TOOLS)
    if page.route is not None:              # untagged text on a hybrid page is the native layer
        return page.route in ("ocr", "visual")
    return page.ocr_used


def _cid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _norm(t: str) -> str:
    return re.sub(r"[\W_]+", " ", t).strip().lower()


_TAIL_STOP = {"is", "are", "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with", "by",
              "from", "that", "which", "be", "can", "as", "at", "it", "its", "this"}


def _junk_heading(text: str) -> bool:
    """Bold labels and fragments that are not section titles."""
    t = text.strip()
    words = re.findall(r"[A-Za-z']+", t)
    return (not t or not (t[0].isalnum() or t[0] in "‘'\"") or " , " in t
            or (words and words[-1].lower() in _TAIL_STOP) or len(words) == 0
            or re.sub(r"^\s*(?:Q[-\s]?\d+|\d+|[A-Za-z])\s*[.)\]]+\s*", "", t).count("(") != t.count(")") - (1 if re.match(r"^\s*(?:Q[-\s]?\d+|\d+|[A-Za-z])\s*[.]?\)", t) else 0))


def outline_chunks(source: str, title: str, headings: list,
                   size: int = 800, force: bool = False) -> list[Chunk]:
    """
    One "document overview" passage per document: its title and the distinct section
    headings in reading order with their first page. Answers questions about the whole
    document ("what is this about?", "what topics does it cover?") that no single
    passage can answer. Slide decks repeat titles on every slide; each is listed once.
    """
    seen, lines = set(), []
    for h in headings:
        text, page = h[0], h[1]
        level = h[2] if len(h) > 2 else 1
        key = _norm(text)
        # outline = real section titles: skip bullet points, run-in labels ("Benefits:"),
        # the title slide itself, and repeats (slide decks repeat titles on every slide)
        if (not key or key in seen or len(text) > 120 or re.match(r"^\s*[➢▪•◦●■►\-*–]", text)
                or text.rstrip().endswith(":") or (title and key in _norm(title)) or _junk_heading(text)):
            continue
        seen.add(key)
        indent = "  " * max(0, min(level, 4) - 1)
        lines.append(f"{indent}- {text}" + (f" (p. {page})" if page is not None else ""))
    if not lines and not title and not force:
        return []
    head = f"Document overview: {source}\n" + (f"Title: {title}\n" if title else "") + "Sections:"
    out, cur = [], []
    for ln in lines or ["- (no headings found)"]:
        if cur and len(head) + sum(len(x) + 1 for x in cur) + len(ln) > size * 2:
            out.append(cur)
            cur = []
        cur.append(ln)
    out.append(cur)
    chunks = []
    for n, part in enumerate(out):
        text = head + ("" if n == 0 else " (continued)") + "\n" + "\n".join(part)
        md = {"source": source, "element": "outline", "method": "parser", "ocr_used": False,
              "content_type": "outline"}
        first = next((h for h in headings if h[1] is not None), None)
        if first and n == 0:
            md["page"] = first[1]
        chunks.append(Chunk(_cid(source, "outline", n), text, md))
    return chunks


def chunk_pages(pages: list[PageText], size: int = 800, overlap: int = 120,
                min_chars: int = 25, outline: bool = True) -> list[Chunk]:
    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []          # heading stack carried across pages of a document
    last_source = None
    doc_headings: list[tuple[str, int | None]] = []
    doc_title = ""
    doc_scanned = False                        # OCR'd notes without headings still get an overview passage

    def close_document():
        if outline and last_source is not None:
            chunks.extend(outline_chunks(last_source, doc_title, doc_headings, size, force=doc_scanned))

    for page in pages:
        if page.source != last_source:
            close_document()
            path, last_source = [], page.source
            doc_headings, doc_title = [], ""
            doc_scanned = False
            heads = []
            for e in page.elements:
                if e.kind != "heading":
                    continue
                t = e.text.strip()
                if heads and re.match(r"^\d+[.)]\s", t):        # "1. Eligibility" starts the body, not the title
                    break
                heads.append(t)
            heads = heads[:3]
            first = next((e for e in page.elements if e.text.strip()), None)
            if heads:
                doc_title = " · ".join(re.sub(r"\s+", " ", h) for h in heads)[:200]
            elif first is not None:
                doc_title = re.sub(r"\s+", " ", first.text.strip().splitlines()[0])[:200]
            if re.match(r"^\s*Q[-\s]?\d", doc_title):         # "Q-1) According to the scenario..." is a question, not a title
                doc_title = ""
        if page.route in ("ocr", "visual") or page.ocr_used:
            doc_scanned = True
        elements = page.elements or [Element("text", page.text)]
        if elements and elements[0].kind == "heading":
            # A page that opens with its own heading starts a new section. Heading levels
            # from different tools/pages (pdfplumber vs OCR) are not comparable, so
            # nesting it under the previous page's heading would be wrong.
            path = []
        buffer: list[str] = []
        buffer_ocr = False
        seq = 0
        emitted: set[str] = set()

        def section() -> str:
            return " > ".join(t for _, t in path)

        def emit(text: str, kind: str, extra: dict | None = None, ocr: bool | None = None):
            nonlocal seq
            text = text.strip()
            if len(text) < min_chars or _norm(text) in emitted:
                return
            emitted.add(_norm(text))
            md = page.metadata()
            md["element"] = kind
            if ocr is not None:
                md["ocr_used"] = ocr
            sec = section()
            if sec:
                md["section"] = sec
            if extra:
                md.update(extra)
            body = f"[{sec}]\n{text}" if sec else text
            chunks.append(Chunk(_cid(md["source"], md.get("page"), kind, seq, text), body, md))
            seq += 1

        def flush():
            nonlocal buffer_ocr
            if buffer:
                for piece in chunk_text("\n\n".join(buffer), size, overlap):
                    emit(piece, "text", ocr=buffer_ocr)
                buffer.clear()
            buffer_ocr = False

        for el in elements:
            if el.kind == "heading":
                flush()
                level = el.level or 1
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, el.text.strip()))
                doc_headings.append((el.text.strip(), page.page, level))
            elif el.kind == "text":
                # drop lines that only repeat a heading (slide titles repeated on every page)
                heads = {_norm(t) for _, t in path} | {_norm(h[0]) for h in doc_headings[-20:]}
                kept = [ln for ln in el.text.splitlines() if _norm(ln) not in heads]
                text = "\n".join(kept).strip()
                if not text:
                    continue
                buffer.append(text)
                buffer_ocr = buffer_ocr or _is_ocr(el, page)
            elif el.kind == "table":
                flush()
                for piece in chunk_table(el.text, size):
                    emit(piece, "table", ocr=_is_ocr(el, page))
            elif el.kind == "figure":
                flush()
                extra = {"ocr_used": True, "figure_tool": el.tool}
                for piece in chunk_text(el.text, size, overlap):
                    emit(piece, "figure", extra)
        flush()

        # a page whose only content was headings still deserves a chunk (unless it is
        # just a repeated title that the outline already covers)
        if not any(c.metadata.get("page") == page.page and c.metadata["source"] == page.source
                   for c in chunks) and len(page.text.strip()) >= min_chars:
            body = "\n".join(ln for ln in page.text.splitlines()
                             if _norm(ln.lstrip("#")) not in {_norm(h[0]) for h in doc_headings[:-1]})
            if len(body.strip()) >= min_chars:
                emit(body, "text")
    close_document()
    return chunks