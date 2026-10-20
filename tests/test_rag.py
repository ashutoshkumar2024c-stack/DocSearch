"""Tests for chunking, retrieval, citations, LLM clients and the full pipeline.

Models that need downloads or servers (MiniLM, Mistral, Ollama) are replaced
by the offline HashingEmbedder and a scripted fake LLM, so the tests run
anywhere. The LLM clients themselves are tested against mocked HTTP/Ollama.
"""

from __future__ import annotations

import os

import json
import re
import sys
import types
from pathlib import Path

import numpy as np

from docsearch.evaluate import evaluate_end_to_end, evaluate_retrieval
from docsearch.ocr.types import PageText
from docsearch.pipeline import DocSearch
from docsearch.rag import (NOT_FOUND, HashingEmbedder, MistralLLM, OllamaLLM, VectorStore,
                           chunk_pages, chunk_text)
from docsearch.rag.answer import parse_citations
from docsearch.rag.store import tokenize

SAMPLES = Path(__file__).parent / "samples"


class FakeLLM:
    """Acts like an obedient grounded model: answers from the best-matching
    source, cites it, and abstains when nothing matches."""

    name = "fake"

    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=None, max_tokens=800):
        self.calls.append(messages)
        content = messages[-1]["content"]
        if content.startswith("Rewrite the follow-up"):
            hist = re.search(r"User: (.*)", content).group(1)
            follow = re.search(r"Follow-up question: (.*)", content).group(1)
            return f"{follow} (about: {hist})"
        q = set(tokenize(content.split("Question:")[-1]))
        best, best_n, best_sent = 0, None, ""
        for m in re.finditer(r"\[(\d+)\] \(.*?\)\n(.*?)(?=\n\n\[\d+\] \(|\n\nQuestion:)", content, re.S):
            for sent in re.split(r"(?<=[.!?])\s+|\n", m.group(2)):
                overlap = len(q & set(tokenize(sent)))
                if overlap > best:
                    best, best_n, best_sent = overlap, int(m.group(1)), sent.strip()
        if best < 2:
            return NOT_FOUND
        return f"{best_sent} [{best_n}]"


def pages():
    return [
        PageText("a.pdf", "Alpha project budget is 12 lakh rupees.\n\n" + "Filler sentence. " * 80, 1),
        PageText("a.pdf", "The audit found invoice INV-2026-0042 was paid twice.", 2, "tesseract", "ocr", 91.0),
        PageText("b.docx", "Team lunch is on Friday at the canteen.", None, "parser"),
    ]


# ------------------------------------------------------------- chunking ---
def test_chunk_sizes_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(300))
    chunks = chunk_text(text, size=500, overlap=80)
    assert all(len(c) <= 500 for c in chunks) and len(chunks) > 5
    # consecutive chunks share text (overlap carried forward)
    assert any(chunks[0][-40:].split()[-1] in chunks[1][:120] for _ in [0])


def test_layout_aware_chunks():
    from docsearch.ocr.types import Element
    els = [Element("heading", "Contract Review", 1), Element("text", "Body text about the contract terms and renewal."),
           Element("heading", "Renewals", 2), Element("figure", "North 82%\nSouth 76%\nWest 91% East 69%", tool="tesseract"),
           Element("table", "| Region | Rate |\n|---|---|\n| West | 91% |")]
    p = PageText.from_elements("r.pdf", els, page=3, method="native+tesseract", route="hybrid")
    chunks = [c for c in chunk_pages([p], min_chars=5) if c.metadata["element"] != "outline"]
    assert [c.metadata["element"] for c in chunks] == ["text", "figure", "table"]
    assert chunks[0].metadata["section"] == "Contract Review"
    assert chunks[1].metadata["section"] == "Contract Review > Renewals"
    assert chunks[1].text.startswith("[Contract Review > Renewals]")
    assert chunks[1].metadata["ocr_used"] is True and chunks[0].metadata["ocr_used"] is False
    assert all(c.metadata["page"] == 3 for c in chunks) and chunks[0].citation == "r.pdf, p. 3"


def test_big_tables_split_with_header_repeated():
    from docsearch.rag.chunking import chunk_table
    md = "| Name | Score |\n|---|---|\n" + "\n".join(f"| student {i} | {i} |" for i in range(200))
    pieces = chunk_table(md, size=400)
    assert len(pieces) > 5
    assert all(p.startswith("| Name | Score |\n|---|---|\n| student") for p in pieces)
    assert sum(p.count("| student") for p in pieces) == 200          # no row lost or duplicated


# ------------------------------------------------------------ retrieval ---
def _store():
    emb = HashingEmbedder()
    chunks = chunk_pages(pages())
    store = VectorStore(emb.dim, emb.name)
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    return emb, store


def test_hybrid_finds_exact_codes():
    emb, store = _store()
    q = "INV-2026-0042"
    for mode in ("bm25", "hybrid"):
        top = store.search(emb.embed_query(q), q, k=1, mode=mode)[0]
        assert top.chunk.metadata["page"] == 2, mode


def test_dedup_remove_and_persistence(tmp_path):
    emb, store = _store()
    n = len(store)
    chunks = chunk_pages(pages())
    assert store.add(chunks, emb.embed_documents([c.text for c in chunks])) == 0   # no duplicates
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == n and np.allclose(loaded.vectors, store.vectors)
    assert loaded.remove_source("b.docx") == 1 and loaded.sources() == ["a.pdf"]


def test_source_filter():
    emb, store = _store()
    q = "budget lunch invoice"
    hits = store.search(emb.embed_query(q), q, k=5, source_filter=["b.docx"])
    assert hits and all(h.chunk.metadata["source"] == "b.docx" for h in hits)


# ------------------------------------------------------------ citations ---
def test_parse_citations():
    assert parse_citations("A [1]. B [2][3]. C [1, 4]. D [9]") == [1, 2, 3, 4, 9]


def test_pipeline_answers_with_citations_and_abstains(tmp_path):
    ds = DocSearch(index_dir=tmp_path / "idx", embedder=HashingEmbedder(), llm=FakeLLM())
    ds.ingest_pages(pages())
    ans = ds.ask("Which invoice was paid twice in the audit?", use_history=False)
    assert "INV-2026-0042" in ans.answer and ans.found
    assert ans.citations[0].page == 2 and ans.citations[0].label().endswith("(OCR)")
    assert ans.invalid_citations == []
    none = ds.ask("What is the capital of Peru?", use_history=False)
    assert not none.found


def test_followup_question_is_condensed(tmp_path):
    llm = FakeLLM()
    ds = DocSearch(index_dir=None, embedder=HashingEmbedder(), llm=llm)
    ds.ingest_pages(pages())
    ds.ask("What is the Alpha project budget?")
    second = ds.ask("Is that in rupees?")
    assert second.standalone_question and "Alpha project budget" in second.standalone_question


def test_invalid_citation_flagged(tmp_path):
    class Liar(FakeLLM):
        def chat(self, messages, temperature=None, max_tokens=800):
            return "The budget is 12 lakh [1] and was approved by the board [7]."
    ds = DocSearch(index_dir=None, embedder=HashingEmbedder(), llm=Liar())
    ds.ingest_pages(pages())
    ans = ds.ask("budget?", use_history=False)
    assert ans.invalid_citations == [7] and [c.number for c in ans.citations] == [1]


# ------------------------------------------------------- full pipeline ---
def test_end_to_end_on_mixed_pdf(tmp_path):
    ds = DocSearch(index_dir=tmp_path / "idx", embedder=HashingEmbedder(), llm=FakeLLM())
    (rep,) = ds.ingest([SAMPLES / "mixed.pdf"])
    assert rep.status == "indexed" and rep.routes == {"native": 1, "ocr": 2, "hybrid": 1}
    assert rep.summary["doc_type"] == "mixed-pdf" and len(rep.plans) == 4

    items = json.loads((SAMPLES / "questions.json").read_text())
    ret = evaluate_retrieval(ds, items, k=4)
    assert ret["hybrid"]["recall@4"] == 1.0
    assert set(ret["hybrid"]["by_element"]) == {"text", "table", "figure"}
    assert set(ret["hybrid"]["by_route"]) == {"native", "ocr", "hybrid"}
    e2e = evaluate_end_to_end(ds, items)
    # The fake LLM only matches words, so answer accuracy is not meaningful here
    # (measure it with a real LLM: `python -m docsearch eval ... --with-llm`).
    # What we check is the plumbing: every citation points at the right page,
    # nothing is cited that wasn't retrieved, and unanswerable questions abstain.
    assert e2e["citation_precision"] == 1.0
    assert e2e["abstention_accuracy"] == 1.0
    assert e2e["invalid_citations"] == 0

    # re-open from disk without re-processing; re-ingest replaces, no duplicates
    ds2 = DocSearch(index_dir=tmp_path / "idx", embedder=HashingEmbedder(), llm=FakeLLM())
    n = len(ds2.store)
    ds2.ingest([SAMPLES / "mixed.pdf"])
    assert len(ds2.store) == n


def test_embedder_mismatch_is_refused(tmp_path):
    ds = DocSearch(index_dir=tmp_path, embedder=HashingEmbedder(), llm=FakeLLM())
    ds.ingest_pages(pages()); ds.save()
    try:
        DocSearch(index_dir=tmp_path, embedder=HashingEmbedder(dim=512), llm=FakeLLM())
    except ValueError as e:
        assert "built with" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ------------------------------------------------------ LLM HTTP clients ---
def test_mistral_client_request(monkeypatch):
    sent = {}

    class Resp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "Answer [1]"}}]}

    fake = types.ModuleType("requests")
    def post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, body=json)
        return Resp()
    fake.post = post
    monkeypatch.setitem(sys.modules, "requests", fake)

    llm = MistralLLM(api_key="test-key")
    assert llm.chat([{"role": "user", "content": "hi"}]) == "Answer [1]"
    assert sent["url"].endswith("/v1/chat/completions")
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["body"]["model"] == "mistral-small-latest" and sent["body"]["temperature"] == 0.1


def test_ollama_llm_strips_thinking(monkeypatch):
    fake = types.ModuleType("ollama")
    class Client:
        def __init__(self, host=None): pass
        def chat(self, **kw):
            return {"message": {"content": "<think>hmm</think>Final answer [2]"}}
    fake.Client = Client
    monkeypatch.setitem(sys.modules, "ollama", fake)
    assert OllamaLLM().chat([{"role": "user", "content": "x"}]) == "Final answer [2]"


def test_selective_visual_retrieval(tmp_path):
    """ColPali-style retriever: only visual/hybrid pages are indexed, and its ranking is fused."""
    class FakeVisual:
        name = "fake-colpali"
        def __init__(self):
            self.keys = []
        def add_pages(self, items):
            self.keys += [k for k, _ in items]
            return len(items)
        def remove_source(self, source):
            n = len(self.keys); self.keys = [k for k in self.keys if not k.startswith(source + "::")]
            return n - len(self.keys)
        def search(self, query, k=10):
            return [(key, 1.0) for key in self.keys][:k]
        def save(self, d): pass
        def stats(self): return {"pages": len(self.keys)}

    vis = FakeVisual()
    ds = DocSearch(index_dir=None, embedder=HashingEmbedder(), llm=FakeLLM(), visual_retriever=vis)
    ds.ingest([SAMPLES / "mixed.pdf"])
    assert vis.keys == ["mixed.pdf::3"]                      # only the hybrid page, not all 4
    hits = ds.retrieve("regional chart", k=4)
    assert any(h.visual_rank is not None and h.chunk.metadata["page"] == 3 for h in hits)
    assert ds.stats()["visual"]["pages"] == 1
    ds.remove("mixed.pdf")
    assert vis.keys == []


def test_ocr_metrics_eval(tmp_path):
    from docsearch.evaluate import evaluate_ocr
    truth = json.loads((SAMPLES / "ground_truth.json").read_text())
    items = [{"file": "mixed.pdf", "page": 2, "text": "\n".join(truth["page2"])},
             {"file": "mixed.pdf", "page": 1, "text": "\n".join(truth["page1"])}]
    ds = DocSearch(index_dir=None, embedder=HashingEmbedder(), llm=FakeLLM())
    r = evaluate_ocr(ds, items, SAMPLES)
    assert r["by_route"]["ocr"]["cer"] < 0.05 and set(r["by_route"]) == {"ocr", "native"}


# ------------------------------------------- slide-deck regressions (Module_4.pdf) ---
def _deck():
    """A lecture deck exported to PDF: every slide repeats its title; one list spans 3 slides."""
    from docsearch.ocr.types import Element as E
    title = "Design Concepts – 6. Component Design - Cohesion"
    hdr = "| Cohesion Type | Explanation |\n|---|---|\n"
    pages = [
        PageText.from_elements("deck.pdf", [E("heading", "Software Engineering", 1), E("heading", "Module 4", 2)], page=1),
        PageText.from_elements("deck.pdf", [E("heading", title, 1), E("text", title),
            E("text", "➢ Cohesion is a fundamental concept in software design that refers to\nhow closely "
                      "the elements within a module are related to each other.")], page=2),
        PageText.from_elements("deck.pdf", [E("heading", title, 1), E("text", title),
            E("table", hdr + "| Coincidental Cohesion | Unrelated functions grouped randomly. |\n"
                             "| Logical Cohesion | Logically similar operations grouped together. |")], page=3),
        PageText.from_elements("deck.pdf", [E("heading", title, 1),
            E("table", hdr + "| Temporal Cohesion | Functions that run at the same time. |\n"
                             "| Communication Cohesion | Functions operate on the same data. |")], page=4),
        PageText.from_elements("deck.pdf", [E("heading", title, 1),
            E("table", hdr + "| Functional Cohesion | The module performs a single task. |")], page=5),
        PageText.from_elements("deck.pdf", [E("heading", "User Interface Design – Golden Rules", 1),
            E("heading", "Place the User in Control", 2), E("text", "Let users interrupt and undo actions."),
            E("heading", "Reduce the User’s Memory Load", 2), E("text", "Use meaningful defaults and shortcuts."),
            E("heading", "Make the Interface Consistent", 2), E("text", "Keep layouts and terms consistent.")], page=6),
        PageText.from_elements("deck.pdf", [E("heading", "User Interface Design – Models", 1),
            E("text", "The user model, the mental model and the implementation model.")], page=7),
    ]
    from docsearch.rag.llm import ExtractiveLLM
    ds = DocSearch(index_dir=None, embedder=HashingEmbedder(), llm=ExtractiveLLM(), top_k=4)
    ds.ingest_pages(pages)
    return ds


def test_repeated_slide_titles_are_not_passages():
    ds = _deck()
    texts = [c.text.split("\n", 1)[-1].strip() for c in ds.store.chunks if c.metadata["element"] == "text"]
    assert "Design Concepts – 6. Component Design - Cohesion" not in texts
    outline = next(c for c in ds.store.chunks if c.metadata["element"] == "outline")
    assert "Title: Software Engineering · Module 4" in outline.text
    assert outline.text.count("Component Design - Cohesion") == 1          # listed once, not per slide


def test_overview_question_uses_outline():
    ans = _deck().ask("What is this document about?", use_history=False)
    assert ans.citations and ans.citations[0].element == "outline"
    assert "Component Design" in ans.answer and "User Interface Design" in ans.answer


def test_list_question_merges_rows_across_slides_and_follow_up():
    ds = _deck()
    a = ds.ask("Types of Cohesion")
    for t in ("Coincidental", "Logical", "Temporal", "Communication", "Functional"):
        assert t in a.answer, t
    b = ds.ask("more types are there")                       # follow-up borrows the topic
    assert b.standalone_question and "Cohesion" in b.standalone_question and b.found
    c = ds.ask("What is cohesion?")                          # a new question is NOT merged
    assert c.standalone_question is None and "fundamental concept" in c.answer
    assert "about system" not in c.answer                    # whole sentences, not line fragments


def test_list_from_section_subheadings():
    ans = _deck().ask("What are the golden rules of user interface design?", use_history=False)
    for rule in ("Place the User in Control", "Reduce the User’s Memory Load", "Make the Interface Consistent"):
        assert rule in ans.answer


def test_mistral_retries_once_then_fails_fast(monkeypatch=None):
    import time as _t, types, sys
    from docsearch.rag import llm as L
    calls = []

    class R:
        def __init__(self, code, body=None): self.status_code, self.headers, self._b, self.text = code, {"Retry-After": "0"}, body, ""
        def json(self): return self._b

    seq = [R(429), R(200, {"choices": [{"message": {"content": "hello"}}]})]
    fake = types.SimpleNamespace(post=lambda *a, **k: (calls.append(1), seq.pop(0))[1])
    old = sys.modules.get("requests"); sys.modules["requests"] = fake
    L.time = types.SimpleNamespace(sleep=lambda s: None, time=_t.time)
    L.MistralLLM._blocked.clear()
    try:
        assert L.MistralLLM(api_key="k").chat([{"role": "user", "content": "hi"}]) == "hello" and len(calls) == 2
        calls.clear(); seq[:] = [R(429), R(429)]
        try:
            L.MistralLLM(api_key="k").chat([{"role": "user", "content": "hi"}]); raise AssertionError("no error")
        except RuntimeError as e:
            assert "rate limit" in str(e) and len(calls) == 2
        calls.clear()                                   # now blocked: no HTTP call, instant error
        try:
            L.MistralLLM(api_key="k").chat([{"role": "user", "content": "hi"}]); raise AssertionError("no error")
        except RuntimeError as e:
            assert "rate-limiting" in str(e) and not calls
    finally:
        L.time = _t
        L.MistralLLM._blocked.clear()
        if old is not None: sys.modules["requests"] = old
        else: sys.modules.pop("requests", None)


def test_self_contained_question_skips_condense_call():
    from docsearch.rag.answer import condense_question

    class Counting:
        name = "c"
        n = 0
        def chat(self, *a, **k):
            Counting.n += 1
            return "rewritten"
    hist = [("What is cohesion?", "A measure")]
    assert condense_question(Counting(), "What are the types of coupling in software design?", hist) \
        == "What are the types of coupling in software design?" and Counting.n == 0
    assert condense_question(Counting(), "what about its types?", hist) == "rewritten" and Counting.n == 1


def test_provider_chain_falls_back_and_skips_failed_provider():
    import sys, types
    from docsearch.rag import llm as L
    calls = {"mistral": 0, "gemini": 0}

    class R:
        def __init__(self, code, body=None): self.status_code, self.headers, self._b, self.text = code, {"Retry-After": "0"}, body, "err"
        def json(self): return self._b

    def post(url, **kw):
        if "mistral" in url:
            calls["mistral"] += 1
            return R(429)
        calls["gemini"] += 1
        assert kw["headers"]["x-goog-api-key"] == "gk" and kw["json"]["systemInstruction"]
        return R(200, {"candidates": [{"content": {"parts": [{"text": "from gemini"}]}}]})

    old = sys.modules.get("requests"); sys.modules["requests"] = types.SimpleNamespace(post=post)
    L.time = types.SimpleNamespace(sleep=lambda s: None, time=__import__("time").time)
    L.MistralLLM._blocked.clear(); L.GeminiLLM._blocked.clear()
    try:
        chain = L.FallbackLLM([L.MistralLLM(api_key="mk"), L.GeminiLLM(api_key="gk"), L.ExtractiveLLM()])
        msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
        assert chain.chat(msgs) == "from gemini" and chain.name.startswith("gemini")
        n = calls["mistral"]
        assert chain.chat(msgs) == "from gemini" and calls["mistral"] == n      # failed provider skipped for a while
        assert "429" in chain.errors["mistral:mistral-small-latest"] or "rate" in chain.errors["mistral:mistral-small-latest"]
        # no keys at all -> offline extractive
        for k in ("MISTRAL_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            os.environ.pop(k, None)
        os.environ["LLM_PROVIDER"] = "auto"
        L.ollama_reachable = lambda *a, **k: False
        assert isinstance(L.get_llm(), L.ExtractiveLLM)
    finally:
        import time as _t
        L.time = _t
        L.MistralLLM._blocked.clear(); L.GeminiLLM._blocked.clear()
        if old is not None: sys.modules["requests"] = old
        else: sys.modules.pop("requests", None)
