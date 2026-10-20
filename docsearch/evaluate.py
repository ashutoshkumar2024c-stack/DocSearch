"""
Evaluation with element-level and route-level breakdowns.

The 2026 survey (Gao et al., ACL 2026) names the lack of element-level evaluation
as an open gap: a single Recall number hides whether a system fails on tables,
on figures, or on scanned pages. Every metric here is therefore also reported
per element type (text / table / figure) and per route (native / ocr / hybrid / visual).

Question file (JSON list):
    {"question": "...", "source": "file.pdf", "page": 2,      # where the answer lives
     "element": "table",                                     # optional: text | table | figure
     "answer_contains": ["4B"]}                              # optional: for end-to-end scoring
    {"question": "...", "source": null}                      # unanswerable -> should abstain

OCR ground-truth file (JSON list), for CER / WER:
    {"file": "scan.pdf", "page": 2, "text": "exact page text ..."}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .ocr.metrics import cer, wer
from .pipeline import DocSearch


def _is_match(hit, item) -> bool:
    md = hit.chunk.metadata
    if md.get("source") != item["source"]:
        return False
    return item.get("page") is None or md.get("page") == item["page"]


def _route_of(ds: DocSearch, item) -> str:
    for c in ds.store.chunks:
        if c.metadata.get("source") == item["source"] and (item.get("page") is None
                                                          or c.metadata.get("page") == item["page"]):
            return c.metadata.get("route", "parser")
    return "missing"


def _group(rows, key, k):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {g: {"n": len(rs), f"recall@{k}": sum(r["rank"] is not None for r in rs) / len(rs),
                "mrr": sum(1 / r["rank"] if r["rank"] else 0 for r in rs) / len(rs)}
            for g, rs in sorted(groups.items())}


def evaluate_retrieval(ds: DocSearch, items: list[dict], k: int = 4,
                       modes=("dense", "bm25", "hybrid")) -> dict:
    answerable = [it for it in items if it.get("source")]
    results = {}
    for mode in modes:
        rows = []
        for it in answerable:
            hits = ds.retrieve(it["question"], k=k, mode=mode)
            rank = next((r for r, h in enumerate(hits, 1) if _is_match(h, it)), None)
            rows.append({"question": it["question"], "rank": rank,
                         "element": it.get("element", "text"), "route": _route_of(ds, it)})
        n = max(len(rows), 1)
        results[mode] = {
            f"recall@{k}": sum(r["rank"] is not None for r in rows) / n,
            "mrr": sum(1 / r["rank"] if r["rank"] else 0 for r in rows) / n,
            "by_element": _group(rows, "element", k),
            "by_route": _group(rows, "route", k),
            "per_question": rows,
        }
    return results


def evaluate_end_to_end(ds: DocSearch, items: list[dict]) -> dict:
    rows = []
    cited_total = cited_ok = invalid = 0
    for it in items:
        ans = ds.ask(it["question"], use_history=False)
        invalid += len(ans.invalid_citations)
        if it.get("source"):
            ok = ans.found and all(s.lower() in ans.answer.lower() for s in it.get("answer_contains", []))
            for c in ans.citations:
                cited_total += 1
                cited_ok += c.source == it["source"] and (it.get("page") is None or c.page == it["page"])
        else:
            ok = not ans.found
        rows.append({"question": it["question"], "answerable": bool(it.get("source")),
                     "element": it.get("element", "text") if it.get("source") else "unanswerable",
                     "correct": bool(ok), "answer": ans.answer,
                     "citations": [c.label() for c in ans.citations]})
    ans_rows = [r for r in rows if r["answerable"]]
    un_rows = [r for r in rows if not r["answerable"]]
    by_el = defaultdict(list)
    for r in ans_rows:
        by_el[r["element"]].append(r["correct"])
    return {
        "answer_accuracy": sum(r["correct"] for r in ans_rows) / max(len(ans_rows), 1),
        "citation_precision": cited_ok / max(cited_total, 1),
        "abstention_accuracy": (sum(r["correct"] for r in un_rows) / len(un_rows)) if un_rows else None,
        "invalid_citations": invalid,
        "by_element": {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by_el.items())},
        "per_question": rows,
    }


def evaluate_ocr(ds: DocSearch, truth: list[dict], base_dir: str | Path = ".") -> dict:
    """CER / WER of extracted text against ground-truth page text, overall and per route."""
    base = Path(base_dir)
    cache, rows = {}, []
    for t in truth:
        f = base / t["file"]
        if f not in cache:
            cache[f] = {p.page: p for p in ds.extractor.extract(f)}
        page = cache[f].get(t.get("page"))
        if page is None:
            rows.append({"file": t["file"], "page": t.get("page"), "route": "missing", "cer": 1.0, "wer": 1.0})
            continue
        hyp = "\n".join(e.text for e in page.elements if e.kind != "figure") or page.text
        rows.append({"file": t["file"], "page": t.get("page"), "route": page.route, "tool": page.method,
                     "cer": cer(t["text"], hyp, lowercase=True), "wer": wer(t["text"], hyp, lowercase=True)})
    by_route = defaultdict(list)
    for r in rows:
        by_route[r["route"]].append(r)
    agg = lambda rs, m: sum(r[m] for r in rs) / max(len(rs), 1)   # noqa: E731
    return {"cer": agg(rows, "cer"), "wer": agg(rows, "wer"),
            "by_route": {k: {"n": len(v), "cer": agg(v, "cer"), "wer": agg(v, "wer")} for k, v in by_route.items()},
            "per_page": rows}


def print_report(retrieval: dict, e2e: dict | None = None, ocr: dict | None = None) -> None:
    print("\nRetrieval")
    print(f"  {'mode':<8} {'Recall@k':>9} {'MRR':>7}")
    for mode, r in retrieval.items():
        rk = next(v for key, v in r.items() if key.startswith("recall@"))
        print(f"  {mode:<8} {rk:>9.2f} {r['mrr']:>7.2f}")
    best = retrieval.get("hybrid") or next(iter(retrieval.values()))
    for title, key in (("by element", "by_element"), ("by route", "by_route")):
        print(f"\n  hybrid, {title}")
        for g, v in best[key].items():
            rk = next(x for kk, x in v.items() if kk.startswith("recall@"))
            print(f"    {g:<10} n={v['n']:<3} Recall@k={rk:.2f}  MRR={v['mrr']:.2f}")
    if e2e:
        print("\nEnd-to-end")
        for key in ("answer_accuracy", "citation_precision", "abstention_accuracy", "invalid_citations"):
            v = e2e[key]
            print(f"  {key:<22} {v if v is None or isinstance(v, int) else f'{v:.2f}'}")
    if ocr:
        print(f"\nOCR  CER={ocr['cer']:.3f}  WER={ocr['wer']:.3f}")
        for k, v in ocr["by_route"].items():
            print(f"  {k:<8} n={v['n']}  CER={v['cer']:.3f}  WER={v['wer']:.3f}")


def load_questions(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
