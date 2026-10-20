"""
Command line for the whole pipeline.

    python -m docsearch serve                             # Streamlit app (same as: streamlit run app.py)
    python -m docsearch ingest docs/*.pdf scans/*.jpg     # build / update the index
    python -m docsearch ask "What was the discharge medication?"
    python -m docsearch chat                              # multi-turn, with follow-ups
    python -m docsearch search "renewal rate west" --mode bm25   # retrieval only, no LLM
    python -m docsearch eval questions.json [--with-llm]
    python -m docsearch list | remove FILE

Common options: --index DIR (default: index)  --embedder minilm|ollama|hashing
                --llm mistral|ollama  --k 4  --ocr tesseract|vision  --fallback vision
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv


def _build(args, need_llm=False):
    from .ocr import Extractor
    from .pipeline import DocSearch

    extractor = Extractor(engine=args.ocr, visual_engine=args.visual_ocr,
                          fallback_engine=args.fallback)
    return DocSearch(index_dir=args.index, embedder=args.embedder, llm=args.llm,
                     extractor=extractor, top_k=args.k, retrieval=args.mode)


def main(argv=None) -> int:
    try:
        load_dotenv()
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m docsearch")
    ap.add_argument("command", choices=["serve", "ingest", "ask", "chat", "search", "eval", "list", "remove"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--index", default="index")
    ap.add_argument("--embedder", default=None, help="minilm (default) | ollama | hashing")
    ap.add_argument("--llm", default=None, help="mistral | ollama (default: mistral if key set)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--mode", default="hybrid", help="hybrid | dense | bm25")
    ap.add_argument("--ocr", default="auto", help="auto | tesseract | easyocr | paddleocr | vision")
    ap.add_argument("--visual-ocr", default=None)
    ap.add_argument("--fallback", default=None)
    ap.add_argument("--with-llm", action="store_true", help="eval: also score generated answers")
    ap.add_argument("--json", metavar="PATH", help="eval: write full results here")
    ap.add_argument("--ocr-truth", metavar="PATH", help="eval: OCR ground-truth JSON for CER/WER")
    ap.add_argument("--host", default=None, help="serve: interface")
    ap.add_argument("--port", type=int, default=None, help="serve: port (default 8501)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if a.command == "serve":
        import subprocess
        import sys
        app = Path(__file__).resolve().parent.parent / "app.py"
        cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
        if a.port:
            cmd += ["--server.port", str(a.port)]
        if a.host:
            cmd += ["--server.address", a.host]
        return subprocess.call(cmd)

    ds = _build(a)

    if a.command == "ingest":
        if not a.args:
            ap.error("ingest needs at least one file")
        for r in ds.ingest(a.args):
            if r.status == "indexed":
                s = r.summary
                print(f"  ✓ {r.file}: {s['doc_type']}, {r.pages} page(s), routes {r.routes}, "
                      f"tools {s['tools']} -> {r.chunks} chunks ({r.seconds:.1f}s)")
            else:
                print(f"  ✗ {r.file}: {r.status} {r.error or ''}")
        print(f"index: {len(ds.store)} chunks from {len(ds.store.sources())} file(s) in {a.index}/")

    elif a.command == "list":
        for s in ds.store.sources():
            n = sum(c.metadata.get("source") == s for c in ds.store.chunks)
            print(f"  {s}  ({n} chunks)")

    elif a.command == "remove":
        for s in a.args:
            print(f"  removed {ds.remove(s)} chunks of {s}")

    elif a.command == "search":
        for n, h in enumerate(ds.retrieve(" ".join(a.args)), 1):
            print(f"\n[{n}] {h.chunk.citation}  score={h.score:.4f} "
                  f"dense_rank={h.dense_rank} bm25_rank={h.sparse_rank}")
            print("    " + h.chunk.text[:300].replace("\n", "\n    "))

    elif a.command == "ask":
        print(ds.ask(" ".join(a.args), use_history=False).format())

    elif a.command == "chat":
        print("Ask about your documents. 'reset' clears history, 'exit' quits.\n")
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit"):
                break
            if q.lower() == "reset":
                ds.reset_chat()
                continue
            if q:
                ans = ds.ask(q)
                if ans.standalone_question:
                    print(f"  (searched for: {ans.standalone_question})")
                print("\nAI: " + ans.format() + "\n")

    elif a.command == "eval":
        from pathlib import Path

        from .evaluate import (evaluate_end_to_end, evaluate_ocr, evaluate_retrieval, load_questions,
                               print_report)

        items = load_questions(a.args[0])
        ret = evaluate_retrieval(ds, items, k=a.k)
        e2e = evaluate_end_to_end(ds, items) if a.with_llm else None
        ocr = None
        if a.ocr_truth:
            ocr = evaluate_ocr(ds, load_questions(a.ocr_truth), Path(a.ocr_truth).parent)
        print_report(ret, e2e, ocr)
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump({"retrieval": ret, "end_to_end": e2e, "ocr": ocr}, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
