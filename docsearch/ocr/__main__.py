"""
Command line for the parsing stage:

    python -m docsearch.ocr report.pdf                  # extract; show the plan and tools per page
    python -m docsearch.ocr scan.pdf --routes-only      # only the router's page decisions (no OCR)
    python -m docsearch.ocr photo.jpg --engine easyocr  # force an engine (default: auto)
    python -m docsearch.ocr --capabilities              # which engines are installed
    python -m docsearch.ocr a.pdf b.png --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .engines import capabilities
from .extract import Extractor, summarize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m docsearch.ocr",
                                 description="Classify documents, assign tools, extract layout elements.")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--engine", default="auto", help="auto | tesseract | easyocr | paddleocr | vision")
    ap.add_argument("--visual-engine", default=None, help="force the figure tool")
    ap.add_argument("--fallback", default=None, help="force the low-confidence fallback engine")
    ap.add_argument("--routes-only", action="store_true")
    ap.add_argument("--capabilities", action="store_true")
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--show-text", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.capabilities:
        for name, info in capabilities().items():
            mark = "✓" if info.get("available") else "✗"
            extra = info.get("version") or info.get("model") or ""
            print(f"  {mark} {name:<11} {extra} {info.get('error', '')}")
        if not args.files:
            return 0

    ex = Extractor(engine=args.engine, visual_engine=args.visual_engine, fallback_engine=args.fallback)
    results, failed = {}, 0
    for f in args.files:
        print(f"\n== {f}")
        if args.routes_only:
            for p in ex.routes(f):
                ev = p.evidence
                print(f"  page {p.page:>3}: {p.route:<7} chars={ev['native_chars']:<6} "
                      f"images={ev['image_coverage']:>4.0%}  {'; '.join(p.reasons)}")
            continue
        t0 = time.perf_counter()
        try:
            pages = ex.extract(f)
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}")
            results[f] = {"error": str(exc)}
            continue
        s = summarize(pages)
        print(f"  type={s['doc_type']}  routes={s['routes']}  tools={s['tools']}  elements={s['elements']}")
        for p in pages:
            conf = f"conf={p.ocr_confidence:.0f}" if p.ocr_confidence is not None else ""
            tools = ",".join(p.plan.tools) if p.plan else p.method
            print(f"  page {p.page or '-':>3}: {p.route or '':<7} {tools:<28} {len(p.text):>6} chars {conf}")
            if p.plan:
                for r in p.plan.reasons:
                    print(f"             - {r}")
            if args.show_text:
                print("    " + p.text.replace("\n", "\n    ")[:3000])
        print(f"  done in {time.perf_counter() - t0:.1f}s")
        results[f] = {"summary": s, "pages": [p.to_dict() for p in pages]}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False, default=str)
        print(f"\nwrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
