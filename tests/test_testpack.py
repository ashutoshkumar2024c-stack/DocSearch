"""The manual test pack stays valid: every file routes as documented and every question is answered."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from docsearch.pipeline import DocSearch  # noqa: E402
from testpack_questions import Q, REFUSE  # noqa: E402


def test_testpack_routes_and_answers():
    ds = DocSearch(index_dir=None, embedder="hashing", llm="extractive")
    routes = {}
    for f in sorted((ROOT / "test_documents").iterdir()):
        if f.name.startswith("QUESTIONS"):
            continue
        r = ds.ingest_file(f)
        assert r.status == "indexed", (f.name, r.error)
        routes[f.name] = set(r.routes)
    assert routes["01_leave_policy_digital.pdf"] == {"native"}
    assert routes["02_invoice_scanned.pdf"] == routes["03_receipt_photo.jpg"] == {"ocr"}
    assert routes["11_mixed_report.pdf"] == {"native", "ocr", "hybrid"}
    assert all(routes[n] == {"parser"} for n in routes if n.split(".")[-1] in ("docx", "pptx", "xlsx", "csv", "html", "md"))
    wrong = [(f, q) for f, q, exp in Q
             if not all(e.lower() in ds.ask(q, use_history=False, sources=[f]).answer.lower() for e in exp)]
    assert not wrong, wrong
    assert all(not ds.ask(q, use_history=False).found for q in REFUSE)
