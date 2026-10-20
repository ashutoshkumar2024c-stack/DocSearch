"""
Logic behind the Streamlit app (``app.py``), kept free of any ``streamlit`` import so it
can be tested without a browser.

Each visitor gets their own in-memory index (nothing is shared between sessions and
nothing is written to a shared disk), which is what you want on Streamlit Community Cloud
where the filesystem is temporary. The embedding model is loaded once and shared.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .ocr import SUPPORTED_EXT
from .pipeline import DocSearch, IngestReport
from .rag.embeddings import get_embedder
from .rag.llm import ExtractiveLLM, FallbackLLM, GeminiLLM, MistralLLM, clean_key, gemini_keys, get_llm

# Keys that may come from Streamlit's secrets (Cloud: "Settings > Secrets") or a .env file.
SECRET_KEYS = ("MISTRAL_API_KEY", "MISTRAL_MODEL", "GEMINI_API_KEY", "GEMINI_API_KEYS", "GOOGLE_API_KEY", "GEMINI_MODEL", "LLM_PROVIDER", "LLM_MODEL", "OLLAMA_HOST",
               "OCR_ENGINE", "VISION_MODEL", "EMBEDDER", "ST_MODEL", "DISABLE_ENGINES")

ROUTE_LABEL = {
    "native": "Text layer", "ocr": "OCR", "hybrid": "Text + OCR",
    "visual": "Vision model", "parser": "Parser",
}
ROUTE_HELP = {
    "native": "The page has a real text layer, so it is read directly (fast, exact).",
    "ocr": "The page is an image or scan, so an OCR engine reads it.",
    "hybrid": "Part of the page is text and part is image; each part uses the right method.",
    "visual": "Charts or diagrams, described by the vision model.",
    "parser": "Office, spreadsheet or text file, read with a format parser.",
}


def apply_secrets(secrets) -> list[str]:
    """Copy known settings from a mapping (st.secrets) into the environment unless the
    environment already sets them. Returns the names that were applied."""
    applied = []
    try:
        keys = set(secrets.keys())
    except Exception:           # no secrets file configured
        return applied
    for k in SECRET_KEYS:
        if k in keys and not os.environ.get(k):
            os.environ[k] = str(secrets[k])
            applied.append(k)
    return applied


def apply_keys(mistral_key: str = "", gemini_key: str = "") -> None:
    """Keys typed into the app also switch on cloud vision OCR (handwriting). They live in this
    process only; this is meant for the single-user local app."""
    from .ocr.engines import set_capabilities
    mk, gk = clean_key(mistral_key), clean_key(gemini_key)
    if mk:
        os.environ["MISTRAL_API_KEY"] = mk
    if gk:
        os.environ["GEMINI_API_KEY"] = gk
    set_capabilities(None)                  # re-detect: cloud_vision may now be available


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def make_llm(provider: str | None = None):
    """Returns (llm, warning). Falls back to the offline extractive mode instead of crashing."""
    try:
        return get_llm(provider), None
    except Exception as exc:
        return ExtractiveLLM(), f"Could not start the language model ({exc}). Using offline extractive answers."


def make_embedder(name: str | None = None):
    return get_embedder(name)


def safe_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\- ()]", "_", name).strip(" .")
    return name or "upload"


def is_supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_EXT


@dataclass
class Session:
    """Everything one visitor owns."""
    engine: DocSearch
    workdir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="docsearch_")))
    files: dict[str, Path] = field(default_factory=dict)          # name -> saved copy
    reports: dict[str, IngestReport] = field(default_factory=dict)

    @property
    def documents(self) -> list[str]:
        return list(self.reports)

    def add(self, name: str, data: bytes, progress=None) -> IngestReport:
        name = safe_name(name)
        if not is_supported(name):
            return IngestReport(name, "failed", error=f"Unsupported file type ({Path(name).suffix or 'none'}).")
        path = self.workdir / name
        path.write_bytes(data)
        report = self.engine.ingest_file(path, source_name=name, replace=True, progress=progress)
        if report.status == "indexed":
            self.files[name] = path
            self.reports[name] = report
        return report

    def remove(self, name: str) -> None:
        self.engine.remove(name)
        path = self.files.pop(name, None)
        self.reports.pop(name, None)
        if path:
            path.unlink(missing_ok=True)

    def set_llm(self, llm) -> None:
        self.engine._llm = llm

    def ask(self, question: str, messages: list[dict], sources: list[str] | None = None,
            k: int | None = None, mode: str | None = None):
        """Answer using this visitor's own chat history. If the language model fails (bad API
        key, no network, rate limit) the question is answered in offline extractive mode and
        `answer.warning` says why, instead of giving the visitor an error."""
        try:
            return self._ask(question, messages, sources, k, mode)
        except Exception as exc:
            original, self.engine._llm = self.engine._llm, ExtractiveLLM()
            try:
                ans = self._ask(question, messages, sources, k, mode)
            finally:
                self.engine._llm = original
            ans.warning = f"{exc} Showing quoted passages instead."
            return ans

    def _ask(self, question, messages, sources, k, mode):
        history, pending = [], None
        for m in messages:
            if m["role"] == "user":
                pending = m["content"]
            elif pending is not None and m.get("found", True):
                history.append((pending, m["content"]))
                pending = None
        return self.engine.ask(question, sources=sources or None, history=history, k=k, mode=mode)


def new_session(embedder, llm, top_k: int = 4, retrieval: str = "hybrid") -> Session:
    return Session(DocSearch(index_dir=None, embedder=embedder, llm=llm, top_k=top_k, retrieval=retrieval))


def page_table(report: IngestReport) -> list[dict]:
    """One row per page for the 'how was this parsed' table."""
    rows = []
    for p in report.plans:
        rows.append({
            "Page": p.get("page"),
            "Route": ROUTE_LABEL.get(p.get("route"), p.get("route")),
            "Tools": ", ".join(p.get("tools") or []) or "-",
            "OCR confidence": p.get("ocr_confidence"),
            "Characters": p.get("chars"),
            "Why": "; ".join(p.get("reasons") or []),
        })
    return rows


def citation_rows(answer) -> list[dict]:
    return [c.to_dict() for c in answer.citations]


def run_evaluation(embedder, llm, samples_dir: str | Path, k: int = 4) -> dict:
    """Evaluate on the bundled sample document in a throw-away index (the visitor's own
    documents are not touched). Returns {"retrieval", "answers", "ocr"}."""
    from .evaluate import evaluate_end_to_end, evaluate_ocr, evaluate_retrieval, load_questions
    samples = Path(samples_dir)
    if not check_llm(llm)[0]:
        llm = ExtractiveLLM()               # a broken key must not break the evaluation page
    ds = DocSearch(index_dir=None, embedder=embedder, llm=llm, top_k=k)
    report = ds.ingest_file(samples / "mixed.pdf")
    if report.status != "indexed":
        raise RuntimeError(report.error or "could not index the sample document")
    items = load_questions(samples / "questions.json")
    out = {"retrieval": evaluate_retrieval(ds, items, k=k), "answers": evaluate_end_to_end(ds, items)}
    truth = samples / "ground_truth.json"
    if truth.exists():
        import json
        raw = json.loads(truth.read_text(encoding="utf-8"))
        pages = [{"file": "mixed.pdf", "page": int(k[4:]), "text": "\n".join(v)}
                 for k, v in raw.items() if re.fullmatch(r"page\d+", k)]
        if pages:
            out["ocr"] = evaluate_ocr(ds, pages, base_dir=samples)
    return out


def provider_status(llm) -> list[dict]:
    """One tiny request per provider: [{"name", "ok", "message"}]. The offline fallback is always ok."""
    out = []
    for p in getattr(llm, "providers", [llm]):
        if p.name.startswith("extractive"):
            out.append({"name": "offline extractive", "ok": True, "message": "always available (quotes passages)"})
            continue
        try:
            reply = p.chat([{"role": "user", "content": "Reply with the single word: OK"}],
                           temperature=0.0, max_tokens=8)
            out.append({"name": p.name, "ok": True, "message": f"replied: {reply.strip()[:30] or '(empty)'}"})
        except Exception as exc:
            out.append({"name": p.name, "ok": False, "message": str(exc)[:220]})
    return out


def check_llm(llm) -> tuple[bool, str]:
    """True if at least one real (non-offline) model answers."""
    real = [r for r in provider_status(llm) if r["name"] != "offline extractive"]
    if not real:
        return True, "Offline mode: no model to test."
    ok = [r for r in real if r["ok"]]
    if ok:
        return True, "; ".join(f"{r['name']} {r['message']}" for r in ok)
    return False, "; ".join(f"{r['name']}: {r['message']}" for r in real)


def build_llm(mistral_key: str = "", mistral_model: str = "", gemini_key: str = "", gemini_model: str = ""):
    """Chain from keys typed into the app (blank = use the environment). Returns (llm, errors)."""
    chain, errors = [], []
    mk = clean_key(mistral_key) or clean_key(os.getenv("MISTRAL_API_KEY"))
    if mk:
        chain.append(MistralLLM(api_key=mk, model=(mistral_model or "").strip() or None))
    for gk in gemini_keys(gemini_key):          # typed key first, then every key from the environment/secrets
        chain.append(GeminiLLM(api_key=gk, model=(gemini_model or "").strip() or None))
    from .rag.llm import OllamaLLM, ollama_reachable
    if ollama_reachable():
        try:
            chain.append(OllamaLLM())
        except Exception as exc:
            errors.append(f"Ollama: {exc}")
    if not chain:
        return ExtractiveLLM(), errors
    return FallbackLLM(chain + [ExtractiveLLM()]), errors


def mistral_with_key(key: str, model: str | None = None):
    """Returns (llm, error). Does not touch the environment."""
    try:
        return MistralLLM(api_key=key, model=(model or "").strip() or None), None
    except Exception as exc:
        return None, str(exc)


# ----------------------------------------------------------------------------- demo helpers
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "test_documents"

SAMPLE_QUESTIONS = {
    "01_leave_policy_digital.pdf": "How many days of maternity leave are given?",
    "02_invoice_scanned.pdf": "What is the grand total of the invoice?",
    "03_receipt_photo.jpg": "What is the total paid?",
    "04_meeting_minutes.docx": "When will the annual tech fest be held?",
    "05_product_launch.pptx": "How much is the development budget?",
    "06_sales_data.xlsx": "How many units were sold in the West region?",
    "07_students.csv": "What is Kavya Nair's CGPA?",
    "08_library_faq.html": "What is the late fee for overdue books?",
    "09_lab_safety.md": "What is the emergency contact number?",
    "10_enrollment_chart.png": "What is the enrollment of the Civil department?",
    "11_mixed_report.pdf": "How much was recovered from Blue Ocean Ltd?",
}


def sample_files() -> list[Path]:
    if not SAMPLE_DIR.is_dir():
        return []
    return [p for p in sorted(SAMPLE_DIR.iterdir()) if p.is_file() and is_supported(p.name) and not p.name.startswith("QUESTIONS")]


def suggested_questions(documents: list[str]) -> list[str]:
    out = ["What is this document about?"]
    out += [SAMPLE_QUESTIONS[d] for d in documents if d in SAMPLE_QUESTIONS][:3]
    if len(out) == 1:
        out += ["What are the main points?", "Are there any tables or figures? What do they show?"]
    return out


def page_summary(report: IngestReport, page_no: int | None) -> dict:
    """The plan and text for one page of a report (page_no None = whole file)."""
    plan = next((p for p in report.plans if p.get("page") == page_no), report.plans[0] if report.plans else {})
    text = next((t for n, t in report.page_texts if n == page_no), "")
    return {"plan": plan, "text": text}