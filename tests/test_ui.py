"""Tests for the Streamlit app: the logic in docsearch/ui.py, and a full run of app.py
against a stand-in for the `streamlit` module (so no browser or Streamlit install is needed)."""

from __future__ import annotations

import functools
import os
import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "tests" / "samples"


# ----------------------------------------------------------------- stand-in for streamlit
class _Dummy:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __bool__(self): return False
    def __getattr__(self, name): return lambda *a, **k: _Dummy()
    def __iter__(self): return iter(())


class _State(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


class _Rerun(Exception):
    pass


class FakeFile:
    def __init__(self, path: Path):
        self.name, self._data = path.name, path.read_bytes()
    def getvalue(self): return self._data


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _State()
        self.secrets = {}
        self.sidebar = _Dummy()
        self.rendered: list[str] = []
        self.errors: list[str] = []
        self.next_files: list = []
        self.next_question: str | None = None
        self.pressed: set[str] = set()
        self._cache: dict = {}

    # caching decorators work like Streamlit's (persist across reruns)
    def _cacher(self, fn=None, **kw):
        def deco(f):
            @functools.wraps(f)
            def wrapper(*a):
                key = (f.__qualname__, a)
                if key not in self._cache:
                    self._cache[key] = f(*a)
                return self._cache[key]
            return wrapper
        return deco(fn) if callable(fn) else deco
    cache_resource = cache_data = _cacher

    def rerun(self): raise _Rerun()
    def columns(self, spec, **k): return [_Dummy() for _ in range(spec if isinstance(spec, int) else len(spec))]
    def tabs(self, labels): return [_Dummy() for _ in labels]
    def file_uploader(self, *a, **k):
        files, self.next_files = self.next_files, []
        return files
    def button(self, label, *a, **k): return any(p in label for p in self.pressed)
    def chat_input(self, *a, **k):
        q, self.next_question = self.next_question, None
        return q
    def selectbox(self, label, options, *a, **k): return options[0]
    def slider(self, label, lo, hi, value=None, *a, **k): return value if value is not None else lo
    def text_area(self, *a, **k): return ""
    def multiselect(self, *a, **k): return []
    def text_input(self, *a, **k): return ""
    def toggle(self, *a, **k): return False
    def markdown(self, text, *a, **k): self.rendered.append(str(text))
    def error(self, text, *a, **k): self.errors.append(str(text))
    def set_page_config(self, *a, **k): pass
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: _Dummy()


def _run_app(st: FakeStreamlit):
    sys.modules["streamlit"] = st
    for _ in range(5):                       # a rerun re-executes the script, like Streamlit
        try:
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
            return
        except _Rerun:
            continue
    raise AssertionError("app kept rerunning")


def _env():
    os.environ.update(EMBEDDER="hashing", LLM_PROVIDER="extractive", MISTRAL_API_KEY="")
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------- tests
def test_app_full_flow():
    _env()
    st = FakeStreamlit()
    try:
        _run_app(st)                                             # empty app renders
        assert st.session_state.session.documents == []

        st.next_files, st.pressed = [FakeFile(SAMPLES / "mixed.pdf")], {"Process"}
        _run_app(st)                                             # upload + process + rerun
        s = st.session_state.session
        assert s.documents == ["mixed.pdf"]
        assert [p["route"] for p in s.reports["mixed.pdf"].plans] == ["native", "ocr", "hybrid", "ocr"]

        st.pressed = set()
        st.next_question = "What was the haemoglobin result?"
        _run_app(st)                                             # ask
        m = st.session_state.messages
        assert [x["role"] for x in m] == ["user", "assistant"]
        assert "10.2" in m[1]["content"] and m[1]["citations"][0]["page"] == 4

        st.next_question = "Who won the football world cup?"
        _run_app(st)
        assert st.session_state.messages[-1]["found"] is False
        assert not st.errors
    finally:
        sys.modules.pop("streamlit", None)


def test_session_helpers(tmp_path):
    _env()
    from docsearch import ui
    emb = ui.make_embedder("hashing")
    llm, warn = ui.make_llm("extractive")
    assert warn is None
    s = ui.new_session(emb, llm)

    bad = s.add("virus.exe", b"MZ")
    assert bad.status == "failed" and "Unsupported" in bad.error
    assert ui.safe_name("../../etc/pass wd.pdf") == "pass wd.pdf"

    r = s.add("mixed.pdf", (SAMPLES / "mixed.pdf").read_bytes())
    assert r.status == "indexed" and ui.page_table(r)[1]["Route"] == "OCR"

    msgs = [{"role": "user", "content": "Who is the patient?"},
            {"role": "assistant", "content": "x", "found": True}]
    assert s.ask("What unit is that measured in?", msgs).standalone_question

    s.remove("mixed.pdf")
    assert s.documents == [] and s.engine.stats()["chunks"] == 0
    assert not (s.workdir / "mixed.pdf").exists()


def test_secrets_bridge():
    from docsearch import ui
    saved = {k: os.environ.pop(k, None) for k in ("MISTRAL_API_KEY", "MISTRAL_MODEL")}
    os.environ["MISTRAL_MODEL"] = "keep-me"
    try:
        assert ui.apply_secrets({"MISTRAL_API_KEY": "k", "MISTRAL_MODEL": "other", "UNRELATED": "x"}) == ["MISTRAL_API_KEY"]
        assert os.environ["MISTRAL_API_KEY"] == "k" and os.environ["MISTRAL_MODEL"] == "keep-me"
        assert "UNRELATED" not in os.environ
        assert ui.apply_secrets(object()) == []                 # no secrets file: no crash
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_bad_llm_config_falls_back():
    _env()
    from docsearch import ui
    os.environ["MISTRAL_API_KEY"] = ""
    llm, warn = ui.make_llm("mistral")                          # provider forced but no key
    assert llm.name.startswith("extractive") and "Could not start" in warn


def test_evaluation_tab_logic():
    _env()
    from docsearch import ui
    res = ui.run_evaluation(ui.make_embedder("hashing"), ui.make_llm("extractive")[0], SAMPLES, k=4)
    assert res["retrieval"]["hybrid"]["recall@4"] == 1.0
    assert res["answers"]["invalid_citations"] == 0 and "cer" in res["ocr"]


class _BrokenLLM:
    name = "mistral:mistral-small-latest"
    def chat(self, *a, **k):
        raise RuntimeError("Mistral rejected the API key (401).")


def test_rejected_key_falls_back_to_quoted_answer():
    _env()
    from docsearch import ui
    s = ui.new_session(ui.make_embedder("hashing"), _BrokenLLM())
    s.add("mixed.pdf", (SAMPLES / "mixed.pdf").read_bytes())
    ans = s.ask("What was the haemoglobin result?", [])
    assert "10.2" in ans.answer and "401" in ans.warning
    assert s.engine._llm.name == "mistral:mistral-small-latest"      # the real model is restored
    ok, msg = ui.check_llm(_BrokenLLM())
    assert not ok and "401" in msg
    assert ui.run_evaluation(ui.make_embedder("hashing"), _BrokenLLM(), SAMPLES)["answers"]["answer_accuracy"] > 0


def test_key_is_cleaned():
    from docsearch.rag.llm import clean_key
    assert clean_key('  "sk-abc123"  ') == "sk-abc123"
    assert clean_key("Bearer sk-abc") == "sk-abc" and clean_key(None) is None


def test_sample_loader_and_parsing_tab():
    """'Load sample documents' parses all 11 test files; the parsing tab then renders page text for each."""
    _env()
    st = FakeStreamlit()
    try:
        _run_app(st)
        st.pressed = {"sample documents"}
        _run_app(st)
        s = st.session_state.session
        assert len(s.documents) == 11
        assert s.reports["02_invoice_scanned.pdf"].page_texts[0][1].count("INV-2026-0417") == 1
        st.pressed = set()
        for name in s.documents:                      # the parsing tab must render for every document
            st.parse_doc = name
            _run_app(st)
        assert not st.errors
    finally:
        sys.modules.pop("streamlit", None)
