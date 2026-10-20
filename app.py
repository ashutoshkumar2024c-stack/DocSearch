"""
DocSearch: Intelligent Document Parsing & Semantic Search Engine (Hybrid OCR-RAG pipeline)
Streamlit application.   Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st

from docsearch import theme, ui
from docsearch.ocr import SUPPORTED_EXT, capabilities
from docsearch.ocr.render import render_page

ROOT = Path(__file__).parent

st.set_page_config(page_title="DocSearch · OCR + RAG", page_icon="📄", layout="wide")

# settings: .env for local use, Streamlit secrets on Streamlit Cloud
ui.load_env()
try:
    ui.apply_secrets(st.secrets)
except Exception:
    pass


# ---------------------------------------------------------------- shared resources
@st.cache_resource(show_spinner="Loading the embedding model (first run downloads about 90 MB)...")
def get_embedder():
    return ui.make_embedder()


@st.cache_data(show_spinner=False, max_entries=96)
def page_image(path: str, page: int | None, dpi: int = 100):
    try:
        return render_page(path, page, dpi=dpi)
    except Exception:
        return None


def session() -> ui.Session:
    if "session" not in st.session_state:
        llm, warning = ui.make_llm()
        st.session_state.llm, st.session_state.llm_warning = llm, warning
        st.session_state.session = ui.new_session(get_embedder(), llm)
        st.session_state.messages = []
        st.session_state.uploader_key = 0
        st.session_state.dark = False
    return st.session_state.session


S = session()
if getattr(get_embedder(), "note", None):
    st.warning(get_embedder().note)
LLM = st.session_state.llm
EXTS = sorted(e.lstrip(".") for e in SUPPORTED_EXT)


def run_ingest(items: list[tuple[str, bytes]]) -> None:
    """Parse + index files with a live progress display, then refresh the page."""
    with st.status("Parsing documents...", expanded=True) as status:
        bar = st.progress(0.0)
        for i, (name, data) in enumerate(items):
            def step(msg, frac, _i=i, _n=len(items), _name=name):
                status.update(label=f"[{_i + 1}/{_n}] {_name}: {msg}")
                bar.progress(min(1.0, (_i + frac) / _n))

            report = S.add(name, data, progress=step)
            if report.status == "indexed":
                routes = ", ".join(f"{theme.route_label(k)} {n}" for k, n in report.routes.items())
                st.write(f"✅ **{name}**: {report.summary.get('doc_type')} · {report.pages} page(s) · {routes} · {report.seconds:.1f}s")
            else:
                st.write(f"❌ **{name}**: {report.error or report.status}")
        status.update(label="Finished", state="complete", expanded=False)
    st.session_state.uploader_key += 1
    st.rerun()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 📄 DocSearch")
    st.toggle("🌙 Dark mode", key="dark")
    dark = bool(st.session_state.get("dark"))
    st.markdown(theme.css(dark), unsafe_allow_html=True)

    st.markdown("##### 1 · Add documents")
    files = st.file_uploader("PDF, Word, PowerPoint, Excel, CSV, HTML, text or images", type=EXTS,
                             accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}",
                             label_visibility="collapsed")
    if files and st.button(f"Process {len(files)} file(s)", type="primary", use_container_width=True):
        run_ingest([(f.name, f.getvalue()) for f in files])

    samples = [p for p in ui.sample_files() if p.name not in S.documents]
    if samples and st.button(f"✨ Load {len(samples)} sample documents", use_container_width=True,
                             help="One of every document type: digital PDF, scan, photo, Word, PowerPoint, Excel, CSV, HTML, Markdown, chart."):
        run_ingest([(p.name, p.read_bytes()) for p in samples])

    st.markdown("##### 2 · Library")
    if not S.documents:
        st.caption("No documents yet.")
    for name in S.documents:
        r = S.reports[name]
        st.markdown(theme.doc_card(name, r.summary.get("doc_type", ""), r.pages, r.chunks, r.routes), unsafe_allow_html=True)
        if st.button("Remove", key=f"rm_{name}", use_container_width=True):
            S.remove(name)
            st.rerun()

    scope = []
    if len(S.documents) > 1:
        scope = st.multiselect("Ask only in", S.documents, placeholder="All documents")

    with st.expander("Retrieval settings"):
        mode = st.selectbox("Method", ["hybrid", "dense", "bm25"],
                            help="hybrid = meaning (embeddings) + keywords (BM25), fused with Reciprocal Rank Fusion.")
        top_k = st.slider("Passages per answer", 2, 12, 4)

    with st.expander("Language models"):
        st.caption("Answers are written by Mistral, then Gemini, then a local Ollama model, whichever works first. "
                   "With none, the app quotes passages. The same keys also let the app read handwriting: pages that local OCR "
                   "cannot read are sent to Gemini (or Mistral OCR). Keys set by the host (Secrets / .env) are used automatically; "
                   "typing one here is only for local use.")
        mk = st.text_input("Mistral API key", type="password")
        mm = st.text_input("Mistral model", value=os.getenv("MISTRAL_MODEL", "mistral-small-latest"))
        gk = st.text_input("Gemini (Google AI Studio) API key", type="password")
        gm = st.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-flash-latest"))
        if st.button("Apply and test", use_container_width=True):
            with st.spinner("Testing..."):
                ui.apply_keys(mk, gk)
                new_llm, _ = ui.build_llm(mk, mm, gk, gm)
                st.session_state.provider_status = ui.provider_status(new_llm)
            S.set_llm(new_llm)
            st.session_state.llm = new_llm
            st.rerun()
        for r in st.session_state.get("provider_status", []):
            st.markdown(theme.badge("✓ " + r["name"] if r["ok"] else "✗ " + r["name"], "ds-ok" if r["ok"] else "ds-bad")
                        + f'<div class="ds-model">{html.escape(r["message"])}</div>', unsafe_allow_html=True)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------- header
n_docs = len(S.documents)
n_pages = sum(r.pages for r in S.reports.values())
n_chunks = sum(r.chunks for r in S.reports.values())
route_totals: dict[str, int] = {}
tools_used: dict[str, int] = {}
for r in S.reports.values():
    for k, n in r.routes.items():
        route_totals[k] = route_totals.get(k, 0) + n
    for k, n in r.summary.get("tools", {}).items():
        tools_used[k] = tools_used.get(k, 0) + n
ocr_pages = route_totals.get("ocr", 0) + route_totals.get("hybrid", 0) + route_totals.get("visual", 0)

st.markdown(theme.hero(["Classify", "Choose tool", "Extract + layout", "Chunk", "Embed + index", "Hybrid retrieve", "Cited answer"]),
            unsafe_allow_html=True)
st.markdown(theme.stat_cards([
    ("Documents", n_docs, "in this session"),
    ("Pages parsed", n_pages, f"{route_totals.get('native', 0)} text layer · {route_totals.get('parser', 0)} parser"),
    ("Pages with OCR", ocr_pages, ", ".join(sorted(k for k in tools_used if k not in ("pdfplumber", "pdf-text"))) or "none yet"),
    ("Passages indexed", n_chunks, "searchable chunks"),
]), unsafe_allow_html=True)


# ---------------------------------------------------------------- answers
def show_answer(msg: dict, key: str) -> None:
    st.markdown(msg["content"])
    if msg.get("standalone"):
        st.caption(f"Understood as: *{msg['standalone']}*")
    if msg.get("invalid"):
        st.warning(f"The model cited sources that do not exist: {msg['invalid']}")
    cites = msg.get("citations") or []
    if cites:
        with st.expander(f"Sources ({len(cites)})"):
            for c in cites:
                kind = {"table": "table", "figure": "figure", "outline": "outline"}.get(c["element"], "text")
                st.markdown(theme.badge(f"[{c['number']}]") + theme.badge(c["source"]) +
                            (theme.badge(f"page {c['page']}") if c["page"] else "") + theme.badge(kind) +
                            (theme.badge("OCR", "ocr") if c.get("ocr_used") else ""), unsafe_allow_html=True)
                if c["section"]:
                    st.caption(f"Section: {c['section']}")
                st.code(c["snippet"], language=None)
                path = S.files.get(c["source"])
                if path is not None and path.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
                    if st.toggle("Show the original page", key=f"pg_{key}_{c['number']}"):
                        img = page_image(str(path), c["page"])
                        if img is not None:
                            st.image(img, use_container_width=True)
    if msg.get("model"):
        st.markdown(f'<div class="ds-model">Answered by {html.escape(msg["model"])} · {len(cites)} source(s) cited</div>',
                    unsafe_allow_html=True)


question = st.chat_input("Ask a question about your documents...", disabled=not S.documents)
queued = st.session_state.pop("queued", None)
question = question or queued

tab_chat, tab_parse, tab_search, tab_eval, tab_sys = st.tabs(
    ["💬 Ask", "🔬 How it was parsed", "🔎 Search", "📊 Evaluate", "⚙️ System"])

# ---- Ask
with tab_chat:
    if not S.documents:
        st.markdown('<div class="ds-empty"><div class="big">📥</div><b>Add a document to begin</b><br>'
                    'Upload files in the sidebar, or load the sample documents to see every parsing route in action.</div>',
                    unsafe_allow_html=True)
        if ui.sample_files() and st.button("✨ Load sample documents", type="primary", key="empty_samples"):
            run_ingest([(p.name, p.read_bytes()) for p in ui.sample_files()])
    elif not st.session_state.messages and not question:
        st.markdown("**Try a question**")
        cols = st.columns(3)
        for col, q in zip(cols, ui.suggested_questions(S.documents)):
            if col.button(q, key=f"sq_{q}", use_container_width=True):
                st.session_state.queued = q
                st.rerun()
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                show_answer(m, f"h{i}")
            else:
                st.markdown(m["content"])

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching and answering..."):
                try:
                    ans = S.ask(question, st.session_state.messages, sources=scope, k=top_k, mode=mode)
                    msg = {"role": "assistant", "content": ans.answer, "found": ans.found,
                           "citations": ui.citation_rows(ans), "standalone": ans.standalone_question,
                           "invalid": ans.invalid_citations, "model": S.engine.llm.name}
                except Exception as exc:
                    msg = {"role": "assistant", "found": False, "citations": [],
                           "content": f"Something went wrong while answering: {exc}"}
            show_answer(msg, "new")
        st.session_state.messages += [{"role": "user", "content": question}, msg]

# ---- How it was parsed
with tab_parse:
    if not S.documents:
        st.markdown('<div class="ds-empty"><div class="big">🔬</div>Process a document to see how each page was handled: '
                    'which tool the router chose, why, and the text it extracted.</div>', unsafe_allow_html=True)
    else:
        name = st.selectbox("Document", S.documents, key="parse_doc")
        r = S.reports[name]
        conf = r.summary.get("ocr_confidence")
        st.markdown(theme.stat_cards([
            ("Document type", r.summary.get("doc_type", "?"), "detected automatically"),
            ("Pages", r.pages, f"{r.seconds:.1f}s to parse"),
            ("OCR confidence", f"{conf:.0f}%" if conf is not None else "n/a", "mean word confidence"),
            ("Passages", r.chunks, ", ".join(f"{k} {v}" for k, v in r.summary.get("elements", {}).items()) or "-"),
        ]), unsafe_allow_html=True)
        st.markdown("**Routes** " + "".join(theme.route_badge(k, n) for k, n in r.routes.items()) +
                    " &nbsp; **Tools** " + "".join(theme.badge(f"{k} · {v}") for k, v in r.summary.get("tools", {}).items()),
                    unsafe_allow_html=True)

        weak = [p_.get("page") for p_ in r.plans
                if p_.get("ocr_confidence") is not None and p_["ocr_confidence"] < 60]
        if weak:
            cloud_ok = bool(capabilities().get("cloud_vision", {}).get("available"))
            st.warning(f"{len(weak)} page(s) were read with low confidence ({', '.join(str(w) for w in weak[:12])}"
                       f"{'…' if len(weak) > 12 else ''}). This usually means handwriting or a poor scan."
                       + ("" if cloud_ok else " Add a Gemini key under Language models and click Apply and test to read them."))
            if cloud_ok and st.button(f"🔁 Re-read {len(weak)} weak page(s) with cloud vision", key=f"reread_{name}", type="primary"):
                run_ingest([(name, S.files[name].read_bytes())])

            # live diagnosis: shows exactly why cloud reading does or does not work, instead of failing silently
            with st.expander("🔍 Why are these pages still weak? (diagnose cloud reading)"):
                caps_now = capabilities()
                st.write("cloud_vision:", caps_now.get("cloud_vision"))
                for p_ in r.plans:
                    if p_.get("page") in weak[:6]:
                        st.caption(f"Page {p_.get('page')}: " + " | ".join(p_.get("reasons") or ["(no notes)"]))
                if st.button("Test: read the first weak page now", key=f"diag_{name}"):
                    try:
                        from docsearch.ocr.engines import CloudVisionEngine, HANDWRITING_PROMPT
                        pg = weak[0] if len(r.plans) > 1 else None
                        img_ = page_image(str(S.files[name]), pg, 150)
                        if img_ is None:
                            raise RuntimeError("could not render the page image")
                        eng_ = CloudVisionEngine()
                        out_ = eng_._run_uncached(img_, HANDWRITING_PROMPT)
                        st.success(f"Worked via {eng_.used}. First lines:")
                        st.code(out_[:600])
                    except Exception as exc_:
                        st.error(f"Cloud reading failed: {exc_}")

        paged = len(r.plans) > 1
        page = 1
        if paged:
            page = st.slider("Page", 1, len(r.plans), 1, key=f"pg_{name}")
        st.markdown(theme.page_strip([p.get("route", "parser") for p in r.plans], page if paged else None),
                    unsafe_allow_html=True)

        info = ui.page_summary(r, page if paged else (r.plans[0].get("page") if r.plans else None))
        plan = info["plan"]
        left, right = st.columns([1, 1])
        with left:
            path = S.files.get(name)
            img = page_image(str(path), plan.get("page") if paged else None, 90) if path is not None else None
            if img is not None:
                st.image(img, caption=f"{name}" + (f", page {page}" if paged else ""), use_container_width=True)
            else:
                st.markdown('<div class="ds-empty">This format has no page image (it is read by a parser).</div>',
                            unsafe_allow_html=True)
        with right:
            c = plan.get("ocr_confidence")
            st.markdown(theme.route_badge(plan.get("route", "parser")) +
                        "".join(theme.badge(t) for t in plan.get("tools", [])) +
                        (theme.badge(f"OCR confidence {c:.0f}%") if c is not None else "") +
                        theme.badge(f"{plan.get('chars', 0)} characters"), unsafe_allow_html=True)
            st.markdown("**Why this tool**")
            for why in plan.get("reasons") or ["(no notes)"]:
                st.markdown(f'<div class="ds-reason">{html.escape(why)}</div>', unsafe_allow_html=True)
            if plan.get("evidence", {}).get("image"):
                with st.expander("Image analysis (what the router measured)"):
                    st.json(plan["evidence"]["image"])
            st.text_area("Extracted text", value=info["text"], height=300, key=f"ta_{name}_{page}",
                         help="Exactly what was read from this page and sent to the index.")

        with st.expander("All pages"):
            rows = [{"Page": p.get("page") or "-", "Route": theme.route_badge(p.get("route", "parser")),
                     "Tools": ", ".join(p.get("tools") or []) or "-",
                     "OCR conf.": f"{p['ocr_confidence']:.0f}%" if p.get("ocr_confidence") is not None else "-",
                     "Characters": p.get("chars")} for p in r.plans]
            st.markdown(theme.table(rows, raw_cols=("Route",)), unsafe_allow_html=True)

# ---- Search
with tab_search:
    st.caption("Retrieval only, no answer is written: see exactly which passages are found and how they score.")
    q = st.text_input("Search query", key="search_q", placeholder="e.g. maternity leave days")
    if q and S.documents:
        hits = S.engine.retrieve(q, k=top_k, mode=mode, sources=scope or None)
        if not hits:
            st.info("Nothing found.")
        for n, h in enumerate(hits, 1):
            md = h.chunk.metadata
            st.markdown(theme.badge(f"#{n}") + theme.badge(md.get("source", "?")) +
                        (theme.badge(f"page {md['page']}") if md.get("page") else "") +
                        theme.badge(md.get("element", "text")) + theme.badge(f"score {h.score:.3f}") +
                        (theme.badge("OCR", "ocr") if md.get("ocr_used") else ""), unsafe_allow_html=True)
            st.code(h.chunk.text[:700], language=None)
    elif q:
        st.info("Add a document first.")

# ---- Evaluate
with tab_eval:
    st.caption("Measures retrieval and answer quality on the bundled sample (a mixed digital + scanned PDF with known "
               "answers). Your own documents are not affected.")
    samples_dir = ROOT / "tests" / "samples"
    if not (samples_dir / "questions.json").exists():
        st.info("Sample files are not part of this deployment.")
    elif st.button("Run evaluation", type="primary"):
        with st.spinner("Parsing the sample and scoring questions..."):
            st.session_state.eval = ui.run_evaluation(get_embedder(), LLM, samples_dir, k=top_k)
    res = st.session_state.get("eval")
    if res:
        key = next(k for k in res["retrieval"]["hybrid"] if k.startswith("recall@"))
        st.markdown("**Retrieval**")
        st.markdown(theme.table([{"Method": m, key: f"{v[key]:.2f}", "MRR": f"{v['mrr']:.2f}"}
                                 for m, v in res["retrieval"].items()]), unsafe_allow_html=True)
        a = res["answers"]
        ab = a["abstention_accuracy"]
        st.markdown(theme.stat_cards([
            ("Answer accuracy", f"{a['answer_accuracy']:.0%}", "correct and cited"),
            ("Citation precision", f"{a['citation_precision']:.0%}", "cites the right file and page"),
            ("Correct refusal", "n/a" if ab is None else f"{ab:.0%}", "unanswerable questions"),
            ("Invalid citations", a["invalid_citations"], "should be 0"),
        ]), unsafe_allow_html=True)
        if res.get("ocr"):
            o = res["ocr"]
            st.markdown(f"**OCR accuracy:** character error rate **{o['cer']:.1%}**, word error rate **{o['wer']:.1%}**")
            st.markdown(theme.table([{"Route": k, "Pages": v["n"], "CER": f"{v['cer']:.1%}", "WER": f"{v['wer']:.1%}"}
                                     for k, v in o["by_route"].items()]), unsafe_allow_html=True)
        with st.expander("Per-question results"):
            st.markdown(theme.table([{"Question": r["question"], "Type": r["element"],
                                      "Result": theme.badge("correct", "ds-ok") if r["correct"] else theme.badge("wrong", "ds-bad")}
                                     for r in a["per_question"]], raw_cols=("Result",)), unsafe_allow_html=True)

# ---- System
with tab_sys:
    st.markdown("**Parsing tools on this machine**")
    st.caption("The router only assigns tools that are installed and falls back to Tesseract automatically.")
    purpose = {"tesseract": "Main OCR for clean printed pages and scans", "pdfplumber": "Headings and tables in digital PDFs",
               "easyocr": "Blurry, faded or photographed pages", "paddleocr": "Chinese, Japanese, Korean text",
               "vision": "Charts, diagrams, handwriting, low-confidence retries (Ollama + Qwen3-VL)",
               "cloud_vision": "Handwriting and very poor scans (page images go to Gemini/Mistral; needs a key)",
               "gpu": "Speeds up the heavy models"}
    st.markdown(theme.table([{"Tool": k, "Status": theme.badge("available", "ds-ok") if v.get("available") else theme.badge("not installed", "ds-bad"),
                              "Used for": purpose.get(k, ""), "Detail": v.get("version") or v.get("error") or ""}
                             for k, v in capabilities().items()], raw_cols=("Status",)), unsafe_allow_html=True)

    st.markdown("<br>**Answering**", unsafe_allow_html=True)
    if st.button("Test language models now"):
        with st.spinner("Testing..."):
            st.session_state.provider_status = ui.provider_status(LLM)
    for r in st.session_state.get("provider_status", []):
        st.markdown(theme.badge(("✓ " if r["ok"] else "✗ ") + r["name"], "ds-ok" if r["ok"] else "ds-bad") +
                    f' <span class="ds-muted">{html.escape(r["message"])}</span>', unsafe_allow_html=True)
    for prov, err in getattr(LLM, "errors", {}).items():
        st.caption(f"Last problem with {prov}: {err[:300]}")
    st.write(f"**Active model:** `{LLM.name}`  ·  **Embedding model:** `{get_embedder().name}`")
    stats = S.engine.stats()
    st.write(f"**Index (this session):** {stats['documents']} document(s), {stats['chunks']} passages "
             + ", ".join(f"{v} {k}" for k, v in stats["elements"].items()))
    st.caption("Accepted files: " + " ".join(f".{e}" for e in EXTS))