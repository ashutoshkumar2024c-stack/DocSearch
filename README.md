# DocSearch: Intelligent Document Parsing & Semantic Search Engine

A hybrid OCR–RAG application. Upload digital PDFs, scanned PDFs, phone photos, Word, PowerPoint, Excel,
CSV or HTML files, all mixed together. Every page is classified and routed to the right tool automatically.
Then ask questions in plain language and get answers that **cite the exact file and page**, with the
page image one click away.

```
 upload ─► 1 classify & route ─► 2 extract + layout ─► 3 chunk ─► 4 embed & index ─► 5 retrieve & answer
           per document/page      text layer, OCR,        by section,   MiniLM + FAISS      hybrid: meaning + keywords
           (picks the tool)       tables, figures         table rows    + BM25 (+ ColPali)   → LLM → cited answer
```

## Start it

**Windows:** install [Python 3.10+](https://www.python.org/downloads/) and
[Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (tick *Add to PATH*), then double-click **`start.bat`**.
**macOS / Linux:** `brew install tesseract` or `sudo apt install tesseract-ocr`, then `./start.sh`.

The first run creates a virtual environment, installs the requirements and starts the Streamlit app (it opens in your browser, normally at http://localhost:8501).

Manual setup:

```bash
python -m venv venv && venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                                # macOS/Linux: cp .env.example .env
streamlit run app.py
```

### Deploy on Streamlit Community Cloud
1. Push the project to GitHub (**do not commit `.env` or `.streamlit/secrets.toml`**; both are in `.gitignore`).
2. On share.streamlit.io choose the repo and set the main file to `app.py`.
3. In *App settings > Secrets* add `MISTRAL_API_KEY = "..."`.

`packages.txt` makes Streamlit Cloud install the Tesseract program. Each visitor gets their own in-memory
index, because the Cloud filesystem is temporary. EasyOCR, PaddleOCR and Ollama are not available there, so
the router uses Tesseract and the Mistral API.

### What the app shows
- **Ask:** cited answers; click *Show the original page* under any source.
- **How it was parsed:** pick a document and a page; see the page image next to the extracted text, the tool the router chose and why, the OCR confidence and the image measurements.
- **Sidebar:** *Load sample documents* parses one file of every type (digital PDF, scan, photo, Word, PowerPoint, Excel, CSV, HTML, Markdown, chart). `test_documents/QUESTIONS.md` lists questions with answers.
- **Dark mode:** the toggle at the top of the sidebar.

### Choose how answers are written (in `.env`, or Streamlit secrets)
| Option | Set | Notes |
|---|---|---|
| Mistral API | `MISTRAL_API_KEY=...` | Free key at console.mistral.ai. Fast and good quality. |
| Gemini (backup) | `GEMINI_API_KEY=...` (a Google AI Studio key) | Used automatically if Mistral fails or is rate-limited |
| Local model | install [Ollama](https://ollama.com), `ollama pull qwen3:4b` | Nothing leaves the machine |
| None | (nothing) | Offline mode: answers quote the best-matching passages, with citations |

**Recommended extras:** `ollama pull qwen3-vl:2b` lets the router describe charts and diagrams, read
handwriting, and retry low-confidence pages. `pip install -r requirements-optional.txt` adds EasyOCR and
PaddleOCR for poor photos and Asian scripts. The **System** page shows what is installed. The router only
assigns tools that are present, and falls back automatically when one is missing.

## The application

| Page | What it does |
|---|---|
| **Library** | Drag-and-drop upload of any mix of files; background processing with live progress; each document shows a *page strip* coloured by how each page was read |
| **Document** | Per page: preview, route, the tools that ran, *why* they were chosen, OCR confidence, and the headings, tables and figures found; all indexed passages |
| **Ask** | Conversations with follow-up questions; inline `[n]` citations; selecting one opens the cited page image and passage; filter by document, retrieval method and number of passages |
| **Search** | Retrieval without an LLM; keyword vs meaning vs hybrid side by side |
| **Evaluate** | Recall@k / MRR per retrieval method, broken down **by element type** (text, table, figure) and **by route** (native, OCR, hybrid); optional answer accuracy, citation precision and abstention; run history |
| **System** | Installed engines and models, configuration, index statistics |

Command line (same pipeline):

```bash
python -m docsearch ingest docs/*.pdf scans/*.jpg
python -m docsearch ask "What was the haemoglobin result?"
python -m docsearch chat
python -m docsearch search "INV-2026-0042" --mode bm25
python -m docsearch eval tests/samples/questions.json --with-llm
python -m docsearch.ocr report.pdf                 # show the plan and tools chosen for each page
python -m docsearch.ocr --capabilities
```

## How it works

### 1. Classification and automatic tool assignment (`docsearch/ocr/router.py`)
**By file type.** Word, PowerPoint, Excel, CSV and HTML go to their native parsers. Their embedded
pictures (diagrams, charts, screenshots) are sent through the same image routing as scans.

**By PDF page**, using evidence from the PDF itself:

| Evidence | Decision |
|---|---|
| no text layer, or a broken one (unmapped glyphs) | **OCR**: render at 300 DPI and choose an engine (below) |
| good text layer, images ≥ 20 % of the page | **Hybrid**: text layer plus figure routing for each image |
| full-page image with a good text layer | **Native**: an already-OCR'd searchable scan; use its layer |
| otherwise | **Native**: pdfplumber layout analysis |

**By image quality** (`quality.py`: sharpness, contrast, character size, lighting, colour, skew,
orientation and script detection):

| Image | Tool |
|---|---|
| chart / diagram, little text | vision model **describes** it (labels, values, takeaway), or Tesseract sparse mode with graphics removed |
| non-Latin script | Tesseract with that language pack, then PaddleOCR / EasyOCR, then the vision model |
| blurry, faded, photographed | EasyOCR, then PaddleOCR, then Tesseract with enhancement |
| clean print | Tesseract with ruled-table detection |
| any page with confidence < 60 | retried with the fallback engine; the better result is kept |
| an engine crashes or is offline | degrades to Tesseract; the reason is recorded |

Upside-down and rotated pages are turned upright first. Every decision is stored with its reasons and
shown in the app.

### 2. Layout analysis (`docsearch/ocr/layout.py`, `office.py`)
Digital pages: font size and weight give headings, gaps give paragraphs, and ruling lines give tables
(kept as Markdown). Scans: OpenCV finds ruled tables and each cell is OCR'd separately; Tesseract's
block structure plus line heights give headings and paragraphs. Charts: bars and fills are removed,
then light and dark text are read separately, so labels are paired with values ("West 91%").

### 3. Chunking (`docsearch/rag/chunking.py`)
Chunks follow the layout. Each chunk carries its heading path, tables are split by rows with the header
repeated, figures are separate chunks, and nothing crosses a page boundary. Chunks are 800 characters,
because all-MiniLM-L6-v2 ignores input after 256 word pieces.

### 4. Indexing and retrieval (`docsearch/rag/store.py`, `embeddings.py`, `visual.py`)
MiniLM embeddings with FAISS (exact cosine search), plus BM25 with light stemming, combined by
Reciprocal Rank Fusion. Keyword matching protects exact codes and numbers that OCR'd text depends on.
Optional **ColPali** indexes *only* the pages the router marked visual, and adds a third ranking.

### 5. Answers (`docsearch/rag/answer.py`, `llm.py`)
Retrieved passages are numbered and labelled (file, page, table/figure, OCR). The model must cite `[n]`
after each claim, and must say *"I couldn't find this in the uploaded documents."* rather than guess.
Citations are mapped back to pages, and citations to sources that were never provided are flagged.
Follow-up questions are rewritten into standalone questions.

## Tests

```bash
python -m pytest -q          # 35 tests: routing, OCR quality, layout, chunking, retrieval, citations, API
```
They run offline: a hashing embedder and a scripted LLM stand in for the real models, and the
Mistral, Ollama, EasyOCR and PaddleOCR clients are tested against mocked responses.
`tests/make_samples.py` builds a 4-page PDF with ground truth: a digital page with a table, a noisy
rotated scan, a page with a chart, and a scanned ruled table.

## Project layout

```
docsearch/
  ocr/        router.py (classifier and tool assignment)  quality.py (image profiling)
              engines.py (Tesseract, EasyOCR, PaddleOCR, Qwen3-VL)  layout.py  office.py
              extract.py (runs the plans)  render.py  metrics.py (CER/WER)
  rag/        chunking.py  embeddings.py  store.py (FAISS + BM25 + RRF)  visual.py (ColPali)
              llm.py (Mistral, Ollama, offline)  answer.py (citations)
  pipeline.py      the five stages in one class
  evaluate.py      element- and route-level evaluation
app.py             the Streamlit application (Ask, How it was parsed, Search, Evaluate, System)
packages.txt       system packages for Streamlit Cloud (Tesseract)
.streamlit/        config.toml, secrets.toml.example
  ui.py            logic behind the Streamlit app (sessions, uploads, evaluation)
tests/             test_ocr.py  test_rag.py  test_ui.py  make_samples.py  samples/
docs/PROJECT_NOTES.md   report ↔ code map, literature gaps, comparison with the earlier code
```
