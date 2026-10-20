# Project notes: report ↔ code, literature gaps, and what changed

Use this for the Review 2 report and the viva. Every claim below points to the file that implements it.

## 1. Every component in the Review 1 report, and where it is

| Report says | Implemented in | Notes |
|---|---|---|
| Document Classification & Routing: native text layer vs image-only | `ocr/router.py` `plan_pdf_page` | Per **page**, not per file; also detects broken text layers, searchable scans and figure-heavy pages |
| Assign the parsing tool automatically by document type | `ocr/router.py` `choose_ocr`, `ocr/quality.py`, `ocr/extract.py` | File type → parser; page → route; image quality / script / kind → OCR engine; confidence → fallback. Reasons shown in the app |
| OCR + layout analysis engine (Tesseract, EasyOCR or PaddleOCR) extracting text **and structural cues such as headers and tables** | `ocr/engines.py`, `ocr/layout.py` | All three engines plus Qwen3-VL. Headings and tables from digital pages (pdfplumber) and scans (OpenCV ruled-table detection + Tesseract blocks) |
| Native PDFs parsed directly | `ocr/layout.py` `native_elements` | pdfplumber, with pypdfium2 fallback |
| Semantic chunking: overlapping, context-preserving segments | `rag/chunking.py` | Layout-aware: section heading paths, whole table rows with header repeated, figures separate, 800 chars / 120 overlap |
| sentence-transformers/all-MiniLM-L6-v2 | `rag/embeddings.py` `MiniLMEmbedder` | Multilingual variant via `ST_MODEL` |
| Vector database (ChromaDB or FAISS) | `rag/store.py` | FAISS `IndexFlatIP` (exact cosine), saved to disk |
| Top-k similarity search | `rag/store.py` `search` | Plus BM25 and Reciprocal Rank Fusion (hybrid) |
| LLM within RAG, answer with **source citations** | `rag/answer.py`, `rag/llm.py` | `[n]` citations mapped to file and page; invalid citations flagged; abstains when not found |
| Option to extend to vision-native (ColPali-style) retrieval for highly visual pages | `rag/visual.py` | Selective: only `visual` / `hybrid` pages are embedded; fused as a third ranking |
| Deployable on standard student / lab hardware | whole project | Defaults are CPU-only (Tesseract, MiniLM, FAISS). GPU tools are optional and detected automatically |
| Healthcare, legal and financial documents | `tests/make_samples.py` | Sample set: audit table, discharge record, contract figure, lab results scan |

## 2. Literature gaps → what this project does about each

| Gap (from the survey) | Our answer | Evidence you can show |
|---|---|---|
| **OCR-dependent pipelines (LayoutLMv3, SCAN) inherit OCR errors** | (a) Image profiling sends each page to the engine best suited to it; (b) per-page confidence with automatic retry on another engine; (c) BM25 + light stemming in hybrid retrieval tolerates OCR noise; (d) OCR provenance and confidence shown on every citation | Evaluate → "by route": OCR pages vs native pages; `evaluate_ocr` CER/WER per route; document page cards show confidence and fallbacks |
| **…and are expensive to fine-tune on annotated layout data (SCAN)** | Training-free layout analysis: font metrics, gaps, ruling lines, OpenCV table detection. Gives SCAN-style coherent regions (sections, whole tables, figures) as chunk boundaries | `ocr/layout.py`; Document → Indexed passages show section paths and Markdown tables |
| **OCR-free Donut loses accuracy on dense / tabular pages** | Tables are detected and kept as structured Markdown (header + rows), and never split mid-row | Evaluate → "by element: table"; sample questions on the digital table (p. 1) and the *scanned* table (p. 4) |
| **ColPali: heavy storage, latency, needs a GPU** | Vision processing only where the router says a page is visual: a VLM description for charts, and ColPali for visual/hybrid pages only. Text pages stay on the cheap path | `DocSearch.stats()["visual"]` gives pages and vectors indexed vs total pages; on the sample PDF ColPali would embed 1 of 4 pages |
| **No standardised, element-level, multilingual benchmark (Gao et al., 2026)** | The evaluation harness reports every metric **per element type** and **per route**, plus OCR CER/WER per route, on any question set. Script detection routes non-Latin pages to a suitable engine; a multilingual embedder can be configured | Evaluate page; `docsearch/evaluate.py`; `python -m docsearch eval questions.json --ocr-truth truth.json` |

Be precise in the viva: the project does **not** create a public benchmark. It provides an element-level
*evaluation method* and reports results that way, which is the gap the survey describes, done at a small scale.

## 3. Your two repositories vs this project

### OCR: V.A.U.L.T. (`document_pipeline.py`, `models/vision.py`)
| V.A.U.L.T. | Now |
|---|---|
| "OCR" was only Qwen3-VL 2B through Ollama: needs a GPU to be practical, can hallucinate, gives no confidence | Four engines chosen per page; Tesseract by default; Qwen3-VL used where it helps (figures, handwriting, low-confidence retry) |
| Routing per **file**: if any page had text, scanned pages in the same PDF were silently skipped | Routing per **page** with five routes and recorded reasons |
| `analyze_image(path, prompt)` ignored `prompt` | Prompt is sent; separate *describe* mode for figures |
| No preprocessing | Denoise, deskew, CLAHE, binarise, upscale, auto-rotate |
| Flat text, no structure | Headings, paragraphs, tables (Markdown), figures |
| Embedded images in DOCX/PPTX ignored | OCR'd or described through figure routing |
| Tightly coupled to its agents/config/Chroma | Standalone package |
| No tests | Unit tests with ground truth (CER/WER) |

### RAG: RAG-Chatbot (`rag.py`, `app.py`)
| RAG-Chatbot | Now |
|---|---|
| PDFs with a text layer only (PyPDFLoader); scans produced nothing | Any supported format through the router |
| One PDF; `rag.py` hard-codes a path on the author's laptop | Library of many files; add, remove and reprocess |
| FAISS rebuilt on every run | Persistent index; files processed once |
| 1000-char chunks cut anywhere (can exceed MiniLM's limit) | Layout-aware 800-char chunks |
| Dense retrieval only | Hybrid dense + BM25 (+ optional ColPali) |
| Answers without sources | `[n]` citations to file and page, page image viewer, invalid-citation check |
| `mistral-tiny` (retired alias), temperature 0.7 | `mistral-small-latest` at 0.1, or local Ollama, or offline mode |
| `ConversationalRetrievalChain` (deprecated, pinned to LangChain 0.2) | Same condense → retrieve → answer flow without LangChain |
| Streamlit demo with one PDF type, no sources | Streamlit app for any document type: per-page routing view, cited answers with page images, search, evaluation and system pages |
| No evaluation | Recall@k, MRR, element/route breakdown, answer accuracy, citation precision, abstention, CER/WER |

## 4. Results to report (fill in from your own runs)

Run on your own documents with a real embedder and LLM, then paste the numbers:

```bash
python -m docsearch ingest your_docs/*
python -m docsearch eval your_questions.json --with-llm --json results.json
```

Suggested tables: (1) Recall@4 / MRR for BM25 vs dense vs hybrid; (2) hybrid Recall@4 by element and by
route; (3) answer accuracy, citation precision and abstention with the LLM; (4) CER per route
(native ≈ 0, OCR > 0) to show where errors come from; (5) processing time per page type.

The bundled sample set is small (4 pages, 13 questions), so treat its 100 % recall as a sanity check,
not a result. Build 30–50 questions over real documents for the report.

## 5. Questions to expect in the viva

* *Why not use ColPali for everything?* Storage (about 1,000 vectors per page) and a GPU requirement. We
  use it only on pages the router marks as visual.
* *Why keep BM25 when you have embeddings?* OCR'd text and exact identifiers (invoice numbers, drug names,
  clause numbers). The per-mode comparison on the Evaluate page shows the effect.
* *How do you know an answer is not hallucinated?* The model must cite; citations are checked against the
  sources actually given; it abstains when the sources lack the answer; the user can open the cited page.
* *What happens when a tool is missing?* `capabilities()` detects it, the router never assigns it, and any
  runtime failure degrades to Tesseract with the reason recorded.
