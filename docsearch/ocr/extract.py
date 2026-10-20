"""
Stages 1-2: classify every document / page, assign tools automatically, extract
layout elements.

    Extractor().extract("file.pdf") -> list[PageText]   (each with .elements and .plan)

The decisions are made in router.py (what to use) and executed here (running
it, checking confidence, falling back). Every PageText carries a PagePlan with
the tools that actually ran and the reasons, so the application can show why a
page was handled the way it was.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps, ImageSequence

from . import office
from .engines import OCREngine, TesseractEngine, capabilities, get_engine, rotate_upright
from .layout import native_elements, tesseract_elements, text_to_elements
from .quality import profile_image
from .router import RouterConfig, choose_ocr, image_regions, plan_pdf_page
from .types import Element, PagePlan, PageText

log = logging.getLogger(__name__)

MIN_USEFUL_CHARS = 20
RENDER_DPI = 300
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
TEXT_EXT = {".txt", ".md", ".json", ".log", ".rst", ".xml", ".yaml", ".yml"}
SUPPORTED_EXT = IMAGE_EXT | TEXT_EXT | {".pdf", ".docx", ".pptx", ".xlsx", ".xlsm", ".csv", ".html", ".htm"}


class ExtractionError(RuntimeError):
    """Raised when a file cannot be turned into useful text."""


def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def useful(text: str) -> bool:
    return len(re.sub(r"\s+", "", text or "")) >= MIN_USEFUL_CHARS


class Extractor:
    """
    Parameters
    ----------
    engine:           "auto" (default) lets the router choose per page. Or force one of
                      "tesseract" | "easyocr" | "paddleocr" | "vision", or pass an engine object.
    visual_engine:    force the tool for figures ("vision" / "tesseract"); default auto.
    fallback_engine:  force the low-confidence fallback; default auto (vision if available).
    caps:             override detected capabilities (tests / forced configurations).
    """

    def __init__(self, engine: str | OCREngine = "auto",
                 visual_engine: str | OCREngine | None = None,
                 fallback_engine: str | OCREngine | None = None,
                 min_confidence: float = 60.0, router: RouterConfig | None = None,
                 dpi: int = RENDER_DPI, caps: dict | None = None, find_tables: bool = True):
        self.router = router or RouterConfig(min_confidence=min_confidence)
        self.dpi = dpi
        self.find_tables = find_tables
        self._caps = caps
        self._engines: dict = {}
        self.forced = None if isinstance(engine, str) and engine in ("auto", "", None) else engine
        self.visual_forced = visual_engine
        self.fallback_forced = fallback_engine

    @property
    def caps(self) -> dict:
        return self._caps if self._caps is not None else capabilities()

    # ---- engines --------------------------------------------------------- #
    def _engine(self, spec, lang=None, psm=3) -> OCREngine:
        if not isinstance(spec, str):
            return spec
        key = (spec, lang, psm)
        if key not in self._engines:
            kw = {}
            if spec == "tesseract":
                kw = {"lang": lang or "eng", "psm": psm}
            elif spec == "easyocr" and lang:
                kw = {"langs": (lang,) if lang == "en" else (lang, "en")}
            elif spec in ("paddleocr", "paddle") and lang:
                kw = {"lang": lang}
            self._engines[key] = get_engine(spec, **kw)
        return self._engines[key]

    def _choice(self, profile, as_figure: bool) -> dict:
        forced = self.visual_forced if as_figure and self.visual_forced else self.forced
        if forced is not None and not isinstance(forced, str):          # engine object
            return {"engine": forced, "mode": "ocr", "lang": None, "psm": 3,
                    "fallback": self.fallback_forced, "reasons": [f"engine '{forced.name}' set by caller"]}
        choice = choose_ocr(profile, self.caps, preferred=forced)
        if self.fallback_forced is not None:
            choice["fallback"] = self.fallback_forced
        return choice

    # ---- one image: profile -> choose -> run -> verify -> fallback -------- #
    def process_image(self, image: Image.Image, as_figure: bool = False):
        """Returns (elements, confidence, method, tools, reasons, evidence)."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        profile = profile_image(image, detect_script=not as_figure)
        if profile.orientation:
            image = rotate_upright(image, profile.orientation)
        choice = self._choice(profile, as_figure)
        reasons = list(choice["reasons"])
        if profile.orientation:
            reasons.append(f"page was rotated {profile.orientation}° - turned upright")

        def run(spec, mode, lang, psm):
            eng = self._engine(spec, lang, psm)
            name = getattr(eng, "name", str(spec))
            if mode == "describe":
                res = eng.describe(image)
                return [Element("figure", normalize(res.text), tool=f"{name}:describe")], None, name
            if isinstance(eng, TesseractEngine) and psm == 11:
                res = eng.recognize_sparse(image)
                return text_to_elements(normalize(res.text), name, res.confidence), res.confidence, name
            if isinstance(eng, TesseractEngine) and not as_figure:
                els, conf = tesseract_elements(eng, image, find_tables=self.find_tables)
                return els, conf, name
            res = eng.recognize(image)
            return text_to_elements(normalize(res.text), name, res.confidence), res.confidence, name

        tools = []
        try:
            elements, conf, method = run(choice["engine"], choice["mode"], choice["lang"], choice["psm"])
            tools.append(method + (":describe" if choice["mode"] == "describe" else ""))
        except Exception as exc:
            log.warning("%s failed: %s", choice["engine"], exc)
            reasons.append(f"{getattr(choice['engine'], 'name', choice['engine'])} failed ({exc}) - "
                           f"using Tesseract instead")
            elements, conf, method = run("tesseract", "ocr", "eng", 11 if profile.kind == "graphic" else 3)
            tools.append(method)

        text = " ".join(e.text for e in elements)
        fb = choice.get("fallback")
        if fb is None and choice["mode"] != "describe" and self.caps.get("cloud_vision", {}).get("available"):
            fb = "cloud_vision"             # the chosen engine failed: handwriting/poor pages can still use the cloud
        weak = (conf is not None and conf < self.router.min_confidence) or not useful(text)
        if fb and weak and choice["mode"] != "describe":
            why = f"confidence {conf:.0f} < {self.router.min_confidence:.0f}" if conf is not None else "almost no text found"
            if conf is not None and conf < 50:
                why += " (probably handwriting)"
            try:
                fb_el, fb_conf, fb_method = run(fb, "ocr", None, 3)
                fb_text = " ".join(e.text for e in fb_el)
                better = useful(fb_text) and (fb_conf is None or conf is None or fb_conf > conf)
                if better:
                    reasons.append(f"{why} -> retried with {fb_method}, result kept")
                    elements, conf, method = fb_el, fb_conf, fb_method
                else:
                    reasons.append(f"{why} -> retried with {fb_method}, original result was better")
                tools.append(fb_method)
            except Exception as exc:
                reasons.append(f"{why} -> fallback {getattr(fb, 'name', fb)} unavailable ({exc})")

        if conf is not None and conf < 50 and method == "tesseract":
            # unreadable (probably handwriting): garbage must not become section titles in the outline
            elements = [Element("text", e.text, tool=e.tool, confidence=e.confidence) if e.kind == "heading" else e
                        for e in elements]
        if as_figure and elements and choice["mode"] != "describe":
            merged = normalize("\n".join(e.text for e in elements))
            elements = [Element("figure", merged, tool=method, confidence=conf)] if useful(merged) else []
        return elements, conf, method, tools, reasons, {"image": profile.to_dict()}

    def figure_element(self, image: Image.Image, label: str = "figure") -> Element | None:
        try:
            els, *_ = self.process_image(image, as_figure=True)
        except Exception as exc:
            log.warning("figure %s skipped: %s", label, exc)
            return None
        return els[0] if els else None

    # ---- public API ------------------------------------------------------ #
    def extract(self, path: str | Path, source_name: str | None = None,
                progress=None) -> list[PageText]:
        """`progress(done, total)` is called after each page of a PDF (for progress bars)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")
        source = source_name or path.name
        ext = path.suffix.lower()

        if ext == ".pdf":
            pages = self._pdf(path, source, progress)
        elif ext in IMAGE_EXT:
            pages = self._image_file(path, source)
        elif ext == ".docx":
            pages = office.parse_docx(path, source, self.figure_element)
        elif ext == ".pptx":
            pages = office.parse_pptx(path, source, self.figure_element)
        elif ext in (".xlsx", ".xlsm"):
            pages = office.parse_xlsx(path, source)
        elif ext == ".csv":
            pages = office.parse_csv(path, source, _read_text(path))
        elif ext in (".html", ".htm"):
            pages = office.parse_html(path, source, _read_text(path))
        elif ext in TEXT_EXT or _looks_like_text(path):
            pages = office.parse_text(path, source, _read_text(path))
        else:
            raise ExtractionError(f"Unsupported file type {ext or '[none]'}: {path.name}")

        pages = [p for p in pages if useful(p.text)]
        if not pages:
            raise ExtractionError(f"No readable text could be extracted from {path.name}")
        return pages

    def routes(self, path: str | Path) -> list[PagePlan]:
        """Classify the pages of a PDF without running OCR."""
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            out = []
            for i in range(len(pdf)):
                page = pdf[i]
                w, h = page.get_size()
                boxes = image_regions(page, self.router.min_region_coverage)
                out.append(plan_pdf_page(i + 1, page.get_textpage().get_text_range(), boxes, w, h,
                                         self.caps, self.router))
            return out
        finally:
            pdf.close()

    # ---- PDF ------------------------------------------------------------- #
    def _pdf(self, path: Path, source: str, progress=None) -> list[PageText]:
        import pypdfium2 as pdfium

        try:
            pdf = pdfium.PdfDocument(str(path))
        except Exception as exc:
            raise ExtractionError(f"Could not open PDF {path.name}: {exc}") from exc
        plumber = None
        if self.caps.get("pdfplumber", {}).get("available", True):
            try:
                import pdfplumber
                plumber = pdfplumber.open(str(path))
            except Exception:
                plumber = None

        scale = self.dpi / 72
        results: list[PageText] = []
        try:
            for i in range(len(pdf)):
                if progress:
                    progress(i, len(pdf))
                page = pdf[i]
                native = page.get_textpage().get_text_range()
                width, height = page.get_size()
                boxes = image_regions(page, self.router.min_region_coverage)
                ignored = 0
                if len(boxes) > 4:
                    # many small pictures on one page are icons/bullets, not content: OCR-ing each
                    # one costs seconds and finds nothing. Keep only substantial regions.
                    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])      # noqa: E731
                    keep = [b for b in boxes if area(b) / (width * height) >= 0.10][:4]
                    ignored, boxes = len(boxes) - len(keep), keep
                plan = plan_pdf_page(i + 1, native, boxes, width, height, self.caps, self.router)

                if ignored:
                    plan.reasons.append(f"{ignored} small repeated images treated as icons and skipped")
                if plan.route in ("native", "hybrid"):
                    elements = []
                    if plumber is not None:
                        try:
                            elements = native_elements(plumber.pages[i])
                        except Exception as exc:
                            log.warning("pdfplumber layout failed on page %d: %s", i + 1, exc)
                    if not elements:
                        elements = text_to_elements(normalize(native), "pdf-text")
                        plan.tools = ["pdf-text"]
                    n_tables = sum(e.kind == "table" for e in elements)
                    n_heads = sum(e.kind == "heading" for e in elements)
                    if n_tables or n_heads:
                        plan.reasons.append(f"layout: {n_heads} heading(s), {n_tables} table(s)")
                    confs = []
                    if plan.route == "hybrid":
                        rendered = page.render(scale=scale).to_pil()
                        for n, (left, bottom, right, top) in enumerate(boxes, 1):
                            crop = rendered.crop((int(left * scale), int((height - top) * scale),
                                                  int(right * scale), int((height - bottom) * scale)))
                            try:
                                els, conf, method, tools, reasons, ev = self.process_image(crop, as_figure=True)
                            except Exception as exc:
                                plan.reasons.append(f"figure {n}: skipped ({exc})")
                                continue
                            plan.tools += [t for t in tools if t not in plan.tools]
                            plan.reasons.append(f"figure {n}: " + "; ".join(reasons))
                            if conf is not None:
                                confs.append(conf)
                            elements += els
                    method = "native" if plan.route == "native" else "native+" + (
                        plan.tools[-1].split(":")[0] if len(plan.tools) > 1 else "none")
                    results.append(PageText.from_elements(
                        source, elements, page=i + 1, method=method, route=plan.route,
                        ocr_confidence=sum(confs) / len(confs) if confs else None,
                        content_type="pdf", plan=plan))
                    continue

                rendered = page.render(scale=scale).to_pil()
                elements, conf, method, tools, reasons, ev = self.process_image(rendered)
                plan.tools, plan.reasons = tools, plan.reasons + reasons
                n_tables = sum(e.kind == "table" for e in elements)
                n_heads = sum(e.kind == "heading" for e in elements)
                if n_tables or n_heads:
                    plan.reasons.append(f"layout: {n_heads} heading(s), {n_tables} table(s)"
                                        + (" via ruled-table detection" if any(e.tool == "opencv+tesseract" for e in elements) else ""))
                plan.evidence.update(ev)
                if elements and all(e.kind == "figure" for e in elements):
                    plan.route = "visual"
                    plan.doc_type = "image-pdf"
                results.append(PageText.from_elements(source, elements, page=i + 1, method=method,
                                                      route=plan.route, ocr_confidence=conf,
                                                      content_type="pdf", plan=plan))
        finally:
            pdf.close()
            if plumber is not None:
                plumber.close()
        return results

    # ---- image files ----------------------------------------------------- #
    def _image_file(self, path: Path, source: str) -> list[PageText]:
        try:
            img = Image.open(path)
        except Exception as exc:
            raise ExtractionError(f"Could not open image {path.name}: {exc}") from exc
        frames = [f.copy() for f in ImageSequence.Iterator(img)] if getattr(img, "n_frames", 1) > 1 else [img]
        out = []
        for n, frame in enumerate(frames, 1):
            elements, conf, method, tools, reasons, ev = self.process_image(frame)
            kind = ev["image"]["kind"]
            route = "visual" if elements and all(e.kind == "figure" for e in elements) else "ocr"
            plan = PagePlan(n if len(frames) > 1 else None, {"scan": "scanned-image", "photo": "photo",
                                                              "graphic": "graphic"}[kind],
                            route, tools, reasons, ev)
            out.append(PageText.from_elements(source, elements, page=n if len(frames) > 1 else None,
                                              method=method, route=route, ocr_confidence=conf,
                                              content_type="image", plan=plan))
        return out


def summarize(pages: list[PageText]) -> dict:
    """Document-level classification from the page plans (for reports and the UI)."""
    routes = Counter(p.route or "parser" for p in pages)
    tools = Counter(t for p in pages if p.plan for t in p.plan.tools)
    types = Counter(p.plan.doc_type for p in pages if p.plan)
    if len(types) == 1:
        doc_type = next(iter(types))
    elif {"digital-pdf", "scanned-pdf"} & set(types):
        doc_type = "mixed-pdf"
    else:
        doc_type = types.most_common(1)[0][0] if types else "unknown"
    confs = [p.ocr_confidence for p in pages if p.ocr_confidence is not None]
    elements = Counter(e.kind for p in pages for e in p.elements)
    return {"doc_type": doc_type, "pages": len(pages), "routes": dict(routes), "tools": dict(tools),
            "elements": dict(elements),
            "ocr_confidence": round(sum(confs) / len(confs), 1) if confs else None}


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _looks_like_text(path: Path) -> bool:
    sample = path.read_bytes()[:8192]
    if not sample or b"\x00" in sample:
        return False
    try:
        decoded = sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    printable = sum(c.isprintable() or c in "\n\r\t" for c in decoded)
    return printable / max(len(decoded), 1) >= 0.85


def extract(path: str | Path, engine: str = "auto", **kwargs) -> list[PageText]:
    return Extractor(engine=engine, **kwargs).extract(path)