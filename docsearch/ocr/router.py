"""
Document Classifier & Router: decides, per document and per page, which tools
parse it, and records why.

Decision table
--------------
File type                       -> tool
  .docx / .pptx / .xlsx / .csv  -> native parsers (python-docx / python-pptx / openpyxl),
                                   embedded images -> figure tool (below)
  .html / .txt / .md / .json    -> text parsers
  .pdf                          -> per page:

PDF page evidence (from the PDF itself, before rendering)
  native_chars    characters in the text layer
  garbage_ratio   share of unreadable glyphs (broken font encodings)
  image_coverage  share of the page covered by image objects

  no / broken text layer                -> route "ocr"    (scanned page)
  good text layer, images >= 20 % page  -> route "hybrid" (text layer + figure tool per image)
  full-page image + good text layer     -> route "native" (already-OCR'd "searchable" scan)
  otherwise                             -> route "native" (pdfplumber: headings + tables)

Image evidence (for scanned pages, photos, image files, figures) from quality.profile_image
  graphic (chart / diagram)        -> vision model *describe*  | else Tesseract sparse-text mode
  non-Latin script                 -> Tesseract with that language pack | PaddleOCR / EasyOCR | vision
  poor quality / camera photo      -> EasyOCR | PaddleOCR | Tesseract (+ vision fallback)
  clean printed scan               -> Tesseract (+ ruled-table detection), fallback on low confidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .quality import SCRIPT_TO_EASYOCR, SCRIPT_TO_TESSERACT, ImageProfile
from .types import PagePlan


@dataclass
class RouterConfig:
    min_native_chars: int = 40
    max_garbage_ratio: float = 0.30
    hybrid_image_coverage: float = 0.20
    scan_image_coverage: float = 0.85
    min_region_coverage: float = 0.03
    min_confidence: float = 60.0       # below this, OCR is retried with the fallback engine


_ALLOWED = re.compile(r"[\w\s.,;:!?'\"()\[\]{}\-–—/\\%&@#*+=<>|$€₹£°•·…“”‘’●■▪◦-]", re.UNICODE)


def garbage_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = sum(1 for c in chars if not _ALLOWED.match(c))
    bad += text.count("(cid:") * 5
    bad += text.count("�")
    return min(1.0, bad / len(chars))


def image_regions(page, min_coverage: float) -> list[tuple[float, float, float, float]]:
    """Bounding boxes (left, bottom, right, top in PDF points) of sizeable images (pypdfium2 page)."""
    import pypdfium2.raw as pdfium_c

    width, height = page.get_size()
    boxes = []
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3):
        left, bottom, right, top = obj.get_bounds()
        left, bottom = max(0, left), max(0, bottom)
        right, top = min(width, right), min(height, top)
        if right <= left or top <= bottom:
            continue
        if (right - left) * (top - bottom) / (width * height) >= min_coverage:
            boxes.append((left, bottom, right, top))
    return boxes


def coverage(boxes, width: float, height: float) -> float:
    if not boxes:
        return 0.0
    grid = np.zeros((100, 100), dtype=bool)
    for left, bottom, right, top in boxes:
        x0, x1 = int(left / width * 100), int(np.ceil(right / width * 100))
        y0, y1 = int(bottom / height * 100), int(np.ceil(top / height * 100))
        grid[y0:y1, x0:x1] = True
    return float(grid.mean())


def _avail(caps: dict, name: str) -> bool:
    return bool(caps.get(name, {}).get("available"))


# --------------------------------------------------------------------------- #
# PDF pages
# --------------------------------------------------------------------------- #
def plan_pdf_page(page_number: int, native_text: str, boxes, width: float, height: float,
                  caps: dict, cfg: RouterConfig | None = None) -> PagePlan:
    cfg = cfg or RouterConfig()
    n_chars = len(re.sub(r"\s+", "", native_text))
    junk = garbage_ratio(native_text)
    cov = coverage(boxes, width, height)
    ev = {"native_chars": n_chars, "garbage_ratio": round(junk, 3),
          "image_coverage": round(cov, 3), "images": len(boxes)}
    layout_tool = "pdfplumber" if _avail(caps, "pdfplumber") else "pdf-text"

    if n_chars < cfg.min_native_chars:
        return PagePlan(page_number, "scanned-pdf", "ocr", [], [
            f"no usable text layer ({n_chars} characters) - page is an image"], ev)
    if junk > cfg.max_garbage_ratio:
        return PagePlan(page_number, "scanned-pdf", "ocr", [], [
            f"text layer is broken ({junk:.0%} unreadable glyphs) - re-reading the page image"], ev)
    if cov >= cfg.scan_image_coverage:
        return PagePlan(page_number, "searchable-scan", "native", [layout_tool], [
            "full-page image with an existing OCR text layer - using that layer"], ev)
    if cov >= cfg.hybrid_image_coverage:
        return PagePlan(page_number, "digital-pdf", "hybrid", [layout_tool], [
            f"clean text layer, plus {len(boxes)} image(s) covering {cov:.0%} of the page"], ev)
    return PagePlan(page_number, "digital-pdf", "native", [layout_tool],
                    ["clean text layer" + (f"; {len(boxes)} small image(s) ignored" if boxes else "")], ev)


# --------------------------------------------------------------------------- #
# Images (scanned pages, photos, image files, figures)
# --------------------------------------------------------------------------- #
def choose_ocr(profile: ImageProfile, caps: dict, preferred: str | None = None,
               as_figure: bool = False) -> dict:
    """
    Pick the tool for one image. Returns
        {"engine": name, "mode": "ocr"|"describe", "lang": ..., "psm": ...,
         "fallback": name|None, "reasons": [...]}
    """
    reasons: list[str] = []
    tess_langs = set(caps.get("tesseract", {}).get("languages", []) or [])
    vision = _avail(caps, "vision")
    cloud = _avail(caps, "cloud_vision")
    easy, paddle, tess = _avail(caps, "easyocr"), _avail(caps, "paddleocr"), _avail(caps, "tesseract")
    script = profile.script or "Latin"
    out = {"engine": None, "mode": "ocr", "lang": None, "psm": 3, "fallback": None, "reasons": reasons}

    def finish(engine, **kw):
        out.update(engine=engine, **kw)
        if out["fallback"] is None and out["mode"] == "ocr" and engine not in ("vision", "cloud_vision"):
            # handwriting and very poor scans: a cloud vision model is the only thing that reads them
            out["fallback"] = ("cloud_vision" if cloud else "vision" if vision
                               else ("easyocr" if easy and engine != "easyocr" else None))
        return out

    if preferred and preferred != "auto":
        if _avail(caps, preferred):
            reasons.append(f"engine '{preferred}' chosen in settings")
            return finish(preferred, lang=SCRIPT_TO_TESSERACT.get(script, "eng")
                          if preferred == "tesseract" else None)
        reasons.append(f"preferred engine '{preferred}' is not available - choosing automatically")

    if profile.issues:
        reasons.append("image: " + profile.kind + ", " + ", ".join(profile.issues))
    else:
        reasons.append(f"image: clean {profile.kind}")

    # 1. charts, diagrams, photos with little text
    if profile.kind == "graphic":
        if vision:
            reasons.append("graphic with little text -> vision model describes it")
            return finish("vision", mode="describe")
        reasons.append("graphic with little text -> Tesseract sparse-text mode (no vision model installed)")
        return finish("tesseract", psm=11, lang="eng")

    # 2. non-Latin scripts
    if script != "Latin":
        tl = SCRIPT_TO_TESSERACT.get(script)
        if tess and tl and tl in tess_langs:
            reasons.append(f"{script} script detected -> Tesseract '{tl}' language pack")
            return finish("tesseract", lang=f"{tl}+eng")
        el = SCRIPT_TO_EASYOCR.get(script)
        if paddle and script in ("Han", "HanS", "HanT", "Japanese", "Hangul", "Korean"):
            reasons.append(f"{script} script detected -> PaddleOCR")
            return finish("paddleocr", lang={"HanT": "chinese_cht", "Japanese": "japan",
                                             "Hangul": "korean", "Korean": "korean"}.get(script, "ch"))
        if easy and el:
            reasons.append(f"{script} script detected -> EasyOCR ('{el}')")
            return finish("easyocr", lang=el)
        if vision:
            reasons.append(f"{script} script detected, no matching OCR language pack -> vision model")
            return finish("vision")
        reasons.append(f"{script} script detected but no engine supports it - install the "
                       f"Tesseract '{tl or '?'}' pack; trying English")

    # 3. poor scans and camera photos
    if profile.quality == "poor":
        if easy:
            reasons.append("poor image quality -> EasyOCR (deep-learning detector, robust to blur/lighting)")
            return finish("easyocr", lang="en")
        if paddle:
            reasons.append("poor image quality -> PaddleOCR")
            return finish("paddleocr", lang="en")
        reasons.append("poor image quality -> Tesseract with enhancement"
                       + (", vision model as fallback" if vision else ""))
        return finish("tesseract", lang="eng")

    # 4. clean printed page
    if tess:
        reasons.append("clean printed text -> Tesseract (fast, with confidence scores)")
        return finish("tesseract", lang="eng")
    for alt in ("easyocr", "paddleocr", "vision"):
        if _avail(caps, alt):
            reasons.append(f"Tesseract is not installed -> {alt}")
            return finish(alt)
    raise RuntimeError("No OCR engine is available. Install Tesseract (see README).")