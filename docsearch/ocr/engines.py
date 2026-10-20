"""
OCR / vision engines, all behind one interface: ``engine.recognize(image) -> OCRResult``.

    tesseract  classical OCR, CPU, per-word confidence, 100+ languages (default for clean print)
    easyocr    deep-learning OCR (CRAFT + CRNN), more robust on photos / poor scans
    paddleocr  deep-learning OCR (PP-OCR), robust and strong on Asian scripts
    vision     a local vision-language model through Ollama (default qwen3-vl:2b, ported from
               V.A.U.L.T.). Reads charts, forms and handwriting; can also *describe* a figure.
               Slow without a GPU and can hallucinate, so it is used selectively.

``capabilities()`` reports which engines are installed and usable on this machine;
the planner only assigns tools that are actually available.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

OCR_PROMPT = (
    "Transcribe all visible document text exactly. Preserve headings, labels, numbers, units "
    "and reading order. Write tables as Markdown tables and headings with '#'. "
    "Do not summarize, interpret, or invent missing text."
)

DESCRIBE_PROMPT = (
    "This image is a figure from a document (chart, diagram, table, form or photo). "
    "Write a factual description that would let someone answer questions about it without "
    "seeing it: the figure type and title; every visible label, legend entry and axis title; "
    "all data values you can read (as a Markdown table if it is a chart or table); and one "
    "sentence on what it shows. Only state what is visible. Do not guess unreadable values."
)


@dataclass
class OCRResult:
    text: str
    confidence: float | None = None      # mean confidence 0-100 if the engine reports one
    engine: str = ""
    lines: list = field(default_factory=list)   # [(text, confidence, (x0, y0, x1, y1))]


class OCREngine(Protocol):
    name: str

    def recognize(self, image: Image.Image) -> OCRResult: ...


# --------------------------------------------------------------------------- #
# Image preprocessing (Tesseract)
# --------------------------------------------------------------------------- #
def estimate_skew(gray: np.ndarray, max_angle: float = 6.0, step: float = 0.25) -> float:
    """
    Projection-profile skew estimate: when text lines are horizontal, row sums of
    the ink mask alternate sharply between text and gaps (maximum variance). Try
    small rotations and keep the sharpest. Robust to speckle noise.
    """
    import cv2

    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) \
        if max(gray.shape) > 1600 else gray
    _, ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    h, w = ink.shape
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def strip_graphics(image: Image.Image) -> Image.Image:
    """
    Keep only character-like ink (for OCR of charts/diagrams). Bars, filled shapes,
    axes and long lines are painted out, so text that touches them is readable.
    A component is character-like if it is not too tall, not a long thin line,
    and not a solid filled block.
    """
    import cv2

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    h = gray.shape[0]
    hs, ws, areas = stats[1:, 3], stats[1:, 2], stats[1:, 4]
    fill = areas / np.maximum(hs * ws, 1)
    charlike = (hs < max(12, h * 0.12)) & (ws < hs * 8) & ~((fill > 0.8) & (areas > 400))
    keep = np.isin(labels, np.where(charlike)[0] + 1)
    out = np.full(gray.shape, 255, np.uint8)
    out[keep] = 0
    if min(out.shape) < 600:
        out = cv2.resize(out, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(out)


def rotate_upright(image: Image.Image, rotate: int) -> Image.Image:
    """Apply Tesseract OSD's 'rotate' (degrees clockwise) to make the page upright."""
    return image.rotate(-rotate, expand=True) if rotate in (90, 180, 270) else image


def preprocess(image: Image.Image, deskew: bool = True) -> Image.Image:
    """Grayscale -> upscale small text -> denoise -> deskew -> contrast (CLAHE) -> Otsu binarise."""
    import cv2

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if min(gray.shape) < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    if deskew:
        angle = estimate_skew(gray)
        if abs(angle) >= 0.3:
            h, w = gray.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


# --------------------------------------------------------------------------- #
# Tesseract
# --------------------------------------------------------------------------- #
class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 3, do_preprocess: bool = True):
        self.lang = lang
        self.psm = psm
        self.do_preprocess = do_preprocess

    def data(self, image: Image.Image, psm: int | None = None) -> dict:
        """Raw word boxes from Tesseract (used by layout analysis)."""
        import pytesseract

        img = preprocess(image) if self.do_preprocess else image
        return pytesseract.image_to_data(img, lang=self.lang,
                                         config=f"--oem 1 --psm {psm or self.psm}",
                                         output_type=pytesseract.Output.DICT)

    def recognize(self, image: Image.Image) -> OCRResult:
        data = self.data(image)
        lines: dict[tuple, list] = {}
        confs: list[float] = []
        for i, word in enumerate(data["text"]):
            word = (word or "").strip()
            conf = float(data["conf"][i])
            if not word or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append((word, conf))
            confs.append(conf)

        text_lines, out_lines, last = [], [], None
        for (block, par, _), words in lines.items():
            if last is not None and (block, par) != last:
                text_lines.append("")
            line = " ".join(w for w, _ in words)
            text_lines.append(line)
            out_lines.append((line, float(np.mean([c for _, c in words])), None))
            last = (block, par)
        return OCRResult("\n".join(text_lines).strip(),
                         float(np.mean(confs)) if confs else None, self.name, out_lines)

    def recognize_sparse(self, image: Image.Image) -> OCRResult:
        """Scattered text (charts, diagrams, forms): strip graphics, find text anywhere
        (--psm 11), then regroup words into visual rows ("North 82%")."""
        import pytesseract
        from PIL import ImageOps

        def words(img, min_conf, strip=True):
            clean = strip_graphics(img) if strip else img
            scale = img.height / clean.height
            data = pytesseract.image_to_data(clean, lang=self.lang, config="--oem 1 --psm 11",
                                             output_type=pytesseract.Output.DICT)
            out = []
            for i, word in enumerate(data["text"]):
                word, conf = (word or "").strip(), float(data["conf"][i])
                if word and conf >= min_conf and re.search(r"\w", word):
                    l, t = data["left"][i] * scale, data["top"][i] * scale
                    w, h = data["width"][i] * scale, data["height"][i] * scale
                    out.append((word, conf, (l, t, l + w, t + h)))
            return out

        # pass 1: dark text on light background; pass 2: light text on dark boxes
        boxes = words(image, 40)
        inverted = words(ImageOps.invert(image.convert("RGB")), 55)
        taken = [b[2] for b in boxes]

        def overlaps(a, b):
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            return ix * iy > 0.3 * (a[2] - a[0]) * (a[3] - a[1])

        boxes += [b for b in inverted if not any(overlaps(b[2], t) for t in taken)]
        # pass 3: text sitting on top of graphics (bar-value labels) is removed together with the
        # graphics in pass 1; read the untouched image and keep only confident new words
        taken = [b[2] for b in boxes]
        boxes += [b for b in words(image, 70, strip=False) if not any(overlaps(b[2], t) for t in taken)]
        def plausible(text: str, conf: float) -> bool:
            words = re.findall(r"[A-Za-z]+", text)
            if len(re.sub(r"[^A-Za-z0-9]", "", text)) < 2:
                return False
            # broken-up OCR ("ro ressive isc osureo") has many 1-3 letter fragments
            if len(words) >= 3 and sum(len(w) <= 3 for w in words) / len(words) > 0.5 and conf < 85:
                return False
            return conf >= 60 or len(text) >= 6

        lines = [ln for ln in _group_lines(boxes) if plausible(ln[0], ln[1])]
        conf = float(np.mean([ln[1] for ln in lines])) if lines else None
        text = "\n".join(ln[0] for ln in lines)
        pairs = _pair_column_values(boxes)
        if pairs:
            text += "\nChart values: " + "; ".join(f"{k}: {v}" for k, v in pairs)
        return OCRResult(text, conf, self.name, lines)


# --------------------------------------------------------------------------- #
# EasyOCR
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4)
def _easyocr_reader(langs: tuple, gpu: bool):
    import easyocr
    return easyocr.Reader(list(langs), gpu=gpu, verbose=False)


def _pair_column_values(boxes: list) -> list[tuple[str, str]]:
    """Column charts: match each category label (bottom row) with the value label printed
    above its bar, by horizontal position. Returns [] unless at least 3 pairs are found."""
    num = re.compile(r"\d[\d,.]*%?")
    labels = [b for b in boxes if re.search(r"[A-Za-z]", b[0]) and len(b[0]) <= 16]
    if len(labels) < 3:
        return []
    cy = lambda b: (b[2][1] + b[2][3]) / 2          # noqa: E731
    cx = lambda b: (b[2][0] + b[2][2]) / 2          # noqa: E731
    h = float(np.median([b[2][3] - b[2][1] for b in boxes]))
    row = []
    for anchor in sorted(labels, key=cy, reverse=True):          # lowest row with >= 3 labels
        row = sorted([b for b in labels if abs(cy(b) - cy(anchor)) < h * 0.8], key=cx)
        if len(row) >= 3:
            break
    if len(row) < 3:
        return []
    spacing = float(np.median(np.diff([cx(b) for b in row])))
    if spacing <= 0:
        return []
    ybase = min(b[2][1] for b in row)
    values = [b for b in boxes if num.fullmatch(b[0]) and cy(b) < ybase and cx(b) > cx(row[0]) - spacing * 0.6]
    out = []
    for lab in row:
        near = [v for v in values if abs(cx(v) - cx(lab)) < spacing * 0.4]
        if near:
            out.append((lab[0].strip(".,"), min(near, key=lambda v: abs(cx(v) - cx(lab)))[0]))
    return out if len(out) >= 3 else []


def _group_lines(boxes: list) -> list:
    """Sort (text, conf, (x0,y0,x1,y1)) boxes into reading-order lines."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2][1] + b[2][3]) / 2)
    heights = [b[2][3] - b[2][1] for b in boxes]
    tol = max(4.0, float(np.median(heights)) * 0.5)
    lines, cur, cur_y = [], [], None
    for b in boxes:
        y = (b[2][1] + b[2][3]) / 2
        if cur and abs(y - cur_y) > tol:
            lines.append(cur)
            cur = []
        cur.append(b)
        cur_y = y if len(cur) == 1 else (cur_y * (len(cur) - 1) + y) / len(cur)
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        ln.sort(key=lambda b: b[2][0])
        text = " ".join(b[0] for b in ln)
        box = (min(b[2][0] for b in ln), min(b[2][1] for b in ln),
               max(b[2][2] for b in ln), max(b[2][3] for b in ln))
        out.append((text, float(np.mean([b[1] for b in ln])), box))
    return out


def _lines_to_text(lines: list) -> str:
    """Join lines, inserting a paragraph break where the vertical gap is large."""
    if not lines:
        return ""
    heights = [ln[2][3] - ln[2][1] for ln in lines if ln[2]]
    h = float(np.median(heights)) if heights else 10.0
    parts = [lines[0][0]]
    for prev, ln in zip(lines, lines[1:]):
        gap = ln[2][1] - prev[2][3] if ln[2] and prev[2] else 0
        parts.append(("\n\n" if gap > h * 0.9 else "\n") + ln[0])
    return "".join(parts)


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, langs: tuple = ("en",), gpu: bool | None = None):
        self.langs = tuple(langs)
        self.gpu = _has_gpu() if gpu is None else gpu

    def recognize(self, image: Image.Image) -> OCRResult:
        reader = _easyocr_reader(self.langs, self.gpu)
        results = reader.readtext(np.array(image.convert("RGB")), detail=1, paragraph=False)
        boxes = []
        for pts, text, conf in results:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            boxes.append((text, float(conf) * 100, (min(xs), min(ys), max(xs), max(ys))))
        lines = _group_lines(boxes)
        conf = float(np.mean([b[1] for b in boxes])) if boxes else None
        return OCRResult(_lines_to_text(lines), conf, self.name, lines)


# --------------------------------------------------------------------------- #
# PaddleOCR (supports both the 2.x `ocr()` and 3.x `predict()` APIs)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4)
def _paddle(lang: str):
    from paddleocr import PaddleOCR
    try:
        return PaddleOCR(lang=lang, use_textline_orientation=True)     # 3.x
    except TypeError:
        return PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)  # 2.x


class PaddleOCREngine:
    name = "paddleocr"

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def recognize(self, image: Image.Image) -> OCRResult:
        ocr = _paddle(self.lang)
        arr = np.array(image.convert("RGB"))[:, :, ::-1]   # BGR
        boxes = []
        if hasattr(ocr, "predict"):
            for res in ocr.predict(arr):
                r = res.json.get("res", res) if hasattr(res, "json") else res
                for text, score, poly in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"]):
                    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
                    boxes.append((text, float(score) * 100, (min(xs), min(ys), max(xs), max(ys))))
        else:
            for page in ocr.ocr(arr, cls=True) or []:
                for pts, (text, score) in page or []:
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    boxes.append((text, float(score) * 100, (min(xs), min(ys), max(xs), max(ys))))
        lines = _group_lines(boxes)
        conf = float(np.mean([b[1] for b in boxes])) if boxes else None
        return OCRResult(_lines_to_text(lines), conf, self.name, lines)


# --------------------------------------------------------------------------- #
# Vision-language model via Ollama (ported from V.A.U.L.T. models/vision.py)
# --------------------------------------------------------------------------- #
_VISION_SYSTEM_PROMPT = (
    "You are a document OCR engine.\n"
    "Your ONLY task is to transcribe visible text from the image.\n\n"
    "RULES:\n"
    "1. Return ONLY text visible in the image.\n"
    "2. Do NOT describe the image.\n"
    "3. Do NOT summarize.\n"
    "4. Do NOT explain your reasoning.\n"
    "5. Do NOT add missing or guessed text.\n"
    "6. Preserve the original order of the text.\n"
    "7. Preserve headings, paragraphs, lists, dates, numbers, codes and punctuation.\n"
    "8. If a word is unclear, make your best visual transcription but do not invent "
    "surrounding content.\n"
    "9. Never say 'Wait', 'Let's check', 'I can see', 'This image contains', or similar.\n"
)
_DESCRIBE_SYSTEM_PROMPT = (
    "You describe figures from documents for a search engine. Be factual and complete; "
    "never invent values. No preamble."
)


def strip_thinking(text: str) -> str:
    text = re.sub(r"(?s)<think>.*?</think>", "", text or "").strip()
    if text.startswith("<think>"):
        text = text[len("<think>"):].strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


class VisionLLMEngine:
    name = "vision"

    def __init__(self, model: str | None = None, host: str | None = None,
                 prompt: str = OCR_PROMPT, max_tokens: int = 4096, max_side: int = 1600):
        self.model = model or os.getenv("VISION_MODEL", "qwen3-vl:2b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.max_side = max_side

    def _ask(self, image: Image.Image, system: str, user: str) -> str:
        import ollama

        img = image.convert("RGB")
        if max(img.size) > self.max_side:              # VLMs downsample anyway; save time/VRAM
            img.thumbnail((self.max_side, self.max_side))
        client = ollama.Client(host=self.host)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            img.save(path)
            response = client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user, "images": [str(path)]}],
                think=False,
                options={"temperature": 0, "num_predict": self.max_tokens},
            )
        message = getattr(response, "message", None)
        if message is None and isinstance(response, dict):
            message = response.get("message")
        if message is None:
            raise RuntimeError("Ollama returned no message.")
        get = message.get if isinstance(message, dict) else (lambda k, d="": getattr(message, k, d))
        content = (get("content", "") or "").strip() or (get("thinking", "") or "").strip()
        text = strip_thinking(content)
        if not text:
            raise RuntimeError(f"{self.model} returned an empty response.")
        return text

    def recognize(self, image: Image.Image) -> OCRResult:
        # V.A.U.L.T. accepted a prompt argument but never sent it; here it is used.
        text = self._ask(image, _VISION_SYSTEM_PROMPT,
                         self.prompt + "\nTranscribe ALL visible text from TOP to BOTTOM. "
                                       "Return ONLY the transcription.")
        return OCRResult(text, None, self.name)

    def describe(self, image: Image.Image) -> OCRResult:
        return OCRResult(self._ask(image, _DESCRIBE_SYSTEM_PROMPT, DESCRIBE_PROMPT), None, self.name)


# --------------------------------------------------------------------------- #
# Cloud vision OCR (Gemini / Mistral): the only option here that reads handwriting
# --------------------------------------------------------------------------- #
HANDWRITING_PROMPT = (
    "Transcribe ALL the text on this page exactly as written, from top to bottom. The page may be "
    "handwritten notes. Keep the line breaks, numbering and bullet points. Write plain text only: no HTML, "
    "no LaTeX, no $ signs. Write arrows as -> or <-. Write a table as a Markdown table with | separators. "
    "For a diagram or figure, write one line in [square brackets] saying what it shows, then copy every "
    "label in it. Do not correct, summarise or add anything. Return only the transcription."
)


def clean_cloud_text(text: str) -> str:
    """Models sometimes answer with HTML or LaTeX; turn those back into plain text."""
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    for latex, plain in ((r"\rightarrow", "->"), (r"\leftarrow", "<-"), (r"\leftrightarrow", "<->"),
                         (r"\downarrow", "(down)"), (r"\uparrow", "(up)"), (r"\times", "x"), (r"\approx", "~")):
        text = text.replace(latex, plain)
    text = re.sub(r"\$\s*([^$\n]{0,40}?)\s*\$", r"\1", text)       # $...$ math delimiters
    text = text.replace("\\_", "_")
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def _jpeg_b64(image: Image.Image, max_side: int = 2000, quality: int = 88) -> str:
    import base64
    import io
    img = image.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def cloud_vision_providers() -> list[str]:
    """Providers whose API key is set, in order of preference."""
    from ..rag.llm import clean_key, gemini_keys
    out = []
    if gemini_keys():
        out.append("gemini")
    if clean_key(os.getenv("MISTRAL_API_KEY")):
        out.append("mistral")
    return out


def _disk_cache_path(key: str):
    """OCR_CACHE_DIR (default ./.ocr_cache). Set OCR_CACHE_DIR=off to disable. Stores page text
    only (no images)."""
    from pathlib import Path
    d = os.getenv("OCR_CACHE_DIR", ".ocr_cache")
    return None if d.lower() in ("", "off", "none", "0") else Path(d) / (key + ".txt")


_LAST_CALL = [0.0]


def _throttle() -> None:
    """Spacing between cloud calls (CLOUD_MIN_INTERVAL seconds, default 4) keeps a free-tier key
    under its requests-per-minute limit instead of tripping 429 and wasting retries."""
    gap = float(os.getenv("CLOUD_MIN_INTERVAL", "4"))
    wait = _LAST_CALL[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[0] = time.time()


def _short_error(resp) -> str:
    try:
        return str(resp.json().get("error", {}).get("message", ""))[:220]
    except Exception:
        return (resp.text or "")[:220]


class CloudVisionEngine:
    """Reads a page image with Gemini (preferred) or Mistral OCR. Pages are sent to that
    provider, so the router uses it only when local OCR is clearly failing (handwriting,
    very poor scans) and a key is configured. Disable with DISABLE_ENGINES=cloud_vision."""

    name = "cloud-vision"

    def __init__(self, timeout: float = 90, retries: int = 4):
        self.timeout, self.retries = timeout, retries
        self.used: str | None = None

    # -- providers --
    def _post(self, url: str, headers: dict, payload: dict, label: str):
        import requests
        resp = None
        for attempt in range(self.retries):
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code not in (429, 500, 503):
                break
            time.sleep(6 * (attempt + 1))                    # free tiers allow only a few requests a minute
        if resp.status_code != 200:
            raise RuntimeError(f"{label} error {resp.status_code}: {resp.text[:160]}")
        return resp.json()

    def _gemini(self, image: Image.Image, prompt: str) -> str:
        """Tries (key x model x request shape) until one answers. A 400 means "this model/shape does not
        accept the request" (it depends on the model generation), so a simpler request or another model is
        tried instead of giving up."""
        from ..rag.llm import _GEMINI_DEFAULT, gemini_keys, gemini_post
        keys = gemini_keys()
        if not keys:
            raise RuntimeError("no Gemini key is set")
        configured = os.getenv("GEMINI_MODEL", _GEMINI_DEFAULT)
        models = []
        for m in (configured, "gemini-flash-latest", "gemini-flash-lite-latest"):
            if m and m not in models:
                models.append(m)
        parts = [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": _jpeg_b64(image)}}]
        shapes = [{"temperature": 0, "maxOutputTokens": 8192}, {}]      # normal, then bare minimum
        errors = []
        for key in keys:
            for model in models:
                for gen in shapes:
                    g = dict(gen)
                    if "2.5-flash" in model and g:
                        g["thinkingConfig"] = {"thinkingBudget": 0}
                    payload = {"contents": [{"role": "user", "parts": parts}]}
                    if g:
                        payload["generationConfig"] = g
                    _throttle()
                    resp, _ = gemini_post(key, model, payload, self.timeout, retries=2, wait=5, max_models=0)
                    if resp.status_code == 200:
                        data = resp.json()
                        cand = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in cand).strip()
                        if text:
                            self.model_used = model
                            return text
                        errors.append(f"{model}: empty answer")
                        break
                    errors.append(f"{model}: {resp.status_code} {_short_error(resp)}")
                    if resp.status_code != 400:
                        break                       # quota/overload/auth: a simpler request will not help
        raise RuntimeError("Gemini failed - " + " | ".join(errors[-4:]))

    def _mistral(self, image: Image.Image, prompt: str) -> str:
        from ..rag.llm import clean_key
        key = clean_key(os.getenv("MISTRAL_API_KEY"))
        _throttle()
        payload = {"model": os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
                   "document": {"type": "image_url", "image_url": "data:image/jpeg;base64," + _jpeg_b64(image)}}
        data = self._post("https://api.mistral.ai/v1/ocr",
                          {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, payload, "Mistral OCR")
        text = "\n\n".join(p.get("markdown", "") for p in data.get("pages", [])).strip()
        if not text:
            raise RuntimeError("Mistral OCR returned no text")
        return text

    _cache: dict = {}            # same page image + prompt -> text, so "re-read" only pays for pages that failed

    def _run(self, image: Image.Image, prompt: str, only: tuple = ("gemini", "mistral")) -> str:
        import hashlib
        ck = hashlib.sha1((prompt + _jpeg_b64(image, 600, 60)).encode()).hexdigest()
        if ck in self._cache:
            self.used = "cache"
            return self._cache[ck]
        disk = _disk_cache_path(ck)                 # survives restarts: a page is paid for once, ever
        try:
            if disk and disk.exists():
                self.used = "cache"
                self._cache[ck] = disk.read_text(encoding="utf-8")
                return self._cache[ck]
        except OSError:
            pass
        text = self._run_uncached(image, prompt, only)
        self._cache[ck] = text
        try:
            if disk:
                disk.parent.mkdir(parents=True, exist_ok=True)
                disk.write_text(text, encoding="utf-8")
        except OSError:
            pass                                    # read-only / ephemeral disk: memory cache still works
        return text

    def _run_uncached(self, image: Image.Image, prompt: str, only: tuple = ("gemini", "mistral")) -> str:
        errors = []
        for prov in cloud_vision_providers():
            if prov not in only:
                continue
            try:
                text = self._gemini(image, prompt) if prov == "gemini" else self._mistral(image, prompt)
                self.used = prov
                return clean_cloud_text(strip_thinking(text))
            except Exception as exc:
                errors.append(f"{prov}: {exc}")
        raise RuntimeError("; ".join(errors) or "no cloud vision key is set (GEMINI_API_KEY or MISTRAL_API_KEY)")

    def recognize(self, image: Image.Image) -> OCRResult:
        return OCRResult(self._run(image, HANDWRITING_PROMPT), None, self.name)

    def describe(self, image: Image.Image) -> OCRResult:
        return OCRResult(self._run(image, DESCRIBE_PROMPT, only=("gemini",)), None, self.name)


# --------------------------------------------------------------------------- #
# Availability / registry
# --------------------------------------------------------------------------- #
def _has_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _tesseract_info() -> dict:
    try:
        import pytesseract
        version = str(pytesseract.get_tesseract_version())
        langs = [l for l in pytesseract.get_languages(config="") if l != "osd"]
        return {"available": True, "version": version, "languages": langs}
    except Exception as exc:
        return {"available": False, "error": f"Tesseract not found: {exc}".strip()}


def _module_info(module: str) -> dict:
    import importlib.util
    if importlib.util.find_spec(module) is None:
        return {"available": False, "error": f"pip install {module}"}
    return {"available": True}


def _ollama_info(model: str, host: str) -> dict:
    import importlib.util
    if importlib.util.find_spec("ollama") is None:
        return {"available": False, "model": model, "error": "pip install ollama (and run Ollama)"}
    try:
        import requests
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=1.5)
        names = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return {"available": False, "model": model, "error": f"Ollama not reachable at {host}"}
    ok = (model in names or (":" not in model and f"{model}:latest" in names)
          or (model.endswith(":latest") and model[:-7] in names))
    return {"available": ok, "model": model,
            **({} if ok else {"error": f"model not pulled: ollama pull {model}"})}


def _cloud_info() -> dict:
    provs = cloud_vision_providers()
    if provs:
        return {"available": True, "version": "via " + " / ".join(provs) + " (handwriting, poor scans)"}
    return {"available": False, "error": "add a Gemini or Mistral key under Language models"}


_CAPS: dict | None = None


def capabilities(refresh: bool = False) -> dict:
    """Which engines/tools can run on this machine right now."""
    global _CAPS
    if _CAPS is not None and not refresh:
        return _CAPS
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    disabled = {e.strip() for e in os.getenv("DISABLE_ENGINES", "").split(",") if e.strip()}
    caps = {
        "tesseract": _tesseract_info(),
        "easyocr": _module_info("easyocr"),
        "paddleocr": _module_info("paddleocr"),
        "vision": _ollama_info(os.getenv("VISION_MODEL", "qwen3-vl:2b"), host),
        "cloud_vision": _cloud_info(),
        "gpu": {"available": _has_gpu()},
        "pdfplumber": _module_info("pdfplumber"),
    }
    if shutil.which("tesseract") is None and caps["tesseract"]["available"] is False:
        caps["tesseract"]["error"] = ("Tesseract program not installed "
                                      "(Windows: UB-Mannheim installer; Ubuntu: apt install tesseract-ocr)")
    for name in disabled:
        if name in caps:
            caps[name] = {"available": False, "error": "disabled by DISABLE_ENGINES"}
    _CAPS = caps
    return caps


def set_capabilities(caps: dict | None) -> None:
    """Override detected capabilities (tests, or forcing a configuration)."""
    global _CAPS
    _CAPS = caps


def get_engine(name: str = "tesseract", **kwargs) -> OCREngine:
    name = name.lower()
    if name in ("tesseract", "tess"):
        return TesseractEngine(**kwargs)
    if name == "easyocr":
        return EasyOCREngine(**kwargs)
    if name in ("paddleocr", "paddle"):
        return PaddleOCREngine(**kwargs)
    if name in ("vision", "vlm", "qwen", "ollama"):
        return VisionLLMEngine(**kwargs)
    if name in ("cloud_vision", "cloud", "gemini", "cloud-vision"):
        return CloudVisionEngine(**kwargs)
    raise ValueError(f"Unknown OCR engine: {name!r} (tesseract | easyocr | paddleocr | vision | cloud_vision)")