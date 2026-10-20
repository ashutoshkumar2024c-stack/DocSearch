"""
Image profiling: measure a page image so the planner can pick the right tool.

Measures (all cheap, OpenCV only, ~50 ms per page):
    sharpness     variance of the Laplacian (low = blurry)
    contrast      grey-level standard deviation (low = faded / washed out)
    char_height   median height of character-like blobs, in pixels (low = tiny text)
    lighting      unevenness of the background (high = phone photo, shadows)
    colorfulness  share of strongly coloured pixels (high = photo or chart)
    text_blobs    number of character-like connected components
    graphic_ink   share of the page covered by large non-character shapes (bars, photos)
    skew          estimated rotation in degrees
    orientation / script   from Tesseract OSD, when available

and derives:
    kind     "scan" (clean printed page) | "photo" (camera picture) | "graphic" (chart/diagram/photo, little text)
    quality  "good" | "poor"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass
class ImageProfile:
    width: int
    height: int
    sharpness: float
    contrast: float
    char_height: float
    lighting: float
    colorfulness: float
    text_blobs: int
    graphic_ink: float
    skew: float
    orientation: int = 0              # degrees the page must be rotated clockwise to be upright
    script: str | None = None         # e.g. "Latin", "Devanagari"
    script_confidence: float = 0.0
    kind: str = "scan"
    quality: str = "good"
    issues: tuple = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issues"] = list(self.issues)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def _osd(image: Image.Image) -> tuple[int, str | None, float]:
    """Orientation + script detection with Tesseract OSD. Returns (rotate, script, conf)."""
    try:
        import pytesseract

        info = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT,
                                        config="--psm 0 -c min_characters_to_try=20")
        return int(info.get("rotate", 0)), info.get("script"), float(info.get("script_conf", 0))
    except Exception:
        return 0, None, 0.0      # too little text for OSD, or osd pack missing


def kind_hint_photo(colorfulness: float) -> bool:
    return colorfulness < 0.08


def profile_image(image: Image.Image, detect_script: bool = True) -> ImageProfile:
    import cv2

    from .engines import estimate_skew

    rgb = np.array(image.convert("RGB"))
    h0, w0 = rgb.shape[:2]
    scale = 1200 / max(h0, w0) if max(h0, w0) > 1200 else 1.0
    if scale != 1.0:
        rgb_s = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        rgb_s = rgb
    gray = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # contrast = gap between paper and ink grey levels (page-wide std is misleading:
    # a clean page is mostly white, so its std is low even when the text is crisp)
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    paper, inkpx = gray[gray > t], gray[gray <= t]
    contrast = float(paper.mean() - inkpx.mean()) if len(paper) and len(inkpx) else 0.0
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((15, 15), np.uint8)), 31)
    lighting = float(background.std())
    hsv = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2HSV)
    colorfulness = float(((hsv[..., 1] > 80) & (hsv[..., 2] > 60)).mean())

    # connected components on the full-resolution ink mask
    gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    hs, ws, areas = stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_AREA]
    page_area = h0 * w0
    min_h, max_h = max(4, h0 * 0.003), h0 * 0.06
    is_char = (hs >= min_h) & (hs <= max_h) & (ws <= hs * 4) & (areas >= 8)
    char_height = float(np.median(hs[is_char])) if is_char.any() else 0.0
    big = (~is_char) & (areas > page_area * 0.0005)
    graphic_ink = float(areas[big].sum() / page_area)
    text_blobs = int(is_char.sum())

    skew = estimate_skew(gray) if text_blobs > 30 else 0.0
    rotate, script, sconf = _osd(image) if detect_script and text_blobs > 80 else (0, None, 0.0)
    if sconf < 1.0:                      # OSD guesses wildly on little or tiny text
        rotate, script = 0, None
    elif script and script != "Latin" and sconf < 6.0:
        script = None                    # handwriting is often "detected" as Arabic etc. with low confidence

    issues = []
    if sharpness < 80:
        issues.append("blurry")
    if contrast < 80:
        issues.append("low contrast")
    if 0 < char_height < 14:
        issues.append("small text")
    if lighting > 18 and kind_hint_photo(colorfulness):
        issues.append("uneven lighting")
    if abs(skew) >= 1:
        issues.append(f"skewed {skew:+.1f}°")
    if rotate:
        issues.append(f"rotated {rotate}°")

    if graphic_ink > 0.08 and colorfulness > 0.08 and text_blobs < 400:
        kind = "graphic"                     # coloured chart / diagram with little text
    elif lighting > 18 and colorfulness < 0.08:
        kind = "photo"                       # camera picture of a page: shadows, uneven paper
    elif text_blobs < 40 and graphic_ink > 0.05:
        kind = "graphic"                     # black-and-white figure, almost no text
    else:
        kind = "scan"
    poor = {"blurry", "low contrast", "uneven lighting"} & set(issues)
    quality = "poor" if poor or kind == "photo" else "good"

    return ImageProfile(w0, h0, sharpness, contrast, char_height, lighting, colorfulness,
                        text_blobs, graphic_ink, skew, rotate, script, sconf, kind, quality,
                        tuple(issues))


# Tesseract language packs for scripts reported by OSD
SCRIPT_TO_TESSERACT = {
    "Latin": "eng", "Devanagari": "hin", "Bengali": "ben", "Tamil": "tam", "Telugu": "tel",
    "Kannada": "kan", "Malayalam": "mal", "Gujarati": "guj", "Gurmukhi": "pan", "Oriya": "ori",
    "Arabic": "ara", "Cyrillic": "rus", "Greek": "ell", "Han": "chi_sim", "HanS": "chi_sim",
    "HanT": "chi_tra", "Japanese": "jpn", "Hangul": "kor", "Korean": "kor", "Thai": "tha",
    "Hebrew": "heb",
}
# EasyOCR / PaddleOCR language codes for the same scripts
SCRIPT_TO_EASYOCR = {
    "Latin": "en", "Devanagari": "hi", "Bengali": "bn", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Arabic": "ar", "Cyrillic": "ru", "Han": "ch_sim", "HanS": "ch_sim",
    "HanT": "ch_tra", "Japanese": "ja", "Hangul": "ko", "Korean": "ko", "Thai": "th",
}