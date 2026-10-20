"""
Layout analysis: turn a page into ordered elements (heading / text / table / figure).

This is the project's lightweight answer to SCAN (EACL 2026). SCAN fine-tunes a
detector to find semantically coherent regions. We get similar region
structure without training any model:

* digital pages  - pdfplumber word boxes: font size and weight -> headings,
                   vertical gaps -> paragraphs, ruling lines -> tables (kept as Markdown)
* scanned pages  - OpenCV finds ruled tables (horizontal + vertical line masks),
                   each cell is OCR'd separately; Tesseract's block/paragraph/line
                   boxes plus line heights -> headings and paragraphs
* other engines  - text is split into paragraphs; Markdown headings/tables that the
                   vision model writes are recognised
"""

from __future__ import annotations

import re
from statistics import median

import numpy as np
from PIL import Image

from .types import Element


# --------------------------------------------------------------------------- #
# Tables -> Markdown
# --------------------------------------------------------------------------- #
def _cell(v) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip().replace("|", "/")


def table_to_markdown(rows: list[list]) -> str:
    rows = [[_cell(c) for c in r] for r in rows if any(_cell(c) for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # drop columns that are empty everywhere (merged-cell artefacts)
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    rows = [[r[j] for j in keep] for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def is_real_table(rows: list[list]) -> bool:
    rows = [r for r in rows if any(_cell(c) for c in r)]
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    cells = [_cell(c) for r in rows for c in r]
    return sum(bool(c) for c in cells) / len(cells) >= 0.4


# --------------------------------------------------------------------------- #
# Digital PDF pages (pdfplumber)
# --------------------------------------------------------------------------- #
def _inside(word, bbox) -> bool:
    x0, top, x1, bottom = bbox
    cx, cy = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def native_elements(page) -> list[Element]:
    """pdfplumber page -> ordered elements."""
    tables = []
    try:
        for t in page.find_tables():
            rows = t.extract()
            if is_real_table(rows):
                md = table_to_markdown(rows)
                if all(md != m for _, m in tables):          # pdfplumber can report a table twice
                    tables.append((t.bbox, md))
    except Exception:
        pass

    words = [w for w in page.extract_words(extra_attrs=["size", "fontname"], use_text_flow=True)
             if not any(_inside(w, bbox) for bbox, _ in tables)]

    # words -> lines
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(w["top"] - lines[-1]["top"]) <= max(2.5, w["size"] * 0.3) \
                and w["x0"] >= lines[-1]["x1"] - 1:
            ln = lines[-1]
            ln["words"].append(w)
            ln["x1"], ln["bottom"] = w["x1"], max(ln["bottom"], w["bottom"])
        else:
            lines.append({"words": [w], "top": w["top"], "bottom": w["bottom"],
                          "x0": w["x0"], "x1": w["x1"]})
    if not lines and not tables:
        return []

    sizes = [w["size"] for w in words] or [10]
    body = median(sizes)
    for ln in lines:
        ws = ln["words"]
        ln["text"] = " ".join(w["text"] for w in ws)
        ln["size"] = max(w["size"] for w in ws)
        ln["bold"] = all(re.search(r"bold|black|heavy|semibold", w["fontname"], re.I) for w in ws)

    def candidate(ln) -> bool:
        t = ln["text"].strip()
        if not (2 <= len(t) <= 120) or t.endswith((".", ",", ";")) or not re.search(r"[A-Za-z]", t):
            return False
        if t[0].islower() or re.search(r"[;{}<>=]|\(\)|//", t):     # sentence fragments, code
            return False
        return ln["size"] >= body * 1.15 or (ln["bold"] and len(t.split()) <= 12 and ln["size"] >= body * 0.98)

    # A run of 3+ consecutive bold lines is a bold paragraph (callout), not a heading.
    gap_h = median([ln["bottom"] - ln["top"] for ln in lines]) if lines else 10
    is_head = [candidate(ln) for ln in lines]
    i = 0
    while i < len(lines):
        if not is_head[i] or lines[i]["size"] >= body * 1.15:
            i += 1
            continue
        j = i
        while (j + 1 < len(lines) and is_head[j + 1] and lines[j + 1]["size"] < body * 1.15
               and lines[j + 1]["top"] - lines[j]["bottom"] < gap_h * 0.9):
            j += 1
        if j - i + 1 >= 3:
            for x in range(i, j + 1):
                is_head[x] = False
        i = j + 1
    head_ids = {id(ln) for ln, h in zip(lines, is_head) if h}

    def heading_like(ln) -> bool:
        return id(ln) in head_ids

    head_sizes = sorted({round(ln["size"], 1) for ln in lines if heading_like(ln)}, reverse=True)

    def level(ln):
        s = round(ln["size"], 1)
        return min(head_sizes.index(s) + 1, 6) if s in head_sizes else 3

    # lines -> paragraphs / headings
    blocks: list[tuple[float, Element]] = []
    para: list[str] = []
    para_top, prev = None, None
    line_h = median([ln["bottom"] - ln["top"] for ln in lines]) if lines else 10

    def flush():
        nonlocal para, para_top
        if para:
            text = re.sub(r"-\n(?=[a-z])", "", "\n".join(para))
            blocks.append((para_top, Element("text", text, tool="pdfplumber")))
        para, para_top = [], None

    for ln in lines:
        if heading_like(ln):
            flush()
            blocks.append((ln["top"], Element("heading", ln["text"].strip(), level(ln), "pdfplumber")))
            prev = ln
            continue
        if prev is not None and (ln["top"] - prev["bottom"] > line_h * 0.9):
            flush()
        if para_top is None:
            para_top = ln["top"]
        para.append(ln["text"])
        prev = ln
    flush()

    for bbox, md in tables:
        blocks.append((bbox[1], Element("table", md, tool="pdfplumber")))
    blocks.sort(key=lambda b: b[0])
    return [e for _, e in blocks]


# --------------------------------------------------------------------------- #
# Scanned pages: ruled-table detection (OpenCV)
# --------------------------------------------------------------------------- #
def detect_ruled_tables(image: Image.Image, min_cells: int = 4):
    """
    Find tables drawn with ruling lines. Returns a list of
    (bbox (x0, y0, x1, y1), rows [[cell_bbox, ...], ...]) in pixel coordinates.
    """
    import cv2

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 30), 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40))))
    grid = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < w * 0.2 or ch < h * 0.03:
            continue
        region = grid[y:y + ch, x:x + cw]
        n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(region), connectivity=4)
        cells = []
        for i in range(1, n):
            cx, cy, ccw, cch, area = stats[i]
            if ccw < 12 or cch < 10 or ccw > cw * 0.98 and cch > ch * 0.98:
                continue
            if cx == 0 or cy == 0 or cx + ccw >= cw or cy + cch >= ch:   # outside the frame
                continue
            cells.append((x + cx, y + cy, x + cx + ccw, y + cy + cch))
        if len(cells) < min_cells:
            continue
        # cluster cells into rows by their vertical centre
        cells.sort(key=lambda b: ((b[1] + b[3]) / 2, b[0]))
        rows, cur = [], [cells[0]]
        for b in cells[1:]:
            if abs((b[1] + b[3]) / 2 - (cur[-1][1] + cur[-1][3]) / 2) < (b[3] - b[1]) * 0.5:
                cur.append(b)
            else:
                rows.append(sorted(cur))
                cur = [b]
        rows.append(sorted(cur))
        if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
            tables.append(((x, y, x + cw, y + ch), rows))
    return sorted(tables, key=lambda t: t[0][1])


def _ocr_cell(image: Image.Image, box, lang: str) -> str:
    import pytesseract

    x0, y0, x1, y1 = box
    crop = image.crop((x0 + 3, y0 + 3, x1 - 3, y1 - 3)).convert("L")
    if crop.width < 5 or crop.height < 5:
        return ""
    if crop.height < 40:
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.BICUBIC)
    return pytesseract.image_to_string(crop, lang=lang, config="--oem 1 --psm 6").strip()


# --------------------------------------------------------------------------- #
# Scanned pages: Tesseract layout
# --------------------------------------------------------------------------- #
def tesseract_elements(engine, image: Image.Image, find_tables: bool = True):
    """Returns (elements, mean_confidence). `engine` is a TesseractEngine."""
    from PIL import ImageDraw

    elements: list[tuple[float, Element]] = []
    work = image.convert("RGB")
    if find_tables:
        for bbox, rows in detect_ruled_tables(work):
            md_rows = [[_ocr_cell(work, cell, engine.lang) for cell in row] for row in rows]
            if is_real_table(md_rows):
                elements.append((bbox[1], Element("table", table_to_markdown(md_rows), tool="opencv+tesseract")))
                work = work.copy()
                ImageDraw.Draw(work).rectangle(bbox, fill="white")     # don't OCR it twice

    data = engine.data(work)
    sy = image.height / max(1, _preprocessed_height(image))
    words = []
    for i, t in enumerate(data["text"]):
        t = (t or "").strip()
        conf = float(data["conf"][i])
        if t and conf >= 0:
            words.append({"text": t, "conf": conf, "block": data["block_num"][i], "par": data["par_num"][i],
                          "line": data["line_num"][i], "top": data["top"][i] * sy, "h": data["height"][i] * sy})
    confs = [w["conf"] for w in words]

    lines: dict = {}
    for w in words:
        lines.setdefault((w["block"], w["par"], w["line"]), []).append(w)
    line_list = [{"key": k, "text": " ".join(x["text"] for x in ws), "top": min(x["top"] for x in ws),
                  "h": median([x["h"] for x in ws])} for k, ws in lines.items()]
    if line_list:
        body_h = median([ln["h"] for ln in line_list])
        paras: dict = {}
        for ln in line_list:
            paras.setdefault(ln["key"][:2], []).append(ln)
        for plines in paras.values():
            text = "\n".join(ln["text"] for ln in plines)
            first = plines[0]
            if (len(plines) == 1 and first["h"] >= body_h * 1.25 and len(text) <= 100
                    and not text.endswith(".")):
                elements.append((first["top"], Element("heading", text, 1 if first["h"] >= body_h * 1.6 else 2,
                                                       "tesseract")))
            else:
                text = re.sub(r"-\n(?=[a-z])", "", text)
                elements.append((first["top"], Element("text", text, tool="tesseract",
                                                       confidence=float(np.mean([w["conf"] for w in words
                                                                                 if (w["block"], w["par"]) == first["key"][:2]])))))
    elements.sort(key=lambda e: e[0])
    return [e for _, e in elements], (float(np.mean(confs)) if confs else None)


def _preprocessed_height(image: Image.Image) -> int:
    # preprocess() doubles images whose shorter side is < 1000 px
    return image.height * 2 if min(image.size) < 1000 else image.height


# --------------------------------------------------------------------------- #
# Plain / Markdown-ish text (EasyOCR, PaddleOCR, vision model)
# --------------------------------------------------------------------------- #
def text_to_elements(text: str, tool: str, confidence: float | None = None) -> list[Element]:
    out: list[Element] = []
    if tool == "cloud-vision":               # keep a heading that sits right above/below a table out of the table block
        spaced, in_tab = [], False
        for ln in text.splitlines():
            is_tab = ln.strip().startswith("|")
            if is_tab != in_tab and spaced and spaced[-1].strip():
                spaced.append("")
            in_tab = is_tab
            spaced.append(ln)
        text = "\n".join(spaced)
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if sum(1 for ln in lines if ln.strip().startswith("|")) >= 2:
            rows = [[c for c in ln.strip().strip("|").split("|")] for ln in lines
                    if ln.strip().startswith("|") and not re.fullmatch(r"\|?[\s:\-|]+\|?", ln.strip())]
            if is_real_table(rows):
                out.append(Element("table", table_to_markdown(rows), tool=tool, confidence=confidence))
                continue
        m = re.match(r"^(#{1,6})\s+(.+)$", lines[0])
        if m:
            out.append(Element("heading", m.group(2).strip(), len(m.group(1)), tool))
            rest = "\n".join(lines[1:]).strip()
            if rest:
                out.append(Element("text", rest, tool=tool, confidence=confidence))
            continue
        if tool == "cloud-vision" and re.match(r"^\[[^\]]{8,}\]", lines[0].strip()) and \
                re.match(r"(?i)^\[\s*(diagram|figure|fig|chart|graph|image|illustration|sketch|drawing|flow|table)", lines[0].strip()):
            out.append(Element("figure", block, tool=tool, confidence=confidence))   # a described diagram
            continue
        if tool == "cloud-vision":              # handwritten notes: "Q-4) Design & Testing:", "Roles:" are headings
            buf: list[str] = []
            for ln in lines:
                lvl = _note_heading_level(ln)
                if lvl:
                    if buf:
                        out.append(Element("text", "\n".join(buf).strip(), tool=tool, confidence=confidence))
                        buf = []
                    out.append(Element("heading", ln.strip().rstrip(":").strip(), lvl, tool))
                else:
                    buf.append(ln)
            if "\n".join(buf).strip():
                out.append(Element("text", "\n".join(buf).strip(), tool=tool, confidence=confidence))
            continue
        out.append(Element("text", block, tool=tool, confidence=confidence))
    return out


_Q_HEADING = re.compile(r"^\s*Q[-\s]?\d+\s*[.)\]]*\s*[-.:)]*\s*[A-Za-z].{2,70}$")


def _note_heading_level(line: str) -> int:
    """Heading level of one line of transcribed notes, or 0. Conservative: question headings
    ("Q-4.) Design & Testing:") and short labels that end with a colon ("Roles:")."""
    ln = line.strip()
    if not ln or len(ln) > 80 or ln.startswith("|"):
        return 0
    if _Q_HEADING.match(ln):
        return 1
    if ln.endswith(":") and len(ln.split()) <= 5 and ln[0].isupper() and not re.match(r"^\(?\d+[.)]|^[a-z]\)", ln):
        return 2
    return 0