"""Generate sample documents with known ground truth for testing the OCR stage.

    python tests/make_samples.py  ->  tests/samples/{mixed.pdf, photo.png, ground_truth.json}

mixed.pdf
    page 1  digitally native text                      -> expect route "native"
    page 2  scanned page (image only, slightly rotated, noisy) -> expect "ocr"
    page 3  native text + an embedded chart image with text   -> expect "hybrid"
    page 4  scanned page with a heading and a ruled table     -> expect "ocr" + table detection
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent / "samples"

PAGE1 = [
    "Quarterly Audit Summary",
    "The finance team reviewed 1,248 invoices during the third quarter.",
    "Discrepancies were found in 37 invoices, mostly duplicate payments.",
    "All duplicate payments above Rs 50,000 were recovered by September 30.",
]
PAGE2 = [
    "Patient Discharge Record",
    "Name: Ravi Kumar   Ward: 4B   Admission: 12/08/2026",
    "Diagnosis: community acquired pneumonia, treated with antibiotics.",
    "Follow up with the pulmonology clinic after two weeks.",
    "Discharge medication: amoxicillin 500 mg three times daily.",
]
PAGE3 = [
    "Contract Review Notes",
    "Clause 7 limits liability to the total fees paid in the previous year.",
    "The figure below shows renewal rates reported by the vendor.",
]
CHART = ["Renewal Rate by Region", "North 82%", "South 76%", "East 69%", "West 91%"]
TABLE1 = [["Vendor", "Invoices", "Duplicates", "Recovered (Rs)"],
          ["Apex Supplies", "412", "14", "3,20,000"],
          ["Blue Ocean Ltd", "388", "9", "1,75,500"],
          ["Corex Traders", "448", "14", "2,90,250"]]
SCAN_TABLE_TITLE = "Lab Results Summary"
SCAN_TABLE = [["Test", "Result", "Unit"],
              ["Haemoglobin", "10.2", "g/dL"],
              ["Platelets", "250", "10^3/uL"],
              ["Glucose", "98", "mg/dL"]]


def font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def scanned_page(lines, angle=1.5, seed=0) -> Image.Image:
    """Render text like a printed page at ~200 DPI, then rotate and add noise."""
    random.seed(seed)
    w, h = 1654, 2339                        # A4 @ 200 DPI
    img = Image.new("L", (w, h), 245)
    d = ImageDraw.Draw(img)
    y = 180
    for i, line in enumerate(lines):
        d.text((150, y), line, fill=25, font=font(46 if i == 0 else 34))
        y += 90 if i == 0 else 64
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=245)
    px = img.load()
    for _ in range(9000):                    # speckle noise
        px[random.randrange(w), random.randrange(h)] = random.choice((40, 200))
    return img.filter(ImageFilter.GaussianBlur(0.6)).convert("RGB")


def scanned_table_page(seed=2) -> Image.Image:
    """Scanned page with a large heading, a sentence and a ruled table."""
    random.seed(seed)
    w, h = 1654, 2339
    img = Image.new("L", (w, h), 246)
    d = ImageDraw.Draw(img)
    d.text((150, 170), SCAN_TABLE_TITLE, fill=20, font=font(64))
    d.text((150, 300), "Values below were measured on the morning of admission.", fill=25, font=font(34))
    x0, y0, cw, rh = 150, 420, [420, 300, 300], 90
    for r, row in enumerate(SCAN_TABLE):
        x = x0
        for c, cell in enumerate(row):
            d.text((x + 20, y0 + r * rh + 25), cell, fill=20, font=font(36))
            x += cw[c]
    total_w = sum(cw)
    for r in range(len(SCAN_TABLE) + 1):
        d.line((x0, y0 + r * rh, x0 + total_w, y0 + r * rh), fill=30, width=4)
    x = x0
    for c in range(len(cw) + 1):
        d.line((x, y0, x, y0 + len(SCAN_TABLE) * rh), fill=30, width=4)
        x += cw[c] if c < len(cw) else 0
    px = img.load()
    for _ in range(6000):
        px[random.randrange(w), random.randrange(h)] = random.choice((60, 210))
    return img.rotate(0.8, resample=Image.BICUBIC, fillcolor=246).filter(ImageFilter.GaussianBlur(0.5)).convert("RGB")


def chart_image() -> Image.Image:
    img = Image.new("RGB", (1200, 700), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 30), CHART[0], fill="black", font=font(44))
    for i, item in enumerate(CHART[1:]):
        region, pct = item.split()
        val = int(pct.rstrip("%"))
        y = 140 + i * 130
        d.text((60, y + 20), region, fill="black", font=font(38))
        d.rectangle((260, y, 260 + val * 7, y + 80), fill=(70, 110, 200))
        d.text((280 + val * 7, y + 20), pct, fill="black", font=font(38))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "mixed.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4

    def write_lines(lines, y0=H - 80):
        y = y0
        for i, line in enumerate(lines):
            c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 16 if i == 0 else 11)
            c.drawString(60, y, line)
            y -= 26 if i == 0 else 18
        return y

    y = write_lines(PAGE1)
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    t = Table(TABLE1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    tw, th = t.wrap(W, H)
    t.drawOn(c, 60, y - th - 10)
    c.showPage()

    buf = io.BytesIO()
    scanned_page(PAGE2).save(buf, format="PNG")
    c.drawImage(ImageReader(io.BytesIO(buf.getvalue())), 0, 0, W, H)
    c.showPage()

    y = write_lines(PAGE3)
    buf = io.BytesIO()
    chart_image().save(buf, format="PNG")
    c.drawImage(ImageReader(io.BytesIO(buf.getvalue())), 60, y - 300, 420, 245)
    c.showPage()

    buf = io.BytesIO()
    scanned_table_page().save(buf, format="PNG")
    c.drawImage(ImageReader(io.BytesIO(buf.getvalue())), 0, 0, W, H)
    c.showPage()
    c.save()

    photo = scanned_page(PAGE2[:3], angle=-3, seed=1).crop((0, 0, 1654, 700))
    photo.save(OUT / "photo.png")

    (OUT / "ground_truth.json").write_text(json.dumps(
        {"page1": PAGE1, "page2": PAGE2, "page3": PAGE3, "chart": CHART, "table1": TABLE1,
         "scan_table_title": SCAN_TABLE_TITLE, "scan_table": SCAN_TABLE,
         "photo": PAGE2[:3]}, indent=2))
    print("wrote", pdf_path, OUT / "photo.png")


if __name__ == "__main__":
    main()
