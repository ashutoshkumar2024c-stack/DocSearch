"""Render pages of a document to images (thumbnails for the UI, inputs for visual retrieval)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, ImageSequence

from .extract import IMAGE_EXT


def page_count(path: str | Path) -> int:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    if ext in IMAGE_EXT:
        with Image.open(path) as img:
            return getattr(img, "n_frames", 1)
    return 0


def render_page(path: str | Path, page: int | None = None, dpi: int = 110) -> Image.Image | None:
    """Return page `page` (1-based; None = first) as an RGB image, or None for non-visual formats."""
    path = Path(path)
    ext = path.suffix.lower()
    idx = (page or 1) - 1
    if ext == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        try:
            if idx >= len(pdf):
                return None
            return pdf[idx].render(scale=dpi / 72).to_pil().convert("RGB")
        finally:
            pdf.close()
    if ext in IMAGE_EXT:
        with Image.open(path) as img:
            frames = list(ImageSequence.Iterator(img))
            if idx >= len(frames):
                return None
            return ImageOps.exif_transpose(frames[idx].copy()).convert("RGB")
    return None


def save_thumbnails(path: str | Path, out_dir: str | Path, max_width: int = 900) -> list[Path]:
    """Write page-N.png previews for PDFs and images. Returns the files written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for n in range(1, page_count(path) + 1):
        img = render_page(path, n)
        if img is None:
            continue
        if img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)), Image.LANCZOS)
        target = out_dir / f"page-{n}.png"
        img.save(target, optimize=True)
        written.append(target)
    return written
