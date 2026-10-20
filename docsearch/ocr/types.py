"""Shared data types for the parsing / OCR stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Route = Literal["native", "ocr", "hybrid", "visual", "parser"]
ElementKind = Literal["heading", "text", "table", "figure"]


@dataclass
class Element:
    """One layout element on a page: a heading, a text block, a table or a figure.

    Tables are stored as Markdown so the LLM sees rows and columns, not a flat
    word soup. Figures hold OCR'd text and/or a vision-model description.
    """

    kind: ElementKind
    text: str
    level: int = 0                    # heading level (1 = top); 0 for non-headings
    tool: str = ""                    # which tool produced it, e.g. "pdfplumber", "tesseract"
    confidence: float | None = None   # OCR confidence 0-100 where the tool reports one

    def render(self) -> str:
        if self.kind == "heading":
            return "#" * max(1, min(self.level, 6)) + " " + self.text
        if self.kind == "figure":
            return f"[Figure]\n{self.text}"
        return self.text


@dataclass
class PagePlan:
    """What the classifier decided for one page/unit, which tools it assigned, and why."""

    page: int | None
    doc_type: str                         # e.g. "digital-pdf", "scanned-pdf", "photo", "docx"
    route: Route
    tools: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    fallback: str | None = None           # engine to retry with on low confidence

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageText:
    """The parsed content of one page (or of a whole file for non-paged formats)."""

    source: str
    text: str
    page: int | None = None
    method: str = "native"                # main tool used: native | parser | tesseract | easyocr | ...
    route: Route | None = None
    ocr_confidence: float | None = None   # mean OCR word confidence 0-100, if any OCR ran
    content_type: str = "text"
    elements: list[Element] = field(default_factory=list)
    plan: PagePlan | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_elements(cls, source: str, elements: list[Element], **kw) -> "PageText":
        text = "\n\n".join(e.render() for e in elements if e.text.strip())
        return cls(source=source, text=text, elements=elements, **kw)

    @property
    def ocr_used(self) -> bool:
        return self.method not in ("native", "parser")

    def metadata(self) -> dict:
        """Flat metadata dict for the vector store (no None values / nesting)."""
        md = {"source": self.source, "method": self.method, "ocr_used": self.ocr_used,
              "content_type": self.content_type}
        if self.page is not None:
            md["page"] = self.page
        if self.route is not None:
            md["route"] = self.route
        if self.ocr_confidence is not None:
            md["ocr_confidence"] = round(float(self.ocr_confidence), 1)
        return md

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ocr_used"] = self.ocr_used
        return d

    def to_langchain(self):
        from langchain_core.documents import Document
        return Document(page_content=self.text, metadata=self.metadata())


# Backwards-compatible alias (earlier versions returned PageRoute from the router)
PageRoute = PagePlan
