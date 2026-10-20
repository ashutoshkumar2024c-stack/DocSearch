"""Document classification, automatic tool routing, layout analysis and OCR."""

from .engines import (EasyOCREngine, OCRResult, PaddleOCREngine, TesseractEngine, VisionLLMEngine,
                      capabilities, get_engine, set_capabilities)
from .extract import SUPPORTED_EXT, ExtractionError, Extractor, extract, summarize
from .metrics import cer, wer
from .quality import ImageProfile, profile_image
from .router import RouterConfig, choose_ocr, plan_pdf_page
from .types import Element, PagePlan, PageRoute, PageText

__all__ = [
    "Extractor", "extract", "summarize", "ExtractionError", "SUPPORTED_EXT",
    "Element", "PageText", "PagePlan", "PageRoute", "RouterConfig", "plan_pdf_page", "choose_ocr",
    "ImageProfile", "profile_image",
    "TesseractEngine", "EasyOCREngine", "PaddleOCREngine", "VisionLLMEngine", "OCRResult",
    "get_engine", "capabilities", "set_capabilities", "cer", "wer",
]
