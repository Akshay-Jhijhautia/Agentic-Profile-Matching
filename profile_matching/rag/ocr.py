"""Optional OCR for scanned resumes: PDFs without a text layer, and images.

Uses RapidOCR (ONNX models, installed with pip, no system dependencies) and
pypdfium2 to render PDF pages. Both are optional: without them, scanned files
are reported as unreadable instead of being read. Imports are deferred so the
models only load when a scanned file actually turns up.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# Page render resolution: readable for 10-11pt resume text, faster than 300 dpi.
RENDER_DPI = 200
INSTALL_HINT = 'install OCR support with: pip install "profile-match[ocr]"'


@lru_cache(maxsize=1)
def available() -> bool:
    return all(importlib.util.find_spec(name) for name in ("rapidocr", "pypdfium2", "onnxruntime"))


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR

    # Resume pages are upright: the orientation classifier only risks flipping long lines.
    # One line per recognition batch keeps long lines from being squeezed ("UI" read as "Ul").
    return RapidOCR(params={"Global.log_level": "critical", "Global.use_cls": False, "Rec.rec_batch_num": 1})


def _recognise(image) -> str:
    return "\n".join(_engine()(image).txts or ())


def pdf_text(path: Path) -> str:
    """The text of every page of a scanned PDF, in order."""
    import pypdfium2

    document = pypdfium2.PdfDocument(str(path))
    try:
        return "\n".join(_recognise(page.render(scale=RENDER_DPI / 72).to_pil()) for page in document).strip()
    finally:
        document.close()


def image_text(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        return _recognise(image.convert("RGB")).strip()
