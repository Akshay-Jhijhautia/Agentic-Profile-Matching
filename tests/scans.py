"""Build scanned (image-only) resumes for OCR tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCANNED_RESUME = [
    "Jordan Rivera",
    "jordan.rivera@example.com | Austin, TX",
    "",
    "SUMMARY",
    "Frontend engineer with 5 years of experience building React and TypeScript applications.",
    "",
    "EXPERIENCE",
    "Senior Frontend Engineer, Acme Payments (Mar 2021 - Present)",
    "- Built a design system in React adopted by 8 product teams.",
    "",
    "SKILLS",
    "React, TypeScript, CSS, Jest",
    "",
    "EDUCATION",
    "B.S. Computer Science, University of Texas at Austin, 2019",
]


def scanned_page(lines: list[str] = SCANNED_RESUME) -> Image.Image:
    """An A4 page at 300 dpi with 11pt text, as a scanner would produce."""
    page = Image.new("RGB", (2480, 3508), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=46)
    for row, line in enumerate(lines):
        draw.text((200, 200 + 70 * row), line, fill="black", font=font)
    return page


def write_scanned_pdf(path: Path, lines: list[str] = SCANNED_RESUME) -> Path:
    scanned_page(lines).save(path, "PDF", resolution=300)
    return path


def write_scanned_image(path: Path, lines: list[str] = SCANNED_RESUME) -> Path:
    scanned_page(lines).save(path)
    return path
