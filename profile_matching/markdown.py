"""Small Markdown formatting helpers shared by the reports."""

from __future__ import annotations


def join(items: list[str], empty: str = "none") -> str:
    return ", ".join(items) if items else empty


def sentences(items: list[str], empty: str = "none") -> str:
    """Sentence-like items as one line: "A; b; c." (trailing periods dropped)."""
    cleaned = [item.strip().rstrip(".") for item in items if item.strip()]
    return "; ".join(cleaned) + "." if cleaned else empty


def cell(text: object) -> str:
    """Text safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def or_dash(value: object) -> str:
    return "–" if value is None else str(value)


def table(header: list[str], rows: list[list[object]]) -> list[str]:
    """A Markdown table as lines, cells escaped."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return lines
