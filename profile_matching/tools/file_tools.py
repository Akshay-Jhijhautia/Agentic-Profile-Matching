"""File system tools (``fs_tools``), exposed to the agent.

Reads are unrestricted so the agent can open any resume or job description;
writes are confined to the configured reports directory.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool

from ..rag import fs_tools
from ..services import Services
from ._base import agent_tool


def resolve_inside(root: Path, filepath: str) -> Path:
    """``filepath`` (relative to ``root``, or absolute) if it stays inside ``root``."""
    target = (root / Path(filepath).expanduser()).resolve()
    if target != root and root not in target.parents:
        raise PermissionError(f"Can only write inside {root}, not {filepath}")
    return target


def make_file_tools(services: Services) -> list[BaseTool]:
    reports_dir = Path(services.settings.reports_dir).expanduser().resolve()

    @agent_tool
    def read_file(filepath: str) -> dict:
        """Read a resume or job description (PDF, DOCX, TXT or MD) and return its text and metadata.

        Args:
            filepath: Path to the file.
        """
        return fs_tools.read_file(filepath)

    @agent_tool
    def list_files(directory: str, extension: str | None = None) -> dict:
        """List the files in a directory (not recursive) with size and modified date.

        Args:
            directory: Directory to list, e.g. "resumes".
            extension: Optional filter such as ".pdf" or "docx".
        """
        return {"success": True, "directory": directory, "files": fs_tools.list_files(directory, extension)}

    @agent_tool
    def write_file(filepath: str, content: str) -> dict:
        """Save text (Markdown or plain text) to a file in the reports directory, replacing it if it exists.

        Args:
            filepath: File name or path relative to the reports directory, e.g. "maria_garcia.md".
            content: The text to write.
        """
        return fs_tools.write_file(str(resolve_inside(reports_dir, filepath)), content)

    @agent_tool
    def search_in_file(filepath: str, keyword: str) -> dict:
        """Find every case-insensitive occurrence of a keyword in one file, with line numbers and context.

        Args:
            filepath: File to search.
            keyword: Word or phrase to find.
        """
        return fs_tools.search_in_file(filepath, keyword)

    return [read_file, list_files, write_file, search_in_file]
