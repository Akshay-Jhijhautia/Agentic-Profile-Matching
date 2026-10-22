"""Shared paths for the test suite.

Tests rank against a frozen copy of the 12 hand-written sample resumes, so the
demo dataset in ``resumes/`` can grow without changing expected rankings.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESUMES_DIR = ROOT / "tests" / "fixtures" / "resumes"
JD_DIR = ROOT / "job_descriptions"
