"""Shared paths for the test suite.

Tests rank the 12 hand-written sample resumes against the sample job
descriptions; both live in ``tests/fixtures`` and are never shipped.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESUMES_DIR = ROOT / "tests" / "fixtures" / "resumes"
JD_DIR = ROOT / "tests" / "fixtures" / "job_descriptions"
