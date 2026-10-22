import os
import shutil
from pathlib import Path

import pytest

from profile_matching import pools, screening
from profile_matching.candidates import trim_resume
from profile_matching.config import Settings
from profile_matching.pools import pool_name, resolve_folder
from profile_matching.rag import fs_tools, ocr
from profile_matching.schemas import Requirements
from profile_matching.services import Services
from tests.paths import RESUMES_DIR
from tests.scans import write_scanned_image, write_scanned_pdf

needs_ocr = pytest.mark.skipif(not ocr.available(), reason="OCR extras not installed")

ROLE = Requirements.model_validate({
    "role_title": "Engineer", "must_have": [], "nice_to_have": ["Python"],
    "min_total_years": None, "min_degree": None,
})


def _folder(root, name: str, files: list[str]):
    folder = root / name
    folder.mkdir()
    for file in files:
        shutil.copy(RESUMES_DIR / file, folder)
    return folder


@pytest.fixture
def offline(tmp_path) -> Services:
    return Services(settings=Settings(db_dir=str(tmp_path / "db"), embedder="hashing"))


# --- naming and folders -------------------------------------------------------------------------------


def test_pool_names_are_readable_and_unique_per_path(tmp_path):
    first, second = tmp_path / "a" / "Applicants 2026", tmp_path / "b" / "Applicants 2026"
    assert pool_name(first).startswith("pool-applicants-2026-")
    assert pool_name(first) != pool_name(second)
    assert pool_name(first) == pool_name(first)


def test_resolve_folder_rejects_missing_and_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Resume folder not found"):
        resolve_folder(tmp_path / "missing")
    (tmp_path / "file.txt").write_text("x")
    with pytest.raises(NotADirectoryError):
        resolve_folder(tmp_path / "file.txt")


# --- syncing --------------------------------------------------------------------------------------------


def test_sync_is_incremental(offline, tmp_path):
    folder = _folder(tmp_path, "applicants", ["resume_alex_lee.txt", "resume_john_doe.pdf"])
    first = offline.use_folder(folder)
    assert (first.ready, len(first.indexed), first.unchanged) == (2, 2, 0)

    again = offline.use_folder(folder)
    assert (again.ready, len(again.indexed), again.unchanged) == (2, 0, 2)

    shutil.copy(RESUMES_DIR / "resume_maria_garcia.pdf", folder)
    (folder / "resume_alex_lee.txt").unlink()
    changed = offline.use_folder(folder)
    assert changed.ready == 2 and len(changed.indexed) == 1 and len(changed.removed) == 1
    assert {p.name for p in offline.candidates.all()} == {"John Doe", "Maria Garcia"}


def test_pools_keep_folders_apart(offline, tmp_path):
    backend = _folder(tmp_path, "backend", ["resume_john_doe.pdf", "resume_alex_lee.txt"])
    frontend = _folder(tmp_path, "frontend", ["resume_maria_garcia.pdf"])
    offline.use_folder(backend)
    backend_pool = offline.pool
    offline.use_folder(frontend)
    frontend_pool = offline.pool
    assert {m.name for m in screening.evaluate(offline, ROLE)} == {"Maria Garcia"}
    offline.pool = backend_pool
    assert {m.name for m in screening.evaluate(offline, ROLE)} == {"John Doe", "Alex Lee"}
    # The same candidate id in two folders never shares a round-2 cache entry.
    keys = {screening.assessment_key(pool.id, ROLE, "john_doe") for pool in (backend_pool, frontend_pool)}
    assert len(keys) == 2


def test_intake_report_explains_every_skipped_file(offline, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)
    folder = _folder(tmp_path, "applicants", ["resume_alex_lee.txt"])
    (folder / "old.doc").write_bytes(b"binary")
    (folder / "sheet.xlsx").write_bytes(b"binary")
    (folder / "empty.txt").write_text("   ")
    write_scanned_pdf(folder / "scan.pdf")
    write_scanned_image(folder / "photo.png")

    report = offline.use_folder(folder)
    reasons = {item["file"]: item["reason"] for item in report.skipped}
    assert report.ready == 1
    assert "save it as .docx or PDF" in reasons["old.doc"]
    assert "Unsupported file type '.xlsx'" in reasons["sheet.xlsx"]
    assert "No text could be extracted" in reasons["empty.txt"]
    assert "scanned PDF" in reasons["scan.pdf"] and "pip install" in reasons["scan.pdf"]
    assert "need OCR support" in reasons["photo.png"]
    summary = report.summary()
    assert summary.startswith("Resumes in ") and "1 ready" in summary and "Skipped 5 file(s):" in summary


def test_an_empty_folder_is_reported_and_not_ready(offline, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    report = offline.use_folder(folder)
    assert report.ready == 0 and "No readable resumes" in report.summary()
    with pytest.raises(ValueError, match="No readable resumes"):
        offline.check_ready()


def test_services_without_a_folder_explain_what_to_do():
    with pytest.raises(ValueError, match="No resume folder is loaded"):
        Services(settings=Settings()).candidates.all()


# --- scanned resumes ------------------------------------------------------------------------------------


@needs_ocr
def test_scanned_pdfs_and_images_are_read_with_ocr(offline, tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    write_scanned_pdf(folder / "resume_jordan_rivera.pdf")
    result = fs_tools.read_file(str(folder / "resume_jordan_rivera.pdf"))
    assert result["metadata"]["ocr"] is True
    assert "Frontend engineer with 5 years of experience" in result["content"]
    image = write_scanned_image(tmp_path / "photo.jpg")
    assert "Jordan Rivera" in fs_tools.read_file(str(image))["content"]

    report = offline.use_folder(folder)
    assert report.ready == 1 and len(report.ocr) == 1 and "Read 1 scanned file(s) with OCR" in report.summary()
    (profile,) = offline.candidates.all()
    assert profile.name == "Jordan Rivera" and "React" in profile.metadata["skills"]


def test_pdfs_with_a_text_layer_skip_ocr():
    result = fs_tools.read_file(str(RESUMES_DIR / "resume_john_doe.pdf"))
    assert "ocr" not in result["metadata"] and "John Doe" in result["content"]


# --- trimming long resumes ------------------------------------------------------------------------------


def test_trimming_keeps_the_sections_that_matter_whole():
    text = "\n".join([
        "Ada Lovelace", "ada@example.com", "", "PROJECTS", "p" * 3000, "",
        "EXPERIENCE", "Engineer, Acme (2019 - Present)", "- Built analytical engines", "",
        "SKILLS", "Python, SQL", "", "SUMMARY", "Senior engineer.",
    ])
    trimmed = trim_resume(text, 300)
    assert len(trimmed) == 300
    assert trimmed.index("SUMMARY") < trimmed.index("EXPERIENCE") < trimmed.index("SKILLS") < trimmed.index("PROJECTS")
    assert "- Built analytical engines" in trimmed and "Python, SQL" in trimmed
    assert trim_resume("short", 300) == "short"


def test_resume_text_is_trimmed_by_section(services):
    profile = services.candidates.resolve("John Doe")
    full = services.candidates.resume_text(profile, max_chars=None)
    assert services.candidates.resume_text(profile, max_chars=len(full) + 10) == full
    assert len(services.candidates.resume_text(profile, max_chars=200)) == 200


def test_unchanged_files_are_not_read_again(offline, tmp_path, monkeypatch):
    folder = _folder(tmp_path, "applicants", ["resume_alex_lee.txt", "resume_john_doe.pdf"])
    offline.use_folder(folder)
    reads = []
    real_read = fs_tools.read_file
    monkeypatch.setattr(fs_tools, "read_file", lambda path: reads.append(path) or real_read(path))

    assert offline.use_folder(folder).unchanged == 2 and reads == []

    # Touched but identical: read once to compare content, kept, then recognised by its stamp.
    touched = folder / "resume_alex_lee.txt"
    stat = touched.stat()
    os.utime(touched, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    report = offline.use_folder(folder)
    assert len(reads) == 1 and report.indexed == [] and report.unchanged == 2
    reads.clear()
    offline.use_folder(folder)
    assert reads == []


@needs_ocr
def test_scanned_files_are_read_with_ocr_only_once(offline, tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    write_scanned_pdf(folder / "resume_jordan_rivera.pdf")
    assert len(offline.use_folder(folder).ocr) == 1
    again = offline.use_folder(folder)
    assert again.ocr == [] and again.unchanged == 1 and "OCR" not in again.summary()


# --- uploads ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("/x/a.pdf /x/b.docx", ["/x/a.pdf", "/x/b.docx"]),
        (r"/Users/me/My\ Resumes/Jane\ Doe.pdf", ["/Users/me/My Resumes/Jane Doe.pdf"]),  # dragged into Terminal
        ("'/x/Jane Doe.pdf' \"/x/Raj K.docx\"", ["/x/Jane Doe.pdf", "/x/Raj K.docx"]),
        ("/x/it's.pdf", ["/x/it's.pdf"]),  # unbalanced quote: split on whitespace
    ],
)
def test_split_paths(text, expected):
    assert pools.split_paths(text) == expected


def test_classify_folder_files_and_mistakes(tmp_path):
    folder = _folder(tmp_path, "batch", ["resume_alex_lee.txt", "resume_john_doe.pdf"])
    files = [str(folder / "resume_alex_lee.txt"), str(folder / "resume_john_doe.pdf")]
    assert pools.classify([str(folder)]) == (folder, [])
    assert pools.classify(files) == (None, [Path(f) for f in files])
    with pytest.raises(FileNotFoundError, match="Not found"):
        pools.classify([str(tmp_path / "nope.pdf")])
    with pytest.raises(ValueError, match="not a mix"):
        pools.classify([str(folder), files[0]])


def test_upload_copies_once_and_renames_clashes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "cv.txt").write_text("Jane Doe")
    (b / "cv.txt").write_text("Raj Kumar")
    batch = tmp_path / "batch"
    assert pools.upload([a / "cv.txt"], batch) == ["cv.txt"]
    assert pools.upload([a / "cv.txt", b / "cv.txt"], batch) == ["cv.txt", "cv-2.txt"]
    assert sorted(p.name for p in batch.iterdir()) == ["cv-2.txt", "cv.txt"]
    assert (batch / "cv-2.txt").read_text() == "Raj Kumar"


def test_batch_for_is_stable_per_file_set(tmp_path):
    files = [tmp_path / "b.pdf", tmp_path / "a.pdf"]
    assert pools.batch_for(files, str(tmp_path)) == pools.batch_for(list(reversed(files)), str(tmp_path))
    assert pools.batch_for(files[:1], str(tmp_path)) != pools.batch_for(files, str(tmp_path))
    assert pools.new_batch(str(tmp_path)) != pools.new_batch(str(tmp_path))


def test_use_paths_uploads_files_and_keeps_originals(offline, tmp_path):
    source = _folder(tmp_path, "downloads", ["resume_alex_lee.txt", "resume_john_doe.pdf"])
    offline.settings = Settings(db_dir=offline.settings.db_dir, embedder="hashing",
                                uploads_dir=str(tmp_path / "uploads"))
    files = [str(source / "resume_alex_lee.txt"), str(source / "resume_john_doe.pdf")]
    report, batch = offline.use_paths(files)
    assert report.folder == "your uploaded resumes" and report.ready == 2
    assert batch.parent == tmp_path / "uploads" and offline.pool.folder == batch
    assert sorted(p.name for p in source.iterdir()) == ["resume_alex_lee.txt", "resume_john_doe.pdf"]
    again, same_batch = offline.use_paths(files)  # one-shot re-runs reuse the batch and its index
    assert same_batch == batch and again.unchanged == 2

    folder_report, no_batch = offline.use_paths([str(source)])
    assert no_batch is None and offline.pool.folder == source and folder_report.ready == 2
