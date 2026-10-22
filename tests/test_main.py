import shutil

import pytest

import main
from profile_matching import config
from profile_matching.schemas import Requirements
from profile_matching.services import Services
from tests.fakes import SCREENING_RESPONSES, FakeChatModel
from tests.paths import RESUMES_DIR

ROLE = Requirements.model_validate({
    "role_title": "Backend Engineer", "must_have": [], "nice_to_have": ["Python"],
    "min_total_years": None, "min_degree": None,
})


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    """Settings from the environment: a temporary index built with the hashing embedder."""
    monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("EMBEDDER", "hashing")
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "applicants"
    path.mkdir()
    for name in ("resume_alex_lee.txt", "resume_john_doe.pdf", "resume_maria_garcia.pdf"):
        shutil.copy(RESUMES_DIR / name, path)
    return path


def test_index_then_list_a_folder(offline_env, folder, capsys):
    assert main.main(["index", str(folder)]) == 0
    assert "3 ready (3 new or changed, 0 unchanged)" in capsys.readouterr().out
    assert main.main(["index", str(folder)]) == 0  # incremental: nothing to redo
    assert "3 ready (0 new or changed, 3 unchanged)" in capsys.readouterr().out
    assert main.main(["list", str(folder)]) == 0
    listed = capsys.readouterr().out
    assert all(name in listed for name in ("Alex Lee", "John Doe", "Maria Garcia"))


def test_index_reports_missing_folders(offline_env, capsys):
    assert main.main(["index", "/no/such/folder"]) == 1
    assert "Resume folder not found" in capsys.readouterr().err


def test_screen_needs_a_job_description_and_chat_refuses_one(offline_env, folder, capsys):
    with pytest.raises(SystemExit):
        main.main(["screen", "--resumes", str(folder)])
    assert "screen needs a job description (text or --jd-file) and --resumes" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main.main(["chat", "React developer", "--resumes", str(folder)])
    assert "chat takes no job description" in capsys.readouterr().err


def test_help_and_unknown_command(capsys):
    assert main.main(["--help"]) == 0
    assert "python main.py screen" in capsys.readouterr().out
    assert main.main([]) == 1
    assert main.main(["deploy"]) == 2
    assert "Unknown command" in capsys.readouterr().err


def test_screen_uploaded_files_one_shot(offline_env, folder, monkeypatch, capsys):
    monkeypatch.setenv("UPLOADS_DIR", str(offline_env / "uploads"))
    monkeypatch.setenv("SCREENING_ROUNDS", "1")
    monkeypatch.setenv("SAVE_REPORTS", "false")
    config.get_settings.cache_clear()
    fake = FakeChatModel({Requirements: ROLE, **SCREENING_RESPONSES})
    monkeypatch.setattr(Services, "from_settings", lambda: Services(model=fake))
    files = [str(folder / "resume_john_doe.pdf"), str(folder / "resume_alex_lee.txt")]
    assert main.main(["screen", "Python backend engineer", "--resumes", *files]) == 0
    captured = capsys.readouterr()
    assert "Uploaded resumes: 2 ready" in captured.err
    assert captured.out.startswith("# Shortlist: Backend Engineer")
