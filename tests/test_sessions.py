import builtins
import dataclasses
import io
import os
import shlex
import shutil
from datetime import datetime, timedelta

import pytest
from rich.console import Console

from profile_matching import app, cli, config, maintenance, sessions
from profile_matching.agent.graph import build_graph
from profile_matching.pools import ResumePool
from profile_matching.schemas import IntentDecision, Requirements
from profile_matching.services import Services
from tests.fakes import SCREENING_RESPONSES, FakeChatModel
from tests.paths import RESUMES_DIR
from tests.test_conversation import ROLE, route_by_keyword


@pytest.fixture
def home(services, tmp_path):
    """``services`` storing sessions under a temporary data home, with a fake model."""
    services.settings = dataclasses.replace(
        services.settings, sessions_db=str(tmp_path / "home" / "sessions.sqlite"), screening_rounds=1,
    )
    services.model = FakeChatModel({Requirements: ROLE, IntentDecision: route_by_keyword, **SCREENING_RESPONSES})
    return services


def _session(services, session_id=None):
    saver = sessions.open_saver(services.settings.sessions_db)
    graph = build_graph(services, checkpointer=saver, conversational=True)
    return cli.ChatSession(graph, Console(file=io.StringIO(), width=120), session_id=session_id), saver


@pytest.fixture
def scripted_input(monkeypatch):
    def script(*lines):
        answers = iter(lines)
        monkeypatch.setattr(builtins, "input", lambda *_: next(answers))
    return script


# --- saving and resuming ------------------------------------------------------------------------------


def test_sessions_are_saved_and_resumed_after_a_restart(home):
    first, _ = _session(home)
    first.send("Find me a Python backend engineer; Django is a plus")
    session_id, shortlist = first.session_id, first.values["shortlist"]

    resumed, saver = _session(home, session_id)  # a new graph and connection, as after a restart
    assert resumed.values["shortlist"] == shortlist
    assert resumed.report == first.report and resumed.report.startswith("# Shortlist")
    (info,) = sessions.list_sessions(resumed.graph, saver)
    assert (info.id, info.role) == (session_id, "Python Backend Engineer")
    assert info.messages == len(first.values["messages"]) and session_id in info.line()

    sessions.delete_session(saver, session_id)
    assert not sessions.exists(saver, session_id) and sessions.list_sessions(resumed.graph, saver) == []


def test_session_ids_are_readable_and_unique():
    first, second = sessions.new_session_id(), sessions.new_session_id()
    assert first != second and first.startswith(f"{datetime.now():%Y%m%d}-")


def test_reset_starts_a_new_session_with_the_loaded_resumes(home, tmp_path):
    folder = tmp_path / "cvs"
    folder.mkdir()
    shutil.copy(RESUMES_DIR / "resume_john_doe.pdf", folder)
    chat, _ = _session(home)
    chat.send(f"/resumes {folder}")
    old_id = chat.session_id
    chat.reset()
    assert chat.session_id != old_id
    assert chat.initial_state["pool_folder"] == str(folder.resolve())
    chat.send("Find me a Python backend engineer; Django is a plus")
    assert {m.name for m in chat.values["candidates"]} == {"John Doe"}


def test_chat_resumes_a_session_and_reloads_its_resumes(home, tmp_path, monkeypatch, scripted_input, capsys):
    folder = tmp_path / "cvs"
    folder.mkdir()
    shutil.copy(RESUMES_DIR / "resume_john_doe.pdf", folder)
    first, _ = _session(home)
    first.send(f"/resumes {folder}")
    home.pool = None  # a fresh process has nothing loaded

    monkeypatch.setattr(Services, "from_settings", lambda: home)
    scripted_input("/exit")
    assert app.main(["chat", "--session", first.session_id]) == 0
    assert home.pool.folder == folder.resolve()
    captured = capsys.readouterr()
    assert "1 ready" in captured.err

    shutil.rmtree(folder)
    home.pool = None
    scripted_input("/exit")
    assert app.main(["chat", "--session", first.session_id]) == 0
    assert "are gone; load new ones" in capsys.readouterr().err


def test_unknown_session_and_conflicting_options(home, monkeypatch, capsys):
    monkeypatch.setattr(Services, "from_settings", lambda: home)
    assert app.main(["chat", "--session", "nope"]) == 1
    assert "No saved session 'nope'" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        app.main(["chat", "--session", "x", "--resumes", str(RESUMES_DIR)])


def test_sessions_command_lists_and_deletes(home, monkeypatch, capsys):
    monkeypatch.setattr(Services, "from_settings", lambda: home)
    assert app.main(["sessions"]) == 0
    assert "No saved sessions yet" in capsys.readouterr().out
    chat, _ = _session(home)
    chat.send("Find me a Python backend engineer; Django is a plus")
    assert app.main(["sessions"]) == 0
    assert chat.session_id in capsys.readouterr().out
    assert app.main(["sessions", "--delete", chat.session_id]) == 0
    assert app.main(["sessions", "--delete", chat.session_id]) == 1


# --- cleaning up --------------------------------------------------------------------------------------------


def test_clean_finds_old_uploads_orphaned_indexes_and_old_sessions(home, tmp_path, monkeypatch, capsys):
    # Its own index database: the shared test one holds other tests' (rightly orphaned) indexes.
    settings = dataclasses.replace(home.settings, uploads_dir=str(tmp_path / "uploads"), embedder="hashing",
                                   db_dir=str(tmp_path / "db"))
    old_batch = tmp_path / "uploads" / "session-old"
    old_batch.mkdir(parents=True)
    shutil.copy(RESUMES_DIR / "resume_alex_lee.txt", old_batch)
    ResumePool.open(old_batch, settings).sync()
    month_ago = (datetime.now() - timedelta(days=40)).timestamp()
    for path in (old_batch / "resume_alex_lee.txt", old_batch):
        os.utime(path, (month_ago, month_ago))
    new_batch = tmp_path / "uploads" / "session-new"
    new_batch.mkdir()

    deleted_folder = tmp_path / "gone"
    deleted_folder.mkdir()
    shutil.copy(RESUMES_DIR / "resume_john_doe.pdf", deleted_folder)
    ResumePool.open(deleted_folder, settings).sync()
    shutil.rmtree(deleted_folder)

    chat, saver = _session(home)
    chat.send("Find me a Python backend engineer; Django is a plus")
    monkeypatch.setattr(sessions, "last_updated", lambda _: {chat.session_id: datetime.now().astimezone() - timedelta(days=40)})

    stale = maintenance.find_stale(settings, older_than_days=30)
    assert {(item.kind, item.name) for item in stale} == {
        ("uploads", "session-old"), ("session", chat.session_id),
        ("index", next(item.collection for item in stale if item.kind == "index")),
    }
    assert "folder" in next(item.detail for item in stale if item.kind == "index")

    monkeypatch.setattr(app, "get_settings", lambda: settings)
    assert app.main(["clean"]) == 0
    assert "would be removed" in capsys.readouterr().out and old_batch.exists()
    assert app.main(["clean", "--yes"]) == 0
    assert not old_batch.exists() and new_batch.exists()
    assert not sessions.exists(saver, chat.session_id)
    monkeypatch.setattr(sessions, "last_updated", lambda _: {})
    assert maintenance.find_stale(settings, older_than_days=30) == []


def test_clean_removes_a_chromadb_index_left_by_an_older_version(home, tmp_path, monkeypatch):
    settings = dataclasses.replace(home.settings, uploads_dir=str(tmp_path / "uploads"), embedder="hashing",
                                   db_dir=str(tmp_path / "db"))
    monkeypatch.setattr(sessions, "last_updated", lambda _: {})
    folder = _resume_folder(tmp_path)
    ResumePool.open(folder, settings).sync()
    legacy_db = tmp_path / "db" / maintenance.LEGACY_CHROMA_FILE
    legacy_db.write_bytes(b"x" * 2_000_000)
    legacy_collection = tmp_path / "db" / "0b7e0d6e-3c41-4ad0-9d3b-0a4b8f6f5a11"
    legacy_collection.mkdir()
    (legacy_collection / "data_level0.bin").write_bytes(b"x")
    (tmp_path / "db" / "notes").mkdir()  # not a ChromaDB folder: left alone

    (stale,) = maintenance.find_stale(settings, older_than_days=30)
    assert (stale.kind, stale.name) == ("index", "ChromaDB index (before 1.1)") and "2.0 MB" in stale.detail
    maintenance.remove(settings, [stale])
    assert not legacy_db.exists() and not legacy_collection.exists() and (tmp_path / "db" / "notes").exists()
    assert ResumePool.open(folder, settings).size == 1  # the current index is untouched
    assert maintenance.find_stale(settings, older_than_days=30) == []


def _resume_folder(tmp_path):
    folder = tmp_path / "cvs"
    folder.mkdir()
    shutil.copy(RESUMES_DIR / "resume_john_doe.pdf", folder)
    return folder


# --- command line and configuration -------------------------------------------------------------------------


def test_version(capsys):
    assert app.main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("profile-match ")


def test_env_files_from_the_folder_and_the_data_home(tmp_path, monkeypatch):
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    (home / ".env").write_text("OPENAI_MODEL=from-home\nSHORTLIST_SIZE=7\n")
    (work / ".env").write_text("OPENAI_MODEL=from-folder\n")
    for name in ("OPENAI_MODEL", "SHORTLIST_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROFILE_MATCH_HOME", str(home))
    monkeypatch.chdir(work)
    config.get_settings.cache_clear()
    try:
        settings = config.get_settings()
    finally:
        config.get_settings.cache_clear()
        for name in ("OPENAI_MODEL", "SHORTLIST_SIZE"):
            os.environ.pop(name, None)
    assert settings.llm_model == "from-folder"  # the current folder's .env wins
    assert settings.shortlist_size == 7  # the data home's .env fills in the rest
    assert settings.sessions_db == str(home / "sessions.sqlite")


def test_a_resumed_upload_session_is_labelled_as_uploads(home, tmp_path, monkeypatch, scripted_input, capsys):
    home.settings = dataclasses.replace(home.settings, uploads_dir=str(tmp_path / "uploads"))
    first, _ = _session(home)
    first.send(f"/add {shlex.quote(str(RESUMES_DIR / 'resume_john_doe.pdf'))}")  # path has spaces
    home.pool = None
    monkeypatch.setattr(Services, "from_settings", lambda: home)
    scripted_input("/exit")
    assert app.main(["chat", "--session", first.session_id]) == 0
    assert "Uploaded resumes: 1 ready" in capsys.readouterr().err

