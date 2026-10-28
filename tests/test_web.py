"""The local web API: guard, sessions, streamed turns, uploads and results."""

import dataclasses
import json
import shutil
from pathlib import Path

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from resume_shortlister import app as cli_app
from resume_shortlister.pools import ResumePool
from resume_shortlister.schemas import IntentDecision, InterviewQuestion, InterviewQuestions, Requirements
from resume_shortlister.services import Services
from resume_shortlister.web.app import TOKEN_HEADER, create_app
from tests.fakes import SCREENING_RESPONSES, FakeChatModel
from tests.paths import RESUMES_DIR
from tests.test_confirmation import IOS_JD, IOS_ROLE
from tests.test_conversation import ROLE, route_by_keyword

TOKEN = "test-token"
PYTHON_REQUEST = "Find me a Python backend engineer; Django is a plus"


def _services_factory(services, tmp_path, rounds: int):
    """Services per session, as the app makes them: the fixture resumes loaded, a fake model."""
    settings = dataclasses.replace(services.settings, sessions_db=str(tmp_path / "sessions.sqlite"),
                                   screening_rounds=rounds)
    model = FakeChatModel({Requirements: ROLE, IntentDecision: route_by_keyword, **SCREENING_RESPONSES})
    return lambda: Services(pool=ResumePool.open(RESUMES_DIR, settings), settings=settings, model=model)


def _client(make_services) -> TestClient:
    return TestClient(create_app(make_services, token=TOKEN), base_url="http://127.0.0.1",
                      headers={TOKEN_HEADER: TOKEN})


@pytest.fixture
def make_services(services, tmp_path):
    return _services_factory(services, tmp_path, rounds=1)


@pytest.fixture
def client(make_services):
    return _client(make_services)


def events(response: httpx.Response) -> list[tuple[str, dict]]:
    """The Server-Sent Events of a streamed turn, as (event name, data)."""
    assert response.status_code == 200, response.text
    found = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        found.append((lines["event"], json.loads(lines["data"])))
    return found


def _turn(client, session_id: str, text: str) -> tuple[list[dict], dict]:
    streamed = events(client.post(f"/api/sessions/{session_id}/turns", json={"text": text}))
    assert streamed[-1][0] == "snapshot"
    return [data for name, data in streamed[:-1]], streamed[-1][1]


def _new(client) -> str:
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


# --- guard and status -----------------------------------------------------------------------------------


def test_only_this_machine_and_this_page_may_call_the_api(client, make_services):
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/status", headers={TOKEN_HEADER: "wrong"}).status_code == 403
    assert TestClient(client.app, base_url="http://127.0.0.1").get("/api/status").status_code == 403  # no token
    # Another host name pointing at this machine (DNS rebinding) is refused, token or not.
    rebound = TestClient(client.app, base_url="http://evil.example", headers={TOKEN_HEADER: TOKEN})
    assert rebound.get("/api/status").status_code == 403
    page = client.get("/")
    assert page.status_code == 200 and f'content="{TOKEN}"' in page.text  # the page carries the token


def test_status_reports_model_and_readiness(client):
    status = client.get("/api/status").json()
    assert status["ready"] is True and status["problem"] is None
    assert status["model"] == "OpenAI gpt-5.6-terra" and status["embedder"] == "hashing"


# --- a screening ----------------------------------------------------------------------------------------


def test_a_turn_streams_progress_then_the_session(client):
    session_id = _new(client)
    streamed, snapshot = _turn(client, session_id, PYTHON_REQUEST)
    kinds = [e["kind"] for e in streamed]
    assert "Screened 12 resumes; 9 pass every must-have." in [e["text"] for e in streamed if e["kind"] == "note"]
    assert kinds.index("report") < len(kinds) - 1 and kinds[-1] == "message"

    assert snapshot["requirements"]["must_have"] == ["Python"] and snapshot["requirements"]["nice_to_have"] == ["Django"]
    assert (snapshot["screened"], snapshot["passing"], snapshot["waiting"]) == (12, 9, True)
    first = snapshot["shortlist"][0]
    assert first["rank"] == 1 and first["must_haves"] == "✓ all met" and first["passes_must_haves"]
    assert snapshot["report"].startswith("# Shortlist")
    assert [m["role"] for m in snapshot["messages"]] == ["user", "assistant"]

    assert client.get(f"/api/sessions/{session_id}").json()["shortlist"] == snapshot["shortlist"]
    (listed,) = client.get("/api/sessions").json()
    assert (listed["id"], listed["role"]) == (session_id, "Python Backend Engineer")


def test_open_requirements_are_confirmed_before_screening(client):
    session = client.app.state.workspace.get(_new(client))
    session.services.model.responses[Requirements] = IOS_ROLE
    streamed, snapshot = _turn(client, session.conversation.session_id, IOS_JD)
    assert snapshot["open_questions"] and snapshot["shortlist"] == [] and snapshot["waiting"]
    assert "Before screening" in streamed[-1]["text"]
    _, snapshot = _turn(client, session.conversation.session_id, "yes")
    assert snapshot["open_questions"] == [] and snapshot["screened"] == 12


def test_a_candidate_detail_explains_their_place_in_every_round(services, tmp_path):
    client = _client(_services_factory(services, tmp_path, rounds=3))
    session_id = _new(client)
    assert client.get(f"/api/sessions/{session_id}/candidates/john_doe").status_code == 404  # no search yet
    _, snapshot = _turn(client, session_id, PYTHON_REQUEST)
    row = snapshot["shortlist"][0]

    detail = client.get(f"/api/sessions/{session_id}/candidates/{row['id']}").json()
    assert (detail["rank"], detail["score"], detail["deep"]) == (row["rank"], row["score"], row["deep"])
    assert detail["recommendation"]["decision"] == row["decision"] and detail["recommendation"]["rule"]
    assert "/" not in detail["resume"]  # the file name, not a path on this computer
    components = detail["score_components"]
    assert components[-1]["component"] == "Deep analysis" and detail["rubric"]
    assert round(sum(c["points"] for c in components)) == detail["score"]
    assert detail["evidence"] and detail["evidence"][0]["verdict"] in ("strong", "partial", "missing")
    assert detail["strengths"] and detail["summary"] and detail["must_have_checks"]
    assert detail["report"].startswith(f"# {detail['name']}")

    # Knocked-out candidates were screened too: their round-1 record, without rounds 2 and 3.
    knocked_out = next(c for c in client.app.state.workspace.get(session_id).conversation.values["candidates"]
                       if not c.passes_must_haves)
    detail = client.get(f"/api/sessions/{session_id}/candidates/{knocked_out.id}").json()
    assert detail["knocked_out"] and detail["deep"] is None and detail["evidence"] == []
    assert client.get(f"/api/sessions/{session_id}/candidates/nobody_here").status_code == 404


def test_interview_questions_for_a_candidate(client):
    session_id = _new(client)
    session = client.app.state.workspace.get(session_id)
    session.services.model.responses[InterviewQuestions] = InterviewQuestions(questions=[
        InterviewQuestion(question="Walk me through a Django migration you owned.", focus="Django",
                          purpose="assess_depth", strong_answer="Schema and data steps, rollback."),
    ])
    _turn(client, session_id, PYTHON_REQUEST)
    plan = client.post(f"/api/sessions/{session_id}/candidates/john_doe/questions").json()
    assert plan["candidate_id"] == "john_doe" and set(plan["focus_areas"]) == {"gaps", "unverified", "strengths"}
    assert plan["questions"][0]["focus"] == "Django"
    assert client.post(f"/api/sessions/{session_id}/candidates/nobody_here/questions").status_code == 404

    def unavailable(_messages):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))

    session.services.model.responses[InterviewQuestions] = unavailable
    failed = client.post(f"/api/sessions/{session_id}/candidates/maria_garcia/questions")
    assert failed.status_code == 502 and "OpenAI" in failed.json()["detail"]


def test_the_report_download(client):
    session_id = _new(client)
    assert client.get(f"/api/sessions/{session_id}/report").status_code == 404  # no search yet
    _turn(client, session_id, PYTHON_REQUEST)
    download = client.get(f"/api/sessions/{session_id}/report")
    assert download.headers["content-type"].startswith("text/markdown")
    assert download.headers["content-disposition"] == f'attachment; filename="{session_id}.md"'


# --- resumes -------------------------------------------------------------------------------------------


def test_uploaded_files_join_the_session(client, make_services):
    session_id = _new(client)
    files = [("files", (name, (RESUMES_DIR / name).read_bytes())) for name in
             ("resume_john_doe.pdf", "resume_alex_lee.txt")]
    streamed = events(client.post(f"/api/sessions/{session_id}/files", files=files))
    snapshot = streamed[-1][1]
    assert snapshot["resumes"] == "your uploaded resumes"
    # The history says what the user did, not the command the app sent for them.
    assert snapshot["messages"][0] == {"role": "user", "text": "Upload 2 files: resume_john_doe.pdf, resume_alex_lee.txt"}
    assert "Uploaded resumes: 2 ready" in streamed[0][1]["text"]
    incoming = Path(make_services().settings.uploads_dir) / "incoming"
    assert not any(incoming.iterdir())  # staged files are cleaned up


def test_a_folder_is_loaded_and_reloaded_after_a_restart(client, make_services, tmp_path):
    folder = tmp_path / "applicants"
    folder.mkdir()
    shutil.copy(RESUMES_DIR / "resume_john_doe.pdf", folder)
    session_id = _new(client)
    assert client.post(f"/api/sessions/{session_id}/folder", json={"path": "/no/such/folder"}).status_code == 422
    streamed = events(client.post(f"/api/sessions/{session_id}/folder", json={"path": str(folder)}))
    assert "1 ready" in streamed[0][1]["text"]
    assert streamed[-1][1]["messages"][0]["text"] == f"Load the resumes in {folder}"

    restarted = TestClient(create_app(make_services, token=TOKEN), base_url="http://127.0.0.1",
                           headers={TOKEN_HEADER: TOKEN})
    snapshot = restarted.get(f"/api/sessions/{session_id}").json()
    assert snapshot["notice"].startswith(f"Resumes in {folder}") and "1 ready" in snapshot["notice"]
    assert restarted.get(f"/api/sessions/{session_id}").json()["notice"] is None  # said once


# --- one turn at a time, failures, deleting ---------------------------------------------------------------


def test_a_session_runs_one_turn_at_a_time(client):
    session_id = _new(client)
    session = client.app.state.workspace.get(session_id)
    with session.busy:
        assert client.post(f"/api/sessions/{session_id}/turns", json={"text": "hello"}).status_code == 409
    assert client.post(f"/api/sessions/{session_id}/turns", json={"text": "  "}).status_code == 422


def test_a_failed_turn_is_an_event_and_the_session_goes_on(client):
    session = client.app.state.workspace.get(_new(client))

    def unreachable(messages):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))

    session.services.model.responses[Requirements] = unreachable
    streamed, snapshot = _turn(client, session.conversation.session_id, PYTHON_REQUEST)
    assert streamed[-1] == {"kind": "error", "text": "could not reach the OpenAI API. Check your connection or "
                                                     "LLM_BASE_URL.", "node": ""}
    assert snapshot["messages"][-1]["text"].startswith("Sorry, that failed") and snapshot["waiting"]

    session.services.model.responses[Requirements] = ROLE
    _, snapshot = _turn(client, session.conversation.session_id, PYTHON_REQUEST)
    assert snapshot["screened"] == 12


def test_progress_notes_name_the_candidate():
    from resume_shortlister.conversation import describe
    from tests.fakes import fake_assessment
    (note,) = describe("assess_candidate", {"assessments": {"pool-1:3f2a:alex_lee": fake_assessment(91)}})
    assert note.text == "Deep analysis of alex_lee: 91"


def test_sessions_can_be_deleted(client):
    session_id = _new(client)
    _turn(client, session_id, PYTHON_REQUEST)
    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/sessions/{session_id}").status_code == 404
    assert client.delete(f"/api/sessions/{session_id}").status_code == 404


def test_web_command_starts_the_server(monkeypatch):
    from resume_shortlister.web import app as web_app
    calls = []
    monkeypatch.setattr(web_app, "serve", lambda port, open_browser, host: calls.append((port, open_browser, host)))
    monkeypatch.delenv("RESUME_SHORTLISTER_WEB_HOST", raising=False)
    assert cli_app.main(["web", "--port", "9001", "--no-browser"]) == 0
    monkeypatch.setenv("RESUME_SHORTLISTER_WEB_HOST", "0.0.0.0")  # as the Docker image sets it
    assert cli_app.main(["web"]) == 0
    assert calls == [(9001, False, "127.0.0.1"), (8765, True, "0.0.0.0")]
