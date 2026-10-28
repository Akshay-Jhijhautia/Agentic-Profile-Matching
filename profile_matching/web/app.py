"""The web API (and, from the UI build, the browser app), served on this machine only.

Every chat action is a turn of the same ``Conversation`` the terminal uses:
loading a folder sends "/resumes <folder>", uploaded files are staged and sent
as "/add <files>". A turn streams its events as Server-Sent Events and ends
with a ``snapshot`` of the session.

The server listens on 127.0.0.1 and answers only requests whose Host is that
address or ``localhost`` (no DNS rebinding). API calls must carry the token
created at startup, which only the app's own page knows, so another website
open in the browser cannot drive the API.
"""

from __future__ import annotations

import json
import secrets
import shlex
import shutil
import threading
import uuid
import webbrowser
from collections.abc import Callable, Iterator
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .. import __version__, analysis
from ..services import Services
from .models import (
    CandidateReport,
    FolderRequest,
    SessionSummary,
    Snapshot,
    Status,
    TurnRequest,
    event_payload,
)
from .workspace import OpenSession, Workspace

TOKEN_HEADER = "X-Profile-Match-Token"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})
DEFAULT_PORT = 8765
# Longest message accepted (a long job description fits easily).
MAX_TEXT_CHARS = 50_000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # per resume file

_PLACEHOLDER_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="profile-match-token" content="{token}">
<title>Profile Match</title></head>
<body style="font-family: sans-serif; margin: 3rem;">
<h1>Profile Match {version}</h1>
<p>The API is running. The browser interface is not built yet; use <code>profile-match chat</code> meanwhile.</p>
</body></html>"""


def _sse(kind: str, payload: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload)}\n\n"


def create_app(
    make_services: Callable[[], Services] = Services.from_settings,
    token: str | None = None,
    allowed_hosts: frozenset[str] = LOCAL_HOSTS,
) -> FastAPI:
    """The app; ``token`` None generates one (``app.state.token``)."""
    app = FastAPI(title="Profile Match", version=__version__, docs_url=None, redoc_url=None)
    app.state.token = token or secrets.token_urlsafe(32)
    workspace = app.state.workspace = Workspace(make_services)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.url.hostname not in allowed_hosts:
            return JSONResponse({"detail": "Profile Match only answers on this machine."}, status_code=403)
        if request.url.path.startswith("/api/") and not secrets.compare_digest(
            request.headers.get(TOKEN_HEADER, ""), app.state.token
        ):
            return JSONResponse({"detail": "Missing or wrong token: open Profile Match from its own page."},
                                status_code=403)
        return await call_next(request)

    def opened(session_id: str) -> OpenSession:
        try:
            return workspace.get(session_id)
        except KeyError:
            raise HTTPException(404, f"No session '{session_id}'") from None

    def stream(session: OpenSession, text: str, cleanup: Callable[[], None] | None = None) -> StreamingResponse:
        """Run one turn and stream its events, then the session's snapshot."""
        if not session.busy.acquire(blocking=False):
            if cleanup:
                cleanup()
            raise HTTPException(409, "This session is still working on the previous message.")

        def events() -> Iterator[str]:
            try:
                for event in session.conversation.turn(text):
                    yield _sse("event", event_payload(event))
                yield _sse("snapshot", Snapshot.of(session.conversation).model_dump(mode="json"))
            finally:
                session.busy.release()
                if cleanup:
                    cleanup()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # --- the page ---

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return _PLACEHOLDER_PAGE.format(token=app.state.token, version=__version__)

    # --- status and sessions ---

    @app.get("/api/status")
    def status() -> Status:
        services = make_services()
        try:
            services.check_ready(require_resumes=False)
            problem = None
        except ValueError as exc:
            problem = str(exc)
        settings = services.settings
        return Status(version=__version__, model=settings.model_label, embedder=settings.embedder,
                      ready=problem is None, problem=problem)

    @app.get("/api/sessions")
    def list_sessions() -> list[SessionSummary]:
        return [SessionSummary.of(info) for info in workspace.list()]

    @app.post("/api/sessions", status_code=201)
    def create_session() -> Snapshot:
        return Snapshot.of(workspace.create().conversation)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> Snapshot:
        session = opened(session_id)
        snapshot = Snapshot.of(session.conversation, notice=session.notice)
        session.notice = None  # said once
        return snapshot

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> None:
        try:
            workspace.delete(session_id)
        except KeyError:
            raise HTTPException(404, f"No session '{session_id}'") from None

    # --- turns ---

    @app.post("/api/sessions/{session_id}/turns")
    def send(session_id: str, request: TurnRequest) -> StreamingResponse:
        text = request.text.strip()
        if not text or len(text) > MAX_TEXT_CHARS:
            raise HTTPException(422, f"Send between 1 and {MAX_TEXT_CHARS} characters.")
        return stream(opened(session_id), text)

    @app.post("/api/sessions/{session_id}/folder")
    def load_folder(session_id: str, request: FolderRequest) -> StreamingResponse:
        path = Path(request.path).expanduser()
        if not path.is_dir():
            raise HTTPException(422, f"Not a folder: {request.path}")
        return stream(opened(session_id), f"/resumes {shlex.quote(str(path))}")

    @app.post("/api/sessions/{session_id}/files")
    def upload_files(session_id: str, files: list[UploadFile] = File(...)) -> StreamingResponse:
        session = opened(session_id)
        staging = Path(workspace.settings.uploads_dir).expanduser() / "incoming" / uuid.uuid4().hex
        staging.mkdir(parents=True)

        def cleanup() -> None:
            shutil.rmtree(staging, ignore_errors=True)

        paths = []
        try:
            for upload in files:
                name = Path(upload.filename or "").name  # never a path from the browser
                if not name:
                    continue
                target = staging / name
                with target.open("wb") as out:
                    shutil.copyfileobj(upload.file, out)
                if target.stat().st_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"{name} is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
                paths.append(str(target))
        except BaseException:
            cleanup()
            raise
        if not paths:
            cleanup()
            raise HTTPException(422, "No files received.")
        # The chat's own upload path: files join this session's uploads batch.
        return stream(session, "/add " + " ".join(shlex.quote(p) for p in paths), cleanup)

    # --- results ---

    @app.get("/api/sessions/{session_id}/candidates/{candidate_id}")
    def candidate(session_id: str, candidate_id: str) -> CandidateReport:
        session = opened(session_id)
        values = session.conversation.values
        try:
            report = analysis.detailed_report(session.services, candidate_id, values.get("requirements"),
                                              values.get("shortlist", []), values.get("candidates", []))
        except (LookupError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from None
        return CandidateReport(id=candidate_id, report=report)

    @app.get("/api/sessions/{session_id}/report", response_class=PlainTextResponse)
    def report(session_id: str) -> PlainTextResponse:
        text = opened(session_id).conversation.values.get("report")
        if not text:
            raise HTTPException(404, "No report yet: describe a role first.")
        return PlainTextResponse(text, media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{session_id}.md"'})

    return app


def serve(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Run the app on 127.0.0.1:``port`` until interrupted, opening it in the browser."""
    import uvicorn

    app = create_app()
    url = f"http://127.0.0.1:{port}/"
    print(f"Profile Match is running at {url} (Ctrl-C to stop).")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
