"""Saved chat sessions: the conversation graph's checkpoints, kept in SQLite.

Every chat is a session (a LangGraph thread) with a short id. A session can
be resumed later with its role, results, history and resumes, listed, or
deleted.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from .agent.state import serializer
from .rag.vector_store import open_sqlite


def new_session_id() -> str:
    """Readable and sortable: ``20261022-1640-a3f2``."""
    return f"{datetime.now():%Y%m%d-%H%M}-{uuid.uuid4().hex[:4]}"


def config_for(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def open_saver(path: str) -> SqliteSaver:
    """The session store at ``path`` (created on first use)."""
    db = Path(path).expanduser()
    db.parent.mkdir(parents=True, exist_ok=True)
    # Already in WAL mode, so the saver's own switch in ``setup`` never waits on a lock.
    saver = SqliteSaver(open_sqlite(db), serde=serializer())
    saver.setup()
    return saver


def exists(saver: SqliteSaver, session_id: str) -> bool:
    return saver.get_tuple(config_for(session_id)) is not None


@dataclass(frozen=True)
class SessionInfo:
    id: str
    updated: datetime
    role: str | None
    resumes: str | None
    messages: int

    def line(self) -> str:
        role = self.role or "no role yet"
        resumes = self.resumes or "no resumes loaded"
        return f"{self.id}  {self.updated:%Y-%m-%d %H:%M}  {role} · {resumes} · {self.messages} messages"


def last_updated(saver: SqliteSaver) -> dict[str, datetime]:
    """When each session was last updated, by session id."""
    latest: dict[str, datetime] = {}
    for checkpoint in saver.list(None):
        session_id = checkpoint.config["configurable"]["thread_id"]
        updated = datetime.fromisoformat(checkpoint.checkpoint["ts"]).astimezone()
        if session_id not in latest or updated > latest[session_id]:
            latest[session_id] = updated
    return latest


def list_sessions(graph: CompiledStateGraph, saver: SqliteSaver, limit: int | None = None) -> list[SessionInfo]:
    """Saved sessions, most recently updated first."""
    ordered = sorted(last_updated(saver).items(), key=lambda item: item[1], reverse=True)[:limit]
    infos = []
    for session_id, updated in ordered:
        values = graph.get_state(config_for(session_id)).values
        requirements = values.get("requirements")
        infos.append(SessionInfo(
            id=session_id, updated=updated, role=requirements.role_title if requirements else None,
            resumes=values.get("resume_folder"), messages=len(values.get("messages", [])),
        ))
    return infos


def delete_session(saver: SqliteSaver, session_id: str) -> None:
    saver.delete_thread(session_id)
