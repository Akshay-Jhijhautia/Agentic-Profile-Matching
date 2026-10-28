"""The chats the web app has open.

Each session gets its own ``Services`` (so its own active resume pool) and
``Conversation``; all share one checkpointer, so they are saved sessions like
the terminal's. A session saved earlier is reopened on first use, with its
resumes reloaded. A session runs one turn at a time.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property

from .. import sessions
from ..agent.graph import build_graph
from ..conversation import Conversation, reload_resumes
from ..services import Services


@dataclass
class OpenSession:
    conversation: Conversation
    services: Services
    notice: str | None = None  # told to the user once, e.g. "the resumes of this session are gone"
    busy: threading.Lock = field(default_factory=threading.Lock)


class Workspace:
    def __init__(self, make_services: Callable[[], Services]):
        self._make_services = make_services
        self._open: dict[str, OpenSession] = {}
        self._lock = threading.Lock()

    # Opened on first use, so building the app (e.g. to export its API schema) touches nothing.
    @cached_property
    def settings(self):
        return self._make_services().settings

    @cached_property
    def saver(self):
        return sessions.open_saver(self.settings.sessions_db)

    def _conversation(self, session_id: str | None = None) -> tuple[Conversation, Services]:
        services = self._make_services()
        graph = build_graph(services, checkpointer=self.saver, conversational=True)
        return Conversation(graph, session_id=session_id), services

    def create(self) -> OpenSession:
        conversation, services = self._conversation()
        opened = OpenSession(conversation, services)
        with self._lock:
            self._open[conversation.session_id] = opened
        return opened

    def get(self, session_id: str) -> OpenSession:
        """The open session, reopening a saved one; ``KeyError`` when there is none."""
        with self._lock:
            if session_id in self._open:
                return self._open[session_id]
            if not sessions.exists(self.saver, session_id):
                raise KeyError(session_id)
            conversation, services = self._conversation(session_id)
            report, problem = reload_resumes(services, conversation.values)
            opened = OpenSession(conversation, services, notice=report.summary() if report else problem)
            self._open[session_id] = opened
            return opened

    def list(self) -> list[sessions.SessionInfo]:
        graph = build_graph(self._make_services(), checkpointer=self.saver, conversational=True)
        return sessions.list_sessions(graph, self.saver)

    def delete(self, session_id: str) -> None:
        with self._lock:
            if not sessions.exists(self.saver, session_id) and session_id not in self._open:
                raise KeyError(session_id)
            self._open.pop(session_id, None)
            sessions.delete_session(self.saver, session_id)
