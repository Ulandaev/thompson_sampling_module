"""In-memory session state store for Phase 1."""

from __future__ import annotations

from ts_module.core.state.base import BaseStateStore, SessionData


class InMemoryStateStore(BaseStateStore):
    """Session state stored in a plain dict.

    Suitable for Phase 1 (synchronous, single-process).
    Replace with a DB-backed store in Phase 3.
    """

    def __init__(self) -> None:
        """Initialize an empty session store."""
        self._sessions: dict[str, SessionData] = {}

    def save_session(self, session_id: str, data: SessionData) -> None:
        """Save or overwrite a session."""
        self._sessions[session_id] = data

    def get_session(self, session_id: str) -> SessionData | None:
        """Return the session or None if not found."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        """Remove the session (no-op if not found)."""
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        """Return number of stored sessions."""
        return len(self._sessions)
