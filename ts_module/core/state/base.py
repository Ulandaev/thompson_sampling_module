"""Abstract session state store and the SessionData dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SessionData:
    """All data associated with a single decide() call.

    Created when a decision is made; updated when feedback arrives.
    """

    session_id: str
    arm_id: str
    context: dict
    timestamp: datetime
    signals_received: list[dict] = field(default_factory=list)
    is_finalized: bool = False


class BaseStateStore(ABC):
    """Abstract store for decision sessions.

    Phase 1 uses InMemoryStateStore.  Phase 3 will add a DB-backed implementation
    without changing this interface.
    """

    @abstractmethod
    def save_session(self, session_id: str, data: SessionData) -> None:
        """Persist or update a session.

        Args:
            session_id: Unique identifier for the session.
            data: SessionData to store.
        """
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> SessionData | None:
        """Retrieve a session by ID.

        Args:
            session_id: Unique identifier to look up.

        Returns:
            SessionData if found, None otherwise.
        """
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Remove a session from the store.

        Args:
            session_id: Unique identifier of the session to delete.
        """
        ...
