"""Session stores for Curator Bot v1.1 (Stage S-01, Agent A1).

Two interchangeable backends behind one uniform interface (`SessionStore`):

  * `InMemorySessionStore` — primary, fast, lost on restart.
  * `SQLiteSessionStore`   — fallback persistence so sessions survive restart.

The interface is deliberately narrow (save / load / delete / all_user_ids) so the
backend is swappable. SessionAgent (session/agent.py) owns lifecycle semantics;
the store just persists/retrieves the serialized record.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Dict, List, Optional, Protocol

from .schema import Session


class SessionStore(Protocol):
    """Uniform persistence interface. Both backends implement this."""

    def save(self, session: Session) -> None: ...

    def load(self, user_id: int) -> Optional[Session]: ...

    def delete(self, user_id: int) -> None: ...

    def all_user_ids(self) -> List[int]: ...

    def close(self) -> None: ...


class InMemorySessionStore:
    """Primary store: a thread-safe dict keyed by user_id."""

    def __init__(self) -> None:
        self._data: Dict[int, Session] = {}
        self._lock = threading.RLock()

    def save(self, session: Session) -> None:
        with self._lock:
            self._data[session.user_id] = session

    def load(self, user_id: int) -> Optional[Session]:
        with self._lock:
            return self._data.get(user_id)

    def delete(self, user_id: int) -> None:
        with self._lock:
            self._data.pop(user_id, None)

    def all_user_ids(self) -> List[int]:
        with self._lock:
            return list(self._data.keys())

    def close(self) -> None:  # nothing to release
        pass


class SQLiteSessionStore:
    """Fallback store: persists serialized sessions to a SQLite DB.

    The DB file is git-ignored (`*.sqlite`). Use `:memory:` for tests that want
    to exercise the SQLite code path without touching disk.
    """

    def __init__(self, path: str = "curator_sessions.sqlite") -> None:
        self.path = path
        # check_same_thread=False so the store can be shared across PTB's threads;
        # we serialize access ourselves with a lock.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id    INTEGER PRIMARY KEY,
                    updated_at REAL    NOT NULL,
                    data       TEXT    NOT NULL
                )
                """
            )
            self._conn.commit()

    def save(self, session: Session) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (user_id, updated_at, data) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET updated_at=excluded.updated_at, "
                "data=excluded.data",
                (session.user_id, session.updated_at, session.to_json()),
            )
            self._conn.commit()

    def load(self, user_id: int) -> Optional[Session]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT data FROM sessions WHERE user_id = ?", (user_id,)
            )
            row = cur.fetchone()
        if row is None:
            return None
        return Session.from_json(row[0])

    def delete(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            self._conn.commit()

    def all_user_ids(self) -> List[int]:
        with self._lock:
            cur = self._conn.execute("SELECT user_id FROM sessions")
            return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
