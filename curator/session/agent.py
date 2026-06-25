"""SessionAgent for Curator Bot v1.1 (Stage S-01, Agent A1).

Owns the session lifecycle: init() / get() / update() / expire(), with a TTL.
Wraps a swappable `SessionStore` backend. The in-memory store is primary; a
SQLite store may be supplied as a durable fallback (so sessions survive restart).

A2 (grading) consumes the session returned by get() — specifically `.photos`,
`.platforms`, `.vibe`, and `.constraints`.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from .schema import PhotoRef, Session, Stage
from .store import InMemorySessionStore, SessionStore


class SessionAgent:
    """Lifecycle manager over a SessionStore."""

    def __init__(
        self,
        store: Optional[SessionStore] = None,
        default_ttl_seconds: int = 1800,
    ) -> None:
        self.store: SessionStore = store or InMemorySessionStore()
        self.default_ttl_seconds = default_ttl_seconds

    # --- lifecycle ----------------------------------------------------------
    def init(self, user_id: int, ttl_seconds: Optional[int] = None) -> Session:
        """Create a fresh session for `user_id`, overwriting any existing one."""
        session = Session(
            user_id=user_id,
            ttl_seconds=ttl_seconds
            if ttl_seconds is not None
            else self.default_ttl_seconds,
            stage=Stage.NEW,
        )
        self.store.save(session)
        return session

    def get(self, user_id: int, *, expire_if_stale: bool = True) -> Optional[Session]:
        """Return the live session, or None if missing/expired.

        If the stored session is past its TTL and `expire_if_stale` is set, it is
        expired (marked + deleted) and None is returned.
        """
        session = self.store.load(user_id)
        if session is None:
            return None
        if session.is_expired():
            if expire_if_stale:
                self.expire(user_id)
            return None
        return session

    def update(self, user_id: int, **fields: Any) -> Optional[Session]:
        """Apply field updates to a live session and persist it.

        Bumps `updated_at` (resetting the TTL window). Returns the updated session,
        or None if there is no live session. Unknown fields raise AttributeError so
        typos surface in tests rather than silently no-op.
        """
        session = self.get(user_id)
        if session is None:
            return None
        for key, value in fields.items():
            if not hasattr(session, key):
                raise AttributeError(f"Session has no field {key!r}")
            setattr(session, key, value)
        session.touch()
        self.store.save(session)
        return session

    def expire(self, user_id: int) -> None:
        """Expire and remove a session (idempotent)."""
        session = self.store.load(user_id)
        if session is not None:
            session.stage = Stage.EXPIRED
            session.touch()
        self.store.delete(user_id)

    def get_or_init(self, user_id: int) -> Session:
        """Convenience: return the live session, creating one if absent/expired."""
        return self.get(user_id) or self.init(user_id)

    # --- domain mutations used by the bot spine -----------------------------
    def set_platforms(self, user_id: int, platforms: List[str]) -> Optional[Session]:
        return self.update(user_id, platforms=list(platforms), stage=Stage.AWAITING_INTENT)

    def toggle_platform(self, user_id: int, platform: str) -> Optional[Session]:
        """Multi-select toggle: add platform if absent, remove if present."""
        session = self.get(user_id)
        if session is None:
            return None
        if platform in session.platforms:
            session.platforms.remove(platform)
        else:
            session.platforms.append(platform)
        session.touch()
        self.store.save(session)
        return session

    def add_photo(self, user_id: int, photo: PhotoRef) -> Optional[Session]:
        """Append a received photo reference; advance stage to RECEIVING_PHOTOS."""
        session = self.get(user_id)
        if session is None:
            return None
        session.photos.append(photo)
        session.stage = Stage.RECEIVING_PHOTOS
        session.touch()
        self.store.save(session)
        return session

    def sweep_expired(self, now: Optional[float] = None) -> int:
        """Expire all stale sessions. Returns the count expired (housekeeping)."""
        now = time.time() if now is None else now
        expired = 0
        for uid in self.store.all_user_ids():
            session = self.store.load(uid)
            if session is not None and session.is_expired(now):
                self.expire(uid)
                expired += 1
        return expired

    def close(self) -> None:
        self.store.close()
