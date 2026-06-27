"""Session timeout warning (Agent A6 / Stage S-06).

A session has a TTL (default 1800s, A1). This module decides — deterministically
and with an INJECTABLE clock — whether a session is inside the "about to expire"
window and should be warned. The PRD/A6 brief fires the warning at **T-15min**:
once a session is within 15 minutes of its TTL expiry, surface a warning to the
user.

Design notes:
  * The clock is injected (``now``) so tests never sleep on the wall clock.
  * ``expires_at`` is ``session.updated_at + session.ttl_seconds`` — the same
    arithmetic A1's ``Session.is_expired`` uses, so the warning fires strictly
    before expiry.
  * The check is pure and side-effect free. The caller (BotAgent) decides when to
    actually surface the warning and de-dupes so it nags at most once per window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .schema import Session, Stage

# Default warning lead time: fire once a session is within 15 minutes of expiry.
WARN_BEFORE_SECONDS: int = 15 * 60

WARNING_MESSAGE = (
    "Heads up — this session will expire in about {minutes} min. "
    "Reply to keep it alive, or /reset to start over."
)


@dataclass
class TimeoutStatus:
    """Result of a timeout check. Pure data; no I/O."""

    should_warn: bool
    seconds_remaining: float
    expires_at: float
    expired: bool = False

    @property
    def minutes_remaining(self) -> int:
        """seconds_remaining rounded up to a whole minute (for display)."""
        if self.seconds_remaining <= 0:
            return 0
        return max(1, int((self.seconds_remaining + 59) // 60))


def expires_at(session: Session) -> float:
    """When the session lapses: updated_at + ttl_seconds (mirrors A1)."""
    return float(session.updated_at) + float(session.ttl_seconds)


def check_timeout(
    session: Optional[Session],
    *,
    now: Optional[float] = None,
    warn_before_seconds: int = WARN_BEFORE_SECONDS,
) -> TimeoutStatus:
    """Decide whether ``session`` is inside the T-minus warning window.

    Warns when ``0 < seconds_remaining <= warn_before_seconds`` — i.e. inside the
    last ``warn_before_seconds`` of the TTL but not yet expired. Silent when there
    is plenty of time left, when the session is already expired, or when there is
    no live session.

    ``now`` is injected for deterministic tests (defaults to wall clock).
    """
    clock = time.time() if now is None else now
    if session is None:
        return TimeoutStatus(
            should_warn=False, seconds_remaining=0.0, expires_at=0.0, expired=True
        )

    exp = expires_at(session)
    remaining = exp - clock
    already_expired = remaining <= 0 or session.stage == Stage.EXPIRED

    should_warn = (not already_expired) and (remaining <= warn_before_seconds)
    return TimeoutStatus(
        should_warn=should_warn,
        seconds_remaining=max(0.0, remaining),
        expires_at=exp,
        expired=already_expired,
    )
