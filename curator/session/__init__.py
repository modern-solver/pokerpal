"""SessionAgent package — session lifecycle + store.

OWNER: Agent A1 (Stage S-01). Implements init()/get()/update()/expire() with TTL,
an in-memory session store with a SQLite fallback, and intent parsing
(platform multi-select, vibe, constraints).
"""

from .agent import SessionAgent
from .intent import ParsedIntent, parse_constraints, parse_intent, parse_platforms
from .schema import (
    PLATFORM_LABELS,
    PLATFORMS,
    PhotoRef,
    Session,
    Stage,
)
from .store import InMemorySessionStore, SessionStore, SQLiteSessionStore

__all__ = [
    "SessionAgent",
    "ParsedIntent",
    "parse_intent",
    "parse_platforms",
    "parse_constraints",
    "Session",
    "PhotoRef",
    "Stage",
    "PLATFORMS",
    "PLATFORM_LABELS",
    "SessionStore",
    "InMemorySessionStore",
    "SQLiteSessionStore",
]
