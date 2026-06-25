"""Session schema for Curator Bot v1.1 (Stage S-01, Agent A1).

Defines the durable session record that the conversation spine builds up and that
downstream stages (A2 grading onward) consume. Kept as a plain dataclass with
explicit (de)serialization so it can be stored uniformly in either the in-memory
store or the SQLite fallback.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# The 6 platforms supported in v1.1. Order is the canonical display order used by
# the platform selector keyboard.
PLATFORMS: tuple[str, ...] = (
    "instagram",
    "facebook",
    "tinder",
    "bumble",
    "hinge",
    "linkedin",
)

# Human-facing labels for the inline keyboard.
PLATFORM_LABELS: Dict[str, str] = {
    "instagram": "Instagram",
    "facebook": "Facebook",
    "tinder": "Tinder",
    "bumble": "Bumble",
    "hinge": "Hinge",
    "linkedin": "LinkedIn",
}


class Stage(str, Enum):
    """Conversation state-machine stage.

    The spine advances NEW -> SELECTING_PLATFORMS -> AWAITING_INTENT ->
    RECEIVING_PHOTOS, after which A2+ takes over (GRADING / RANKING / DONE are
    placeholders consumed by later stages, defined here so the enum is stable).
    """

    NEW = "new"
    SELECTING_PLATFORMS = "selecting_platforms"
    AWAITING_INTENT = "awaiting_intent"
    RECEIVING_PHOTOS = "receiving_photos"
    GRADING = "grading"  # entered by A2/A3
    RANKING = "ranking"  # entered by A5
    DONE = "done"
    EXPIRED = "expired"


@dataclass
class PhotoRef:
    """A reference to a received photo. We store references only — never the bytes.

    `file_id` is the Telegram file identifier (re-downloadable later by A2/A3).
    A2's grading stage consumes this list.
    """

    file_id: str
    unique_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    received_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhotoRef":
        return cls(
            file_id=d["file_id"],
            unique_id=d.get("unique_id"),
            width=d.get("width"),
            height=d.get("height"),
            received_at=d.get("received_at", time.time()),
        )


@dataclass
class Session:
    """Durable session record.

    Fields:
        user_id:    Telegram user id (primary key).
        platforms:  multi-selected target platforms (subset of PLATFORMS).
        vibe:       free-text mood/theme string (e.g. "summer roadtrip energy").
        constraints: parsed constraint list (e.g. ["no_location_names",
                     "max_chars:100"]).
        photos:     received photo references (intake only; grading is A2/A3).
        stage:      state-machine stage (Stage enum value).
        created_at / updated_at: epoch timestamps.
        ttl_seconds: lifetime; session is expired once now > updated_at + ttl.
    """

    user_id: int
    platforms: List[str] = field(default_factory=list)
    vibe: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    photos: List[PhotoRef] = field(default_factory=list)
    stage: Stage = Stage.NEW
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl_seconds: int = 1800  # 30 min default TTL

    # --- lifecycle helpers ---------------------------------------------------
    def touch(self) -> None:
        """Bump the updated_at timestamp (resets the TTL window)."""
        self.updated_at = time.time()

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        if self.stage == Stage.EXPIRED:
            return True
        return now > (self.updated_at + self.ttl_seconds)

    # --- serialization (uniform across both store backends) ------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "platforms": list(self.platforms),
            "vibe": self.vibe,
            "constraints": list(self.constraints),
            "photos": [p.to_dict() for p in self.photos],
            "stage": self.stage.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Session":
        return cls(
            user_id=d["user_id"],
            platforms=list(d.get("platforms", [])),
            vibe=d.get("vibe"),
            constraints=list(d.get("constraints", [])),
            photos=[PhotoRef.from_dict(p) for p in d.get("photos", [])],
            stage=Stage(d.get("stage", Stage.NEW.value)),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            ttl_seconds=d.get("ttl_seconds", 1800),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> "Session":
        return cls.from_dict(json.loads(raw))
