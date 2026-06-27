"""GradingResult data structure + Tier-3 eligibility gate (Agent A2 / Stage S-02).

A `GradingResult` is the per-photo output of Tiers 1+2 that downstream stages
consume:
    - A3 (Tier 3 / Groq vision) reads `tier3_eligible` to decide whether to make a
      (paid-budget) API call. It MUST NOT call Tier 3 for ineligible photos.
    - A5 (ranking) reads `platform_fit` scores + flags.

The API-call gate (PRD rule G-06, owned by this stage) is:
    tier3_eligible = (not blurry) AND (not low_res)
                     AND (not duplicate_suppressed)
                     AND (not dating-face-gated-out)

`platform_fit` holds a 0..10 local fit score per *target* platform. Dating face-gate
(G-08) caps Tinder/Bumble at the configured cap and attaches `no_face_detected`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tier1 import PhotoMetadata


@dataclass
class GradingResult:
    """Per-photo Tier 1+2 result."""

    # Identity (mirrors PhotoRef so results can be matched back to the session).
    file_id: Optional[str] = None
    unique_id: Optional[str] = None
    index: int = 0  # position in the submitted batch

    metadata: Optional[PhotoMetadata] = None

    # Tier 1 flags (denormalised for quick access).
    low_res: bool = False
    blurry: bool = False
    duplicate_suppressed: bool = False

    # Tier 2: 0..10 local platform-fit score per target platform.
    platform_fit: Dict[str, float] = field(default_factory=dict)
    # Composite-cap applied by the dating face-gate, per platform (G-08). When a
    # platform is capped here, its platform_fit AND any later composite is capped.
    capped_scores: Dict[str, float] = field(default_factory=dict)

    # Suitability / advisory flags (e.g. no_face_detected, linkedin_inappropriate,
    # duplicate_suppressed, low_res, blurry).
    flags: List[str] = field(default_factory=list)

    # User-facing warnings (e.g. dating face-gate message). Surfaced by the bot.
    warnings: List[str] = field(default_factory=list)

    # API-call gate result (PRD G-06). True => A3 may run Tier 3 on this photo.
    tier3_eligible: bool = True

    # --- Tier 3 (Agent A3 / Stage S-03) enrichment --------------------------
    # The Tier-3 semantic score object (4 axes + provenance). None until Tier 3
    # runs (or stays None for ineligible photos that skip the Groq call). Typed
    # Any to avoid a hard import cycle with tier3.py.
    tier3: Optional[Any] = None
    # BLIP-2 scene-description string (1-2 sentences). Consumed by A4 (captions).
    scene_description: str = ""
    # Signals derived from the scene description (mood/sunglasses/text_overlay/
    # activity) — feeds A2's Tier-2 signal hooks and surfaces to downstream stages.
    signals: Dict[str, Any] = field(default_factory=dict)
    # Per-platform composite score (Tier 1+2+3). Set by A3 composite aggregation.
    composite: Dict[str, float] = field(default_factory=dict)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def add_warning(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "unique_id": self.unique_id,
            "index": self.index,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "low_res": self.low_res,
            "blurry": self.blurry,
            "duplicate_suppressed": self.duplicate_suppressed,
            "platform_fit": dict(self.platform_fit),
            "capped_scores": dict(self.capped_scores),
            "flags": list(self.flags),
            "warnings": list(self.warnings),
            "tier3_eligible": self.tier3_eligible,
            "tier3": self.tier3.axes() if self.tier3 is not None else None,
            "tier3_source": getattr(self.tier3, "source", None),
            "scene_description": self.scene_description,
            "signals": dict(self.signals),
            "composite": dict(self.composite),
        }


def compute_tier3_eligible(result: GradingResult) -> bool:
    """The API-call gate (G-06). A3 consumes this.

    Eligible iff the photo is technically postable AND not a suppressed duplicate
    AND not fully dating-face-gated-out. A photo whose ONLY targets are dating
    platforms and which has no face is gated out (no point spending a Tier-3 call
    on an undateable photo). A photo also targeting a non-dating platform stays
    eligible so Tier 3 can still grade it for that platform.
    """
    if result.low_res or result.blurry or result.duplicate_suppressed:
        return False
    if "no_face_detected" in result.flags:
        # Gated out only if every target platform it has a fit score for is capped.
        scored = set(result.platform_fit) or set(result.capped_scores)
        if scored and scored.issubset(set(result.capped_scores)):
            return False
    return True
