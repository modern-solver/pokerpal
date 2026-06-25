"""GradingAgent — Tiers 1+2 orchestration (Agent A2 / Stage S-02).

ZERO network, ZERO cost, ZERO API calls. This module runs the local pre-filter
(Tier 1) and the local platform-fit scorer (Tier 2) over a batch of photos and
returns a list of `GradingResult`. It also:

  * suppresses near-duplicate photos (pHash distance < 10), keeping the higher-
    scored one (PRD G-05);
  * applies the dating face-gate for Tinder/Bumble (G-08): face_count == 0 -> cap
    the platform score, attach `no_face_detected`, surface a user warning, and
    exclude from the Tier-3 gate;
  * attaches the LinkedIn suitability flag (G-09) when the optional scene
    description names an inappropriate setting (party/beach/festival/heavy filter);
  * computes the Tier-3 eligibility gate (G-06) that A3 consumes.

Tier 3 (Groq vision) is Agent A3 and is intentionally NOT called from here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..session.schema import PLATFORMS, Session
from .result import GradingResult, compute_tier3_eligible
from .tier1 import PHASH_NEAR_DUPE_DISTANCE, PhotoMetadata, phash_distance
from .tier2 import score_platform_fit
from .weights import get_weights

DATING_PLATFORMS = ("tinder", "bumble")

# G-08 user-facing warning.
NO_FACE_WARNING = (
    "This photo has no visible face — not recommended for dating platforms."
)

# G-09 LinkedIn inappropriate-context keywords.
LINKEDIN_INAPPROPRIATE_KEYWORDS = (
    "party",
    "beach",
    "festival",
    "club",
    "nightclub",
    "nightlife",
    "rave",
    "heavy filter",
)


def _scene_has_inappropriate(scene_description: Optional[str]) -> bool:
    if not scene_description:
        return False
    text = scene_description.lower()
    return any(kw in text for kw in LINKEDIN_INAPPROPRIATE_KEYWORDS)


def _mean_fit(result: GradingResult) -> float:
    """Mean platform_fit used to pick the survivor among near-duplicates."""
    if not result.platform_fit:
        return 0.0
    return sum(result.platform_fit.values()) / len(result.platform_fit)


class GradingAgent:
    """Runs local Tiers 1+2 over a photo batch. No Tier 3, no API calls."""

    def __init__(self, weights: Optional[Dict[str, Any]] = None) -> None:
        # Loaded once; tunable from YAML without code change (A6).
        self.weights = weights or get_weights()

    # --- public API ---------------------------------------------------------
    def grade_batch(
        self,
        metadatas: Sequence[PhotoMetadata],
        platforms: Sequence[str],
        *,
        scene_descriptions: Optional[Sequence[Optional[str]]] = None,
        signals: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        photo_refs: Optional[Sequence[Any]] = None,
    ) -> List[GradingResult]:
        """Grade a batch of pre-extracted Tier-1 metadata against target platforms.

        Args:
            metadatas: Tier-1 metadata per photo (extract via tier1.extract_metadata,
                or supply synthetic dicts/objects in tests).
            platforms: target platforms (subset of PLATFORMS).
            scene_descriptions: optional per-photo scene strings (A3/BLIP-2 hook).
                Default: all None -> scene-driven rules no-op.
            signals: optional per-photo signal dicts (mood_score, sunglasses, ...).
            photo_refs: optional per-photo PhotoRef for identity (file_id/unique_id).

        Returns a list of GradingResult, one per input photo (including suppressed
        duplicates, which carry duplicate_suppressed=True).
        """
        targets = [p for p in platforms if p in PLATFORMS]
        n = len(metadatas)
        scene_descriptions = list(scene_descriptions or [None] * n)
        signals = list(signals or [None] * n)
        photo_refs = list(photo_refs or [None] * n)

        results: List[GradingResult] = []
        for i, meta in enumerate(metadatas):
            ref = photo_refs[i] if i < len(photo_refs) else None
            res = self._grade_one(
                index=i,
                meta=meta,
                targets=targets,
                scene_description=scene_descriptions[i] if i < len(scene_descriptions) else None,
                signals=signals[i] if i < len(signals) else None,
                ref=ref,
            )
            results.append(res)

        self._suppress_duplicates(results)

        # Finalise the Tier-3 gate after dedup so suppressed dupes are excluded.
        for res in results:
            res.tier3_eligible = compute_tier3_eligible(res)
        return results

    def grade_session(
        self,
        session: Session,
        *,
        scene_descriptions: Optional[Sequence[Optional[str]]] = None,
        signals: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        extract: bool = True,
    ) -> List[GradingResult]:
        """Grade a Session's photos. Requires Pillow-readable images when extract=True.

        Note: PhotoRef stores Telegram file_ids, not bytes — A3 downloads bytes. In
        A2 this path is exercised mainly by callers that have already attached image
        data. When extract=False, callers must pass metadata via grade_batch instead.
        This method exists so the SessionAgent integration point is explicit.
        """
        raise NotImplementedError(
            "grade_session requires downloaded image bytes (A1/A3 wiring); A2 "
            "exposes grade_batch over pre-extracted metadata. See grade_batch."
        )

    # --- internals ----------------------------------------------------------
    def _grade_one(
        self,
        *,
        index: int,
        meta: PhotoMetadata,
        targets: List[str],
        scene_description: Optional[str],
        signals: Optional[Dict[str, Any]],
        ref: Any,
    ) -> GradingResult:
        res = GradingResult(index=index, metadata=meta)
        if ref is not None:
            res.file_id = getattr(ref, "file_id", None)
            res.unique_id = getattr(ref, "unique_id", None)

        # Carry Tier-1 flags forward.
        res.low_res = bool(meta.low_res)
        res.blurry = bool(meta.blurry)
        if res.low_res:
            res.add_flag("low_res")
        if res.blurry:
            res.add_flag("blurry")

        no_face = (meta.face_count == 0)

        for platform in targets:
            cfg = self.weights[platform]
            fit = score_platform_fit(
                platform,
                meta,
                scene_description=scene_description,
                signals=signals,
                weights=self.weights,
            )

            # Dating face-gate (G-08): mandatory face missing -> cap.
            face_cfg = cfg.get("face", {})
            if face_cfg.get("mandatory") and no_face:
                cap = float(face_cfg.get("cap_score", 3.0))
                fit = min(fit, cap)
                res.capped_scores[platform] = cap
                res.add_flag("no_face_detected")
                res.add_warning(NO_FACE_WARNING)

            res.platform_fit[platform] = round(fit, 3)

        # LinkedIn suitability flag (G-09) — flag only, still scored.
        if "linkedin" in targets and _scene_has_inappropriate(scene_description):
            res.add_flag("linkedin_inappropriate")

        return res

    def _suppress_duplicates(self, results: List[GradingResult]) -> None:
        """Mark near-duplicates (pHash distance < 10), suppressing the lower-scored.

        Compares every still-active pair; the lower mean-fit photo of a near-dupe
        pair gets duplicate_suppressed=True (PRD G-05). Photos with no usable phash
        are never suppressed (we can't prove duplication without a hash).
        """
        active = [r for r in results if r.metadata and r.metadata.phash]
        for a in active:
            if a.duplicate_suppressed:
                continue
            for b in active:
                if b is a or b.duplicate_suppressed:
                    continue
                dist = phash_distance(a.metadata.phash, b.metadata.phash)
                if dist is None or dist >= PHASH_NEAR_DUPE_DISTANCE:
                    continue
                # Near-duplicate: suppress the lower-scored one.
                loser = a if _mean_fit(a) <= _mean_fit(b) else b
                loser.duplicate_suppressed = True
                loser.add_flag("duplicate_suppressed")
                if loser is a:
                    break  # `a` is gone; move to next outer photo
