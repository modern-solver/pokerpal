"""RankingAgent — weighted-composite ranking + output cards (Agent A5 / S-05).

Ranks the session's photos for a TARGET platform off
``GradingResult.composite[platform]`` — the source of truth A3 produced (do NOT
re-derive axes; the composite already encodes the YAML weights and the G-08 2.0
cap for face-gated no-face dating photos). Produces:

  * ``rank(...)``        -> top-N ``RankedPhoto`` (default top 3), each with a
                            rationale from the score breakdown + its caption
                            variants joined by photo_index;
  * ``build_cards(...)`` -> ``OutputCard`` list for the BotAgent to present.

Ordering is deterministic and stable: descending composite, ties broken by
(a) Tier-3-graded photos before fit-only photos, then (b) ascending photo_index
(original submission order) — so the same inputs always rank identically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..caption.result import CaptionResult
from ..grading.result import GradingResult
from ..grading.weights import get_weights
from .card import OutputCard, RankedPhoto
from .rationale import build_breakdown, rationale_line

DEFAULT_TOP_N = 3


class RankingAgent:
    """Ranks graded+captioned photos for a platform and builds output cards."""

    def __init__(
        self,
        weights: Optional[Dict[str, Any]] = None,
        top_n: int = DEFAULT_TOP_N,
    ) -> None:
        self.weights = weights or get_weights()
        self.top_n = top_n

    # --- ranking ------------------------------------------------------------
    def rank(
        self,
        platform: str,
        results: Sequence[GradingResult],
        captions: Optional[Sequence[CaptionResult]] = None,
        *,
        top_n: Optional[int] = None,
    ) -> List[RankedPhoto]:
        """Return the top-N ranked photos for ``platform``.

        ``results`` are the per-photo GradingResults (the ranking source of truth
        is ``composite[platform]``). ``captions`` are A4 CaptionResults for the
        SAME platform; they are joined to each photo by ``photo_index`` so each
        ranked card carries its 3 numbered variants.
        """
        n = self.top_n if top_n is None else top_n
        caption_by_index = self._index_captions(platform, captions)

        # Stable, deterministic sort. Python's sort is stable; we sort by a key
        # that fully specifies the order (so ties resolve identically every run).
        ordered = sorted(
            results,
            key=lambda r: (
                -float(r.composite.get(platform, 0.0)),  # higher score first
                0 if r.tier3 is not None else 1,          # graded before fit-only
                self._photo_index(r),                     # original order
            ),
        )

        ranked: List[RankedPhoto] = []
        for position, res in enumerate(ordered[: max(0, n)], start=1):
            ranked.append(self._to_ranked(position, platform, res, caption_by_index))
        return ranked

    def build_cards(
        self,
        platform: str,
        results: Sequence[GradingResult],
        captions: Optional[Sequence[CaptionResult]] = None,
        *,
        top_n: Optional[int] = None,
    ) -> List[OutputCard]:
        """Rank then wrap as numbered OutputCards for the presentation layer."""
        ranked = self.rank(platform, results, captions, top_n=top_n)
        return [OutputCard(card_index=i, photo=rp) for i, rp in enumerate(ranked, 1)]

    # --- helpers ------------------------------------------------------------
    def _to_ranked(
        self,
        position: int,
        platform: str,
        res: GradingResult,
        caption_by_index: Dict[int, CaptionResult],
    ) -> RankedPhoto:
        breakdown = build_breakdown(res, platform, self.weights)
        rationale = rationale_line(breakdown)
        idx = self._photo_index(res)
        cap = caption_by_index.get(idx)
        variants = list(cap.variants) if cap else []
        return RankedPhoto(
            rank=position,
            photo_index=idx,
            platform=platform,
            score=float(res.composite.get(platform, 0.0)),
            rationale=rationale,
            file_id=res.file_id,
            breakdown=breakdown,
            variants=variants,
            flags=list(res.flags),
            warnings=list(res.warnings),
        )

    @staticmethod
    def _photo_index(res: GradingResult) -> int:
        return int(getattr(res, "index", 0) or 0)

    @staticmethod
    def _index_captions(
        platform: str, captions: Optional[Sequence[CaptionResult]]
    ) -> Dict[int, CaptionResult]:
        """Join CaptionResults to photos by photo_index (filtered to ``platform``)."""
        out: Dict[int, CaptionResult] = {}
        for cap in captions or []:
            if cap.platform != platform:
                continue
            out[int(cap.photo_index)] = cap
        return out
