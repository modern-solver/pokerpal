"""Composite aggregation — Tier 1+2+3 (Agent A3 / Stage S-03).

Combines the Tier-3 4-axis semantic score with the Tier-2 ``platform_fit`` using
the per-platform weights in ``platform_weights.yml``. The weight axes are
``composition / lighting / technical / platform_fit`` (they sum to ~1.0).

Axis mapping (PRD: "composition/lighting/technical map to Tier-3 axes; platform_fit
is Tier-2"):
    composition  <- Tier-3 ``composition``
    lighting     <- Tier-3 ``lighting``
    technical    <- Tier-3 ``subject``   (sharpness / focus accuracy of the subject)
    platform_fit <- Tier-2 ``platform_fit`` (the local, deterministic fit score)

Tier-3 ``mood`` is not a weight axis on its own; it is blended evenly into the three
semantic axes ((axis + mood) / 2) so emotional impact / colour harmony influences
the composite without changing the YAML schema. This keeps every component on the
same 1..10 scale, so the weighted sum is itself 1..10.

G-08 reconciliation (the A2 carry-forward Reviewer R flagged):
    The spec says BOTH "cap dating no-face photos at 3/10" (A2's Tier-2 fit cap) AND
    G-08 "set composite_score = 2.0". In the COMPOSITE the binding rule is G-08: a
    dating photo that was face-gated out (``no_face_detected`` AND the platform is in
    ``capped_scores``) gets composite_score = 2.0 EXACTLY, and it is never sent to
    Tier 3. The Tier-2 3/10 cap remains the platform_fit value; the 2.0 is the final
    composite for that platform. Both coexist: 3/10 is the fit, 2.0 is the composite.
"""

from __future__ import annotations

from typing import Dict, Optional

from .result import GradingResult
from .tier3 import Tier3Score
from .weights import get_weights

# G-08 final composite for a face-gated dating photo.
G08_NO_FACE_COMPOSITE = 2.0


def _semantic_axes_with_mood(t3: Tier3Score) -> Dict[str, float]:
    """Map Tier-3 axes onto the weight axes, blending mood evenly into each."""
    mood = t3.mood
    return {
        "composition": (t3.composition + mood) / 2.0,
        "lighting": (t3.lighting + mood) / 2.0,
        "technical": (t3.subject + mood) / 2.0,
    }


def composite_for_platform(
    platform: str,
    result: GradingResult,
    tier3: Optional[Tier3Score],
    weights: Optional[Dict] = None,
) -> float:
    """Compute the 0..10 composite_score for one platform.

    G-08 takes precedence: a face-gated dating photo returns exactly 2.0. Otherwise
    the weighted blend of the (mood-blended) Tier-3 semantic axes and the Tier-2
    platform_fit is returned. If no Tier-3 score is available (e.g. photo was
    ineligible, or Tier 3 was skipped), the composite falls back to the Tier-2
    platform_fit alone so the photo still ranks.
    """
    # G-08 reconciliation: face-gated dating photo -> composite is exactly 2.0.
    if "no_face_detected" in result.flags and platform in result.capped_scores:
        return G08_NO_FACE_COMPOSITE

    fit = float(result.platform_fit.get(platform, 0.0))

    if tier3 is None:
        # No semantic score (ineligible / skipped) — rank on local fit only.
        return round(max(0.0, min(10.0, fit)), 3)

    w = (weights or get_weights())[platform]["weights"]
    semantic = _semantic_axes_with_mood(tier3)
    score = (
        w["composition"] * semantic["composition"]
        + w["lighting"] * semantic["lighting"]
        + w["technical"] * semantic["technical"]
        + w["platform_fit"] * fit
    )
    return round(max(0.0, min(10.0, score)), 3)


def attach_composite(
    result: GradingResult,
    tier3: Optional[Tier3Score],
    platforms,
    weights: Optional[Dict] = None,
) -> Dict[str, float]:
    """Compute + attach the per-platform composite dict to a GradingResult.

    Sets ``result.composite[platform] = score`` for each target platform and
    returns the dict. Honours G-08 per platform.
    """
    weights = weights or get_weights()
    composite: Dict[str, float] = {}
    for platform in platforms:
        composite[platform] = composite_for_platform(platform, result, tier3, weights)
    result.composite = composite
    return composite
