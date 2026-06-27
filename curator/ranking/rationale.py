"""Rationale builder — explains a pick FROM the score breakdown (Agent A5 / S-05).

The PRD requires a "rationale line from the score breakdown". The rationale here
is NOT generic boilerplate: it is derived from the same numbers the composite was
built from —

  * the per-platform composite score (the ranking source of truth, A3);
  * the Tier-3 semantic axes (composition / lighting / technical<-subject / mood)
    blended exactly as composite.py blends them, so the "what drove the score"
    line reflects the real top axis;
  * the Tier-2 ``platform_fit`` (its weight is the largest axis for some
    platforms, so it can itself be the driver);
  * the suitability flags + warnings (``no_face_detected`` / G-08 cap,
    ``linkedin_inappropriate`` / G-09, blurry / low_res / duplicate_suppressed).

We do NOT re-derive the composite — we read the breakdown the grading stages
produced and explain it. ``build_breakdown`` returns the structured data;
``rationale_line`` renders the short human string from that data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..grading.composite import G08_NO_FACE_COMPOSITE
from ..grading.result import GradingResult
from ..grading.weights import get_weights

# Human labels for the four weight axes (technical maps to the Tier-3 subject axis).
_AXIS_LABEL = {
    "composition": "composition",
    "lighting": "lighting",
    "technical": "subject sharpness",
    "platform_fit": "platform fit",
}

# Flags worth surfacing in a rationale, with a short user-facing phrase.
_FLAG_PHRASE = {
    "no_face_detected": "no visible face — not recommended for dating platforms",
    "linkedin_inappropriate": "casual/party scene — weak for LinkedIn",
    "blurry": "soft focus",
    "low_res": "low resolution",
    "duplicate_suppressed": "near-duplicate of another photo",
}


def _semantic_axes_with_mood(tier3: Any) -> Dict[str, float]:
    """Mirror composite.py's mood-blend so the breakdown matches the real score.

    Returns the three semantic weight axes (composition / lighting / technical)
    with Tier-3 ``mood`` blended evenly in, exactly as the composite did.
    """
    mood = float(getattr(tier3, "mood", 5.0))
    return {
        "composition": (float(getattr(tier3, "composition", 5.0)) + mood) / 2.0,
        "lighting": (float(getattr(tier3, "lighting", 5.0)) + mood) / 2.0,
        "technical": (float(getattr(tier3, "subject", 5.0)) + mood) / 2.0,
    }


def build_breakdown(
    result: GradingResult,
    platform: str,
    weights: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decompose the composite for one platform into the parts that drove it.

    The ``contributions`` are weight*value per axis — the actual additive terms of
    the weighted composite (or, for the no-Tier-3 case, fit alone). The
    ``top_axis`` is the largest contributor. G-08 (face-gated dating, composite
    2.0) and the suitability flags are surfaced so the rationale can name them.
    """
    weights = weights or get_weights()
    score = float(result.composite.get(platform, 0.0))

    g08_capped = (
        "no_face_detected" in result.flags and platform in result.capped_scores
    )

    breakdown: Dict[str, Any] = {
        "platform": platform,
        "score": score,
        "g08_capped": g08_capped,
        "has_tier3": result.tier3 is not None,
        "fit": float(result.platform_fit.get(platform, 0.0)),
        "flags": list(result.flags),
        "warnings": list(result.warnings),
        "contributions": {},
        "top_axis": None,
    }

    if g08_capped:
        # G-08: the score is the fixed 2.0 cap; the driver is the missing face.
        breakdown["g08_composite"] = G08_NO_FACE_COMPOSITE
        breakdown["top_axis"] = "no_face_detected"
        return breakdown

    fit = breakdown["fit"]
    if result.tier3 is None:
        # No semantic score — composite is fit alone.
        breakdown["contributions"] = {"platform_fit": fit}
        breakdown["top_axis"] = "platform_fit"
        return breakdown

    w = weights[platform]["weights"]
    semantic = _semantic_axes_with_mood(result.tier3)
    contributions = {
        "composition": w["composition"] * semantic["composition"],
        "lighting": w["lighting"] * semantic["lighting"],
        "technical": w["technical"] * semantic["technical"],
        "platform_fit": w["platform_fit"] * fit,
    }
    breakdown["contributions"] = contributions
    breakdown["axis_values"] = {
        "composition": semantic["composition"],
        "lighting": semantic["lighting"],
        "technical": semantic["technical"],
        "platform_fit": fit,
    }
    breakdown["top_axis"] = max(contributions, key=contributions.get)
    return breakdown


def rationale_line(breakdown: Dict[str, Any]) -> str:
    """Render the short human rationale string from a ``build_breakdown`` dict.

    Always mentions the driving axis (or the G-08 face cap) and appends any
    surfaced suitability flag/warning so the explanation reflects the breakdown,
    not a template.
    """
    score = breakdown["score"]
    flags: List[str] = breakdown.get("flags", [])

    # G-08: face-gated dating photo — the warning IS the explanation.
    if breakdown.get("g08_capped"):
        phrase = _FLAG_PHRASE["no_face_detected"]
        return f"Scored {score:.1f}/10 — {phrase}."

    top = breakdown.get("top_axis")
    top_label = _AXIS_LABEL.get(top, top or "overall quality")

    if not breakdown.get("has_tier3"):
        lead = (
            f"Scored {score:.1f}/10 on local {top_label} "
            "(semantic grading skipped)"
        )
    else:
        lead = f"Scored {score:.1f}/10 — strongest on {top_label}"

    # Append the most relevant suitability flag, if any (priority order).
    note = ""
    for fl in ("no_face_detected", "linkedin_inappropriate", "blurry", "low_res",
               "duplicate_suppressed"):
        if fl in flags and fl in _FLAG_PHRASE:
            note = f"; note: {_FLAG_PHRASE[fl]}"
            break

    return f"{lead}{note}."
