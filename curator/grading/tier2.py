"""Tier 2 — Platform-fit scoring (Agent A2 / Stage S-02). ZERO network, ZERO cost.

Deterministic rules over Tier-1 metadata. Reads point rules from platform_weights.yml
(so they tune without code changes) and produces a 0..10 `platform_fit` score per
target platform. No model call — every rule operates on a `PhotoMetadata` dict.

Some rule types depend on signals only available later (A3/BLIP-2): scene_context,
smile_mood, sunglasses, text_overlay, horizon_level. For A2 these are driven by an
OPTIONAL `scene_description` string + optional `signals` dict so the hooks exist and
are testable, and they no-op (neutral) when those signals are absent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .tier1 import ASPECT_BUCKETS, PhotoMetadata
from .weights import get_weights

# Aspect-bucket matching tolerance (fractional). A 1.0±10% ratio is "square".
_ASPECT_TOLERANCE = 0.12


def classify_aspect_bucket(ratio: Optional[float]) -> Optional[str]:
    """Map a width/height ratio to the nearest named bucket within tolerance."""
    if ratio is None:
        return None
    best_name: Optional[str] = None
    best_err = None
    for name, target in ASPECT_BUCKETS.items():
        err = abs(ratio - target) / target
        if best_err is None or err < best_err:
            best_err, best_name = err, name
    if best_err is not None and best_err <= _ASPECT_TOLERANCE:
        return best_name
    # Outside tolerance: coarse fallback by orientation so we never return None for
    # a real image (extreme ratios still get the closest landscape/portrait bucket).
    if ratio >= 1.3:
        return "landscape_16_9" if ratio >= 1.55 else "standard_4_3"
    if ratio <= 0.7:
        return "story_9_16"
    return best_name  # near-square-ish but off-tolerance


def _scene_contains(scene_description: Optional[str], keywords) -> bool:
    if not scene_description:
        return False
    text = scene_description.lower()
    return any(kw.lower() in text for kw in keywords)


def _apply_rule(
    rule: Dict[str, Any],
    meta: PhotoMetadata,
    scene_description: Optional[str],
    signals: Dict[str, Any],
) -> float:
    """Return the point delta for one fit rule. Unknown signals -> neutral (0)."""
    rtype = rule.get("type")

    if rtype == "aspect_ratio":
        bucket = classify_aspect_bucket(meta.aspect_ratio)
        if bucket is None:
            return 0.0
        return float(rule.get("buckets", {}).get(bucket, 0))

    if rtype == "saturation":
        if meta.saturation_mean is None:
            return 0.0
        s = meta.saturation_mean
        total = 0.0
        for th in rule.get("thresholds", []):
            op = th.get("op")
            if op == "gt" and s > th["value"]:
                total += th["points"]
            elif op == "lt" and s < th["value"]:
                total += th["points"]
            elif op == "between" and th["low"] <= s <= th["high"]:
                total += th["points"]
        return float(total)

    if rtype == "warm_tones":
        return float(rule.get("points", 0)) if meta.warm_dominant else 0.0

    if rtype == "face_present":
        return float(rule.get("points", 0)) if (meta.face_count or 0) >= 1 else 0.0

    if rtype == "face_absent_penalty":
        return float(rule.get("points", 0)) if (meta.face_count == 0) else 0.0

    if rtype == "face_size":
        frac = meta.largest_face_frac
        if frac is None:
            return 0.0
        return float(rule.get("points", 0)) if frac > rule.get("frac", 0.2) else 0.0

    if rtype == "solo_subject":
        return float(rule.get("points", 0)) if meta.face_count == 1 else 0.0

    if rtype == "group_photo":
        n = meta.face_count or 0
        return float(rule.get("points", 0)) if n >= rule.get("min_faces", 2) else 0.0

    if rtype == "group_penalty":
        n = meta.face_count or 0
        return float(rule.get("points", 0)) if n >= rule.get("min_faces", 3) else 0.0

    if rtype == "horizon_level":
        # Tier-1 hint not measured in A2; driven by optional signal. Neutral if absent.
        return float(rule.get("points", 0)) if signals.get("horizon_level") else 0.0

    if rtype == "text_overlay":
        return float(rule.get("points", 0)) if signals.get("text_overlay") else 0.0

    if rtype == "scene_context":
        return (
            float(rule.get("points", 0))
            if _scene_contains(scene_description, rule.get("keywords", []))
            else 0.0
        )

    if rtype == "smile_mood":
        mood = signals.get("mood_score")
        if mood is None:
            return 0.0
        return float(rule.get("points", 0)) if mood > rule.get("threshold", 7) else 0.0

    if rtype == "sunglasses":
        return float(rule.get("points", 0)) if signals.get("sunglasses") else 0.0

    if rtype == "heavy_filter":
        if meta.saturation_mean is None:
            return 0.0
        return (
            float(rule.get("points", 0))
            if meta.saturation_mean > rule.get("value", 0.85)
            else 0.0
        )

    # Unknown rule type: neutral (don't crash on forward-compat config).
    return 0.0


def score_platform_fit(
    platform: str,
    meta: PhotoMetadata,
    *,
    scene_description: Optional[str] = None,
    signals: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute the 0..10 platform_fit score for one platform from Tier-1 metadata.

    `scene_description` (default None) is the A2 hook for A3/BLIP-2 scene text — when
    None, scene-driven rules no-op. `signals` carries optional A3 hints
    (mood_score, sunglasses, text_overlay, horizon_level).
    """
    signals = signals or {}
    cfg = (weights or get_weights())[platform]
    base = float(cfg.get("fit_score_scale", {}).get("base", 5.0))

    points = 0.0
    for rule in cfg.get("fit_rules", []):
        points += _apply_rule(rule, meta, scene_description, signals)

    score = base + points
    return max(0.0, min(10.0, score))
