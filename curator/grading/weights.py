"""platform_weights.yml loader for Tier 2 fit scoring (Agent A2 / Stage S-02).

Loads and lightly validates the per-platform scoring profiles. Tier 2 reads the
returned structure so weights and point rules tune from YAML WITHOUT code changes
(Agent A6 tunes them later). Pure local I/O — no network, no model call.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import yaml

from ..session.schema import PLATFORMS

# Default config path: curator/config/platform_weights.yml relative to this file.
DEFAULT_WEIGHTS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "config", "platform_weights.yml")
)

# Grading axes whose weights must be present and sum to ~1.0.
AXES = ("composition", "lighting", "technical", "platform_fit")
_WEIGHT_SUM_TOLERANCE = 0.001


def load_weights(path: str | None = None) -> Dict[str, Any]:
    """Load and validate the platform weights config.

    Returns a dict keyed by platform name. Raises ValueError if a platform is
    missing, an axis is missing, or axis weights do not sum to ~1.0 — so config
    typos surface loudly rather than silently mis-scoring.
    """
    path = path or DEFAULT_WEIGHTS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for platform in PLATFORMS:
        if platform not in data:
            raise ValueError(f"platform_weights.yml missing platform: {platform!r}")
        profile = data[platform]
        weights = profile.get("weights")
        if not weights:
            raise ValueError(f"platform {platform!r} missing 'weights' block")
        missing = [ax for ax in AXES if ax not in weights]
        if missing:
            raise ValueError(f"platform {platform!r} weights missing axes: {missing}")
        total = sum(float(weights[ax]) for ax in AXES)
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"platform {platform!r} weights sum to {total:.3f}, must be ~1.0"
            )
        profile.setdefault("fit_rules", [])
        profile.setdefault("fit_score_scale", {"base": 5.0})
        profile.setdefault("face", {"mandatory": False, "cap_score": None})

    return data


@lru_cache(maxsize=4)
def get_weights(path: str | None = None) -> Dict[str, Any]:
    """Cached accessor for the default (or given) weights file."""
    return load_weights(path)
