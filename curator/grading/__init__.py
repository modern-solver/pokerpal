"""GradingAgent package — three-tier grading pipeline.

OWNERS:
    Agent A2 (Stage S-02) — Tier 1 local pre-filter (Pillow resolution, OpenCV
        Laplacian blur, imagehash dedup) + Tier 2 platform-fit scoring from
        config/platform_weights.yml + dating face-gate + LinkedIn suitability flag.
    Agent A3 (Stage S-03) — Tier 3 semantic grading (Groq Llama 4 Scout vision),
        BLIP-2 scene description, JSON parse + retry + HF Serverless 429 fallback,
        and composite aggregation.

A2 public surface (Tiers 1+2, zero API cost):
    - GradingAgent.grade_batch(...) -> list[GradingResult]
    - tier1.extract_metadata(image) -> PhotoMetadata   (Pillow/OpenCV/imagehash)
    - tier2.score_platform_fit(platform, meta, ...)    (deterministic, YAML-driven)
    - result.compute_tier3_eligible(result)            (the API-call gate, G-06)

A3 consumes GradingResult.tier3_eligible to decide which photos get a Tier-3 call.
"""

from .agent import GradingAgent, DATING_PLATFORMS, NO_FACE_WARNING
from .result import GradingResult, compute_tier3_eligible
from .tier1 import (
    PhotoMetadata,
    extract_metadata,
    phash_distance,
    BLUR_LAPLACIAN_THRESHOLD,
    MIN_RESOLUTION_PX,
    PHASH_NEAR_DUPE_DISTANCE,
)
from .tier2 import score_platform_fit, classify_aspect_bucket
from .weights import get_weights, load_weights

__all__ = [
    "GradingAgent",
    "GradingResult",
    "PhotoMetadata",
    "compute_tier3_eligible",
    "extract_metadata",
    "phash_distance",
    "score_platform_fit",
    "classify_aspect_bucket",
    "get_weights",
    "load_weights",
    "DATING_PLATFORMS",
    "NO_FACE_WARNING",
    "BLUR_LAPLACIAN_THRESHOLD",
    "MIN_RESOLUTION_PX",
    "PHASH_NEAR_DUPE_DISTANCE",
]
