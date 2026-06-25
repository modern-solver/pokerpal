"""Tier 3 — Semantic grading orchestration (Agent A3 / Stage S-03).

Runs ONLY on photos the local gate (A2) marked ``tier3_eligible`` — Reviewer R
verifies Groq is never called on blurry / low-res / dupe / face-gated photos.

Per photo it:
  1. calls Groq Llama 4 Scout vision (``VisionGrader``) with the structured JSON
     grading prompt;
  2. parses the 4-axis JSON (composition / lighting / subject / mood);
  3. on JSON-parse failure, RETRIES the Groq call ONCE (rule G-07);
  4. on a SECOND parse failure, falls back to HF Serverless BLIP-2 scene
     description and ESTIMATES the four axes from text + Tier-2 platform_fit;
  5. on a Groq 429 (rate limit), falls back to HF Serverless BLIP-2 immediately
     (no second Groq call — that would just 429 again).

The Tier-3 4-axis score, the source path, latency, and any derived scene/signals
are returned in a ``Tier3Score`` for the composite stage to aggregate.

All external calls are injected (``VisionGrader`` + ``SceneDescriptionService``),
so every path — success, parse-retry-then-fallback, 429-fallback — is unit-tested
with mocks, no keys, no network, no heavy libs.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .groq_client import GroqRateLimitError, VisionGrader
from .scene import SceneDescriptionService, SceneResult, derive_signals

TIER3_AXES = ("composition", "lighting", "subject", "mood")

# Latency budget for the Groq vision call (PRD exit gate: < 5s). Surfaced on the
# score so the smoke test / ranking can assert/inspect it.
GROQ_LATENCY_BUDGET_S = 5.0


@dataclass
class Tier3Score:
    """Result of Tier-3 semantic grading for one photo."""

    # 4-axis semantic scores (1..10). On fallback these are ESTIMATED.
    composition: float = 5.0
    lighting: float = 5.0
    subject: float = 5.0
    mood: float = 5.0

    # Provenance: "groq" | "groq_retry" | "hf_fallback" | "skipped".
    source: str = "skipped"
    # JSON parse retries actually consumed (0, 1).
    retries: int = 0
    # Wall-clock seconds for the Groq vision call (None if not attempted).
    latency_s: Optional[float] = None
    # Scene description (populated on the HF-fallback path; else empty).
    scene_description: str = ""
    # Signals derived from the scene description (feeds A2 Tier-2 hooks).
    signals: Dict[str, Any] = field(default_factory=dict)

    def axes(self) -> Dict[str, float]:
        return {
            "composition": self.composition,
            "lighting": self.lighting,
            "subject": self.subject,
            "mood": self.mood,
        }

    def mean(self) -> float:
        return sum(self.axes().values()) / 4.0


def _clamp_axis(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 5.0
    return max(1.0, min(10.0, f))


def parse_grading_json(raw: str) -> Dict[str, float]:
    """Parse the 4-axis grading JSON from a raw model response.

    Tolerant of code fences and surrounding prose: extracts the first JSON object,
    strips ``// comments`` (the prompt template shows commented keys). Raises
    ``ValueError`` if no parseable object with the expected axes is found — that
    drives the single retry then the HF fallback (rule G-07).
    """
    if not raw or not raw.strip():
        raise ValueError("empty grading response")

    text = raw.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Grab the first {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in response: {raw!r}")
    blob = match.group(0)
    # Strip // line comments the template demonstrates.
    blob = re.sub(r"//[^\n]*", "", blob)

    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid grading JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("grading JSON is not an object")

    present = [ax for ax in TIER3_AXES if ax in data]
    if not present:
        raise ValueError(f"grading JSON missing all axes: {list(data)}")

    # Use present axes; any missing axis defaults to neutral 5.
    return {ax: _clamp_axis(data.get(ax, 5.0)) for ax in TIER3_AXES}


def estimate_axes_from_scene(
    scene: SceneResult,
    tier2_fit_mean: Optional[float] = None,
) -> Dict[str, float]:
    """Estimate the 4 axes when Groq is unavailable (HF Serverless BLIP-2 fallback).

    BLIP-2 gives a scene description, not numbers. We anchor on the Tier-2
    platform_fit mean (already a 0..10 local quality proxy) and nudge ``subject`` /
    ``mood`` from the derived signals (smile/positivity). This is intentionally
    conservative — the fallback should not invent high semantic scores.
    """
    base = 5.0 if tier2_fit_mean is None else max(1.0, min(10.0, tier2_fit_mean))
    signals = scene.signals or derive_signals(scene.description)
    mood = signals.get("mood_score")
    mood_axis = _clamp_axis(mood) if mood is not None else base
    # Activity / clear subject in the scene lifts the subject axis slightly.
    subject_axis = _clamp_axis(base + (1.0 if signals.get("activity") else 0.0))
    return {
        "composition": _clamp_axis(base),
        "lighting": _clamp_axis(base),
        "subject": subject_axis,
        "mood": _clamp_axis(mood_axis),
    }


class Tier3Grader:
    """Orchestrates the Groq vision call + parse/retry/fallback for one photo.

    Args:
        vision: injected ``VisionGrader`` (GroqVisionGrader in prod, stub in tests).
        scene_service: injected ``SceneDescriptionService`` for the HF fallback +
            local scene description.
    """

    def __init__(
        self,
        vision: Optional[VisionGrader] = None,
        scene_service: Optional[SceneDescriptionService] = None,
    ) -> None:
        self.vision = vision
        self.scene_service = scene_service or SceneDescriptionService()

    def grade_photo(
        self,
        image_bytes: bytes,
        *,
        tier2_fit_mean: Optional[float] = None,
        scene: Optional[SceneResult] = None,
    ) -> Tier3Score:
        """Run Tier-3 on a single (already-eligible) photo.

        ``scene`` may be supplied if scene description was computed earlier
        (caption stage reuses it); otherwise the HF-fallback path computes one on
        demand. ``tier2_fit_mean`` anchors the fallback estimate.
        """
        if self.vision is None:
            return self._fallback(image_bytes, tier2_fit_mean, scene, reason="no_vision")

        # --- attempt 1 + single retry on JSON parse failure (rule G-07) -----
        last_err: Optional[Exception] = None
        for attempt in range(2):  # attempt 0 = first call, attempt 1 = the one retry
            try:
                start = time.perf_counter()
                raw = self.vision.grade(image_bytes)
                latency = time.perf_counter() - start
            except GroqRateLimitError:
                # 429 -> immediate HF Serverless fallback (no retry; would 429 again).
                return self._fallback(
                    image_bytes, tier2_fit_mean, scene, reason="rate_limit"
                )
            try:
                axes = parse_grading_json(raw)
            except ValueError as exc:
                last_err = exc
                continue  # retry once, then fall through to fallback
            return Tier3Score(
                **axes,
                source="groq" if attempt == 0 else "groq_retry",
                retries=attempt,
                latency_s=latency,
            )

        # --- second parse failure -> HF Serverless BLIP-2 fallback ----------
        return self._fallback(
            image_bytes, tier2_fit_mean, scene, reason="parse_fail", retries=1
        )

    # --- fallback path ------------------------------------------------------
    def _fallback(
        self,
        image_bytes: bytes,
        tier2_fit_mean: Optional[float],
        scene: Optional[SceneResult],
        *,
        reason: str,
        retries: int = 0,
    ) -> Tier3Score:
        if scene is None:
            scene = self.scene_service.describe_via_serverless(image_bytes)
        axes = estimate_axes_from_scene(scene, tier2_fit_mean)
        return Tier3Score(
            **axes,
            source="hf_fallback",
            retries=retries,
            latency_s=None,
            scene_description=scene.description,
            signals=scene.signals,
        )
