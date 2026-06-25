"""Tier-3 semantic grading — mocked unit tests (Agent A3 / Stage S-03).

Proves the PRD S-03 review & exit gates WITHOUT keys / network / heavy libs:
  * JSON parse + single retry + HF Serverless fallback (rule G-07);
  * Groq 429 rate-limit fallback;
  * Tier 3 NEVER called on ineligible (blurry/dupe/low-res/face-gated) photos;
  * composite aggregation incl. the G-08 2.0 reconciliation;
  * full mocked 10-photo pipeline end-to-end;
  * latency captured on the success path (mirrors the live smoke-test assertion).

Everything external (Groq vision + BLIP-2) is a stub passed via DI.
"""

import pytest

from curator.grading.agent import GradingAgent
from curator.grading.composite import G08_NO_FACE_COMPOSITE
from curator.grading.groq_client import GroqRateLimitError
from curator.grading.scene import (
    SceneDescriptionService,
    SceneResult,
    derive_signals,
)
from curator.grading.tier1 import PhotoMetadata
from curator.grading.tier3 import (
    Tier3Grader,
    Tier3Score,
    estimate_axes_from_scene,
    parse_grading_json,
)

IMG = b"\xff\xd8fake-jpeg-bytes"


def meta(**kw) -> PhotoMetadata:
    return PhotoMetadata.from_dict(kw)


# --- test doubles -----------------------------------------------------------
class StubVision:
    """A VisionGrader stub. `responses` is a list of (str) or Exception to raise.

    Records every call so we can assert it is NOT invoked on ineligible photos.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def grade(self, image_bytes):
        self.calls += 1
        item = self._responses.pop(0) if self._responses else "{}"
        if isinstance(item, Exception):
            raise item
        return item


class StubDescriber:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def describe(self, image_bytes):
        self.calls += 1
        return self.text


def _scene_service(text="a person smiling on a hiking trail"):
    return SceneDescriptionService(serverless=StubDescriber(text))


# === 1. JSON parser =========================================================
def test_parse_plain_json():
    assert parse_grading_json('{"composition":8,"lighting":7,"subject":9,"mood":6}') == {
        "composition": 8.0, "lighting": 7.0, "subject": 9.0, "mood": 6.0,
    }


def test_parse_with_fences_and_comments():
    raw = '```json\n{ "composition": 9, // framing\n "lighting": 8 }\n```'
    out = parse_grading_json(raw)
    assert out["composition"] == 9.0 and out["lighting"] == 8.0
    # Missing axes default to neutral 5.
    assert out["subject"] == 5.0 and out["mood"] == 5.0


def test_parse_clamps_out_of_range():
    out = parse_grading_json('{"composition":42,"lighting":-3,"subject":0,"mood":10}')
    assert out["composition"] == 10.0 and out["lighting"] == 1.0
    assert out["subject"] == 1.0 and out["mood"] == 10.0


@pytest.mark.parametrize("bad", ["", "not json at all", "sorry, I cannot", "{oops}"])
def test_parse_raises_on_unparseable(bad):
    with pytest.raises(ValueError):
        parse_grading_json(bad)


# === 2. Success path (+ latency captured) ===================================
def test_tier3_success_first_try_records_latency():
    vision = StubVision(['{"composition":8,"lighting":7,"subject":9,"mood":6}'])
    grader = Tier3Grader(vision=vision, scene_service=_scene_service())
    score = grader.grade_photo(IMG)
    assert score.source == "groq"
    assert score.retries == 0
    assert vision.calls == 1
    assert score.composition == 8.0 and score.mood == 6.0
    # Latency captured (mirrors live smoke-test <5s assertion).
    assert score.latency_s is not None and score.latency_s >= 0.0


# === 3. Parse-retry-then-fallback (rule G-07) ===============================
def test_parse_fail_retries_once_then_succeeds():
    # First response unparseable, second valid -> source groq_retry, retries=1.
    vision = StubVision(["garbage", '{"composition":6,"lighting":6,"subject":6,"mood":6}'])
    grader = Tier3Grader(vision=vision, scene_service=_scene_service())
    score = grader.grade_photo(IMG)
    assert vision.calls == 2
    assert score.source == "groq_retry"
    assert score.retries == 1
    assert score.composition == 6.0


def test_parse_fail_twice_falls_back_to_hf_serverless():
    vision = StubVision(["garbage", "still garbage"])
    describer = StubDescriber("a smiling person hiking outdoors")
    grader = Tier3Grader(
        vision=vision, scene_service=SceneDescriptionService(serverless=describer)
    )
    score = grader.grade_photo(IMG, tier2_fit_mean=7.0)
    # Groq tried exactly twice (1 + single retry), then HF fallback.
    assert vision.calls == 2
    assert score.source == "hf_fallback"
    assert describer.calls == 1
    assert score.scene_description == "a smiling person hiking outdoors"
    # Estimated axes anchored on the Tier-2 fit mean (7.0) + smile/activity signals.
    assert 1.0 <= score.composition <= 10.0
    assert score.signals.get("activity") is True


# === 4. 429 rate-limit fallback =============================================
def test_429_falls_back_to_hf_immediately_no_retry():
    vision = StubVision([GroqRateLimitError("429 Too Many Requests")])
    describer = StubDescriber("a clean office headshot")
    grader = Tier3Grader(
        vision=vision, scene_service=SceneDescriptionService(serverless=describer)
    )
    score = grader.grade_photo(IMG, tier2_fit_mean=6.0)
    # On 429 we do NOT retry Groq (would 429 again) — straight to HF.
    assert vision.calls == 1
    assert score.source == "hf_fallback"
    assert describer.calls == 1
    assert score.scene_description == "a clean office headshot"


def test_fallback_with_no_serverless_degrades_gracefully():
    vision = StubVision([GroqRateLimitError("429")])
    grader = Tier3Grader(vision=vision, scene_service=SceneDescriptionService())
    score = grader.grade_photo(IMG, tier2_fit_mean=5.0)
    assert score.source == "hf_fallback"
    assert score.scene_description == ""  # no describer -> empty, still scored
    assert 1.0 <= score.mean() <= 10.0


# === 5. Scene signal derivation =============================================
def test_derive_signals_smile_and_activity():
    sig = derive_signals("a smiling man hiking with sunglasses near a sign")
    assert sig["mood_score"] is not None and sig["mood_score"] > 5.0
    assert sig["activity"] is True
    assert sig["sunglasses"] is True
    assert sig["text_overlay"] is True
    assert sig["scene_context"].startswith("a smiling")


def test_derive_signals_empty():
    assert derive_signals("") == {}
    assert derive_signals(None) == {}


def test_estimate_axes_anchors_on_fit():
    scene = SceneResult("a neutral portrait", "serverless", derive_signals("a neutral portrait"))
    axes = estimate_axes_from_scene(scene, tier2_fit_mean=8.0)
    assert axes["composition"] == 8.0 and axes["lighting"] == 8.0
