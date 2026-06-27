"""Composite aggregation + Tier-3 orchestration tests (Agent A3 / Stage S-03).

Covers:
  * composite blends Tier-3 4-axis with Tier-2 platform_fit via YAML weights;
  * G-08 reconciliation: face-gated dating photo -> composite EXACTLY 2.0, and is
    NEVER sent to Tier 3;
  * Tier 3 not called on ineligible photos (blurry / low-res / dupe);
  * ineligible / no-Tier-3 photos still get a composite from Tier-2 fit;
  * full mocked 10-photo pipeline end-to-end with eligibility-gated Groq calls.

All mocked — no keys, no network, no heavy libs.
"""

import pytest

from curator.grading.agent import GradingAgent
from curator.grading.composite import (
    G08_NO_FACE_COMPOSITE,
    composite_for_platform,
)
from curator.grading.tier1 import PhotoMetadata
from curator.grading.tier3 import Tier3Grader, Tier3Score

IMG = b"\xff\xd8img"


def meta(**kw) -> PhotoMetadata:
    return PhotoMetadata.from_dict(kw)


class CountingVision:
    """Vision stub returning a fixed valid grade; counts + records image bytes."""

    def __init__(self, response='{"composition":8,"lighting":8,"subject":8,"mood":8}'):
        self.response = response
        self.calls = 0
        self.seen = []

    def grade(self, image_bytes):
        self.calls += 1
        self.seen.append(image_bytes)
        return self.response


# === composite math =========================================================
def test_composite_blends_tier3_and_fit():
    agent = GradingAgent()
    res = agent.grade_batch([meta(width=1080, height=1350, face_count=1, phash="01")],
                            ["instagram"])[0]
    fit = res.platform_fit["instagram"]
    t3 = Tier3Score(composition=10, lighting=10, subject=10, mood=10, source="groq")
    score = composite_for_platform("instagram", res, t3, agent.weights)
    # All semantic axes blend to 10 (max). The weighted composite is then
    #   w_fit*fit + (1-w_fit)*10, which must sit between fit and 10 inclusive.
    assert fit - 1e-6 <= score <= 10.0 + 1e-6
    # And strictly above fit whenever fit < 10 (the semantic 10 pulls it up).
    if fit < 10.0:
        assert score > fit
    # Sanity against the closed form using the IG platform_fit weight (0.35).
    w_fit = agent.weights["instagram"]["weights"]["platform_fit"]
    assert score == pytest.approx(w_fit * fit + (1 - w_fit) * 10.0, abs=1e-3)


def test_composite_without_tier3_uses_fit():
    agent = GradingAgent()
    res = agent.grade_batch([meta(width=1080, height=1350, face_count=1, phash="02")],
                            ["instagram"])[0]
    score = composite_for_platform("instagram", res, None, agent.weights)
    assert score == round(res.platform_fit["instagram"], 3)


# === G-08 reconciliation ====================================================
@pytest.mark.parametrize("platform", ["tinder", "bumble"])
def test_g08_face_gated_dating_composite_is_2(platform):
    agent = GradingAgent()
    res = agent.grade_batch([meta(width=1080, height=1920, face_count=0, phash="aa")],
                            [platform])[0]
    # Tier-2 fit was capped at 3.0; composite must be EXACTLY 2.0 (G-08).
    assert res.capped_scores[platform] == 3.0
    t3 = Tier3Score(composition=9, lighting=9, subject=9, mood=9, source="groq")
    score = composite_for_platform(platform, res, t3, agent.weights)
    assert score == G08_NO_FACE_COMPOSITE == 2.0


def test_g08_face_present_dating_not_forced_to_2():
    agent = GradingAgent()
    res = agent.grade_batch(
        [meta(width=1080, height=1920, face_count=1, largest_face_frac=0.3, phash="bb")],
        ["tinder"],
    )[0]
    t3 = Tier3Score(composition=8, lighting=8, subject=8, mood=8, source="groq")
    score = composite_for_platform("tinder", res, t3, agent.weights)
    assert score != 2.0
    assert score > 2.0


# === Tier 3 NOT called on ineligible photos =================================
def test_tier3_not_called_on_blurry_lowres_dupe():
    vision = CountingVision()
    agent = GradingAgent(tier3_grader=Tier3Grader(vision=vision))
    batch = [
        meta(width=1080, height=1080, face_count=1, laplacian_variance=5.0, phash="0001"),  # blurry
        meta(width=300, height=300, face_count=1, phash="0002"),                            # low_res
        meta(width=1080, height=1080, face_count=1, saturation_mean=0.6, phash="ffffffffffffffff"),  # dupe A
        meta(width=1080, height=1080, face_count=1, saturation_mean=0.6, phash="fffffffffffffffe"),  # dupe B (near A)
        meta(width=1080, height=1350, face_count=1, saturation_mean=0.6, phash="00ff"),     # GOOD
    ]
    results = agent.grade_batch(batch, ["instagram"])
    images = [IMG] * len(batch)
    agent.grade_tier3(results, images, ["instagram"])

    # The three ineligible photos (blurry idx0, low_res idx1, the suppressed dupe)
    # must NOT be sent to Groq. Groq is called once per eligible photo only.
    assert results[0].tier3_eligible is False  # blurry
    assert results[1].tier3_eligible is False  # low_res
    # Of the near-dupe pair (idx 2 & 3), exactly one is suppressed -> ineligible.
    assert results[2].duplicate_suppressed != results[3].duplicate_suppressed
    suppressed = results[2] if results[2].duplicate_suppressed else results[3]
    assert suppressed.tier3_eligible is False
    assert suppressed.tier3 is None  # suppressed dupe never graded by Tier 3

    eligible = [r for r in results if r.tier3_eligible]
    assert vision.calls == len(eligible)
    # Blurry/low-res/suppressed carry no Tier-3 score but DO get a composite (fit-based).
    for r in results:
        assert "instagram" in r.composite
        if not r.tier3_eligible:
            assert r.tier3 is None


def test_tier3_not_called_on_face_gated_dating_photo():
    vision = CountingVision()
    agent = GradingAgent(tier3_grader=Tier3Grader(vision=vision))
    # No-face photo targeting ONLY dating platforms -> gated out, no Groq call.
    res = agent.grade_batch([meta(width=1080, height=1920, face_count=0, phash="cc")],
                            ["tinder", "bumble"])
    agent.grade_tier3(res, [IMG], ["tinder", "bumble"])
    assert vision.calls == 0
    assert res[0].tier3 is None
    # G-08 composite 2.0 on both dating platforms.
    assert res[0].composite["tinder"] == 2.0
    assert res[0].composite["bumble"] == 2.0


# === full mocked 10-photo pipeline end-to-end ===============================
def _ten():
    return [
        meta(width=1080, height=1350, face_count=1, saturation_mean=0.6, largest_face_frac=0.25, phash="0000000000000001"),
        meta(width=400, height=400, face_count=1, saturation_mean=0.5, phash="0000000000000002"),                          # low_res
        meta(width=1080, height=1080, face_count=1, laplacian_variance=5.0, phash="0000000000000004"),                     # blurry
        meta(width=1080, height=1920, face_count=0, phash="0000000000000008"),                                             # no face
        meta(width=1920, height=1080, face_count=2, saturation_mean=0.4, phash="0000000000000010"),
        meta(width=1080, height=1080, face_count=1, saturation_mean=0.45, phash="0000000000000020"),
        meta(width=1080, height=1350, face_count=3, saturation_mean=0.7, phash="0000000000000040"),
        meta(width=1080, height=1920, face_count=1, largest_face_frac=0.30, saturation_mean=0.5, phash="0000000000000080"),
        meta(width=1080, height=1080, face_count=1, saturation_mean=0.6, phash="ffffffffffffffff"),                        # dupe A
        meta(width=1080, height=1080, face_count=1, saturation_mean=0.6, phash="fffffffffffffffe"),                        # dupe B
    ]


def test_full_pipeline_10_photos_end_to_end():
    vision = CountingVision()
    agent = GradingAgent(tier3_grader=Tier3Grader(vision=vision))
    platforms = ["instagram", "facebook", "tinder", "bumble", "hinge", "linkedin"]
    batch = _ten()
    results = agent.grade_batch(batch, platforms)
    images = [IMG] * 10
    enriched = agent.grade_tier3(results, images, platforms)

    assert len(enriched) == 10
    # Every photo has a composite for all 6 platforms.
    for r in enriched:
        assert set(r.composite) == set(platforms)
        for v in r.composite.values():
            assert 0.0 <= v <= 10.0

    # Groq called exactly once per eligible photo (never on ineligible).
    n_eligible = sum(1 for r in enriched if r.tier3_eligible)
    assert vision.calls == n_eligible
    assert n_eligible >= 1
    # Only eligible photos carry a Tier-3 score.
    for r in enriched:
        assert (r.tier3 is not None) == r.tier3_eligible
