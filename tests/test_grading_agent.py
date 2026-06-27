"""Tests for GradingAgent batch orchestration (Agent A2 / Stage S-02).

Covers the PRD S-02 Review & Exit gates:
  * 10-photo batch returns a correctly scored list with flags (low_res / blurry /
    duplicate / face flags + per-platform fit scores);
  * dating face-gate (Tinder/Bumble): face_count == 0 -> capped score + warning,
    NOT sent to Tier 3;
  * LinkedIn suitability flag from scene description;
  * the Tier-3 eligibility gate (G-06).

All synthetic metadata — no images, no network.
"""

import pytest

from curator.grading.agent import (
    DATING_PLATFORMS,
    GradingAgent,
    NO_FACE_WARNING,
)
from curator.grading.result import GradingResult, compute_tier3_eligible
from curator.grading.tier1 import PhotoMetadata


def meta(**kw) -> PhotoMetadata:
    return PhotoMetadata.from_dict(kw)


@pytest.fixture
def agent():
    return GradingAgent()


def _ten_photo_batch():
    """10 synthetic photos exercising every flag path. Distinct phashes unless a
    deliberate near-dupe pair (indices 8 & 9)."""
    return [
        meta(width=1080, height=1350, face_count=1, saturation_mean=0.6, largest_face_frac=0.25, phash="0000000000000001"),  # 0 good IG/dating
        meta(width=400, height=400, face_count=1, saturation_mean=0.5, phash="0000000000000002"),                            # 1 low_res
        meta(width=1080, height=1080, face_count=1, laplacian_variance=5.0, phash="0000000000000004"),                       # 2 blurry
        meta(width=1080, height=1920, face_count=0, phash="0000000000000008"),                                               # 3 no face
        meta(width=1920, height=1080, face_count=2, saturation_mean=0.4, phash="0000000000000010"),                          # 4 group landscape
        meta(width=1080, height=1080, face_count=1, saturation_mean=0.45, phash="0000000000000020"),                         # 5 linkedin-ish
        meta(width=1080, height=1350, face_count=3, saturation_mean=0.7, phash="0000000000000040"),                          # 6 group
        meta(width=1080, height=1920, face_count=1, largest_face_frac=0.30, saturation_mean=0.5, phash="0000000000000080"),  # 7 tinder closeup
        meta(width=1080, height=1080, face_count=1, saturation_mean=0.6, phash="ffffffffffffffff"),                          # 8 dupe A
        meta(width=1080, height=1080, face_count=0, saturation_mean=0.1, phash="fffffffffffffffe"),                          # 9 dupe B (near 8)
    ]


def test_batch_of_ten_returns_scored_list_with_flags(agent):
    batch = _ten_photo_batch()
    platforms = ["instagram", "facebook", "tinder", "bumble", "hinge", "linkedin"]
    results = agent.grade_batch(batch, platforms)

    assert len(results) == 10
    # Every result has a fit score for all 6 platforms.
    for r in results:
        assert set(r.platform_fit) == set(platforms)
        for v in r.platform_fit.values():
            assert 0.0 <= v <= 10.0

    # Flag presence checks.
    assert results[1].low_res and "low_res" in results[1].flags
    assert results[2].blurry and "blurry" in results[2].flags
    assert "no_face_detected" in results[3].flags  # no-face on dating targets
    # Near-dupe pair (8,9): exactly one suppressed (the lower-scored, index 9).
    assert results[8].duplicate_suppressed != results[9].duplicate_suppressed
    assert results[9].duplicate_suppressed is True


def test_all_six_platform_fits_computed_locally(agent):
    r = agent.grade_batch([meta(width=1080, height=1350, face_count=1, phash="01")],
                          ["instagram", "facebook", "tinder", "bumble", "hinge", "linkedin"])[0]
    assert len(r.platform_fit) == 6


# --- Dating face-gate (G-08) ------------------------------------------------
@pytest.mark.parametrize("platform", DATING_PLATFORMS)
def test_face_gate_caps_and_warns(agent, platform):
    m = meta(width=1080, height=1920, face_count=0, phash="aa")
    r = agent.grade_batch([m], [platform])[0]
    assert r.platform_fit[platform] <= 3.0
    assert r.capped_scores[platform] == 3.0
    assert "no_face_detected" in r.flags
    assert NO_FACE_WARNING in r.warnings


@pytest.mark.parametrize("platform", DATING_PLATFORMS)
def test_face_gate_not_eligible_for_tier3(agent, platform):
    """G-08: a no-face photo targeting ONLY dating platforms is gated out of Tier 3."""
    m = meta(width=1080, height=1920, face_count=0, phash="bb")
    r = agent.grade_batch([m], [platform])[0]
    assert r.tier3_eligible is False


def test_face_gate_does_not_block_when_also_non_dating(agent):
    """A no-face photo also targeting IG stays Tier-3 eligible (IG can still grade)."""
    m = meta(width=1080, height=1350, face_count=0, phash="cc")
    r = agent.grade_batch([m], ["tinder", "instagram"])[0]
    assert r.capped_scores.get("tinder") == 3.0
    assert "no_face_detected" in r.flags
    assert r.tier3_eligible is True


def test_face_present_dating_not_capped(agent):
    m = meta(width=1080, height=1920, face_count=1, largest_face_frac=0.25, phash="dd")
    r = agent.grade_batch([m], ["tinder"])[0]
    assert "no_face_detected" not in r.flags
    assert "tinder" not in r.capped_scores
    assert r.platform_fit["tinder"] > 3.0
    assert r.tier3_eligible is True


# --- LinkedIn suitability flag (G-09) ---------------------------------------
def test_linkedin_inappropriate_flag_set(agent):
    m = meta(width=1080, height=1080, face_count=1, saturation_mean=0.45, phash="ee")
    r = agent.grade_batch([m], ["linkedin"], scene_descriptions=["a wild beach party with heavy filter"])[0]
    assert "linkedin_inappropriate" in r.flags
    # Still scored (flag only, not a cap).
    assert r.platform_fit["linkedin"] >= 0.0


def test_linkedin_clean_scene_no_flag(agent):
    m = meta(width=1080, height=1080, face_count=1, saturation_mean=0.45, phash="ef")
    r = agent.grade_batch([m], ["linkedin"], scene_descriptions=["a clean neutral office headshot"])[0]
    assert "linkedin_inappropriate" not in r.flags


def test_linkedin_flag_no_op_when_scene_none(agent):
    """A2 scene hook defaults to None -> no flag (real scene desc comes from A3)."""
    m = meta(width=1080, height=1080, face_count=1, phash="f0")
    r = agent.grade_batch([m], ["linkedin"])[0]
    assert "linkedin_inappropriate" not in r.flags


# --- Tier-3 gate (G-06) -----------------------------------------------------
def test_blurry_low_res_dupe_excluded_from_gate(agent):
    blurry = meta(width=1080, height=1080, face_count=1, laplacian_variance=5.0, phash="11")
    lowres = meta(width=300, height=300, face_count=1, phash="22")
    rs = agent.grade_batch([blurry, lowres], ["instagram"])
    assert rs[0].tier3_eligible is False  # blurry
    assert rs[1].tier3_eligible is False  # low_res


def test_gate_function_matches_g06_definition():
    r = GradingResult()
    r.platform_fit = {"instagram": 7.0}
    assert compute_tier3_eligible(r) is True
    r.blurry = True
    assert compute_tier3_eligible(r) is False
