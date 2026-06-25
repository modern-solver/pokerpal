"""Tests for Tier 2 deterministic platform-fit scoring (Agent A2 / Stage S-02).

100% metadata-driven — no images, no cv2, no network. Exercises the exact
per-platform point rules from PRD Section 05 using synthetic PhotoMetadata dicts.
"""

import pytest

from curator.grading.tier1 import PhotoMetadata
from curator.grading.tier2 import classify_aspect_bucket, score_platform_fit


def meta(**kw) -> PhotoMetadata:
    return PhotoMetadata.from_dict(kw)


# --- aspect bucket classification -------------------------------------------
@pytest.mark.parametrize(
    "ratio,bucket",
    [
        (9 / 16, "story_9_16"),
        (4 / 5, "portrait_4_5"),
        (1.0, "square_1_1"),
        (4 / 3, "standard_4_3"),
        (16 / 9, "landscape_16_9"),
    ],
)
def test_aspect_bucket_exact(ratio, bucket):
    assert classify_aspect_bucket(ratio) == bucket


def test_aspect_bucket_none_for_unknown_ratio():
    assert classify_aspect_bucket(None) is None


# --- Instagram: 4:5 +4, sat>.55 +2, warm +1, face +3, base 5 -----------------
def test_instagram_portrait_face_high_sat():
    m = meta(width=1080, height=1350, saturation_mean=0.6, face_count=1, warm_dominant=True)
    # base5 + aspect4 + sat2 + warm1 + face3 = 15 -> clamped 10
    assert score_platform_fit("instagram", m) == 10.0


def test_instagram_landscape_lowsat_penalised():
    m = meta(width=1920, height=1080, saturation_mean=0.2, face_count=0, warm_dominant=False)
    # base5 + aspect(-3) + sat(-1) = 1.0
    assert score_platform_fit("instagram", m) == 1.0


# --- Facebook: group photo + landscape --------------------------------------
def test_facebook_group_landscape():
    m = meta(width=1920, height=1080, face_count=3)
    s = score_platform_fit("facebook", m, scene_description="a group of friends at a gathering")
    # base5 + landscape4 + scene(group)2 + face1 + group2 = 14 -> 10
    assert s == 10.0


# --- Tinder: face_size close-up + solo + portrait ---------------------------
def test_tinder_closeup_solo_portrait():
    m = meta(width=1080, height=1920, face_count=1, largest_face_frac=0.30)
    # base4 + face_size4 + aspect(story)4 + solo2 = 14 -> 10
    assert score_platform_fit("tinder", m) == 10.0


def test_tinder_group_penalised():
    m = meta(width=1080, height=1920, face_count=4, largest_face_frac=0.05)
    # base4 + aspect4 + group_penalty(-2) = 6.0
    assert score_platform_fit("tinder", m) == 6.0


# --- Bumble: activity scene context -----------------------------------------
def test_bumble_activity_scene():
    m = meta(width=1080, height=1920, face_count=1, saturation_mean=0.4)
    s = score_platform_fit("bumble", m, scene_description="a person hiking on a mountain")
    # base4 + face3 + scene(activity)3 + aspect4 + sat(<.45)1 = 15 -> 10
    assert s == 10.0


# --- Hinge: lifestyle/travel scene ------------------------------------------
def test_hinge_lifestyle_travel():
    m = meta(width=1080, height=1350, face_count=1)
    s = score_platform_fit("hinge", m, scene_description="traveling outdoor on a mountain")
    # base5 + lifestyle4 + travel/outdoor2 + aspect(4:5)3 = 14 -> 10
    assert s == 10.0


def test_hinge_scene_none_is_neutral_hook():
    m = meta(width=1080, height=1350, face_count=1)
    # No scene description -> scene rules no-op. base5 + aspect3 = 8.0
    assert score_platform_fit("hinge", m) == 8.0


# --- LinkedIn: headshot + clean bg + moderate sat ---------------------------
def test_linkedin_professional_headshot():
    m = meta(width=1080, height=1080, face_count=1, saturation_mean=0.45)
    s = score_platform_fit("linkedin", m, scene_description="a clean neutral studio background")
    # base5 + face4 + clean2 + sat(between)1 + square3 = 15 -> 10
    assert s == 10.0


def test_linkedin_party_context_penalised():
    m = meta(width=1080, height=1080, face_count=1, saturation_mean=0.9)
    s = score_platform_fit("linkedin", m, scene_description="a wild party at a club")
    # base5 + face4 + sat(>.55)-2 + square3 + party(-4) = 6.0
    assert s == 6.0


def test_score_clamped_to_0_10():
    m = meta(width=1920, height=1080, saturation_mean=0.95, face_count=4, largest_face_frac=0.01)
    s = score_platform_fit("tinder", m)
    assert 0.0 <= s <= 10.0


def test_unknown_metadata_is_neutral():
    # All-unknown metadata (no cv2, no measurements) must not crash; scene rules no-op.
    m = PhotoMetadata(width=1000, height=1000)
    m.finalize()
    for platform in ("instagram", "facebook", "tinder", "bumble", "hinge", "linkedin"):
        s = score_platform_fit(platform, m)
        assert 0.0 <= s <= 10.0
