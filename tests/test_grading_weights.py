"""Tests for the platform_weights.yml loader (Agent A2 / Stage S-02).

Pure local — no images, no network. Validates schema completeness for all 6
platforms and the weight-sum invariant.
"""

import pytest

from curator.grading.weights import AXES, load_weights, get_weights
from curator.session.schema import PLATFORMS


def test_all_six_platforms_present():
    cfg = load_weights()
    for platform in PLATFORMS:
        assert platform in cfg, f"missing platform {platform}"
    assert len(PLATFORMS) == 6


def test_each_platform_weights_sum_to_one():
    cfg = load_weights()
    for platform in PLATFORMS:
        total = sum(float(cfg[platform]["weights"][ax]) for ax in AXES)
        assert abs(total - 1.0) < 1e-6, f"{platform} weights sum to {total}"


def test_dating_platforms_have_mandatory_face_gate():
    cfg = load_weights()
    for platform in ("tinder", "bumble"):
        face = cfg[platform]["face"]
        assert face["mandatory"] is True
        assert float(face["cap_score"]) == 3.0
    # Non-dating must not be mandatory.
    for platform in ("instagram", "facebook", "hinge", "linkedin"):
        assert cfg[platform]["face"].get("mandatory") is not True


def test_get_weights_is_cached():
    assert get_weights() is get_weights()


def test_loader_rejects_bad_weight_sum(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text(
        "instagram:\n"
        "  weights: {composition: 0.5, lighting: 0.5, technical: 0.5, platform_fit: 0.5}\n"
    )
    with pytest.raises(ValueError):
        load_weights(str(bad))
