"""Tests for Tier 1 metadata extraction + flags (Agent A2 / Stage S-02).

Resolution/blur/dedup logic is tested on synthetic metadata (no images). Where an
actual image is needed, a small one is synthesised in-memory with Pillow and never
committed. OpenCV-dependent fields (blur/face) skip gracefully when cv2 is absent.
"""

import io

import pytest

from curator.grading.tier1 import (
    BLUR_LAPLACIAN_THRESHOLD,
    MIN_RESOLUTION_PX,
    PhotoMetadata,
    extract_metadata,
    phash_distance,
)


def test_low_res_flag_from_small_image():
    m = PhotoMetadata.from_dict({"width": 400, "height": 400})  # 160k px < 0.5MP
    assert m.megapixels == pytest.approx(0.16)
    assert m.low_res is True
    assert "low_res" in m.flags


def test_high_res_not_flagged():
    m = PhotoMetadata.from_dict({"width": 1080, "height": 1080})  # 1.16MP
    assert m.low_res is False
    assert "low_res" not in m.flags


def test_resolution_threshold_boundary():
    assert MIN_RESOLUTION_PX == 500_000


def test_blur_flag_when_below_threshold():
    m = PhotoMetadata.from_dict({"width": 1080, "height": 1080, "laplacian_variance": 10.0})
    assert m.blurry is True
    assert "blurry" in m.flags


def test_blur_not_flagged_above_threshold():
    m = PhotoMetadata.from_dict(
        {"width": 1080, "height": 1080, "laplacian_variance": BLUR_LAPLACIAN_THRESHOLD + 1}
    )
    assert m.blurry is False


def test_unknown_blur_is_not_blurry():
    # cv2 unavailable -> laplacian_variance None -> never flagged blurry (fail open).
    m = PhotoMetadata.from_dict({"width": 1080, "height": 1080})
    assert m.laplacian_variance is None
    assert m.blurry is False


def test_phash_distance_identical_is_zero():
    assert phash_distance("ffffffffffffffff", "ffffffffffffffff") == 0


def test_phash_distance_near_dupe():
    d = phash_distance("ffffffffffffffff", "fffffffffffffff0")
    assert d is not None and d < 10


def test_phash_distance_unknown_is_none():
    assert phash_distance(None, "ffffffffffffffff") is None


def test_extract_metadata_on_synthetic_image():
    """Synthesise a small image in-memory (not committed) and extract metadata."""
    Image = pytest.importorskip("PIL.Image")
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (1000, 1250), (200, 120, 60))  # warm-ish portrait
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    loaded = PILImage.open(buf)

    meta = extract_metadata(loaded)
    assert meta.width == 1000 and meta.height == 1250
    assert meta.aspect_ratio == pytest.approx(0.8)
    assert meta.megapixels == pytest.approx(1.25)
    assert meta.low_res is False
    assert 0.0 <= meta.saturation_mean <= 1.0
    assert 0.0 <= meta.brightness_mean <= 1.0
    assert meta.phash is not None  # imagehash available in sandbox
