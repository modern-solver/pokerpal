"""Tier 1 — Local pre-filter (Agent A2 / Stage S-02). ZERO network, ZERO cost.

Extracts per-photo technical metadata using ONLY local libraries (Pillow, OpenCV,
imagehash). No API call, no model, no network is ever made here — Tier 1+2 are a
hard zero-cost boundary that Reviewer R verifies.

Graceful degradation: OpenCV (cv2) is an optional heavy dependency in the sandbox.
Blur (Laplacian variance) and Haar face detection require it; when cv2 is missing
those fields are reported as `None` (UNKNOWN) and the corresponding flags are NOT
raised (we never flag a photo blurry/no-face on missing-tooling — fail open). All
metadata-driven Tier 2 scoring works regardless, because it consumes the metadata
dict, which can also be supplied synthetically in tests.

Thresholds (PRD Tier 1):
    resolution  : width*height < 0.5MP   -> low_res = True
    blur        : Laplacian variance < 50 -> blurry = True
    dedup       : pHash distance < 10     -> near-duplicate (handled in agent.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- thresholds (PRD Tier 1) ------------------------------------------------
MIN_MEGAPIXELS = 0.5
MIN_RESOLUTION_PX = int(MIN_MEGAPIXELS * 1_000_000)  # 500_000
BLUR_LAPLACIAN_THRESHOLD = 50.0
PHASH_NEAR_DUPE_DISTANCE = 10  # distance < 10 -> near-dupe (PRD G-05, relaxed)

# Aspect-ratio bucket boundaries (width/height ratio). Used by Tier 2.
# Buckets are matched in tier2.classify_aspect_bucket.
ASPECT_BUCKETS = {
    "story_9_16": 9 / 16,      # 0.5625 — tall portrait / story
    "portrait_4_5": 4 / 5,     # 0.80  — IG portrait
    "square_1_1": 1.0,         # 1.0
    "standard_4_3": 4 / 3,     # 1.333 — classic landscape
    "landscape_16_9": 16 / 9,  # 1.778 — wide landscape
}


def _try_import_cv2():
    try:
        import cv2  # noqa: F401  (local, offline import only)

        return cv2
    except Exception:
        return None


def _try_import_numpy():
    try:
        import numpy as np

        return np
    except Exception:
        return None


@dataclass
class PhotoMetadata:
    """Tier-1 technical metadata for a single photo.

    Fields set to None mean "unknown" (e.g. cv2 unavailable) — NOT a failing value.
    Tier 2 treats unknowns as neutral (no bonus, no penalty).
    """

    width: Optional[int] = None
    height: Optional[int] = None
    megapixels: Optional[float] = None
    aspect_ratio: Optional[float] = None  # width / height
    saturation_mean: Optional[float] = None  # 0..1
    brightness_mean: Optional[float] = None  # 0..1
    warm_dominant: Optional[bool] = None
    laplacian_variance: Optional[float] = None
    face_count: Optional[int] = None
    largest_face_frac: Optional[float] = None  # largest face area / frame area
    phash: Optional[str] = None  # imagehash average_hash hex string
    # flags
    low_res: bool = False
    blurry: bool = False
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "megapixels": self.megapixels,
            "aspect_ratio": self.aspect_ratio,
            "saturation_mean": self.saturation_mean,
            "brightness_mean": self.brightness_mean,
            "warm_dominant": self.warm_dominant,
            "laplacian_variance": self.laplacian_variance,
            "face_count": self.face_count,
            "largest_face_frac": self.largest_face_frac,
            "phash": self.phash,
            "low_res": self.low_res,
            "blurry": self.blurry,
            "flags": list(self.flags),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhotoMetadata":
        """Build metadata from a plain dict (used by tests with synthetic data)."""
        m = cls(
            width=d.get("width"),
            height=d.get("height"),
            megapixels=d.get("megapixels"),
            aspect_ratio=d.get("aspect_ratio"),
            saturation_mean=d.get("saturation_mean"),
            brightness_mean=d.get("brightness_mean"),
            warm_dominant=d.get("warm_dominant"),
            laplacian_variance=d.get("laplacian_variance"),
            face_count=d.get("face_count"),
            largest_face_frac=d.get("largest_face_frac"),
            phash=d.get("phash"),
            low_res=bool(d.get("low_res", False)),
            blurry=bool(d.get("blurry", False)),
            flags=list(d.get("flags", [])),
        )
        # Derive resolution/aspect/flags if size given but derived fields absent.
        m.finalize()
        return m

    def finalize(self) -> "PhotoMetadata":
        """Fill derived fields + raise low_res/blurry flags from raw measurements."""
        if self.width and self.height:
            if self.megapixels is None:
                self.megapixels = (self.width * self.height) / 1_000_000
            if self.aspect_ratio is None:
                self.aspect_ratio = self.width / self.height
            if self.width * self.height < MIN_RESOLUTION_PX:
                self.low_res = True
        if self.low_res and "low_res" not in self.flags:
            self.flags.append("low_res")
        # Blur only when we actually measured it (cv2 available). Unknown != blurry.
        if self.laplacian_variance is not None:
            if self.laplacian_variance < BLUR_LAPLACIAN_THRESHOLD:
                self.blurry = True
        if self.blurry and "blurry" not in self.flags:
            self.flags.append("blurry")
        return self


# --- Haar cascade path resolution (cv2 ships the XML data) -------------------
def _frontalface_cascade(cv2):
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return None
        return cascade
    except Exception:
        return None


def extract_metadata(image: Any) -> PhotoMetadata:
    """Extract Tier-1 metadata from a Pillow Image (or path).

    Pure local. Pillow handles resolution / aspect / saturation / brightness /
    pHash; OpenCV (if present) handles Laplacian blur + Haar face count. Anything
    requiring an unavailable lib is left as None (unknown) and never raises.
    """
    from PIL import Image, ImageStat  # local import keeps module import cheap

    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    img = img.convert("RGB")

    meta = PhotoMetadata()
    meta.width, meta.height = img.size

    # Saturation + brightness via HSV (Pillow). S,V are 0..255 -> normalise 0..1.
    hsv = img.convert("HSV")
    stat = ImageStat.Stat(hsv)
    # HSV channels: 0=H, 1=S, 2=V
    meta.saturation_mean = stat.mean[1] / 255.0
    meta.brightness_mean = stat.mean[2] / 255.0
    # Warm-tone dominance: mean hue in the 15-45 band (Pillow H is 0..255 ~ 0..360).
    mean_hue_deg = (stat.mean[0] / 255.0) * 360.0
    meta.warm_dominant = 15.0 <= mean_hue_deg <= 45.0

    # Perceptual hash (imagehash, local).
    try:
        import imagehash

        meta.phash = str(imagehash.average_hash(img))
    except Exception:
        meta.phash = None

    # Blur + faces require OpenCV.
    cv2 = _try_import_cv2()
    np = _try_import_numpy()
    if cv2 is not None and np is not None:
        arr = np.array(img)  # RGB
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        meta.laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        cascade = _frontalface_cascade(cv2)
        if cascade is not None:
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            meta.face_count = int(len(faces))
            if len(faces) > 0:
                frame_area = float(meta.width * meta.height)
                largest = max(int(w) * int(h) for (_x, _y, w, h) in faces)
                meta.largest_face_frac = largest / frame_area if frame_area else None

    meta.finalize()
    return meta


def phash_distance(hash_a: Optional[str], hash_b: Optional[str]) -> Optional[int]:
    """Hamming distance between two average_hash hex strings. None if unavailable."""
    if not hash_a or not hash_b:
        return None
    try:
        import imagehash

        return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
    except Exception:
        # Fallback: bit-count diff of the hex ints (works without imagehash too).
        try:
            a = int(hash_a, 16)
            b = int(hash_b, 16)
            return bin(a ^ b).count("1")
        except Exception:
            return None
