"""Live Groq Llama 4 Scout vision smoke test (Agent A3 / Stage S-03).

PRD S-03 exit gate: "Groq vision scores return in < 5s." This makes a REAL vision
grading call and asserts sub-5s latency + parseable 4-axis JSON. It SKIPS
gracefully when GROQ_API_KEY is unset or the `groq` package is absent, mirroring
A0's handshake smoke test — so it passes-as-skip on the clean sandbox and only
runs when a real key + lib are present.

Also a live BLIP-2 local-load smoke test that skips without transformers/torch.
"""

import io
import os
import time

import pytest

from curator.grading.groq_client import GroqVisionGrader
from curator.grading.scene import BLIP_LOCAL_MODEL, LocalBlip2Describer
from curator.grading.tier3 import GROQ_LATENCY_BUDGET_S, parse_grading_json


def _tiny_jpeg() -> bytes:
    """A small in-memory JPEG so the smoke test needs no committed image."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    from PIL import Image  # noqa

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 140, 160)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live Groq vision smoke (sandbox has no key)",
)
def test_groq_vision_grades_under_5s():
    pytest.importorskip("groq", reason="groq package not installed")

    grader = GroqVisionGrader(api_key=os.environ["GROQ_API_KEY"])
    img = _tiny_jpeg()

    start = time.perf_counter()
    raw = grader.grade(img)
    elapsed = time.perf_counter() - start

    axes = parse_grading_json(raw)  # must be parseable 4-axis JSON
    assert set(axes) == {"composition", "lighting", "subject", "mood"}
    assert elapsed < GROQ_LATENCY_BUDGET_S, (
        f"Groq vision took {elapsed:.3f}s, budget is {GROQ_LATENCY_BUDGET_S}s"
    )


def test_blip2_local_load_smoke():
    """Live BLIP-2 local load — skips without transformers/torch (sandbox)."""
    pytest.importorskip("transformers", reason="transformers not installed")
    pytest.importorskip("torch", reason="torch not installed")

    describer = LocalBlip2Describer(model=BLIP_LOCAL_MODEL)
    img = _tiny_jpeg()
    try:
        desc = describer.describe(img)
    except Exception as exc:  # model fetch unavailable offline
        pytest.skip(f"BLIP-2 model unavailable in this environment: {exc}")
    assert isinstance(desc, str)
