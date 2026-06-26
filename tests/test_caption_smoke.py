"""Live Groq Llama 3.3 70B caption smoke test (Agent A4 / Stage S-04).

Makes a REAL caption-generation call and asserts a parseable JSON caption comes
back. SKIPS gracefully when GROQ_API_KEY is unset or the `groq` package is absent
— mirroring A0's handshake and A3's vision smoke test, so it passes-as-skip on
the clean sandbox and only runs when a real key + lib are present.
"""

import json
import os

import pytest

from curator.caption import CaptionAgent
from curator.caption.groq_text import GroqTextGenerator


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live Groq caption smoke (sandbox has no key)",
)
def test_groq_caption_generates_post_caption():
    pytest.importorskip("groq", reason="groq package not installed")

    gen = GroqTextGenerator(api_key=os.environ["GROQ_API_KEY"])
    agent = CaptionAgent(generator=gen)
    result = agent.generate(
        "instagram",
        "a person hiking on a mountain trail at golden hour",
        vibe="adventurous summer energy",
    )
    assert result.output_mode == "post_caption"
    assert len(result.variants) == 3
    for v in result.variants:
        assert v.char_count <= 150
        assert set(v.payload) == {"caption", "hashtags", "char_count"}
        # payload round-trips as JSON (it is what A5 will render)
        json.dumps(v.payload)
