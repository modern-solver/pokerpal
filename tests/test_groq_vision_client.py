"""Groq vision client wiring tests (Agent A3 / Stage S-03).

Verifies the structured prompt is the EXACT PRD template, the image is sent as a
base64 data URL on a vision content part, and a 429 surfaces as
GroqRateLimitError. Uses an injected fake Groq client — no key, no network, no
`groq` package.
"""

import base64

import pytest

from curator.grading.groq_client import (
    GRADING_SYSTEM_PROMPT,
    GRADING_USER_PROMPT,
    GroqRateLimitError,
    GroqVisionGrader,
    _is_rate_limit,
)

IMG = b"\xff\xd8jpeg"


class FakeMessage:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeMessage(content)]


class FakeGroqClient:
    """Mimics groq.Groq().chat.completions.create, recording the call kwargs."""

    def __init__(self, content='{"composition":7,"lighting":7,"subject":7,"mood":7}',
                 raise_exc=None):
        self.content = content
        self.raise_exc = raise_exc
        self.last_kwargs = None

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.last_kwargs = kwargs
                if outer.raise_exc is not None:
                    raise outer.raise_exc
                return FakeCompletion(outer.content)

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_prompt_matches_prd_template_verbatim():
    assert GRADING_SYSTEM_PROMPT.startswith("You are a photo quality assessor")
    assert "Respond ONLY with JSON" in GRADING_SYSTEM_PROMPT
    for axis in ("composition", "lighting", "subject", "mood"):
        assert f'"{axis}"' in GRADING_USER_PROMPT
    assert "1-10" in GRADING_USER_PROMPT


def test_grade_builds_vision_message_with_data_url():
    client = FakeGroqClient()
    grader = GroqVisionGrader(client=client)
    out = grader.grade(IMG)
    assert out == '{"composition":7,"lighting":7,"subject":7,"mood":7}'

    msgs = client.last_kwargs["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == GRADING_SYSTEM_PROMPT
    user = msgs[1]
    assert user["role"] == "user"
    parts = user["content"]
    text_part = next(p for p in parts if p["type"] == "text")
    img_part = next(p for p in parts if p["type"] == "image_url")
    assert text_part["text"] == GRADING_USER_PROMPT
    expected_url = "data:image/jpeg;base64," + base64.b64encode(IMG).decode()
    assert img_part["image_url"]["url"] == expected_url


def test_grade_uses_vision_model_and_deterministic_temp():
    client = FakeGroqClient()
    GroqVisionGrader(client=client, model="meta-llama/llama-4-scout-x").grade(IMG)
    assert client.last_kwargs["model"] == "meta-llama/llama-4-scout-x"
    assert client.last_kwargs["temperature"] == 0.0


def test_429_normalised_to_rate_limit_error():
    exc = Exception("Error 429 rate limit exceeded")
    client = FakeGroqClient(raise_exc=exc)
    with pytest.raises(GroqRateLimitError):
        GroqVisionGrader(client=client).grade(IMG)


def test_non_429_error_propagates():
    client = FakeGroqClient(raise_exc=ValueError("bad request"))
    with pytest.raises(ValueError):
        GroqVisionGrader(client=client).grade(IMG)


def test_is_rate_limit_detection():
    assert _is_rate_limit(Exception("429 Too Many Requests"))
    assert _is_rate_limit(Exception("Rate limit reached"))
    err = type("E", (), {"status_code": 429})()
    assert _is_rate_limit(err)
    assert not _is_rate_limit(Exception("500 server error"))


def test_grade_without_key_or_client_raises():
    with pytest.raises(RuntimeError):
        GroqVisionGrader(api_key=None, client=None).grade(IMG)
