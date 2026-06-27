"""Groq Llama 3.3 70B text client wiring tests (Agent A4 / Stage S-04).

Verifies the caption text client builds a (system, user) chat call on the
correct model with the right params, surfaces a 429 as GroqRateLimitError, and
raises cleanly without a key/client. Injected fake client — no key, no network,
no `groq` package. Mirrors A3's vision-client tests.
"""

import pytest

from curator.caption.groq_text import (
    GROQ_TEXT_MODEL,
    GroqRateLimitError,
    GroqTextGenerator,
    _is_rate_limit,
)


class FakeMessage:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeMessage(content)]


class FakeGroqClient:
    def __init__(self, content='{"caption":"hi","hashtags":[],"char_count":2}',
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


def test_model_id_default_is_llama_33_70b():
    assert GROQ_TEXT_MODEL == "llama-3.3-70b-versatile"


def test_generate_builds_system_user_messages():
    client = FakeGroqClient()
    gen = GroqTextGenerator(client=client)
    out = gen.generate("SYS", "USER")
    assert out == '{"caption":"hi","hashtags":[],"char_count":2}'
    msgs = client.last_kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "USER"}


def test_generate_uses_text_model():
    client = FakeGroqClient()
    GroqTextGenerator(client=client).generate("s", "u")
    assert client.last_kwargs["model"] == GROQ_TEXT_MODEL


def test_429_normalised_to_rate_limit_error():
    client = FakeGroqClient(raise_exc=Exception("Error 429 rate limit exceeded"))
    with pytest.raises(GroqRateLimitError):
        GroqTextGenerator(client=client).generate("s", "u")


def test_non_429_error_propagates():
    client = FakeGroqClient(raise_exc=ValueError("bad request"))
    with pytest.raises(ValueError):
        GroqTextGenerator(client=client).generate("s", "u")


def test_is_rate_limit_detection():
    assert _is_rate_limit(Exception("429 Too Many Requests"))
    assert _is_rate_limit(Exception("Rate limit reached"))
    err = type("E", (), {"status_code": 429})()
    assert _is_rate_limit(err)
    assert not _is_rate_limit(Exception("500 server error"))


def test_generate_without_key_or_client_raises():
    with pytest.raises(RuntimeError):
        GroqTextGenerator(api_key=None, client=None).generate("s", "u")
