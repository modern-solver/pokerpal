"""BLIP-2 scene-description service tests (Agent A3 / Stage S-03).

Tests the local-first / HF-Serverless-fallback orchestration + the HF Serverless
REST wrapper with an injected HTTP transport — no torch/transformers, no network,
no HF_TOKEN.
"""

import json

import pytest

from curator.grading.scene import (
    HFServerlessDescriber,
    LocalBlip2Describer,
    SceneDescriptionService,
    derive_signals,
)

IMG = b"img-bytes"


class GoodDescriber:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def describe(self, image_bytes):
        self.calls += 1
        return self.text


class FailingDescriber:
    def __init__(self):
        self.calls = 0

    def describe(self, image_bytes):
        self.calls += 1
        raise RuntimeError("model unavailable")


def test_service_prefers_local():
    local = GoodDescriber("local caption")
    serverless = GoodDescriber("serverless caption")
    svc = SceneDescriptionService(local=local, serverless=serverless)
    res = svc.describe(IMG)
    assert res.source == "local"
    assert res.description == "local caption"
    assert serverless.calls == 0


def test_service_falls_back_to_serverless_when_local_fails():
    svc = SceneDescriptionService(local=FailingDescriber(), serverless=GoodDescriber("hi"))
    res = svc.describe(IMG)
    assert res.source == "serverless"
    assert res.description == "hi"


def test_service_graceful_none_when_both_absent():
    res = SceneDescriptionService().describe(IMG)
    assert res.source == "none"
    assert res.description == ""
    assert res.signals == {}


def test_describe_via_serverless_forces_serverless():
    svc = SceneDescriptionService(local=GoodDescriber("local"), serverless=GoodDescriber("remote"))
    res = svc.describe_via_serverless(IMG)
    assert res.source == "serverless"
    assert res.description == "remote"


# --- HF Serverless REST wrapper (injected transport) ------------------------
def test_hf_serverless_parses_caption_payload():
    def fake_post(url, *, headers, data):
        assert "Authorization" in headers
        assert data == IMG
        return 200, json.dumps([{"generated_text": "a dog on a beach"}]).encode()

    d = HFServerlessDescriber(token="hf_x", http_post=fake_post)
    assert d.describe(IMG) == "a dog on a beach"


def test_hf_serverless_raises_without_token():
    d = HFServerlessDescriber(token=None)
    with pytest.raises(RuntimeError):
        d.describe(IMG)


def test_hf_serverless_raises_on_non_200():
    def fake_post(url, *, headers, data):
        return 503, b"loading"

    d = HFServerlessDescriber(token="hf_x", http_post=fake_post)
    with pytest.raises(RuntimeError):
        d.describe(IMG)


def test_local_blip2_raises_when_libs_absent():
    """In the sandbox transformers/torch are absent -> clean RuntimeError (not crash)."""
    pytest.importorskip  # noqa: just to keep import used
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        pytest.skip("heavy libs present — local path would actually load a model")
    except Exception:
        pass
    with pytest.raises(RuntimeError):
        LocalBlip2Describer().describe(IMG)


def test_signals_feed_into_service_result():
    svc = SceneDescriptionService(serverless=GoodDescriber("a smiling woman cooking dinner"))
    res = svc.describe(IMG)
    assert res.signals["activity"] is True
    assert res.signals["mood_score"] > 5.0
    assert res.signals == derive_signals("a smiling woman cooking dinner")
