"""CaptionAgent tests — length-category captions (short/long/haiku), no hashtags."""

import json

import pytest

from curator.caption.agent import CaptionAgent, N_OPTIONS, _clean, _parse_captions
from curator.caption.prompts import get_prompts, length_spec, normalize_length


class FakeGen:
    """Stub TextGenerator returning a fixed JSON captions payload."""

    def __init__(self, captions, record=None):
        self._caps = captions
        self.calls = 0
        self.record = record

    def generate(self, system, user):
        self.calls += 1
        if self.record is not None:
            self.record.append((system, user))
        return json.dumps({"captions": self._caps})


# --- length modes -----------------------------------------------------------
def test_three_options_for_each_length_mode():
    for mode in ("short", "long", "haiku"):
        gen = FakeGen([f"{mode} one", f"{mode} two", f"{mode} three"])
        r = CaptionAgent(generator=gen).generate("instagram", "a beach at sunset", length_mode=mode)
        assert r.length_mode == mode
        assert len(r.variants) == N_OPTIONS
        assert [v.label for v in r.variants] == ["Option 1", "Option 2", "Option 3"]


def test_short_cap_enforced_200():
    long_text = "word " * 100  # ~500 chars
    gen = FakeGen([long_text, long_text, long_text])
    r = CaptionAgent(generator=gen).generate("instagram", "scene", length_mode="short")
    for v in r.variants:
        assert v.char_count <= 200


def test_long_cap_enforced_1000():
    long_text = "word " * 400
    gen = FakeGen([long_text, long_text, long_text])
    r = CaptionAgent(generator=gen).generate("facebook", "scene", length_mode="long")
    for v in r.variants:
        assert v.char_count <= 1000


def test_haiku_keeps_three_lines():
    h = "Golden light descends\nshadows stretch across the sand\nthe day exhales slow"
    gen = FakeGen([h, h, h])
    r = CaptionAgent(generator=gen).generate("instagram", "beach", length_mode="haiku")
    assert r.variants[0].text.count("\n") == 2  # three lines


# --- no hashtags ------------------------------------------------------------
def test_hashtags_are_stripped():
    gen = FakeGen(["Sunset vibes #beach #golden", "Calm seas #ocean", "Still water #zen"])
    r = CaptionAgent(generator=gen).generate("instagram", "beach", length_mode="short")
    for v in r.variants:
        assert "#" not in v.text
        assert "hashtags" not in v.payload


def test_clean_strips_hashtags_and_quotes():
    assert "#" not in _clean('"Hello world #tag"')
    assert _clean('"quoted"') == "quoted"


# --- payload shape ----------------------------------------------------------
def test_payload_is_caption_only():
    gen = FakeGen(["a", "b", "c"])
    r = CaptionAgent(generator=gen).generate("linkedin", "office", length_mode="short")
    assert set(r.variants[0].payload) == {"caption", "char_count"}


# --- parsing + fallback -----------------------------------------------------
def test_parse_captions_fence_tolerant():
    raw = '```json\n{"captions": ["one", "two", "three"]}\n```'
    assert _parse_captions(raw) == ["one", "two", "three"]


def test_local_fallback_without_generator():
    r = CaptionAgent(generator=None).generate("instagram", "a quiet forest", length_mode="short")
    assert len(r.variants) == 3
    assert all(v.source == "fallback" for v in r.variants)
    assert all(v.char_count <= 200 for v in r.variants)


def test_bad_json_falls_back():
    class Bad:
        def generate(self, s, u):
            return "not json at all"
    r = CaptionAgent(generator=Bad()).generate("instagram", "scene", length_mode="short")
    assert len(r.variants) == 3
    assert all(v.source == "fallback" for v in r.variants)


def test_fewer_than_three_captions_are_padded():
    gen = FakeGen(["only one"])
    r = CaptionAgent(generator=gen).generate("instagram", "scene", length_mode="short")
    assert len(r.variants) == 3  # padded with fallback


# --- prompt construction ----------------------------------------------------
def test_prompt_carries_platform_scene_vibe_length():
    rec = []
    gen = FakeGen(["a", "b", "c"], record=rec)
    CaptionAgent(generator=gen).generate(
        "tinder", "a person hiking", vibe="adventurous", constraints=["no location names"],
        length_mode="short",
    )
    system, user = rec[0]
    assert "JSON" in system and "hashtag" in system.lower()
    assert "tinder" in user and "hiking" in user and "adventurous" in user
    assert "no location names" in user


def test_normalize_length_defaults_to_short():
    assert normalize_length("bogus") == "short"
    assert normalize_length("HAIKU") == "haiku"
    assert length_spec(get_prompts(), "long")["max_chars"] == 1000
