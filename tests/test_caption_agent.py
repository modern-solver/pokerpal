"""CaptionAgent tests (Agent A4 / Stage S-04).

Covers the S-04 Review & Test gate + Exit gate, all WITHOUT a real Groq key /
package / network — the Groq text client is injected as a stub:
  * correct OUTPUT TYPE per platform (rule C-08);
  * 3 tone variants per photo (PRD Section 03);
  * banned-phrase rejection for dating bios (rule C-09);
  * Hinge returns a {prompt, answer} pair drawn from the standard library (C-10);
  * length caps enforced (over-length retried then trimmed; final <= cap);
  * LinkedIn tag policy + professional-tone rules (rule C-11);
  * output-mode switching loads the right template (C-08);
  * prompt construction carries platform/tone/{blip_desc}/vibe/constraints.
"""

import json

import pytest

from curator.caption import (
    CaptionAgent,
    GROQ_TEXT_MODEL,
    get_prompts,
    output_mode_for,
)
from curator.caption.hinge import is_library_prompt, select_hinge_prompt
from curator.caption.prompts import PLATFORM_OUTPUT_MODE
from curator.session.schema import PLATFORMS, Session


# --- fake injected Groq text client -----------------------------------------
class FakeGen:
    """Returns a scripted JSON string; records the (system, user) prompts seen.

    ``payload`` may be a dict (same every call) or a list (one per call, cycled).
    """

    def __init__(self, payload, raise_exc=None):
        self.payload = payload
        self.raise_exc = raise_exc
        self.calls = []
        self._i = 0

    def generate(self, system, user):
        self.calls.append((system, user))
        if self.raise_exc is not None:
            raise self.raise_exc
        if isinstance(self.payload, list):
            item = self.payload[min(self._i, len(self.payload) - 1)]
            self._i += 1
        else:
            item = self.payload
        return item if isinstance(item, str) else json.dumps(item)


def _agent(payload, **kw):
    return CaptionAgent(generator=FakeGen(payload), **kw)


# --- output mode / switcher (C-08) ------------------------------------------
def test_platform_output_mode_map_matches_spec():
    assert PLATFORM_OUTPUT_MODE == {
        "instagram": "post_caption",
        "facebook": "post_caption",
        "tinder": "dating_bio",
        "bumble": "dating_bio",
        "hinge": "prompt_answer",
        "linkedin": "linkedin_post",
    }
    for p in PLATFORMS:
        assert output_mode_for(p) == PLATFORM_OUTPUT_MODE[p]


def test_instagram_produces_post_caption():
    a = _agent({"caption": "Trail magic", "hashtags": ["#hike", "#sun", "#trail", "#wild", "#go"]})
    r = a.generate("instagram", "person hiking at sunset")
    assert r.output_mode == "post_caption"
    v = r.variants[0]
    assert set(v.payload) == {"caption", "hashtags", "char_count"}
    assert v.payload["caption"] == "Trail magic"
    assert v.hashtags  # IG appends hashtags


def test_facebook_produces_post_caption():
    a = _agent({"caption": "A lovely day out by the lake with friends.", "hashtags": ["#lake", "#friends", "#weekend"]})
    r = a.generate("facebook", "friends by a lake")
    assert r.output_mode == "post_caption"
    assert set(r.variants[0].payload) == {"caption", "hashtags", "char_count"}


def test_tinder_produces_dating_bio_no_hashtags():
    a = _agent({"bio": "Equally good at ordering wine and assembling IKEA"})
    r = a.generate("tinder", "smiling portrait")
    assert r.output_mode == "dating_bio"
    v = r.variants[0]
    assert set(v.payload) == {"bio", "char_count"}
    assert v.hashtags == []  # dating: never hashtags


def test_bumble_produces_dating_bio():
    a = _agent({"bio": "Perpetually planning my next trip"})
    r = a.generate("bumble", "person traveling")
    assert r.output_mode == "dating_bio"
    assert set(r.variants[0].payload) == {"bio", "char_count"}


def test_hinge_produces_prompt_answer_pair():
    a = _agent({"answer": "Finding good light and a good taco"})
    r = a.generate("hinge", "a travel scene on a mountain trail")
    assert r.output_mode == "prompt_answer"
    v = r.variants[0]
    assert set(v.payload) == {"prompt", "answer", "char_count"}
    assert v.payload["prompt"]  # non-empty
    assert v.payload["answer"]
    # the prompt MUST come from the standard library, never invented (C-10).
    lib = get_prompts()["hinge_prompts"]
    assert is_library_prompt(lib, v.payload["prompt"])


def test_linkedin_produces_linkedin_post():
    a = _agent({"post": "Reflecting on a milestone for the team this quarter.", "hashtags": ["#leadership", "#growth", "#team"]})
    r = a.generate("linkedin", "person at a conference, banner behind")
    assert r.output_mode == "linkedin_post"
    assert set(r.variants[0].payload) == {"post", "hashtags", "char_count"}


def test_all_six_platforms_emit_correct_output_type():
    """Exit gate: all 6 platforms generate the correct output type."""
    payloads = {
        "instagram": {"caption": "x", "hashtags": ["#a", "#b", "#c", "#d", "#e"]},
        "facebook": {"caption": "x", "hashtags": ["#a", "#b", "#c"]},
        "tinder": {"bio": "x"},
        "bumble": {"bio": "x"},
        "hinge": {"answer": "x"},
        "linkedin": {"post": "x", "hashtags": ["#leadership", "#growth", "#career"]},
    }
    expected_keys = {
        "post_caption": {"caption", "hashtags", "char_count"},
        "dating_bio": {"bio", "char_count"},
        "prompt_answer": {"prompt", "answer", "char_count"},
        "linkedin_post": {"post", "hashtags", "char_count"},
    }
    for platform in PLATFORMS:
        a = _agent(payloads[platform])
        r = a.generate(platform, "a generic scene")
        assert r.output_mode == PLATFORM_OUTPUT_MODE[platform]
        assert set(r.variants[0].payload) == expected_keys[r.output_mode]


# --- 3 tone variants (PRD Section 03) ---------------------------------------
def test_three_tone_variants_per_photo_in_order():
    a = _agent({"caption": "hi", "hashtags": ["#a", "#b", "#c", "#d", "#e"]})
    r = a.generate("instagram", "scene")
    assert len(r.variants) == 3
    assert r.tones == ["Witty", "Heartfelt", "Minimal"]


def test_tinder_tone_cycle_order_for_retry():
    """C-07: dating tone variants are produced in the documented order."""
    a = _agent({"bio": "hi"})
    r = a.generate("tinder", "scene")
    assert r.tones == ["Playful", "Confident", "Mysterious"]


def test_tones_override_supports_retry_cycle():
    """A6's /retry can pass a rotated tone order; A4 exposes the capability."""
    a = _agent({"bio": "hi"})
    r = a.generate("tinder", "scene", tones=["Confident", "Mysterious", "Playful"])
    assert r.tones == ["Confident", "Mysterious", "Playful"]


# --- banned-phrase rejection (C-09) -----------------------------------------
def test_banned_phrase_triggers_regeneration():
    """A bio with a banned phrase is rejected; a clean retry is accepted."""
    gen = FakeGen([
        {"bio": "Just a partner in crime looking for fun"},   # banned -> reject
        {"bio": "I build furniture badly and brew coffee well"},  # clean
    ])
    a = CaptionAgent(generator=gen)
    r = a.generate("tinder", "smiling portrait", tones=["Playful"])
    v = r.variants[0]
    assert v.attempts >= 2  # the first (banned) output was rejected
    assert "partner in crime" not in v.text.lower()


def test_banned_phrase_scrubbed_if_model_never_complies():
    """If every retry contains a banned phrase, the final text is scrubbed clean."""
    gen = FakeGen({"bio": "I love to laugh and laugh"})  # always banned
    a = CaptionAgent(generator=gen)
    r = a.generate("tinder", "portrait", tones=["Playful"])
    assert "love to laugh" not in r.variants[0].text.lower()


def test_banned_list_loaded_from_yaml():
    from curator.caption import banned_phrases
    bp = banned_phrases(get_prompts())
    for must in ("love to laugh", "fluent in sarcasm", "partner in crime",
                 "loves adventures", "looking for my person"):
        assert must in bp


# --- Hinge library selection (C-10) -----------------------------------------
def test_hinge_selects_travel_prompt_for_travel_scene():
    lib = get_prompts()["hinge_prompts"]
    chosen = select_hinge_prompt(lib, "a person hiking a mountain trail while traveling")
    assert chosen == "Best travel story"
    assert is_library_prompt(lib, chosen)


def test_hinge_selects_food_prompt_for_food_scene():
    lib = get_prompts()["hinge_prompts"]
    chosen = select_hinge_prompt(lib, "a person cooking dinner in a restaurant kitchen")
    assert chosen == "The way to my heart is..."


def test_hinge_defaults_to_first_library_prompt_when_no_match():
    lib = get_prompts()["hinge_prompts"]
    chosen = select_hinge_prompt(lib, "")
    assert chosen == lib[0]["prompt"]


def test_hinge_prompt_shared_across_tone_variants():
    a = _agent({"answer": "x"})
    r = a.generate("hinge", "a travel scene by the ocean")
    prompts = {v.prompt for v in r.variants}
    assert len(prompts) == 1  # one library prompt per photo, all tones share it


def test_select_hinge_prompt_empty_library_raises():
    with pytest.raises(ValueError):
        select_hinge_prompt([], "anything")


# --- length caps (enforced; retry then trim) --------------------------------
def test_overlength_caption_is_trimmed_to_cap():
    long = "x" * 400
    a = _agent({"caption": long, "hashtags": ["#a", "#b", "#c", "#d", "#e"]})
    r = a.generate("instagram", "scene")  # IG cap 150
    v = r.variants[0]
    assert v.char_count <= 150
    assert v.trimmed is True


def test_within_cap_caption_not_trimmed():
    a = _agent({"caption": "short and sweet", "hashtags": ["#a", "#b", "#c", "#d", "#e"]})
    r = a.generate("instagram", "scene")
    assert r.variants[0].trimmed is False
    assert r.variants[0].char_count <= 150


def test_length_retry_prefers_within_cap_output():
    """First output over cap, retry within cap -> accept the shorter one untrimmed."""
    gen = FakeGen([
        {"bio": "y" * 200},   # over the 100-char Tinder cap
        {"bio": "concise and punchy"},  # within cap
    ])
    a = CaptionAgent(generator=gen)
    r = a.generate("tinder", "portrait", tones=["Playful"])
    v = r.variants[0]
    assert v.char_count <= 100
    assert v.attempts >= 2


@pytest.mark.parametrize("platform,cap", [
    ("instagram", 150), ("facebook", 500), ("tinder", 100),
    ("bumble", 150), ("hinge", 150), ("linkedin", 300),
])
def test_each_platform_cap_enforced(platform, cap):
    key = {"instagram": "caption", "facebook": "caption", "tinder": "bio",
           "bumble": "bio", "hinge": "answer", "linkedin": "post"}[platform]
    a = _agent({key: "z" * 800, "hashtags": ["#leadership", "#growth", "#career", "#x", "#y"]})
    r = a.generate(platform, "a scene")
    for v in r.variants:
        assert v.char_count <= cap, f"{platform} variant {v.tone} over cap"


# --- LinkedIn policy (C-11) -------------------------------------------------
def test_linkedin_strips_banned_hashtags_and_tops_up():
    a = _agent({"post": "A reflection on the quarter.", "hashtags": ["#hustle", "#blessed", "#grind"]})
    r = a.generate("linkedin", "person at a conference")
    tags = [t.lower() for t in r.variants[0].hashtags]
    assert "#hustle" not in tags and "#blessed" not in tags and "#grind" not in tags
    assert len(r.variants[0].hashtags) >= 3  # topped up to the professional minimum


def test_linkedin_professional_fallback_tags_used():
    a = _agent({"post": "A reflection.", "hashtags": []})
    r = a.generate("linkedin", "headshot")
    assert len(r.variants[0].hashtags) >= 3
    # fallbacks come from the professional list
    pro = {f"#{t}".lower() for t in get_prompts()["linkedin"]["professional_hashtags"]}
    assert all(t.lower() in pro for t in r.variants[0].hashtags)


# --- prompt construction ----------------------------------------------------
def test_prompt_carries_platform_tone_scene_vibe_constraints():
    gen = FakeGen({"caption": "x", "hashtags": ["#a", "#b", "#c", "#d", "#e"]})
    a = CaptionAgent(generator=gen)
    sess = Session(user_id=1, vibe="summer roadtrip energy",
                   constraints=["no_location_names", "max_chars:150"])
    a.generate("instagram", "a red convertible on a desert highway",
               session=sess, tones=["Witty"])
    system, user = gen.calls[0]
    assert "valid JSON" in system  # SYSTEM is JSON-only
    assert "Platform: instagram" in user
    assert "Tone requested: Witty" in user
    assert "a red convertible on a desert highway" in user  # {blip_desc}
    assert "summer roadtrip energy" in user  # session vibe
    assert "no_location_names" in user  # constraints


def test_system_prompt_is_json_only():
    p = get_prompts()
    assert "ONLY valid JSON" in p["architecture"]["system"]


def test_hashtag_instruction_absent_for_dating():
    gen = FakeGen({"bio": "x"})
    a = CaptionAgent(generator=gen)
    a.generate("tinder", "portrait", tones=["Playful"])
    _, user = gen.calls[0]
    assert "Do not include any hashtags." in user


# --- graceful degradation ----------------------------------------------------
def test_rate_limit_falls_back_locally_without_crashing():
    from curator.caption.groq_text import GroqRateLimitError
    gen = FakeGen({"caption": "x"}, raise_exc=GroqRateLimitError("429"))
    a = CaptionAgent(generator=gen)
    r = a.generate("instagram", "a sunny beach scene")
    assert len(r.variants) == 3
    assert all(v.source == "fallback" for v in r.variants)
    assert all(v.char_count <= 150 for v in r.variants)


def test_no_generator_uses_local_fallback_no_groq_import():
    """Constructing/running with no generator must not import groq (sandbox-safe)."""
    a = CaptionAgent(generator=None)
    r = a.generate("linkedin", "a professional headshot in an office")
    assert r.output_mode == "linkedin_post"
    assert r.variants[0].source == "fallback"


def test_model_id_is_llama_33_70b():
    assert GROQ_TEXT_MODEL == "llama-3.3-70b-versatile"
