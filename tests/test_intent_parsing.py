"""Intent parsing tests (Stage S-01, Agent A1).

Covers platform multi-select, vibe, constraints, and graceful handling of
malformed input.
"""

from curator.session.intent import (
    parse_constraints,
    parse_intent,
    parse_platforms,
)


def test_parse_platforms_multiselect_and_aliases():
    assert parse_platforms("instagram and tinder") == ["instagram", "tinder"]
    assert parse_platforms("IG, FB, LI") == ["instagram", "facebook", "linkedin"]


def test_parse_platforms_dedup_preserves_order():
    assert parse_platforms("hinge hinge bumble") == ["hinge", "bumble"]


def test_parse_platforms_no_false_match_in_words():
    # "li" must not match inside "lighting"; "ig" must not match inside "big".
    assert parse_platforms("great lighting and big mood") == []


def test_parse_constraints_variants():
    assert "no_location_names" in parse_constraints("please no location names")
    assert "no_hashtags" in parse_constraints("without hashtags please")
    assert "no_emoji" in parse_constraints("no emojis at all")
    assert "max_chars:100" in parse_constraints("keep it under 100 chars")
    assert "max_chars:80" in parse_constraints("max 80 characters")


def test_parse_intent_full():
    intent = parse_intent(
        "vibe: chill beach sunset for instagram and hinge, under 100 chars, no location names"
    )
    assert "instagram" in intent.platforms and "hinge" in intent.platforms
    assert "max_chars:100" in intent.constraints
    assert "no_location_names" in intent.constraints
    assert intent.vibe is not None
    assert "beach" in intent.vibe.lower()
    # platform words stripped from the vibe
    assert "instagram" not in intent.vibe.lower()


def test_parse_intent_vibe_only():
    intent = parse_intent("moody noir city nights")
    assert intent.platforms == []
    assert intent.constraints == []
    assert intent.vibe == "moody noir city nights"


def test_parse_intent_malformed_inputs_do_not_raise():
    for bad in (None, "", "   ", 12345, [], {}):
        intent = parse_intent(bad)  # type: ignore[arg-type]
        assert intent.platforms == []
        assert intent.constraints == []
        assert intent.vibe is None
