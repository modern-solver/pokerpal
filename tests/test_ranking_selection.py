"""Selection-handler tests (Agent A5 / Stage S-05).

Covers BOTH picks (photo pick + variant pick) and graceful handling of invalid
input (out-of-range / garbage / empty), per the review gate. Plus the render of
each output mode's variant and the post-ready FinalCard.
"""

import pytest

from curator.caption.result import CaptionVariant
from curator.ranking.card import OutputCard, RankedPhoto, render_variant
from curator.ranking.selection import (
    SelectionError,
    build_final_card,
    cycle_variant,
    parse_choice,
    select,
    select_photo,
    select_variant,
)


def variant(tone, mode, payload):
    return CaptionVariant(tone=tone, output_mode=mode,
                          text=str(payload.get("caption") or payload.get("bio")
                                   or payload.get("answer") or payload.get("post")),
                          char_count=10, payload=payload)


def make_card(card_index, photo_index, n_variants=3):
    variants = [
        variant(f"t{i}", "post_caption",
                {"caption": f"cap{photo_index}_{i}", "hashtags": [], "char_count": 5})
        for i in range(n_variants)
    ]
    photo = RankedPhoto(rank=card_index, photo_index=photo_index, platform="instagram",
                        score=8.0, rationale="strong", file_id=f"file{photo_index}",
                        variants=variants)
    return OutputCard(card_index=card_index, photo=photo)


CARDS = [make_card(1, 5), make_card(2, 3), make_card(3, 9)]


# === parse_choice ===========================================================
@pytest.mark.parametrize("raw,expected", [
    (2, 2), ("2", 2), ("photo 2", 2), ("#3", 3), (" 1 ", 1),
    (None, None), ("", None), ("abc", None), ([], None), (True, None),
])
def test_parse_choice(raw, expected):
    assert parse_choice(raw) == expected


# === photo pick =============================================================
def test_select_photo_valid():
    chosen = select_photo(CARDS, "2")
    assert isinstance(chosen, OutputCard)
    assert chosen.card_index == 2
    assert chosen.photo.photo_index == 3


def test_select_photo_out_of_range():
    err = select_photo(CARDS, "9")
    assert isinstance(err, SelectionError)
    assert err.reason == "out_of_range"
    assert not err.ok


def test_select_photo_garbage():
    err = select_photo(CARDS, "banana")
    assert isinstance(err, SelectionError)
    assert err.reason == "unparseable"


def test_select_photo_no_cards():
    err = select_photo([], "1")
    assert isinstance(err, SelectionError)
    assert err.reason == "no_cards"


# === variant pick ===========================================================
def test_select_variant_valid():
    photo = CARDS[0].photo
    idx = select_variant(photo, "2")
    assert idx == 1  # 0-based


def test_select_variant_out_of_range():
    err = select_variant(CARDS[0].photo, "5")
    assert isinstance(err, SelectionError)
    assert err.reason == "out_of_range"


def test_select_variant_garbage():
    err = select_variant(CARDS[0].photo, "???")
    assert isinstance(err, SelectionError)
    assert err.reason == "unparseable"


def test_select_variant_no_variants():
    photo = RankedPhoto(rank=1, photo_index=0, platform="instagram", score=5.0,
                        rationale="x", variants=[])
    err = select_variant(photo, "1")
    assert isinstance(err, SelectionError)
    assert err.reason == "no_variants"


# === one-shot select -> FinalCard ===========================================
def test_select_builds_final_card():
    final = select(CARDS, "2", "3")
    assert final.platform == "instagram"
    assert final.photo_index == 3
    assert final.variant_index == 3
    assert final.file_id == "file3"
    assert final.rendered.startswith("cap3_2")


def test_select_propagates_photo_error():
    err = select(CARDS, "99", "1")
    assert isinstance(err, SelectionError)
    assert err.reason == "out_of_range"


def test_select_propagates_variant_error():
    err = select(CARDS, "1", "nope")
    assert isinstance(err, SelectionError)
    assert err.reason == "unparseable"


# === render_variant per output mode =========================================
def test_render_post_caption_with_hashtags():
    v = variant("w", "post_caption", {"caption": "sunset vibes",
                                      "hashtags": ["#sun", "#sea"], "char_count": 12})
    assert render_variant(v) == "sunset vibes\n#sun #sea"


def test_render_dating_bio():
    v = variant("b", "dating_bio", {"bio": "ramen enthusiast", "char_count": 15})
    assert render_variant(v) == "ramen enthusiast"


def test_render_prompt_answer():
    v = variant("p", "prompt_answer", {"prompt": "The way to my heart is...",
                                       "answer": "good lighting", "char_count": 13})
    rendered = render_variant(v)
    assert "The way to my heart is..." in rendered
    assert "good lighting" in rendered


def test_render_linkedin_post():
    v = variant("pro", "linkedin_post", {"post": "Reflecting on Q2",
                                         "hashtags": ["#growth"], "char_count": 16})
    assert render_variant(v) == "Reflecting on Q2\n#growth"


# === A6 hook: cycle_variant =================================================
def test_cycle_variant_wraps():
    photo = CARDS[0].photo  # 3 variants
    assert cycle_variant(photo, 0) == 1
    assert cycle_variant(photo, 2) == 0  # wraps
