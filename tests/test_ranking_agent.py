"""RankingAgent + rationale tests (Agent A5 / Stage S-05).

Covers:
  * ranking off GradingResult.composite (the A3 source of truth), deterministic
    top-N with defined, stable tie-breaks;
  * G-08: a no-face dating photo (composite 2.0) ranks last and its rationale
    names the face warning;
  * rationale actually reflects the score breakdown (the driving axis / fit /
    flags), not a generic line;
  * caption variants joined by photo_index onto each ranked card.

All local — no keys, no network.
"""

import pytest

from curator.caption.result import CaptionResult, CaptionVariant
from curator.grading.composite import G08_NO_FACE_COMPOSITE
from curator.grading.result import GradingResult
from curator.grading.tier3 import Tier3Score
from curator.ranking import RankingAgent
from curator.ranking.rationale import build_breakdown, rationale_line


def make_result(index, file_id, composite, *, tier3=None, flags=None,
                platform_fit=None, capped=None, warnings=None):
    r = GradingResult(index=index, file_id=file_id)
    r.composite = dict(composite)
    r.platform_fit = dict(platform_fit or {})
    r.capped_scores = dict(capped or {})
    r.tier3 = tier3
    r.flags = list(flags or [])
    r.warnings = list(warnings or [])
    return r


def make_caption(platform, photo_index, mode="post_caption"):
    variants = [
        CaptionVariant(tone="Witty", output_mode=mode, text=f"witty {photo_index}",
                       char_count=10, payload={"caption": f"witty {photo_index}",
                                               "hashtags": ["#a"], "char_count": 10}),
        CaptionVariant(tone="Heartfelt", output_mode=mode, text=f"heart {photo_index}",
                       char_count=10, payload={"caption": f"heart {photo_index}",
                                               "hashtags": [], "char_count": 10}),
        CaptionVariant(tone="Minimal", output_mode=mode, text=f"min {photo_index}",
                       char_count=8, payload={"caption": f"min {photo_index}",
                                              "hashtags": [], "char_count": 8}),
    ]
    return CaptionResult(platform=platform, output_mode=mode, max_length=150,
                         variants=variants, photo_index=photo_index)


# === ranking off composite ==================================================
def test_rank_orders_by_composite_descending():
    results = [
        make_result(0, "a", {"instagram": 6.0}, tier3=Tier3Score(source="groq")),
        make_result(1, "b", {"instagram": 9.0}, tier3=Tier3Score(source="groq")),
        make_result(2, "c", {"instagram": 7.5}, tier3=Tier3Score(source="groq")),
    ]
    ranked = RankingAgent().rank("instagram", results)
    assert [r.file_id for r in ranked] == ["b", "c", "a"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert ranked[0].score == 9.0


def test_rank_top_n_caps_results():
    results = [make_result(i, f"f{i}", {"instagram": float(i)},
                           tier3=Tier3Score(source="groq")) for i in range(5)]
    ranked = RankingAgent(top_n=3).rank("instagram", results)
    assert len(ranked) == 3
    assert ranked[0].score == 4.0  # highest first


def test_rank_tie_break_is_deterministic_and_stable():
    # Equal composite: graded-before-fit-only, then ascending photo_index.
    results = [
        make_result(2, "fit_only", {"instagram": 8.0}, tier3=None),
        make_result(0, "graded_a", {"instagram": 8.0}, tier3=Tier3Score(source="groq")),
        make_result(1, "graded_b", {"instagram": 8.0}, tier3=Tier3Score(source="groq")),
    ]
    ranked = RankingAgent().rank("instagram", results)
    # graded ones first (by index 0 then 1), fit-only last.
    assert [r.file_id for r in ranked] == ["graded_a", "graded_b", "fit_only"]
    # Stable across repeated runs.
    ranked2 = RankingAgent().rank("instagram", list(reversed(results)))
    assert [r.file_id for r in ranked2] == ["graded_a", "graded_b", "fit_only"]


# === G-08: no-face dating photo ranks last + capped =========================
def test_g08_no_face_dating_photo_ranks_last_and_capped():
    good = make_result(0, "good", {"tinder": 7.5},
                       tier3=Tier3Score(source="groq"),
                       platform_fit={"tinder": 7.0})
    noface = make_result(1, "noface", {"tinder": G08_NO_FACE_COMPOSITE},
                         flags=["no_face_detected"],
                         platform_fit={"tinder": 3.0},
                         capped={"tinder": 3.0},
                         warnings=["This photo has no visible face — not recommended "
                                   "for dating platforms."])
    ranked = RankingAgent().rank("tinder", [good, noface])
    assert ranked[0].file_id == "good"
    assert ranked[-1].file_id == "noface"
    assert ranked[-1].score == 2.0
    # Rationale names the face warning.
    assert "face" in ranked[-1].rationale.lower()


# === rationale reflects the breakdown =======================================
def test_rationale_reflects_top_axis_from_breakdown():
    # Tier-3 with very high lighting -> lighting should dominate the IG breakdown.
    t3 = Tier3Score(composition=5, lighting=10, subject=5, mood=5, source="groq")
    res = make_result(0, "x", {"instagram": 7.0}, tier3=t3,
                      platform_fit={"instagram": 4.0})
    bd = build_breakdown(res, "instagram")
    # The breakdown's contributions are real weight*value terms.
    assert set(bd["contributions"]) == {"composition", "lighting", "technical",
                                        "platform_fit"}
    # Lighting is the strongest axis given the high lighting score.
    assert bd["top_axis"] == "lighting"
    line = rationale_line(bd)
    assert "lighting" in line.lower()
    assert "7.0/10" in line


def test_rationale_fit_only_when_no_tier3():
    res = make_result(0, "x", {"instagram": 5.5}, tier3=None,
                      platform_fit={"instagram": 5.5})
    bd = build_breakdown(res, "instagram")
    assert bd["has_tier3"] is False
    assert bd["top_axis"] == "platform_fit"
    line = rationale_line(bd)
    assert "skipped" in line.lower()
    assert "platform fit" in line.lower()


def test_rationale_surfaces_linkedin_inappropriate_flag():
    t3 = Tier3Score(composition=8, lighting=8, subject=8, mood=8, source="groq")
    res = make_result(0, "x", {"linkedin": 6.0}, tier3=t3,
                      platform_fit={"linkedin": 5.0},
                      flags=["linkedin_inappropriate"])
    line = rationale_line(build_breakdown(res, "linkedin"))
    assert "linkedin" in line.lower() or "casual" in line.lower()


def test_rationale_matches_actual_composite_top_axis_via_agent():
    # End-to-end through the agent: rationale's named axis == argmax contribution.
    t3 = Tier3Score(composition=9, lighting=4, subject=4, mood=4, source="groq")
    res = make_result(0, "x", {"instagram": 6.0}, tier3=t3,
                      platform_fit={"instagram": 3.0})
    ranked = RankingAgent().rank("instagram", [res])[0]
    bd = ranked.breakdown
    top = max(bd["contributions"], key=bd["contributions"].get)
    label = {"composition": "composition", "lighting": "lighting",
             "technical": "subject sharpness", "platform_fit": "platform fit"}[top]
    assert label in ranked.rationale.lower()


# === caption join ===========================================================
def test_cards_carry_joined_caption_variants():
    results = [
        make_result(0, "a", {"instagram": 8.0}, tier3=Tier3Score(source="groq")),
        make_result(1, "b", {"instagram": 9.0}, tier3=Tier3Score(source="groq")),
    ]
    captions = [make_caption("instagram", 0), make_caption("instagram", 1)]
    cards = RankingAgent().build_cards("instagram", results, captions)
    assert len(cards) == 2
    # Top card is photo index 1 (score 9); its variants joined by photo_index.
    assert cards[0].photo.photo_index == 1
    assert len(cards[0].photo.variants) == 3
    rendered = cards[0].rendered_variants
    assert rendered[0].startswith("witty 1")


def test_caption_join_ignores_other_platform_captions():
    results = [make_result(0, "a", {"instagram": 8.0},
                           tier3=Tier3Score(source="groq"))]
    captions = [make_caption("tinder", 0, mode="dating_bio")]  # wrong platform
    cards = RankingAgent().build_cards("instagram", results, captions)
    assert cards[0].photo.variants == []
