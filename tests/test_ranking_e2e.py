"""End-to-end session test for the ranking + output flow (Agent A5 / S-05).

photos -> graded (mocked composite via grading agent with a stubbed vision) ->
captioned (mocked generator) -> ranked -> card shown via BotAgent -> selection
(photo pick + variant pick) -> final post-ready card returned.

Proves the PRD exit gate: full session runs end-to-end; user sees ranked cards
for the target platform; selection confirmed and card returned. No keys/network.
"""

from types import SimpleNamespace

from curator.bot import BotAgent
from curator.caption.agent import CaptionAgent
from curator.grading.agent import GradingAgent
from curator.grading.tier1 import PhotoMetadata
from curator.grading.tier3 import Tier3Grader
from curator.ranking import RankingAgent
from curator.session import SessionAgent
from curator.session.schema import Stage

IMG = b"\xff\xd8img"


class StubVision:
    """Deterministic Tier-3 vision stub returning varied scores per call."""

    def __init__(self):
        self.calls = 0

    def grade(self, image_bytes):
        self.calls += 1
        # Alternate strong/weak so ranking has a clear order.
        score = 9 if self.calls % 2 == 1 else 6
        return f'{{"composition":{score},"lighting":{score},"subject":{score},"mood":{score}}}'


def meta(**kw):
    return PhotoMetadata.from_dict(kw)


def make_photo_update(user_id, file_id):
    size = SimpleNamespace(file_id=file_id, file_unique_id=f"u_{file_id}",
                           width=1080, height=1350)
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        message=SimpleNamespace(text=None, photo=[size],
                                from_user=SimpleNamespace(id=user_id)),
        callback_query=None,
    )


def test_full_session_end_to_end_instagram():
    user_id = 4242
    platform = "instagram"
    sessions = SessionAgent()
    bot = BotAgent(sessions=sessions)

    # 1. Start + intake photos through the bot spine.
    bot.handle_start(SimpleNamespace(effective_user=SimpleNamespace(id=user_id),
                                     message=None, callback_query=None))
    file_ids = ["photoA", "photoB", "photoC"]
    for fid in file_ids:
        bot.handle_photo(make_photo_update(user_id, fid))
    session = sessions.get(user_id)
    assert len(session.photos) == 3

    # 2. Grade (mocked Tier-3) + composite.
    vision = StubVision()
    grader = GradingAgent(tier3_grader=Tier3Grader(vision=vision))
    batch = [
        meta(width=1080, height=1350, face_count=1, saturation_mean=0.6,
             largest_face_frac=0.25, phash="0000000000000001"),
        meta(width=1080, height=1350, face_count=1, saturation_mean=0.6,
             largest_face_frac=0.25, phash="0000000000000002"),
        meta(width=1080, height=1350, face_count=1, saturation_mean=0.6,
             largest_face_frac=0.25, phash="0000000000000004"),
    ]
    results = grader.grade_batch(batch, [platform])
    # Tie file_ids back onto the results (mirrors the real download/grade join).
    for r, fid in zip(results, file_ids):
        r.file_id = fid
    grader.grade_tier3(results, [IMG] * 3, [platform])
    assert all(platform in r.composite for r in results)

    # 3. Caption (mocked generator -> local fallback deterministic mode).
    cap_agent = CaptionAgent(generator=None)
    captions = [
        cap_agent.generate(platform, r.scene_description or "a sunny portrait",
                           photo_index=r.index)
        for r in results
    ]
    assert all(len(c.variants) == 3 for c in captions)

    # 4. Rank + present cards through the bot.
    cards = RankingAgent().build_cards(platform, results, captions)
    assert len(cards) == 3
    presented = bot.present_ranked_cards(user_id, cards)
    assert presented.kind == "ranking_cards"
    assert "top 3" in presented.text.lower()
    assert sessions.get(user_id).stage == Stage.RANKING
    # Ranked by composite desc: the strong-scored photos rank above weak ones.
    scores = [c.photo.score for c in cards]
    assert scores == sorted(scores, reverse=True)

    # 5. Selection: photo pick then variant pick.
    photo_pick = bot.handle_photo_selection(user_id, "1")
    assert photo_pick.kind == "select_photo_ok"
    final = bot.handle_variant_selection(user_id, "2")
    assert final.kind == "final_card"
    assert "Post-ready for instagram" in final.text

    # 6. Selection logged; session marked DONE.
    assert sessions.get(user_id).stage == Stage.DONE
    # Pending state cleared (loop closed).
    assert user_id not in bot._pending_cards


def test_bot_selection_handles_invalid_input_gracefully():
    user_id = 99
    bot = BotAgent()
    bot.handle_start(SimpleNamespace(effective_user=SimpleNamespace(id=user_id),
                                     message=None, callback_query=None))
    # Pick before any cards -> graceful.
    r = bot.handle_photo_selection(user_id, "1")
    assert r.kind == "select_no_cards"

    # Present a single card then send garbage / out-of-range picks.
    results = [
        __import__("curator.grading.result", fromlist=["GradingResult"]).GradingResult(
            index=0, file_id="f0")
    ]
    results[0].composite = {"instagram": 7.0}
    cards = RankingAgent().build_cards("instagram", results, [])
    bot.present_ranked_cards(user_id, cards)

    bad = bot.handle_photo_selection(user_id, "banana")
    assert bad.kind == "select_photo_invalid"
    oor = bot.handle_photo_selection(user_id, "99")
    assert oor.kind == "select_photo_invalid"
    # Variant pick before a photo pick -> graceful.
    no_photo = bot.handle_variant_selection(user_id, "1")
    assert no_photo.kind == "select_no_photo"
