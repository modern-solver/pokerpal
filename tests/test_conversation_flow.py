"""End-to-end conversational flow (the orchestration spine).

Drives the whole conversation offline (no Groq key -> local-fallback captions,
no Tier-3 vision) to prove there are no dead ends:
  start -> pick platform -> vibe -> photos -> done -> ranked cards
        -> pick a photo -> pick a caption -> post-ready final card.
"""

import io
from types import SimpleNamespace

import pytest

from curator.bot.agent import BotAgent, Stage


def _img_bytes(color=(180, 120, 60)) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (800, 1000), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _text(uid, text):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=uid),
        message=SimpleNamespace(text=text, photo=None, from_user=SimpleNamespace(id=uid)),
        callback_query=None,
    )


def _photo(uid, file_id):
    size = SimpleNamespace(file_id=file_id, file_unique_id="u" + file_id, width=800, height=1000)
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=uid),
        message=SimpleNamespace(text=None, photo=[size], from_user=SimpleNamespace(id=uid)),
        callback_query=None,
    )


@pytest.fixture(autouse=True)
def _no_groq(monkeypatch):
    # Force the offline path: local-fallback captions, no Tier-3 vision.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_done_signal_triggers_grading_when_photos_present():
    agent = BotAgent()
    agent.route(_text(1, "/start"))
    agent.sessions.update(1, platforms=["instagram"])
    agent.route(_photo(1, "a"))
    res = agent.route(_text(1, "done"))
    assert res.kind == "trigger_grading"


def test_done_with_no_photos_is_guided_not_silent():
    agent = BotAgent()
    agent.route(_text(1, "/start"))
    res = agent.route(_text(1, "done"))
    assert res.kind == "done_no_photos"
    assert res.text  # never silent


def test_vibe_text_nudges_user_to_send_photos():
    agent = BotAgent()
    agent.route(_text(1, "/start"))
    res = agent.route(_text(1, "moody adventurous sunset"))
    assert res.kind == "text_intent"
    assert "photo" in res.text.lower()  # proactively prompts the next step


def test_full_flow_to_post_ready_card():
    agent = BotAgent()
    agent.route(_text(2, "/start"))
    agent.sessions.update(2, platforms=["instagram"])
    agent.route(_text(2, "adventurous"))
    agent.route(_photo(2, "a"))
    agent.route(_photo(2, "b"))

    # The PTB layer would download bytes; here we hand them in directly.
    images = {"a": _img_bytes((200, 130, 70)), "b": _img_bytes((80, 90, 140))}
    cards = agent.grade_and_present(2, images)
    assert cards.kind == "ranking_cards"
    assert "best shots" in cards.text.lower()
    assert agent.sessions.get(2).stage == Stage.RANKING

    # A bare number now routes to selection (not intent parsing).
    pick = agent.route(_text(2, "1"))
    assert pick.kind == "select_photo_ok"

    final = agent.route(_text(2, "1"))
    assert final.kind == "final_card"
    assert "post-ready" in final.text.lower()
    assert agent.sessions.get(2).stage == Stage.DONE


def test_grade_and_present_handles_unreadable_photos_gracefully():
    agent = BotAgent()
    agent.route(_text(3, "/start"))
    agent.sessions.update(3, platforms=["instagram"])
    agent.route(_photo(3, "a"))
    # No bytes provided for file_id "a" -> graceful message, not a crash.
    res = agent.grade_and_present(3, {})
    assert res.kind == "grade_no_photos"
    assert res.text


def test_done_command_handler():
    agent = BotAgent()
    agent.route(_text(4, "/start"))
    agent.sessions.update(4, platforms=["tinder"])
    agent.route(_photo(4, "x"))
    res = agent.handle_done(_text(4, "/done"))
    assert res.kind == "trigger_grading"
