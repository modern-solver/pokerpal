"""Help, /about, and capability-Q&A tests (go-live readiness).

Covers:
  * /help returns the full walkthrough + command reference;
  * /about returns capabilities + limits;
  * free-text questions about the bot are answered (kind="faq") instead of being
    parsed as a posting intent, while real intent text still flows to parsing;
  * the canonical command list is valid for Telegram setMyCommands / BotFather.
"""

from types import SimpleNamespace

from curator.bot.agent import BotAgent
from curator.bot.faq import (
    BOT_COMMANDS,
    answer_question,
    command_reference,
    looks_like_question,
)


def _text_update(text: str, user_id: int = 1):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        message=SimpleNamespace(
            text=text, photo=None, from_user=SimpleNamespace(id=user_id)
        ),
    )


# --- /help and /about -------------------------------------------------------
def test_help_returns_full_guide_and_commands():
    res = BotAgent().handle_help(_text_update("/help"))
    assert res.kind == "help"
    assert "How to get the best out of" in res.text
    # every command appears in the reference block
    for name, _desc in BOT_COMMANDS:
        assert f"/{name}" in res.text


def test_about_lists_capabilities_and_limits():
    res = BotAgent().handle_about(_text_update("/about"))
    assert res.kind == "about"
    assert "What I do" in res.text
    assert "limit" in res.text.lower()


def test_help_and_about_route_via_router():
    agent = BotAgent()
    assert agent.route(_text_update("/help")).kind == "help"
    assert agent.route(_text_update("/about")).kind == "about"
    assert agent.route(_text_update("/commands")).kind == "help"  # alias


# --- capability Q&A through free text ---------------------------------------
def test_question_about_platforms_is_answered_not_parsed():
    res = BotAgent().route(_text_update("what platforms do you support?"))
    assert res.kind == "faq"
    assert "Instagram" in res.text and "LinkedIn" in res.text


def test_question_about_grading_is_answered():
    res = BotAgent().route(_text_update("how do you grade my photos?"))
    assert res.kind == "faq"
    assert "tier" in res.text.lower()


def test_question_about_cost_is_answered():
    assert "free" in answer_question("is this free?").lower()


def test_question_about_limits_is_answered():
    res = BotAgent().route(_text_update("what are your limitations?"))
    assert res.kind == "faq"
    assert "video" in res.text.lower() or "face" in res.text.lower()


def test_question_about_privacy_is_answered():
    assert "session" in answer_question("do you store my photos?").lower()


def test_unknown_question_still_gives_overview():
    # No keyword hit, but it's phrased as a question -> overview, not a dead end.
    ans = answer_question("what is the meaning of life?")
    assert "Instagram" in ans and "/help" in ans


# --- intent text must NOT be hijacked by the FAQ ----------------------------
def test_intent_text_still_parses_as_intent():
    # A vibe/constraint line is not a question and must reach intent parsing.
    res = BotAgent().route(_text_update("witty and beachy, no location names"))
    assert res.kind in {"text_intent", "text_noop"}
    assert res.kind != "faq"


def test_looks_like_question_discriminates():
    assert looks_like_question("what can you do?")
    assert looks_like_question("how does grading work")
    assert looks_like_question("can you post for me?")
    assert not looks_like_question("witty, heartfelt, minimal")
    assert not looks_like_question("instagram and tinder")
    assert not looks_like_question("")


# --- command list is valid for Telegram setMyCommands -----------------------
def test_bot_commands_valid_for_telegram():
    assert BOT_COMMANDS, "command list must not be empty"
    seen = set()
    for name, desc in BOT_COMMANDS:
        assert name == name.lower(), f"command {name} must be lowercase"
        assert 1 <= len(name) <= 32
        assert all(c.isalnum() or c == "_" for c in name), name
        assert 1 <= len(desc) <= 256, f"{name} description length"
        assert name not in seen, f"duplicate command {name}"
        seen.add(name)


def test_command_reference_lists_every_command():
    ref = command_reference()
    for name, _desc in BOT_COMMANDS:
        assert f"/{name}" in ref
