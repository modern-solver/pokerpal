"""Override-command + timeout-warning tests (Agent A6 / Stage S-06).

Covers each override command (happy path + invalid-state graceful handling) and
the session timeout warning with an INJECTED clock (deterministic; no sleeps).

Proves the review-gate items:
  * /retry cycles tones IN ORDER (Playful -> Confident -> Mysterious), not random.
  * /shorter enforces a tighter cap.
  * timeout warning fires inside T-15min and is silent outside it.
"""

from types import SimpleNamespace

from curator.bot import BotAgent
from curator.caption.result import CaptionVariant
from curator.grading.result import GradingResult
from curator.ranking import RankingAgent
from curator.ranking.card import OutputCard, RankedPhoto
from curator.session import SessionAgent
from curator.session.schema import Stage
from curator.session.timeout import check_timeout


# --- helpers ----------------------------------------------------------------
def cmd(user_id, text):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        message=SimpleNamespace(text=text, photo=None,
                                from_user=SimpleNamespace(id=user_id)),
        callback_query=None,
    )


def graded(index, file_id, scores):
    """A GradingResult with per-platform composites set."""
    r = GradingResult(index=index, file_id=file_id)
    r.composite = dict(scores)
    return r


def make_variant(tone, mode, text):
    payload = {"char_count": len(text)}
    if mode == "dating_bio":
        payload["bio"] = text
    elif mode == "post_caption":
        payload["caption"] = text
        payload["hashtags"] = []
    v = CaptionVariant(label=tone, length_mode=mode, text=text,
                       char_count=len(text), payload=payload)
    return v


def make_card(card_index, photo_index, platform, tones, mode="post_caption",
              score=8.0):
    variants = [make_variant(t, mode, f"{t} caption text {photo_index}")
                for t in tones]
    photo = RankedPhoto(rank=card_index, photo_index=photo_index, platform=platform,
                        score=score, rationale="strong", file_id=f"f{photo_index}",
                        variants=variants, breakdown={"scene_description": "a portrait"})
    return OutputCard(card_index=card_index, photo=photo)


def present(bot, user_id, cards, results=None, captions=None, platform=None):
    return bot.present_ranked_cards(user_id, cards, results=results,
                                    captions=captions, platform=platform)


# === /next ==================================================================
def test_next_advances_through_cards():
    bot = BotAgent()
    uid = 1
    cards = [make_card(1, 0, "instagram", ["Witty", "Heartfelt", "Minimal"]),
             make_card(2, 1, "instagram", ["Witty", "Heartfelt", "Minimal"]),
             make_card(3, 2, "instagram", ["Witty", "Heartfelt", "Minimal"])]
    present(bot, uid, cards)
    r1 = bot.handle_next(cmd(uid, "/next"))
    assert r1.kind == "next_card"
    assert "rank #2" in r1.text  # second card
    r2 = bot.handle_next(cmd(uid, "/next"))
    assert r2.kind == "next_card"
    assert "rank #3" in r2.text
    r3 = bot.handle_next(cmd(uid, "/next"))
    assert r3.kind == "next_exhausted"


def test_next_before_any_cards_is_graceful():
    bot = BotAgent()
    r = bot.handle_next(cmd(5, "/next"))
    assert r.kind == "next_no_cards"


def test_next_routes_via_router():
    bot = BotAgent()
    uid = 7
    cards = [make_card(1, 0, "instagram", ["Witty", "Heartfelt", "Minimal"]),
             make_card(2, 1, "instagram", ["Witty", "Heartfelt", "Minimal"])]
    present(bot, uid, cards)
    r = bot.route(cmd(uid, "/next"))
    assert r.kind == "next_card"


# === /retry — cycles tone IN ORDER (C-07) ===================================
def test_retry_cycles_dating_tones_in_order():
    bot = BotAgent()
    uid = 2
    # Tinder tone order per the PRD: Playful -> Confident -> Mysterious.
    tones = ["Playful", "Confident", "Mysterious"]
    card = make_card(1, 0, "tinder", tones, mode="dating_bio")
    present(bot, uid, [card])
    # The user has the default (Playful) shown; /retry steps in order.
    seen = []
    for _ in range(4):  # 3 tones then wrap back to the first
        r = bot.handle_retry(cmd(uid, "/retry"))
        assert r.kind == "retry_cycled"
        tone = r.text.split("]")[0].lstrip("[")
        seen.append(tone)
    # Proves IN-ORDER cycling (not random) with wrap-around.
    assert seen == ["Confident", "Mysterious", "Playful", "Confident"]


def test_retry_cycles_options_in_order_not_random():
    """The cycle must step through the caption options IN ORDER (not random)."""
    bot = BotAgent()
    uid = 21
    labels = ["Option 1", "Option 2", "Option 3"]
    card = make_card(1, 0, "instagram", labels, mode="short")
    present(bot, uid, [card])
    walked = []
    for _ in range(len(labels)):
        r = bot.handle_retry(cmd(uid, "/retry"))
        walked.append(r.text.split("]")[0].lstrip("["))
    rotated = labels[1:] + labels[:1]  # default (1) already shown -> starts at 2
    assert walked == rotated


def test_retry_before_any_card_is_graceful():
    bot = BotAgent()
    r = bot.handle_retry(cmd(9, "/retry"))
    assert r.kind == "retry_no_card"


# === /shorter — enforces a tighter cap ======================================
def test_shorter_enforces_tighter_cap():
    bot = BotAgent()
    uid = 3
    long_text = "x" * 120  # the current caption is 120 chars
    card = make_card(1, 0, "instagram", ["Witty"], mode="post_caption")
    card.photo.variants[0].text = long_text
    card.photo.variants[0].char_count = len(long_text)
    card.photo.variants[0].payload["caption"] = long_text
    present(bot, uid, [card])
    r = bot.handle_shorter(cmd(uid, "/shorter"))
    assert r.kind == "shorter_ok"
    # The replaced variant must be strictly shorter than the original.
    new_len = card.photo.variants[0].char_count
    assert new_len < 120
    # ... and within the tighter cap the bot advertised.
    cap = bot._tighter_cap(120)
    assert new_len <= cap


def test_shorter_regenerates_via_caption_agent():
    """/shorter calls the (injected) CaptionAgent and replaces with the result."""
    from curator.caption.agent import CaptionAgent

    class ShortGen:
        # A stub Groq text client that returns short captions JSON.
        def generate(self, system, user):
            return '{"captions": ["brief", "brief two", "brief three"]}'

    bot = BotAgent(caption_agent=CaptionAgent(generator=ShortGen()))
    uid = 31
    card = make_card(1, 0, "instagram", ["Witty"], mode="post_caption")
    card.photo.variants[0].text = "y" * 140
    card.photo.variants[0].char_count = 140
    card.photo.variants[0].payload["caption"] = "y" * 140
    present(bot, uid, [card])
    r = bot.handle_shorter(cmd(uid, "/shorter"))
    assert r.kind == "shorter_ok"
    assert card.photo.variants[0].char_count < 140
    assert "brief" in card.photo.variants[0].text


def test_shorter_before_any_card_is_graceful():
    bot = BotAgent()
    r = bot.handle_shorter(cmd(11, "/shorter"))
    assert r.kind == "shorter_no_card"


# === /platform — re-rank for a new platform =================================
def test_platform_switch_reranks():
    bot = BotAgent()
    uid = 4
    # Two graded photos; photo 0 is best for instagram, photo 1 best for tinder.
    results = [
        graded(0, "f0", {"instagram": 9.0, "tinder": 4.0}),
        graded(1, "f1", {"instagram": 5.0, "tinder": 8.0}),
    ]
    cards = RankingAgent().build_cards("instagram", results, [])
    present(bot, uid, cards, results=results, captions=[], platform="instagram")
    assert cards[0].photo.photo_index == 0  # photo 0 leads on instagram

    r = bot.handle_platform(cmd(uid, "/platform tinder"))
    assert r.kind == "platform_switched"
    assert "Tinder" in r.text
    # Active platform updated + re-ranked: photo 1 now leads.
    new_cards = bot._pending_cards[uid]
    assert new_cards[0].photo.platform == "tinder"
    assert new_cards[0].photo.photo_index == 1


def test_platform_requires_arg():
    bot = BotAgent()
    uid = 14
    results = [graded(0, "f0", {"instagram": 9.0})]
    cards = RankingAgent().build_cards("instagram", results, [])
    present(bot, uid, cards, results=results, captions=[], platform="instagram")
    r = bot.handle_platform(cmd(uid, "/platform"))
    assert r.kind == "platform_need_arg"


def test_platform_unknown_is_graceful():
    bot = BotAgent()
    uid = 15
    results = [graded(0, "f0", {"instagram": 9.0})]
    cards = RankingAgent().build_cards("instagram", results, [])
    present(bot, uid, cards, results=results, captions=[], platform="instagram")
    r = bot.handle_platform(cmd(uid, "/platform myspace"))
    assert r.kind == "platform_unknown"


def test_platform_before_results_is_graceful():
    bot = BotAgent()
    r = bot.handle_platform(cmd(16, "/platform tinder"))
    assert r.kind == "platform_no_results"


# === /reset =================================================================
def test_reset_clears_all_working_state():
    bot = BotAgent()
    uid = 6
    cards = [make_card(1, 0, "instagram", ["Witty", "Heartfelt", "Minimal"])]
    present(bot, uid, cards, results=[graded(0, "f0", {"instagram": 8.0})],
            captions=[], platform="instagram")
    bot.handle_photo_selection(uid, "1")  # focus a photo
    assert uid in bot._pending_cards
    r = bot.handle_reset(cmd(uid, "/reset"))
    assert r.kind == "reset"
    assert uid not in bot._pending_cards
    assert uid not in bot._pending_photo
    assert uid not in bot._pending_results
    # A fresh session exists, ready to start over.
    assert bot.sessions.get(uid).stage == Stage.SELECTING_PLATFORMS


def test_reset_routes_via_router():
    bot = BotAgent()
    r = bot.route(cmd(8, "/reset"))
    assert r.kind == "reset"


# === unknown command falls through gracefully ===============================
def test_unknown_command_does_not_crash():
    bot = BotAgent()
    bot.route(cmd(20, "/start"))
    r = bot.route(cmd(20, "/wat"))  # unknown -> intent parse, no crash
    assert r is not None


# === /help ==================================================================
def test_help_lists_override_commands():
    bot = BotAgent()
    r = bot.handle_help(cmd(1, "/help"))
    for c in ("/next", "/retry", "/shorter", "/platform", "/reset"):
        assert c in r.text


# === timeout warning — deterministic injected clock =========================
def test_timeout_check_inside_window_warns():
    sessions = SessionAgent(default_ttl_seconds=1800)
    s = sessions.init(100)
    # updated_at == created now; expires_at = updated_at + 1800.
    # Inside the last 15 min: now = expires_at - 600 (10 min left).
    now = s.updated_at + 1800 - 600
    status = check_timeout(s, now=now)
    assert status.should_warn is True
    assert status.minutes_remaining == 10
    assert status.expired is False


def test_timeout_check_outside_window_silent():
    sessions = SessionAgent(default_ttl_seconds=1800)
    s = sessions.init(101)
    # 20 min left -> outside the 15-min window.
    now = s.updated_at + 1800 - 1200
    status = check_timeout(s, now=now)
    assert status.should_warn is False


def test_timeout_check_expired_is_silent():
    sessions = SessionAgent(default_ttl_seconds=1800)
    s = sessions.init(102)
    now = s.updated_at + 1800 + 5  # past expiry
    status = check_timeout(s, now=now)
    assert status.should_warn is False
    assert status.expired is True


def test_timeout_check_no_session_silent():
    status = check_timeout(None, now=123.0)
    assert status.should_warn is False
    assert status.expired is True


def test_bot_timeout_warning_with_injected_clock_fires_once():
    holder = {"t": 0.0}
    bot = BotAgent(clock=lambda: holder["t"])
    uid = 200
    bot.handle_start(cmd(uid, "/start"))
    s = bot.sessions.get(uid)
    # Drive the clock into the warning window.
    holder["t"] = s.updated_at + 1800 - 300  # 5 min left
    first = bot.timeout_warning(uid)
    assert first is not None
    assert first.kind == "timeout_warning"
    assert "min" in first.text
    # Idempotent within the same window: a second tick is silent.
    second = bot.timeout_warning(uid)
    assert second is None


def test_bot_timeout_warning_silent_when_plenty_of_time():
    holder = {"t": 0.0}
    bot = BotAgent(clock=lambda: holder["t"])
    uid = 201
    bot.handle_start(cmd(uid, "/start"))
    s = bot.sessions.get(uid)
    holder["t"] = s.updated_at + 60  # only 1 min elapsed of 30
    assert bot.timeout_warning(uid) is None


def test_bot_timeout_warning_rearms_after_activity():
    holder = {"t": 0.0}
    bot = BotAgent(clock=lambda: holder["t"])
    uid = 202
    bot.handle_start(cmd(uid, "/start"))
    s = bot.sessions.get(uid)
    holder["t"] = s.updated_at + 1800 - 300
    assert bot.timeout_warning(uid) is not None  # warned
    # Activity bumps updated_at; move clock back outside the window.
    bot.sessions.update(uid, vibe="fresh")
    s2 = bot.sessions.get(uid)
    holder["t"] = s2.updated_at + 60  # plenty of time again
    assert bot.timeout_warning(uid) is None  # re-armed (window cleared)
    # Re-enter the window -> warns again.
    holder["t"] = s2.updated_at + 1800 - 200
    assert bot.timeout_warning(uid) is not None
