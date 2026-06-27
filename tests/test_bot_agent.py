"""BotAgent router + handler tests (Stage S-01, Agent A1).

No live Telegram token: we drive handlers with lightweight namespace objects.

Review gate:  router tested against malformed input (None/empty/unexpected).
Exit gate:    session init -> photo-receipt loop works; platform picker shows
              all 6 options.
"""

from types import SimpleNamespace

from curator.bot import BotAgent
from curator.bot.keyboards import (
    PLATFORM_CALLBACK_PREFIX,
    PLATFORMS_DONE_CALLBACK,
    platform_button_rows,
)
from curator.session import SessionAgent
from curator.session.schema import PLATFORMS, Stage


# --- lightweight fake update builders ---------------------------------------
def make_text_update(user_id: int, text: str):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        message=SimpleNamespace(text=text, photo=None, from_user=SimpleNamespace(id=user_id)),
        callback_query=None,
    )


def make_photo_update(user_id: int, file_id: str, width=800, height=1000):
    size = SimpleNamespace(
        file_id=file_id, file_unique_id=f"u_{file_id}", width=width, height=height
    )
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        message=SimpleNamespace(text=None, photo=[size], from_user=SimpleNamespace(id=user_id)),
        callback_query=None,
    )


def make_callback_update(user_id: int, data: str):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        message=None,
        callback_query=SimpleNamespace(
            data=data, from_user=SimpleNamespace(id=user_id), message=SimpleNamespace()
        ),
    )


# --- platform picker --------------------------------------------------------
def test_platform_picker_shows_all_six_options():
    rows = platform_button_rows()
    callbacks = [b["callback_data"] for row in rows for b in row]
    for key in PLATFORMS:
        assert f"{PLATFORM_CALLBACK_PREFIX}{key}" in callbacks
    assert PLATFORMS_DONE_CALLBACK in callbacks
    assert len(PLATFORMS) == 6


def test_start_presents_six_platform_keyboard():
    agent = BotAgent()
    result = agent.handle_start(make_text_update(1, "/start"))
    assert result.kind == "start"
    assert result.keyboard_rows is not None
    callbacks = [b["callback_data"] for row in result.keyboard_rows for b in row]
    platform_cbs = [c for c in callbacks if c.startswith(PLATFORM_CALLBACK_PREFIX)]
    assert len(platform_cbs) == 6


# --- multi-select flow ------------------------------------------------------
def test_platform_multiselect_toggle_and_done():
    agent = BotAgent()
    agent.handle_start(make_text_update(1, "/start"))
    agent.route(make_callback_update(1, f"{PLATFORM_CALLBACK_PREFIX}instagram"))
    agent.route(make_callback_update(1, f"{PLATFORM_CALLBACK_PREFIX}tinder"))
    session = agent.sessions.get(1)
    assert session.platforms == ["instagram", "tinder"]

    # toggle instagram off
    agent.route(make_callback_update(1, f"{PLATFORM_CALLBACK_PREFIX}instagram"))
    assert agent.sessions.get(1).platforms == ["tinder"]

    done = agent.route(make_callback_update(1, PLATFORMS_DONE_CALLBACK))
    assert done.kind == "platforms_done"
    # Done now shows the caption-length selector; a length pick advances the stage.
    assert done.keyboard_rows  # length buttons shown
    length = agent.route(make_callback_update(1, "length:haiku"))
    assert length.kind == "length_set"
    assert agent.sessions.get(1).length_mode == "haiku"
    assert agent.sessions.get(1).stage == Stage.AWAITING_INTENT


def test_done_with_no_selection_is_rejected():
    agent = BotAgent()
    agent.handle_start(make_text_update(1, "/start"))
    res = agent.route(make_callback_update(1, PLATFORMS_DONE_CALLBACK))
    assert res.kind == "platforms_done_empty"


# --- exit gate: init -> photo-receipt loop ----------------------------------
def test_session_init_to_photo_receipt_loop():
    agent = BotAgent()
    agent.route(make_text_update(7, "/start"))  # init
    r1 = agent.route(make_photo_update(7, "a"))
    r2 = agent.route(make_photo_update(7, "b"))
    r3 = agent.route(make_photo_update(7, "c"))
    assert r1.kind == r2.kind == r3.kind == "photo_received"
    assert "1/" in r1.text and "2/" in r2.text and "3/" in r3.text
    session = agent.sessions.get(7)
    assert [p.file_id for p in session.photos] == ["a", "b", "c"]
    assert session.stage == Stage.RECEIVING_PHOTOS


def test_photo_stores_reference_not_bytes_and_picks_largest():
    agent = BotAgent()
    agent.route(make_text_update(1, "/start"))
    # photo list with ascending sizes; largest (last) should be stored
    small = SimpleNamespace(file_id="small", file_unique_id="us", width=100, height=100)
    large = SimpleNamespace(file_id="large", file_unique_id="ul", width=900, height=1200)
    upd = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        message=SimpleNamespace(text=None, photo=[small, large], from_user=SimpleNamespace(id=1)),
        callback_query=None,
    )
    agent.route(upd)
    photo = agent.sessions.get(1).photos[-1]
    assert photo.file_id == "large" and photo.width == 900


# --- text/intent routing ----------------------------------------------------
def test_text_intent_merges_into_session():
    agent = BotAgent()
    agent.route(make_text_update(1, "/start"))
    res = agent.route(make_text_update(1, "instagram, moody sunset, under 80 chars"))
    assert res.kind == "text_intent"
    s = agent.sessions.get(1)
    assert "instagram" in s.platforms
    assert s.vibe and "sunset" in s.vibe.lower()
    assert "max_chars:80" in s.constraints


# --- malformed / robustness (Review gate) -----------------------------------
def test_router_handles_none_update():
    agent = BotAgent()
    res = agent.route(None)
    assert res.kind == "ignored_empty"


def test_router_handles_empty_namespace():
    agent = BotAgent()
    res = agent.route(SimpleNamespace())
    assert res.kind in {"ignored_unknown", "ignored_empty"}


def test_router_handles_unexpected_update_type():
    agent = BotAgent()
    for junk in (123, "a string", ["list"], {"k": "v"}, 3.14):
        res = agent.route(junk)  # must not raise
        assert res is not None


def test_router_handles_message_without_text_or_photo():
    agent = BotAgent()
    upd = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        message=SimpleNamespace(text=None, photo=None, from_user=SimpleNamespace(id=1)),
        callback_query=None,
    )
    res = agent.route(upd)
    assert res.kind == "unsupported_message"


def test_router_handles_callback_with_none_data():
    agent = BotAgent()
    upd = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        message=None,
        callback_query=SimpleNamespace(data=None, from_user=SimpleNamespace(id=1)),
    )
    res = agent.route(upd)  # data is None -> falls through, no crash
    assert res is not None


def test_router_does_not_crash_on_exploding_update():
    """An update whose attribute access raises must be contained by the router."""

    class Exploding:
        @property
        def callback_query(self):
            raise RuntimeError("boom")

        @property
        def message(self):
            raise RuntimeError("boom")

    agent = BotAgent()
    res = agent.route(Exploding())
    assert res.kind == "error"


def test_photo_without_user_prompts_start():
    agent = BotAgent()
    upd = SimpleNamespace(
        effective_user=None,
        message=SimpleNamespace(
            text=None,
            photo=[SimpleNamespace(file_id="x", file_unique_id="u", width=1, height=1)],
            from_user=None,
        ),
        callback_query=None,
    )
    res = agent.route(upd)
    assert res.kind == "no_user"


def test_shared_session_agent_persists_across_handlers():
    """BotAgent over an explicit SessionAgent threads state across updates."""
    sessions = SessionAgent()
    agent = BotAgent(sessions=sessions)
    agent.route(make_text_update(3, "/start"))
    agent.route(make_photo_update(3, "p1"))
    assert len(sessions.get(3).photos) == 1
