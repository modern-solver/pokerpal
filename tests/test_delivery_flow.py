"""Conversational DELIVERY-layer integration tests.

These drive the actual PTB handler callbacks registered by register_handlers
(via a fake Application that captures them) with mock Telegram I/O, and assert
the user RECEIVES a non-empty reply at EVERY step of the conversation:

    /start -> tap platform -> tap Done -> vibe -> photo -> "done" (grade) ->
    cards (as photos) -> pick photo -> pick caption -> final card.

This is the layer the route()-level unit tests skip — and where the
"non-responsive after X" faults live. Offline: no Groq key (local fallback).
"""

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("telegram", reason="python-telegram-bot not installed")

from curator.bot.handlers import register_handlers  # noqa: E402


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _img():
    Image = pytest.importorskip("PIL.Image")
    im = Image.new("RGB", (700, 900), (190, 140, 90))
    b = io.BytesIO(); im.save(b, format="JPEG", quality=80); return b.getvalue()


class FakeApp:
    """Captures the handler callbacks register_handlers installs."""

    def __init__(self):
        self.cmd = {}      # command name -> callback
        self.on_callback = None
        self.on_photo = None
        self.on_text = None

    def add_handler(self, h):
        from telegram.ext import CommandHandler, CallbackQueryHandler, MessageHandler
        from telegram.ext import filters as f
        if isinstance(h, CommandHandler):
            for c in h.commands:
                self.cmd[c] = h.callback
        elif isinstance(h, CallbackQueryHandler):
            self.on_callback = h.callback
        elif isinstance(h, MessageHandler):
            # PHOTO filter vs TEXT filter
            if h.filters == f.PHOTO:
                self.on_photo = h.callback
            else:
                self.on_text = h.callback


@pytest.fixture
def bot(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)  # offline: local fallback
    app = FakeApp()
    agent = register_handlers(app)
    return app, agent


def _ctx():
    """A mock PTB context whose bot records every outbound call."""
    bot = SimpleNamespace(
        send_message=AsyncMock(),
        send_photo=AsyncMock(),
        get_file=AsyncMock(
            return_value=SimpleNamespace(
                download_as_bytearray=AsyncMock(return_value=bytearray(_img()))
            )
        ),
    )
    return SimpleNamespace(bot=bot)


def _msg_update(uid, text=None):
    sent = []
    async def reply_text(t, **kw):
        sent.append(t)
    msg = SimpleNamespace(
        text=text, photo=None, from_user=SimpleNamespace(id=uid),
        chat=SimpleNamespace(id=uid), reply_text=reply_text,
    )
    upd = SimpleNamespace(
        effective_user=SimpleNamespace(id=uid),
        effective_chat=SimpleNamespace(id=uid),
        message=msg, callback_query=None,
    )
    return upd, sent


def _photo_update(uid, file_id):
    sent = []
    async def reply_text(t, **kw):
        sent.append(t)
    size = SimpleNamespace(file_id=file_id, file_unique_id="u" + file_id, width=700, height=900)
    msg = SimpleNamespace(
        text=None, photo=[size], from_user=SimpleNamespace(id=uid),
        chat=SimpleNamespace(id=uid), reply_text=reply_text,
    )
    upd = SimpleNamespace(
        effective_user=SimpleNamespace(id=uid),
        effective_chat=SimpleNamespace(id=uid),
        message=msg, callback_query=None,
    )
    return upd, sent


def _cb_update(uid, data):
    edited = []
    async def edit_message_text(t, **kw):
        edited.append(t)
    cq = SimpleNamespace(
        data=data, from_user=SimpleNamespace(id=uid),
        message=SimpleNamespace(chat=SimpleNamespace(id=uid)),
        answer=AsyncMock(), edit_message_text=edit_message_text,
    )
    upd = SimpleNamespace(
        effective_user=SimpleNamespace(id=uid),
        effective_chat=SimpleNamespace(id=uid),
        message=None, callback_query=cq,
    )
    return upd, edited


def test_full_conversation_delivers_at_every_step(bot):
    _run(_full_conversation(bot))


async def _full_conversation(bot):
    app, agent = bot
    uid = 555
    ctx = _ctx()

    # /start
    u, sent = _msg_update(uid, "/start")
    await app.cmd["start"](u, ctx)
    assert sent and "platform" in sent[0].lower(), "start not delivered"

    # tap a platform (callback) -> edits message in place
    u, edited = _cb_update(uid, "platform:instagram")
    await app.on_callback(u, ctx)
    assert edited, "platform toggle not delivered"

    # tap Done (callback, no keyboard) -> THE earlier silent bug
    u, edited = _cb_update(uid, "platforms:done")
    await app.on_callback(u, ctx)
    assert edited and "targeting" in edited[-1].lower(), "Done confirmation not delivered"

    # vibe text -> nudges to photos
    u, sent = _msg_update(uid, "adventurous golden hour")
    await app.on_text(u, ctx)
    assert sent and "photo" in sent[-1].lower(), "vibe nudge not delivered"

    # two photos
    for fid in ("a", "b"):
        u, sent = _photo_update(uid, fid)
        await app.on_photo(u, ctx)
        assert sent and "received" in sent[-1].lower(), "photo ack not delivered"

    # "done" -> grading message + cards delivered as photos
    u, sent = _msg_update(uid, "done")
    await app.on_text(u, ctx)
    assert sent and "grading" in sent[-1].lower(), "grading ack not delivered"
    # cards delivered via context.bot: a header + N photos + a prompt
    assert ctx.bot.send_photo.await_count >= 1, "ranked photos were not sent"
    assert ctx.bot.send_message.await_count >= 2, "card header/prompt not sent"

    # pick a photo number -> selection prompt delivered
    u, sent = _msg_update(uid, "1")
    await app.on_text(u, ctx)
    assert sent and "caption" in sent[-1].lower(), "photo-pick reply not delivered"

    # pick a caption number -> post-ready final card delivered
    u, sent = _msg_update(uid, "1")
    await app.on_text(u, ctx)
    assert sent and "post-ready" in sent[-1].lower(), "final card not delivered"


def test_done_command_path_delivers_cards(bot):
    _run(_done_command_path(bot))


async def _done_command_path(bot):
    app, agent = bot
    uid = 777
    ctx = _ctx()
    await app.cmd["start"](*( _msg_update(uid, "/start")[0], ctx))
    agent.sessions.update(uid, platforms=["tinder"])
    u, _ = _photo_update(uid, "p1"); await app.on_photo(u, ctx)
    u, sent = _msg_update(uid, "/done"); await app.cmd["done"](u, ctx)
    assert any("grading" in s.lower() for s in sent), "done ack missing"
    assert ctx.bot.send_photo.await_count >= 1, "cards not delivered for /done"


def test_help_about_done_no_photos_deliver(bot):
    _run(_help_about_path(bot))


async def _help_about_path(bot):
    app, agent = bot
    uid = 888
    ctx = _ctx()
    await app.cmd["start"](*(_msg_update(uid, "/start")[0], ctx))

    u, sent = _msg_update(uid, "/help"); await app.cmd["help"](u, ctx)
    assert sent and len(sent[-1]) > 30

    u, sent = _msg_update(uid, "/about"); await app.cmd["about"](u, ctx)
    assert sent and "limit" in sent[-1].lower()

    # done with no photos -> guidance, never silent
    u, sent = _msg_update(uid, "done"); await app.on_text(u, ctx)
    assert sent and "photo" in sent[-1].lower()
    assert ctx.bot.send_photo.await_count == 0
