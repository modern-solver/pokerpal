"""Regression tests for the PTB delivery layer (_reply).

These guard the bug where the platform-selection "Done" confirmation was silently
dropped: a callback-query update has no ``update.message``, and the old _reply
only replied to callbacks when a keyboard was present, so the keyboard-less Done
confirmation reached neither branch and the bot went silent right after the user
chose their platforms.

We drive _reply with async mocks (no real python-telegram-bot needed for the
keyboard-less paths, which is exactly the Done case).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from curator.bot.handlers import _reply
from curator.bot.agent import HandlerResult


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_done_confirmation_callback_is_delivered():
    """Done returns keyboard_rows=None over a callback -> must still reply."""
    cq = SimpleNamespace(
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        message=SimpleNamespace(),  # an attached message exists -> edit path
        from_user=SimpleNamespace(id=42),
    )
    update = SimpleNamespace(callback_query=cq, message=None)
    result = HandlerResult(text="Great — targeting: Instagram.", kind="platforms_done")

    _run(_reply(update, result))

    cq.answer.assert_awaited_once()  # spinner acknowledged
    cq.edit_message_text.assert_awaited_once()  # the confirmation WAS sent
    sent_text = cq.edit_message_text.call_args.args[0]
    assert "targeting" in sent_text


def test_platform_toggle_callback_edits_in_place():
    """A toggle carries keyboard_rows; build a real markup only if PTB present."""
    import pytest

    pytest.importorskip("telegram", reason="python-telegram-bot not installed")
    cq = SimpleNamespace(
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        message=SimpleNamespace(),
        from_user=SimpleNamespace(id=7),
    )
    update = SimpleNamespace(callback_query=cq, message=None)
    rows = [[{"text": "✅ Instagram", "callback_data": "platform:instagram"}]]
    result = HandlerResult(text="Which platforms?", kind="platform_toggle", keyboard_rows=rows)

    _run(_reply(update, result))

    cq.answer.assert_awaited_once()
    cq.edit_message_text.assert_awaited_once()


def test_plain_message_reply_path_unchanged():
    """A normal text/photo update still replies via message.reply_text."""
    msg = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(callback_query=None, message=msg)
    result = HandlerResult(text="Received 1/12 photos.", kind="photo_received")

    _run(_reply(update, result))

    msg.reply_text.assert_awaited_once()
    assert "Received" in msg.reply_text.call_args.args[0]
