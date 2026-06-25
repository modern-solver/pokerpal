"""Unit test for the /start handler (Exit gate: 'Bot responds to /start').

No live Telegram token required — we drive the handler with mock objects and assert
it replies with the welcome message. This is the S-00 evidence that the bot
responds to /start.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from curator.main import START_MESSAGE, start


def test_start_replies_with_welcome_message():
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    context = MagicMock()

    returned = asyncio.run(start(update, context))

    update.message.reply_text.assert_awaited_once_with(START_MESSAGE)
    assert returned == START_MESSAGE
    assert "Curator Bot" in returned


def test_start_mentions_all_six_platforms():
    text = START_MESSAGE.lower()
    for platform in ("instagram", "facebook", "tinder", "bumble", "hinge", "linkedin"):
        assert platform in text


def test_start_handles_missing_message_gracefully():
    # A defensive update with no .message should not raise.
    update = MagicMock()
    update.message = None
    assert asyncio.run(start(update, MagicMock())) == START_MESSAGE
