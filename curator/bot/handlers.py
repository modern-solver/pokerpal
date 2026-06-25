"""python-telegram-bot adapter for Curator Bot v1.1 (Stage S-01, Agent A1).

Bridges the token-free BotAgent (bot/agent.py) to PTB's async callback signatures
and registers the handlers on an Application. Imported lazily by curator.main so the
package stays importable without python-telegram-bot installed.

A single shared BotAgent instance backs all handlers so session state is consistent
across updates within a process.
"""

from __future__ import annotations

from typing import Optional

from .agent import BotAgent, HandlerResult
from .keyboards import build_platform_keyboard


async def _reply(update, result: HandlerResult) -> None:
    """Send a HandlerResult back over Telegram (text + optional keyboard)."""
    markup = None
    if result.keyboard_rows is not None:
        # Rebuild a PTB keyboard from the same selection the agent computed.
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        rows = [
            [
                InlineKeyboardButton(b["text"], callback_data=b["callback_data"])
                for b in row
            ]
            for row in result.keyboard_rows
        ]
        markup = InlineKeyboardMarkup(rows)

    cq = getattr(update, "callback_query", None)
    if cq is not None:
        await cq.answer()
        if result.keyboard_rows is not None and getattr(cq, "message", None):
            await cq.edit_message_text(result.text, reply_markup=markup)
            return
    if getattr(update, "message", None) is not None and result.text:
        await update.message.reply_text(result.text, reply_markup=markup)


def register_handlers(application, agent: Optional[BotAgent] = None) -> BotAgent:
    """Register S-01 handlers on a PTB Application. Returns the backing BotAgent."""
    from telegram.ext import (
        CallbackQueryHandler,
        CommandHandler,
        MessageHandler,
        filters,
    )

    agent = agent or BotAgent()

    async def on_start(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_start(update))

    async def on_callback(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_platform_callback(update))

    async def on_photo(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_photo(update))

    async def on_text(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_text(update))

    application.add_handler(CommandHandler("start", on_start))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_text)
    )
    return agent
