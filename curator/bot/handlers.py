"""python-telegram-bot adapter for Curator Bot v1.1 (Stage S-01, Agent A1).

Bridges the token-free BotAgent (bot/agent.py) to PTB's async callback signatures
and registers the handlers on an Application. Imported lazily by curator.main so the
package stays importable without python-telegram-bot installed.

A single shared BotAgent instance backs all handlers so session state is consistent
across updates within a process.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from .agent import BotAgent, HandlerResult


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
        # Always acknowledge the tap so Telegram stops the loading spinner.
        await cq.answer()
        if not result.text:
            return
        msg = getattr(cq, "message", None)
        if msg is not None:
            # Edit the card in place: updates the checkmarks on a platform
            # toggle (markup present) and replaces the buttons with the
            # confirmation text on Done (markup is None). A callback update has
            # no `update.message`, so this is the ONLY path that replies to a
            # tap — earlier it was gated on keyboard_rows, which silently
            # dropped the keyboard-less "Done" confirmation.
            await cq.edit_message_text(result.text, reply_markup=markup)
        else:
            # No attached message to edit — fall back to a fresh send.
            await cq.get_bot().send_message(cq.from_user.id, result.text, reply_markup=markup)
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

    async def _run_grading(update, context):  # noqa: ANN001
        """Download the session's photos and grade+present (the async work).

        Triggered when the agent returns kind="trigger_grading". The blocking
        grade/caption/rank pipeline runs in a thread so the bot stays responsive.
        """
        uid = agent._user_id(update)
        session = agent.sessions.get(uid) if uid is not None else None
        images: dict = {}
        if session is not None:
            for ph in session.photos:
                try:
                    tg_file = await context.bot.get_file(ph.file_id)
                    images[ph.file_id] = bytes(await tg_file.download_as_bytearray())
                except Exception:  # noqa: BLE001 - skip a photo we can't fetch
                    continue
        result = await asyncio.to_thread(agent.grade_and_present, uid, images)
        await _reply(update, result)

    async def on_text(update, context):  # noqa: ANN001
        result = agent.handle_text(update)
        await _reply(update, result)
        if result.kind == "trigger_grading":
            await _run_grading(update, context)

    # Override commands (A6 / S-06). Each routes through the token-free BotAgent.
    async def on_help(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_help(update))

    async def on_about(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_about(update))

    async def on_done(update, context):  # noqa: ANN001
        result = agent.handle_done(update)
        await _reply(update, result)
        if result.kind == "trigger_grading":
            await _run_grading(update, context)

    async def on_next(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_next(update))

    async def on_retry(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_retry(update))

    async def on_shorter(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_shorter(update))

    async def on_platform(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_platform(update))

    async def on_reset(update, context):  # noqa: ANN001
        await _reply(update, agent.handle_reset(update))

    application.add_handler(CommandHandler("start", on_start))
    application.add_handler(CommandHandler("help", on_help))
    application.add_handler(CommandHandler("commands", on_help))
    application.add_handler(CommandHandler("about", on_about))
    application.add_handler(CommandHandler("done", on_done))
    application.add_handler(CommandHandler("next", on_next))
    application.add_handler(CommandHandler("retry", on_retry))
    application.add_handler(CommandHandler("shorter", on_shorter))
    application.add_handler(CommandHandler("platform", on_platform))
    application.add_handler(CommandHandler("reset", on_reset))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_text)
    )
    return agent
