"""Curator Bot v1.1 — application entry point.

OWNER: Agent A0 (Stage S-00) ships a MINIMAL Telegram bot that responds to /start.
Agent A1 (Stage S-01) extends this with the full message router, 6-platform
selector, intent parsing, and photo-receipt handler.

Design notes:
    - The /start handler (`start`) is factored out so it is unit-testable WITHOUT
      a live Telegram token (see tests/test_start_handler.py).
    - The run loop is guarded behind `if __name__ == '__main__'` and a token check
      that logs a clear message and exits cleanly if TELEGRAM_TOKEN is absent
      (the sandbox has no token).
"""

from __future__ import annotations

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("curator.bot")

# Welcome copy shown on /start. A1 may expand into the platform selector flow.
START_MESSAGE = (
    "Welcome to Curator Bot v1.1\n\n"
    "Send me your photos and I will grade them and write platform-specific "
    "captions for Instagram, Facebook, Tinder, Bumble, Hinge, and LinkedIn.\n\n"
    "(More setup coming — this is the S-00 scaffold.)"
)


async def start(update, context):  # noqa: ANN001 - PTB callback signature
    """Handle the /start command by replying with the welcome message.

    Kept dependency-light and side-effect-isolated so it can be unit-tested with a
    mock `update`/`context`. Returns the sent text for easy assertion.
    """
    if update is not None and getattr(update, "message", None) is not None:
        await update.message.reply_text(START_MESSAGE)
    return START_MESSAGE


def build_application(token: str):
    """Build a python-telegram-bot Application with the S-01 handlers registered.

    Imported lazily so the module imports cleanly even if python-telegram-bot is
    not installed (e.g. lightweight test/CI environments).

    A0 shipped only the minimal /start handler. A1 (Stage S-01) registers the full
    conversation spine — /start with the 6-platform selector, the callback-query
    handler for multi-select, the photo-receipt handler, and the free-text intent
    router — via curator.bot.register_handlers.
    """
    from telegram.ext import Application

    from curator.bot import register_handlers

    application = Application.builder().token(token).build()
    register_handlers(application)
    return application


def main() -> int:
    """Entry point for `python -m curator.main`.

    Returns a process exit code. Exits cleanly (non-crash) when no token is set,
    so the scaffold is safe to invoke in the sandbox.
    """
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        logger.warning(
            "TELEGRAM_TOKEN is not set — cannot start the bot. "
            "Set it in your .env (see .env.example) and register the bot with "
            "@BotFather. Exiting without starting the run loop."
        )
        return 0

    logger.info("Starting Curator Bot (polling)...")
    application = build_application(token)
    application.run_polling()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
