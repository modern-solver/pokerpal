"""BotAgent package — Telegram conversation spine (Stage S-01, Agent A1).

Exposes the token-free `BotAgent` (handlers + router), the PTB handler registrar,
and the platform-selector keyboard helpers.
"""

from .agent import BotAgent, HandlerResult, START_MESSAGE
from .handlers import register_handlers
from .keyboards import (
    PLATFORM_CALLBACK_PREFIX,
    PLATFORMS_DONE_CALLBACK,
    build_platform_keyboard,
    platform_button_rows,
)

__all__ = [
    "BotAgent",
    "HandlerResult",
    "START_MESSAGE",
    "register_handlers",
    "platform_button_rows",
    "build_platform_keyboard",
    "PLATFORM_CALLBACK_PREFIX",
    "PLATFORMS_DONE_CALLBACK",
]
