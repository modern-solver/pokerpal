"""Inline keyboards for Curator Bot v1.1 (Stage S-01, Agent A1).

The platform selector presents all 6 platforms as a multi-select inline keyboard.
A selected platform is prefixed with a check mark; tapping toggles it. A final
"Done" button confirms the selection.

Callback data convention:
  * "platform:<key>"  — toggle a platform
  * "platforms:done"  — confirm selection
  * "length:<mode>"   — pick a caption length (short|long|haiku)
  * "cap:<card>:<opt>" — pick caption option <opt> on ranked card <card>
"""

from __future__ import annotations

from typing import Iterable, List

from curator.session.schema import PLATFORM_LABELS, PLATFORMS

PLATFORM_CALLBACK_PREFIX = "platform:"
PLATFORMS_DONE_CALLBACK = "platforms:done"
LENGTH_CALLBACK_PREFIX = "length:"
CAPTION_CALLBACK_PREFIX = "cap:"

# Caption length options shown as buttons (mode -> button label).
LENGTH_OPTIONS = [
    ("short", "✍️ Short (<200)"),
    ("long", "📖 Long (<1000)"),
    ("haiku", "🍃 Haiku"),
]


def platform_button_rows(selected: Iterable[str] = ()) -> List[List[dict]]:
    """Return the selector as plain rows of {text, callback_data} dicts.

    Backend-agnostic so it is unit-testable without python-telegram-bot. The PTB
    adapter (`build_platform_keyboard`) wraps these into InlineKeyboardButtons.
    Two platforms per row, then a confirm row.
    """
    selected_set = set(selected)
    buttons: List[dict] = []
    for key in PLATFORMS:
        label = PLATFORM_LABELS[key]
        mark = "✅ " if key in selected_set else ""  # white check mark
        buttons.append(
            {"text": f"{mark}{label}", "callback_data": f"{PLATFORM_CALLBACK_PREFIX}{key}"}
        )

    rows: List[List[dict]] = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    rows.append([{"text": "Done →", "callback_data": PLATFORMS_DONE_CALLBACK}])
    return rows


def length_button_rows(selected: str = "") -> List[List[dict]]:
    """Single-select caption-length keyboard (Short / Long / Haiku)."""
    rows: List[List[dict]] = []
    for mode, label in LENGTH_OPTIONS:
        mark = "✅ " if mode == selected else ""
        rows.append(
            [{"text": f"{mark}{label}", "callback_data": f"{LENGTH_CALLBACK_PREFIX}{mode}"}]
        )
    return rows


def caption_button_rows(card_index: int, n_options: int) -> List[List[dict]]:
    """A row of 'Use 1/2/3' buttons to pick a caption on one ranked card."""
    return [
        [
            {
                "text": f"Use {i}",
                "callback_data": f"{CAPTION_CALLBACK_PREFIX}{card_index}:{i}",
            }
            for i in range(1, n_options + 1)
        ]
    ]


def build_platform_keyboard(selected: Iterable[str] = ()):
    """Wrap `platform_button_rows` into a PTB InlineKeyboardMarkup.

    Imported lazily so this module imports cleanly without python-telegram-bot
    installed (test/CI environments use `platform_button_rows` directly).
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    rows = [
        [InlineKeyboardButton(b["text"], callback_data=b["callback_data"]) for b in row]
        for row in platform_button_rows(selected)
    ]
    return InlineKeyboardMarkup(rows)
