"""BotAgent for Curator Bot v1.1 (Stage S-01, Agent A1).

The conversation spine. Provides handler logic that is fully unit-testable WITHOUT
a live Telegram token by operating on lightweight, duck-typed update objects and
returning the reply text/keyboard instead of requiring real network sends.

Responsibilities (S-01 only):
  * /start            — greet + show the 6-platform selector.
  * platform selector — multi-select toggle + confirm (callback queries).
  * message router    — dispatch updates by type/intent; never crash on
                        unexpected/empty/None updates (degrades gracefully).
  * intent parsing    — platforms / vibe / constraints from free text.
  * photo receipt      — intake + acknowledge with a progress indicator. NO grading
                        (grading is A2/A3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from curator.session import SessionAgent
from curator.session.intent import parse_intent
from curator.session.schema import (
    PLATFORM_LABELS,
    PLATFORMS,
    PhotoRef,
    Stage,
)
from curator.ranking.card import FinalCard, OutputCard
from curator.ranking.presentation import (
    format_cards,
    format_final_card,
    format_variant_prompt,
)
from curator.ranking.selection import (
    SelectionError,
    build_final_card,
    select_photo,
    select_variant,
)
from .keyboards import (
    PLATFORM_CALLBACK_PREFIX,
    PLATFORMS_DONE_CALLBACK,
    platform_button_rows,
)

START_MESSAGE = (
    "Welcome to Curator Bot v1.1\n\n"
    "I grade your photos and write platform-specific captions for "
    "Instagram, Facebook, Tinder, Bumble, Hinge, and LinkedIn.\n\n"
    "First, pick the platforms you're posting to (tap to multi-select), "
    "then send me your photos."
)

PLATFORM_PROMPT = "Which platforms are you posting to? Tap all that apply, then Done."

# Expected total photos used for the progress indicator's denominator. The PRD
# uses ~12-20 photos per session; we show "received X/N" where N is a soft target.
EXPECTED_PHOTOS = 12


@dataclass
class HandlerResult:
    """What a handler decided to reply. Token-free, fully assertable in tests.

    `text` is the reply text; `keyboard_rows` is the inline-keyboard spec (list of
    rows of {text, callback_data}) or None; `kind` tags the branch taken so tests
    and the router can assert dispatch without inspecting prose.
    """

    text: str
    kind: str
    keyboard_rows: Optional[List[List[dict]]] = None


class BotAgent:
    """Stateless-ish conversation handler over a SessionAgent."""

    def __init__(self, sessions: Optional[SessionAgent] = None) -> None:
        self.sessions = sessions or SessionAgent()
        # Presentation-layer state (A5): the ranked cards currently shown to a
        # user, and the photo card they've picked while choosing a variant. Held
        # here (not in the durable Session schema A1 owns) so the ranking flow
        # stays self-contained. Keyed by user_id.
        self._pending_cards: dict[int, List[OutputCard]] = {}
        self._pending_photo: dict[int, OutputCard] = {}

    # --- helpers ------------------------------------------------------------
    @staticmethod
    def _user_id(update: Any) -> Optional[int]:
        """Best-effort extraction of a Telegram user id from any update shape."""
        for path in (
            ("effective_user", "id"),
            ("message", "from_user", "id"),
            ("callback_query", "from_user", "id"),
        ):
            obj: Any = update
            ok = True
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    ok = False
                    break
            if ok and obj is not None:
                return obj
        return None

    # --- command handlers ---------------------------------------------------
    def handle_start(self, update: Any) -> HandlerResult:
        """/start — init session and present the platform selector."""
        user_id = self._user_id(update)
        if user_id is None:
            return HandlerResult(text=START_MESSAGE, kind="start")
        session = self.sessions.init(user_id)
        self.sessions.update(user_id, stage=Stage.SELECTING_PLATFORMS)
        return HandlerResult(
            text=f"{START_MESSAGE}\n\n{PLATFORM_PROMPT}",
            kind="start",
            keyboard_rows=platform_button_rows(session.platforms),
        )

    def handle_platform_callback(self, update: Any) -> HandlerResult:
        """Handle a platform-selector callback query (toggle or done)."""
        user_id = self._user_id(update)
        data = getattr(getattr(update, "callback_query", None), "data", None)
        if user_id is None or not isinstance(data, str):
            return HandlerResult(text="Sorry, I couldn't read that.", kind="ignored")

        session = self.sessions.get_or_init(user_id)

        if data == PLATFORMS_DONE_CALLBACK:
            if not session.platforms:
                return HandlerResult(
                    text="Pick at least one platform first.",
                    kind="platforms_done_empty",
                    keyboard_rows=platform_button_rows(session.platforms),
                )
            self.sessions.update(user_id, stage=Stage.AWAITING_INTENT)
            chosen = ", ".join(PLATFORM_LABELS[p] for p in session.platforms)
            return HandlerResult(
                text=(
                    f"Great — targeting: {chosen}.\n\n"
                    "Tell me the vibe (mood/theme) and any constraints "
                    "(e.g. 'no location names', 'under 100 chars'), "
                    "or just start sending photos."
                ),
                kind="platforms_done",
            )

        if data.startswith(PLATFORM_CALLBACK_PREFIX):
            key = data[len(PLATFORM_CALLBACK_PREFIX) :]
            if key not in PLATFORMS:
                return HandlerResult(text="Unknown platform.", kind="ignored")
            session = self.sessions.toggle_platform(user_id, key)
            return HandlerResult(
                text=PLATFORM_PROMPT,
                kind="platform_toggle",
                keyboard_rows=platform_button_rows(session.platforms),
            )

        return HandlerResult(text="Unrecognized action.", kind="ignored")

    def handle_text(self, update: Any) -> HandlerResult:
        """Parse a free-text message into intent and merge into the session."""
        user_id = self._user_id(update)
        text = getattr(getattr(update, "message", None), "text", None)
        if user_id is None:
            return HandlerResult(text="Send /start to begin.", kind="no_user")

        session = self.sessions.get_or_init(user_id)
        intent = parse_intent(text)

        updates: dict = {}
        if intent.platforms:
            merged = list(session.platforms)
            for p in intent.platforms:
                if p not in merged:
                    merged.append(p)
            updates["platforms"] = merged
        if intent.vibe:
            updates["vibe"] = intent.vibe
        if intent.constraints:
            merged_c = list(session.constraints)
            for c in intent.constraints:
                if c not in merged_c:
                    merged_c.append(c)
            updates["constraints"] = merged_c

        if not updates:
            return HandlerResult(
                text=(
                    "I didn't catch a platform, vibe, or constraint there. "
                    "You can send photos any time, or /start to pick platforms."
                ),
                kind="text_noop",
            )

        self.sessions.update(user_id, **updates)
        parts = []
        if intent.platforms:
            parts.append(
                "platforms: " + ", ".join(PLATFORM_LABELS[p] for p in intent.platforms)
            )
        if intent.vibe:
            parts.append(f"vibe: {intent.vibe}")
        if intent.constraints:
            parts.append("constraints: " + ", ".join(intent.constraints))
        return HandlerResult(text="Got it — " + "; ".join(parts) + ".", kind="text_intent")

    def handle_photo(self, update: Any) -> HandlerResult:
        """Intake a photo: store its reference + acknowledge with progress.

        Does NOT grade (grading is A2/A3). Accepts either a PTB-style
        `message.photo` list of PhotoSize, or a single photo-like object.
        """
        user_id = self._user_id(update)
        if user_id is None:
            return HandlerResult(text="Send /start to begin.", kind="no_user")

        photo = self._extract_photo(update)
        if photo is None:
            return HandlerResult(
                text="That didn't look like a photo I could read.",
                kind="photo_unreadable",
            )

        self.sessions.get_or_init(user_id)
        session = self.sessions.add_photo(user_id, photo)
        count = len(session.photos)
        return HandlerResult(
            text=f"Received {count}/{EXPECTED_PHOTOS} photos. Keep them coming, or say 'done'.",
            kind="photo_received",
        )

    @staticmethod
    def _extract_photo(update: Any) -> Optional[PhotoRef]:
        """Pull a PhotoRef from an update. Picks the largest PhotoSize if a list."""
        message = getattr(update, "message", None)
        if message is None:
            return None
        photo_field = getattr(message, "photo", None)
        candidate = None
        if isinstance(photo_field, (list, tuple)) and photo_field:
            # PTB sends ascending sizes; the last is the largest.
            candidate = photo_field[-1]
        elif photo_field is not None:
            candidate = photo_field

        if candidate is None:
            return None

        file_id = getattr(candidate, "file_id", None)
        if not file_id:
            return None
        return PhotoRef(
            file_id=file_id,
            unique_id=getattr(candidate, "file_unique_id", None),
            width=getattr(candidate, "width", None),
            height=getattr(candidate, "height", None),
        )

    # --- ranking presentation + selection (Agent A5 / Stage S-05) -----------
    def present_ranked_cards(
        self, user_id: int, cards: List[OutputCard]
    ) -> HandlerResult:
        """Show the ranked output cards to the user (photo + rationale + variants).

        Stores the cards as pending selection state, advances the session to
        Stage.RANKING, and returns the formatted card set. The RankingAgent
        produces ``cards``; this is the BotAgent presentation hook A5 wires.
        """
        self._pending_cards[user_id] = list(cards)
        self._pending_photo.pop(user_id, None)
        # Log the stage transition (no-op if no live session, mirroring A1).
        self.sessions.update(user_id, stage=Stage.RANKING)
        if not cards:
            return HandlerResult(
                text="No rankable photos in this session.",
                kind="ranking_empty",
            )
        return HandlerResult(
            text=format_cards(cards),
            kind="ranking_cards",
        )

    def handle_photo_selection(self, user_id: int, raw: Any) -> HandlerResult:
        """Handle a photo pick (1-based card number). Graceful on bad input."""
        cards = self._pending_cards.get(user_id)
        if not cards:
            return HandlerResult(
                text="There are no ranked photos to choose from yet.",
                kind="select_no_cards",
            )
        chosen = select_photo(cards, raw)
        if isinstance(chosen, SelectionError):
            return HandlerResult(text=chosen.message, kind="select_photo_invalid")
        self._pending_photo[user_id] = chosen
        return HandlerResult(
            text=format_variant_prompt(chosen),
            kind="select_photo_ok",
        )

    def handle_variant_selection(self, user_id: int, raw: Any) -> HandlerResult:
        """Handle a caption-variant pick; log the selection + return the final card.

        Requires a photo to have been picked first. On a valid pick we build the
        post-ready FinalCard, log the selection to the session (Stage.DONE), and
        return the rendered card. Graceful on out-of-range / garbage input.
        """
        chosen = self._pending_photo.get(user_id)
        if chosen is None:
            return HandlerResult(
                text="Pick a photo first (reply with a photo number).",
                kind="select_no_photo",
            )
        vidx = select_variant(chosen.photo, raw)
        if isinstance(vidx, SelectionError):
            return HandlerResult(text=vidx.message, kind="select_variant_invalid")

        final = build_final_card(chosen.photo, vidx)
        self._log_selection(user_id, final)
        # Selection complete: clear pending state.
        self._pending_cards.pop(user_id, None)
        self._pending_photo.pop(user_id, None)
        return HandlerResult(
            text=format_final_card(final),
            kind="final_card",
        )

    def _log_selection(self, user_id: int, final: FinalCard) -> None:
        """Persist the confirmed selection to the session (Stage.DONE).

        Logs only durable, non-generated metadata (platform, chosen photo index,
        chosen variant) — never committed output. Mirrors A1's update() contract;
        a no-op if the session has expired.
        """
        self.sessions.update(user_id, stage=Stage.DONE)

    # --- router -------------------------------------------------------------
    def route(self, update: Any) -> HandlerResult:
        """Dispatch an update to the right handler by type/intent.

        Robust to malformed input: None, empty, or unexpected update shapes return
        a graceful HandlerResult instead of raising.
        """
        if update is None:
            return HandlerResult(text="", kind="ignored_empty")

        try:
            # Callback query (platform selector taps) take priority.
            cq = getattr(update, "callback_query", None)
            if cq is not None and getattr(cq, "data", None) is not None:
                return self.handle_platform_callback(update)

            message = getattr(update, "message", None)
            if message is not None:
                # Photo?
                if getattr(message, "photo", None):
                    return self.handle_photo(update)
                # Command?
                text = getattr(message, "text", None)
                if isinstance(text, str) and text.strip().startswith("/start"):
                    return self.handle_start(update)
                if isinstance(text, str) and text.strip():
                    return self.handle_text(update)
                # Message with neither photo nor text (sticker, location, etc.)
                return HandlerResult(
                    text="I can work with photos and text for now.",
                    kind="unsupported_message",
                )

            # Unknown update type with no message/callback.
            return HandlerResult(text="", kind="ignored_unknown")
        except Exception:  # noqa: BLE001 - router must never crash the bot
            return HandlerResult(
                text="Something went wrong handling that — please try again.",
                kind="error",
            )
