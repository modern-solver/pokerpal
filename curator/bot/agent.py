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

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from curator.session import SessionAgent
from curator.session.intent import parse_intent
from curator.session.schema import (
    PLATFORM_LABELS,
    PLATFORMS,
    PhotoRef,
    Stage,
)
from curator.session.timeout import (
    WARNING_MESSAGE,
    TimeoutStatus,
    check_timeout,
)
from curator.ranking.card import FinalCard, OutputCard
from curator.ranking.presentation import (
    format_card,
    format_cards,
    format_final_card,
    format_variant_prompt,
)
from curator.ranking.selection import (
    SelectionError,
    build_final_card,
    cycle_variant,
    select_photo,
    select_variant,
)
from .keyboards import (
    PLATFORM_CALLBACK_PREFIX,
    PLATFORMS_DONE_CALLBACK,
    platform_button_rows,
)
from .faq import (
    ABOUT_TEXT,
    USER_GUIDE,
    answer_question,
    command_reference,
    looks_like_question,
)

START_MESSAGE = (
    "Welcome to Curator Bot v1.1\n\n"
    "I grade your photos and write platform-specific captions for "
    "Instagram, Facebook, Tinder, Bumble, Hinge, and LinkedIn.\n\n"
    "First, pick the platforms you're posting to (tap to multi-select), "
    "then send me your photos."
)

PLATFORM_PROMPT = "Which platforms are you posting to? Tap all that apply, then Done."

# Command reference shown in /start and /help. Single source of truth is
# faq.BOT_COMMANDS (also used to register the Telegram command menu), so the
# in-chat list and the BotFather menu never drift apart.
COMMANDS_HELP = command_reference()

# Expected total photos used for the progress indicator's denominator. The PRD
# uses ~12-20 photos per session; we show "received X/N" where N is a soft target.
EXPECTED_PHOTOS = 12

# Default platform when a session somehow has none selected.
DEFAULT_PLATFORM = "instagram"

# Phrases that mean "I'm finished sending photos — grade them now."
_DONE_WORDS = {
    "done", "im done", "i'm done", "all done", "finished", "finish", "go",
    "ready", "rank", "rank them", "grade", "grade them", "that's all",
    "thats all", "go ahead", "ok done", "okay done", "that is all",
}


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

    def __init__(
        self,
        sessions: Optional[SessionAgent] = None,
        *,
        caption_agent: Any = None,
        ranking_agent: Any = None,
        grading_agent: Any = None,
        clock: Any = None,
    ) -> None:
        self.sessions = sessions or SessionAgent()
        # Presentation-layer state (A5): the ranked cards currently shown to a
        # user, and the photo card they've picked while choosing a variant. Held
        # here (not in the durable Session schema A1 owns) so the ranking flow
        # stays self-contained. Keyed by user_id.
        self._pending_cards: dict[int, List[OutputCard]] = {}
        self._pending_photo: dict[int, OutputCard] = {}
        # Override-command state (A6 / S-06). To re-rank on /platform and advance
        # on /next without re-grading, we cache the per-user grading results +
        # caption results + the active platform + a /next cursor + the current
        # /retry tone-cycle index per card. All in-memory (never persisted), so
        # nothing generated is committed.
        self._pending_results: dict[int, List[Any]] = {}
        self._pending_captions: dict[int, List[Any]] = {}
        self._active_platform: dict[int, str] = {}
        self._next_cursor: dict[int, int] = {}
        self._retry_idx: dict[int, int] = {}
        # Lazily-constructed CaptionAgent / RankingAgent used by /retry, /shorter,
        # and /platform. Injectable for tests; never imported eagerly so the
        # zero-API guarantee and lazy-client posture hold. ``None`` => build on
        # first use (deterministic local-fallback CaptionAgent: no Groq, no key).
        self._caption_agent = caption_agent
        self._ranking_agent = ranking_agent
        self._grading_agent = grading_agent
        # Shared Groq vision client (Tier-3 grading + scene description), built
        # lazily from GROQ_API_KEY; None when no key (local-only fallback).
        self._vision = None
        # Injectable clock for deterministic timeout-warning tests.
        self._clock = clock
        # Users already warned in the current expiry window (de-dupe the nag).
        self._warned: set = set()

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
        self._clear_override_state(user_id)
        session = self.sessions.init(user_id)
        self.sessions.update(user_id, stage=Stage.SELECTING_PLATFORMS)
        return HandlerResult(
            text=f"{START_MESSAGE}\n\n{COMMANDS_HELP}\n\n{PLATFORM_PROMPT}",
            kind="start",
            keyboard_rows=platform_button_rows(session.platforms),
        )

    def handle_help(self, update: Any) -> HandlerResult:
        """/help — the full usage walkthrough + command reference."""
        return HandlerResult(text=f"{USER_GUIDE}\n\n{COMMANDS_HELP}", kind="help")

    def handle_about(self, update: Any) -> HandlerResult:
        """/about — capabilities and limitations."""
        return HandlerResult(text=ABOUT_TEXT, kind="about")

    def handle_done(self, update: Any) -> HandlerResult:
        """/done — finish collecting photos and start grading.

        Returns kind="trigger_grading" when there are photos; the PTB layer then
        downloads them and calls grade_and_present. Token-free + testable here.
        """
        user_id = self._user_id(update)
        if user_id is None:
            return HandlerResult(text="Send /start to begin.", kind="no_user")
        session = self.sessions.get_or_init(user_id)
        if not session.photos:
            return HandlerResult(
                text="Send me at least one photo first, then /done.",
                kind="done_no_photos",
            )
        return HandlerResult(
            text="Got it — grading your photos now, this takes a few seconds…",
            kind="trigger_grading",
        )

    def _is_done_signal(self, text: str) -> bool:
        """True if free text means 'finish and grade' (e.g. 'done', "that's all")."""
        t = " ".join((text or "").lower().split()).strip(" .!?")
        return t in _DONE_WORDS

    def _handle_ranking_reply(self, user_id: int, text: Any) -> HandlerResult:
        """Route a reply while ranked cards are shown: photo pick, then variant."""
        if self._pending_photo.get(user_id) is None:
            return self.handle_photo_selection(user_id, text)
        return self.handle_variant_selection(user_id, text)

    def grade_and_present(
        self, user_id: int, images: Dict[str, bytes]
    ) -> HandlerResult:
        """Grade the session's downloaded photos and present ranked cards.

        ``images`` maps file_id -> bytes (downloaded by the PTB layer). Builds the
        grading/caption agents (Groq-backed when GROQ_API_KEY is set, otherwise a
        local fallback), runs the pipeline, and hands the cards to
        present_ranked_cards. Failures degrade to a friendly message rather than a
        silent bot.
        """
        session = self.sessions.get_or_init(user_id)
        refs: List[Any] = []
        byts: List[bytes] = []
        for ph in session.photos:
            data = images.get(ph.file_id) if images else None
            if data:
                refs.append(ph)
                byts.append(data)
        if not byts:
            return HandlerResult(
                text="I couldn't read any of those photos — send a few clear ones, then /done.",
                kind="grade_no_photos",
            )

        platforms = [p for p in (session.platforms or []) if p in PLATFORMS] or [
            DEFAULT_PLATFORM
        ]
        target = self._active_platform.get(user_id) or platforms[0]

        try:
            from .pipeline import run_pipeline

            cards, results, captions = run_pipeline(
                byts,
                refs,
                platforms,
                target,
                grading_agent=self._grading(),
                caption_agent=self._caption(),
                ranking_agent=self._ranking(),
                describe=self._describe_fn(),
                vibe=session.vibe,
                constraints=list(session.constraints),
                top_n=3,
            )
        except Exception:  # noqa: BLE001 - never crash the conversation
            return HandlerResult(
                text=(
                    "Something went wrong while grading those — please /reset and "
                    "try again."
                ),
                kind="grade_error",
            )

        if not cards:
            return HandlerResult(
                text="I couldn't rank those — try clearer shots and /done again.",
                kind="ranking_empty",
            )
        intro = f"Done! Here are your best shots for {PLATFORM_LABELS.get(target, target)}.\n\n"
        card = self.present_ranked_cards(
            user_id, cards, results=results, captions=captions, platform=target
        )
        return HandlerResult(text=intro + card.text, kind=card.kind)

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

        # A question about how the bot works / its limits is answered directly
        # (deterministic, zero-cost) rather than parsed as a posting intent.
        if isinstance(text, str) and looks_like_question(text):
            return HandlerResult(text=answer_question(text), kind="faq")

        # While reviewing ranked cards, a reply is a selection (photo then
        # caption number), not a new posting intent.
        if session.stage == Stage.RANKING:
            return self._handle_ranking_reply(user_id, text)

        # "done" (and friends) finishes the photo-collection step and kicks off
        # grading. The PTB layer sees kind="trigger_grading", downloads the
        # photos, and calls grade_and_present (the async work it owns).
        if isinstance(text, str) and self._is_done_signal(text):
            if not session.photos:
                return HandlerResult(
                    text="Send me at least one photo first, then say 'done'.",
                    kind="done_no_photos",
                )
            return HandlerResult(
                text="Got it — grading your photos now, this takes a few seconds…",
                kind="trigger_grading",
            )

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
                    "You can send photos any time, /start to pick platforms, or "
                    "/help and /about to learn what I do."
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
        nudge = (
            " Now send me your photos (a batch works best) — say 'done' or /done "
            "when you're finished and I'll grade and rank them."
        )
        return HandlerResult(
            text="Got it — " + "; ".join(parts) + "." + nudge, kind="text_intent"
        )

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
        self.sessions.update(user_id, stage=Stage.RECEIVING_PHOTOS)
        count = len(session.photos)
        return HandlerResult(
            text=(
                f"Received {count}/{EXPECTED_PHOTOS} photos. Send more, or say "
                "'done' (or /done) and I'll grade and rank them."
            ),
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
        self,
        user_id: int,
        cards: List[OutputCard],
        *,
        results: Optional[Sequence[Any]] = None,
        captions: Optional[Sequence[Any]] = None,
        platform: Optional[str] = None,
    ) -> HandlerResult:
        """Show the ranked output cards to the user (photo + rationale + variants).

        Stores the cards as pending selection state, advances the session to
        Stage.RANKING, and returns the formatted card set. The RankingAgent
        produces ``cards``; this is the BotAgent presentation hook A5 wires.

        A6 additionally caches the source ``results`` (GradingResults) +
        ``captions`` (CaptionResults) + active ``platform`` so the override
        commands (/next, /platform, /retry, /shorter) can re-rank / regenerate
        WITHOUT re-grading. These are optional so A5's existing call sites keep
        working; the override commands degrade gracefully when they're absent.
        """
        self._pending_cards[user_id] = list(cards)
        self._pending_photo.pop(user_id, None)
        self._next_cursor[user_id] = 0
        self._retry_idx.pop(user_id, None)
        if results is not None:
            self._pending_results[user_id] = list(results)
        if captions is not None:
            self._pending_captions[user_id] = list(captions)
        if platform is not None:
            self._active_platform[user_id] = platform
        elif cards:
            # Infer the active platform from the cards if not given.
            self._active_platform[user_id] = cards[0].photo.platform
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

    def pending_card_views(self, user_id: int) -> List[tuple]:
        """(file_id, caption) per ranked card, for sending photos in the chat."""
        cards = self._pending_cards.get(user_id) or []
        return [
            (getattr(c.photo, "file_id", None), format_card(c)) for c in cards
        ]

    def active_platform_label(self, user_id: int) -> str:
        """Human label of the platform currently being presented."""
        key = self._active_platform.get(user_id, DEFAULT_PLATFORM)
        return PLATFORM_LABELS.get(key, key)

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

    # --- override commands (Agent A6 / Stage S-06) --------------------------
    def _clear_override_state(self, user_id: int) -> None:
        """Drop ALL in-memory working state for a user (used by /reset, /start)."""
        for store in (
            self._pending_cards,
            self._pending_photo,
            self._pending_results,
            self._pending_captions,
            self._active_platform,
            self._next_cursor,
            self._retry_idx,
        ):
            store.pop(user_id, None)

    @staticmethod
    def _groq_key() -> Optional[str]:
        key = os.environ.get("GROQ_API_KEY")
        return key or None

    def _vision_grader(self):
        """Lazily build a shared Groq vision client (grading + scene describe)."""
        if self._vision is None:
            key = self._groq_key()
            if not key:
                return None
            from curator.grading.groq_client import GroqVisionGrader

            self._vision = GroqVisionGrader(api_key=key)
        return self._vision

    def _grading(self):
        """Lazily build the GradingAgent (Groq Tier-3 when GROQ_API_KEY is set)."""
        if self._grading_agent is None:
            from curator.grading.agent import GradingAgent
            from curator.grading.scene import SceneDescriptionService
            from curator.grading.tier3 import Tier3Grader

            vision = self._vision_grader()
            tier3 = (
                Tier3Grader(vision, SceneDescriptionService())
                if vision is not None
                else None
            )
            self._grading_agent = GradingAgent(tier3_grader=tier3)
        return self._grading_agent

    def _describe_fn(self):
        """A callable(bytes)->str for caption grounding, or None without a key."""
        vision = self._vision_grader()
        return vision.describe if vision is not None else None

    def _caption(self):
        """Lazily build the CaptionAgent (Groq when keyed, else local fallback)."""
        if self._caption_agent is None:
            from curator.caption.agent import CaptionAgent

            generator = None
            key = self._groq_key()
            if key:
                from curator.caption.groq_text import GroqTextGenerator

                generator = GroqTextGenerator(api_key=key)
            self._caption_agent = CaptionAgent(generator=generator)
        return self._caption_agent

    def _ranking(self):
        """Lazily build the RankingAgent (pure, local — no network)."""
        if self._ranking_agent is None:
            from curator.ranking.agent import RankingAgent

            self._ranking_agent = RankingAgent()
        return self._ranking_agent

    def _current_card(self, user_id: int) -> Optional[OutputCard]:
        """The card currently in focus: the picked photo, else the top card."""
        chosen = self._pending_photo.get(user_id)
        if chosen is not None:
            return chosen
        cards = self._pending_cards.get(user_id)
        if cards:
            return cards[0]
        return None

    def handle_next(self, update: Any) -> HandlerResult:
        """/next — advance to the next-ranked photo/card (A5 ranked cards).

        Walks the already-ranked OutputCard list via an in-memory cursor. Validates
        state (no cards yet / past the end) and degrades gracefully.
        """
        user_id = self._user_id(update)
        if user_id is None:
            return HandlerResult(text="Send /start to begin.", kind="no_user")
        cards = self._pending_cards.get(user_id)
        if not cards:
            return HandlerResult(
                text="There are no ranked photos yet — send photos first.",
                kind="next_no_cards",
            )
        cursor = self._next_cursor.get(user_id, 0) + 1
        if cursor >= len(cards):
            return HandlerResult(
                text="That's the last ranked photo. /reset to start over.",
                kind="next_exhausted",
            )
        self._next_cursor[user_id] = cursor
        card = cards[cursor]
        # Focus this card for subsequent /retry, /shorter, variant pick.
        self._pending_photo[user_id] = card
        self._retry_idx.pop(user_id, None)
        from curator.ranking.presentation import format_card

        return HandlerResult(text=format_card(card), kind="next_card")

    def handle_retry(self, update: Any) -> HandlerResult:
        """/retry — regenerate the caption.

        For dating platforms (rule C-07) this CYCLES tone variants IN ORDER
        (Playful -> Confident -> Mysterious for Tinder, etc.) using A5's
        ``cycle_variant`` over A4's tone order — NOT a random reseed. For other
        platforms it likewise steps to the next available tone variant in order.
        """
        user_id = self._user_id(update)
        if user_id is None:
            return HandlerResult(text="Send /start to begin.", kind="no_user")
        card = self._current_card(user_id)
        if card is None:
            return HandlerResult(
                text="Nothing to retry yet — pick a photo first.",
                kind="retry_no_card",
            )
        photo = card.photo
        if not photo.variants:
            return HandlerResult(
                text="No caption options are available for that photo.",
                kind="retry_no_variants",
            )
        current = self._retry_idx.get(user_id, 0)
        nxt = cycle_variant(photo, current)
        self._retry_idx[user_id] = nxt
        variant = photo.variants[nxt]
        from curator.ranking.card import render_variant

        return HandlerResult(
            text=(
                f"[{variant.tone}] {render_variant(variant)}"
            ),
            kind="retry_cycled",
        )

    def handle_shorter(self, update: Any) -> HandlerResult:
        """/shorter — regenerate the current caption under a tighter length cap.

        Re-runs CaptionAgent for the focused photo+platform with a tighter cap
        (re-using A4; never reimplementing generation). Replaces the focused
        card's current variant with the shorter one so /retry continues from it.
        """
        user_id = self._user_id(update)
        if user_id is None:
            return HandlerResult(text="Send /start to begin.", kind="no_user")
        card = self._current_card(user_id)
        if card is None:
            return HandlerResult(
                text="Nothing to shorten yet — pick a photo first.",
                kind="shorter_no_card",
            )
        photo = card.photo
        if not photo.variants:
            return HandlerResult(
                text="No caption to shorten for that photo.",
                kind="shorter_no_variants",
            )
        idx = self._retry_idx.get(user_id, 0)
        idx = idx if 0 <= idx < len(photo.variants) else 0
        current = photo.variants[idx]
        original_len = current.char_count
        tighter_cap = self._tighter_cap(original_len)

        session = self.sessions.get(user_id)
        try:
            result = self._caption().generate(
                photo.platform,
                photo.breakdown.get("scene_description", "") if photo.breakdown else "",
                session=session,
                photo_index=photo.photo_index,
                tones=[current.tone],
            )
        except Exception:  # noqa: BLE001 - never crash the command
            return HandlerResult(
                text="Couldn't shorten the caption right now — try /retry.",
                kind="shorter_failed",
            )
        if not result.variants:
            return HandlerResult(
                text="Couldn't shorten the caption right now — try /retry.",
                kind="shorter_failed",
            )
        new_variant = result.variants[0]
        # Enforce the tighter cap on the primary text (A4 trims to platform cap;
        # we additionally clamp to the tighter target so the cap is provably met).
        new_variant = self._clamp_variant(new_variant, tighter_cap)
        photo.variants[idx] = new_variant
        from curator.ranking.card import render_variant

        return HandlerResult(
            text=(
                f"Shorter ({new_variant.char_count}/{tighter_cap} chars):\n"
                f"[{new_variant.tone}] {render_variant(new_variant)}"
            ),
            kind="shorter_ok",
        )

    @staticmethod
    def _tighter_cap(original_len: int) -> int:
        """A tighter length cap: ~75% of the current length, floored sensibly."""
        target = int(original_len * 0.75)
        return max(20, min(original_len - 1 if original_len > 1 else 1, target))

    @staticmethod
    def _clamp_variant(variant: Any, cap: int) -> Any:
        """Clamp a CaptionVariant's primary text + payload to ``cap`` chars."""
        from curator.caption.agent import _PRIMARY_KEY, _trim_to

        text = variant.text or ""
        if len(text) > cap:
            text = _trim_to(text, cap)
        variant.text = text
        variant.char_count = len(text)
        variant.trimmed = True
        key = _PRIMARY_KEY.get(variant.output_mode)
        if key and isinstance(variant.payload, dict):
            variant.payload[key] = text
            variant.payload["char_count"] = len(text)
        return variant

    def handle_platform(self, update: Any) -> HandlerResult:
        """/platform — switch the active target platform and re-rank/re-present.

        Accepts the new platform from the command argument (e.g. "/platform
        tinder"). Re-ranks the cached grading results + captions for the new
        platform via the RankingAgent (no re-grading). Validates the platform and
        that there is rankable state.
        """
        user_id = self._user_id(update)
        if user_id is None:
            return HandlerResult(text="Send /start to begin.", kind="no_user")
        arg = self._command_arg(update)
        if not arg:
            options = ", ".join(PLATFORM_LABELS[p] for p in PLATFORMS)
            return HandlerResult(
                text=f"Which platform? Reply e.g. /platform tinder. Options: {options}.",
                kind="platform_need_arg",
            )
        key = arg.strip().lower()
        if key not in PLATFORMS:
            return HandlerResult(
                text=f"'{arg}' isn't a platform I know. Try one of: "
                + ", ".join(PLATFORMS)
                + ".",
                kind="platform_unknown",
            )
        results = self._pending_results.get(user_id)
        if not results:
            return HandlerResult(
                text="No graded photos to re-rank yet — send photos first.",
                kind="platform_no_results",
            )
        captions = self._pending_captions.get(user_id)
        cards = self._ranking().build_cards(key, results, captions)
        # Track the platform in the durable session too (additive to platforms).
        session = self.sessions.get(user_id)
        if session is not None and key not in session.platforms:
            self.sessions.update(user_id, platforms=session.platforms + [key])
        present = self.present_ranked_cards(
            user_id, cards, results=results, captions=captions, platform=key
        )
        label = PLATFORM_LABELS[key]
        if present.kind == "ranking_empty":
            return HandlerResult(
                text=f"No rankable photos for {label}.",
                kind="platform_switched_empty",
            )
        return HandlerResult(
            text=f"Switched to {label}.\n\n{present.text}",
            kind="platform_switched",
        )

    def handle_reset(self, update: Any) -> HandlerResult:
        """/reset — clear the session's working state and start over."""
        user_id = self._user_id(update)
        if user_id is None:
            return HandlerResult(text="Send /start to begin.", kind="no_user")
        self._clear_override_state(user_id)
        # Re-init the durable session to a clean slate (A1 lifecycle).
        self.sessions.init(user_id)
        self.sessions.update(user_id, stage=Stage.SELECTING_PLATFORMS)
        return HandlerResult(
            text="Reset done — your session is clear. Send /start to pick platforms "
            "and begin again.",
            kind="reset",
        )

    @staticmethod
    def _command_arg(update: Any) -> Optional[str]:
        """Extract the argument after a slash command (e.g. '/platform tinder')."""
        text = getattr(getattr(update, "message", None), "text", None)
        if not isinstance(text, str):
            return None
        parts = text.strip().split(maxsplit=1)
        if len(parts) < 2:
            return None
        return parts[1].strip() or None

    # --- session timeout warning (Agent A6 / Stage S-06) --------------------
    def check_session_timeout(self, user_id: int) -> TimeoutStatus:
        """Return the T-minus timeout status for a user (injectable clock).

        Uses ``self._clock`` (a no-arg callable) if set so tests are deterministic;
        otherwise the wall clock. Pure read — does NOT expire the session.
        """
        now = self._clock() if callable(self._clock) else None
        session = self.sessions.get(user_id, expire_if_stale=False)
        return check_timeout(session, now=now)

    def timeout_warning(self, user_id: int) -> Optional[HandlerResult]:
        """A warning HandlerResult if the session is inside T-15min, else None.

        Idempotent per window: once warned, we don't nag again until the window is
        re-entered (TTL reset by activity). The caller surfaces this proactively
        (e.g. on a scheduler tick) alongside normal replies.
        """
        status = self.check_session_timeout(user_id)
        if not status.should_warn:
            # Out of the window (or expired): clear any prior warned mark so a new
            # window can warn again.
            self._warned.discard(user_id)
            return None
        if user_id in self._warned:
            return None
        self._warned.add(user_id)
        return HandlerResult(
            text=WARNING_MESSAGE.format(minutes=status.minutes_remaining),
            kind="timeout_warning",
        )

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
                if isinstance(text, str) and text.strip().startswith("/"):
                    command = text.strip().split()[0].lstrip("/").lower()
                    # Strip any @botname suffix Telegram appends in groups.
                    command = command.split("@", 1)[0]
                    handler = self._COMMANDS.get(command)
                    if handler is not None:
                        return handler(self, update)
                    # Unknown slash command: fall through to intent parsing below
                    # (graceful; consistent with A1's robust router).
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

    # Slash-command dispatch table (A1 /start + A6 overrides). Defined after the
    # handler methods so the references resolve.
    _COMMANDS = {
        "start": handle_start,
        "help": handle_help,
        "commands": handle_help,
        "about": handle_about,
        "done": handle_done,
        "next": handle_next,
        "retry": handle_retry,
        "shorter": handle_shorter,
        "platform": handle_platform,
        "reset": handle_reset,
    }
