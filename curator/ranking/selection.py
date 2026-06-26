"""Selection handling — photo pick + variant pick (Agent A5 / Stage S-05).

Closes the loop: the user is shown a set of numbered ``OutputCard``s, picks a
photo (card number), then picks one of its 3 numbered caption variants. This
module validates BOTH picks (graceful on out-of-range / garbage input, mirroring
A1's router robustness) and assembles the post-ready ``FinalCard``.

Pure / token-free. It also exposes the hooks A6's override commands will call
(``cycle_variant`` for /retry, ``select`` for the picks) without implementing the
commands themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

from .card import FinalCard, OutputCard, RankedPhoto, render_variant


@dataclass
class SelectionError:
    """A graceful, non-raising selection failure (out-of-range / unparseable)."""

    reason: str
    message: str

    ok: bool = False


def parse_choice(raw: Union[str, int, None]) -> Optional[int]:
    """Parse a 1-based numbered choice from user input. None if unparseable.

    Accepts an int, or a string that contains a number (e.g. "2", "photo 2",
    "#3"). Returns the integer, or None for garbage — the caller decides the
    graceful message. Never raises.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):  # guard: bool is an int subclass
        return None
    if isinstance(raw, int):
        return raw
    if not isinstance(raw, str):
        return None
    digits = ""
    for ch in raw.strip():
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def select_photo(
    cards: Sequence[OutputCard], raw: Union[str, int, None]
) -> Union[OutputCard, SelectionError]:
    """Validate a photo pick (1-based card number) against the presented cards."""
    if not cards:
        return SelectionError("no_cards", "There are no photos to choose from yet.")
    choice = parse_choice(raw)
    if choice is None:
        return SelectionError(
            "unparseable",
            f"Pick a photo by number (1-{len(cards)}).",
        )
    if not (1 <= choice <= len(cards)):
        return SelectionError(
            "out_of_range",
            f"That's not one of the options — pick 1-{len(cards)}.",
        )
    return cards[choice - 1]


def select_variant(
    photo: RankedPhoto, raw: Union[str, int, None]
) -> Union[int, SelectionError]:
    """Validate a variant pick (1-based) for a chosen photo's variants.

    Returns the 0-based index into ``photo.variants`` on success, else a
    SelectionError.
    """
    variants = photo.variants
    if not variants:
        return SelectionError(
            "no_variants", "No caption options are available for that photo."
        )
    choice = parse_choice(raw)
    if choice is None:
        return SelectionError(
            "unparseable", f"Pick a caption by number (1-{len(variants)})."
        )
    if not (1 <= choice <= len(variants)):
        return SelectionError(
            "out_of_range",
            f"That's not one of the captions — pick 1-{len(variants)}.",
        )
    return choice - 1


def build_final_card(photo: RankedPhoto, variant_index_zero_based: int) -> FinalCard:
    """Assemble the post-ready FinalCard from a chosen photo + variant index."""
    variant = photo.variants[variant_index_zero_based]
    return FinalCard(
        platform=photo.platform,
        photo_index=photo.photo_index,
        variant_index=variant_index_zero_based + 1,
        tone=variant.tone,
        output_mode=variant.output_mode,
        rendered=render_variant(variant),
        payload=dict(variant.payload),
        file_id=photo.file_id,
        score=photo.score,
        rationale=photo.rationale,
    )


def select(
    cards: Sequence[OutputCard],
    photo_choice: Union[str, int, None],
    variant_choice: Union[str, int, None],
) -> Union[FinalCard, SelectionError]:
    """One-shot: validate both picks and return the post-ready FinalCard.

    Returns a SelectionError (never raises) on any invalid pick so the bot can
    reply gracefully and re-prompt.
    """
    chosen = select_photo(cards, photo_choice)
    if isinstance(chosen, SelectionError):
        return chosen
    vidx = select_variant(chosen.photo, variant_choice)
    if isinstance(vidx, SelectionError):
        return vidx
    return build_final_card(chosen.photo, vidx)


# --- hooks for A6 override commands (NOT implemented here) -------------------
def cycle_variant(photo: RankedPhoto, current_zero_based: int) -> int:
    """Hook for A6's /retry (rule C-07): step to the next tone variant in order.

    Returns the next 0-based variant index, wrapping around. A6 wires the actual
    /retry command to this; A5 only exposes the cursor logic so the cycle order
    (the A4 variant list order) is honoured in one place.
    """
    n = len(photo.variants)
    if n == 0:
        return 0
    return (current_zero_based + 1) % n
