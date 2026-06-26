"""Card -> display text formatting (Agent A5 / Stage S-05).

Token-free formatting of an ``OutputCard`` (and the post-ready ``FinalCard``)
into the text the BotAgent sends. Kept separate from the data structures so the
presentation copy lives in one place and is unit-testable without Telegram.

The bot sends the photo by ``file_id`` (handled by the PTB adapter); these
helpers produce the accompanying caption/keyboard text.
"""

from __future__ import annotations

from typing import List

from .card import FinalCard, OutputCard


def format_card(card: OutputCard) -> str:
    """Render one ranked card: header + rationale + numbered caption variants."""
    photo = card.photo
    lines: List[str] = [
        f"Photo {card.card_index} (rank #{photo.rank}, {photo.platform}) — "
        f"score {photo.score:.1f}/10",
        photo.rationale,
        "",
        "Caption options:",
    ]
    for i, rendered in enumerate(card.rendered_variants, start=1):
        tone = photo.variants[i - 1].tone if i - 1 < len(photo.variants) else ""
        tone_tag = f" [{tone}]" if tone else ""
        lines.append(f"{i}.{tone_tag} {rendered}")
    return "\n".join(lines)


def format_cards(cards: List[OutputCard]) -> str:
    """Render the full ranked set the user chooses from."""
    if not cards:
        return "No rankable photos in this session."
    header = (
        f"Here are your top {len(cards)} photos. "
        "Reply with a photo number to pick one."
    )
    blocks = [header, ""]
    for card in cards:
        blocks.append(format_card(card))
        blocks.append("")
    return "\n".join(blocks).rstrip()


def format_variant_prompt(card: OutputCard) -> str:
    """Prompt shown after a photo pick: choose one of the numbered captions."""
    n = len(card.photo.variants)
    return (
        f"You picked photo {card.card_index}. "
        f"Now reply with a caption number (1-{n})."
    )


def format_final_card(final: FinalCard) -> str:
    """Render the confirmed, post-ready card."""
    return (
        f"Post-ready for {final.platform}:\n\n"
        f"{final.rendered}\n\n"
        f"(photo score {final.score:.1f}/10 · {final.tone} tone · "
        f"variant {final.variant_index})"
    )
