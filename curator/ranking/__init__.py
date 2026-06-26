"""RankingAgent package — weighted composite ranking + output composer.

OWNER: Agent A5 (Stage S-05). Computes the ranking from
``GradingResult.composite`` (the A3 source of truth — NOT re-derived), ranks
top-N with a rationale line built from the score breakdown, builds the BotAgent
output card (photo + rationale + 3 numbered variants), and handles user
selection (photo pick + variant pick), returning the post-ready final card.

Public API:
    RankingAgent          — rank(...) / build_cards(...)
    RankedPhoto, OutputCard, FinalCard — data structures
    render_variant        — render a caption variant from its mode payload
    select / select_photo / select_variant / build_final_card — selection handler
    cycle_variant         — hook for A6's /retry tone-cycle (C-07)
    build_breakdown / rationale_line — the score-breakdown -> rationale derivation
"""

from .agent import DEFAULT_TOP_N, RankingAgent
from .card import FinalCard, OutputCard, RankedPhoto, render_variant
from .rationale import build_breakdown, rationale_line
from .selection import (
    SelectionError,
    build_final_card,
    cycle_variant,
    parse_choice,
    select,
    select_photo,
    select_variant,
)

__all__ = [
    "RankingAgent",
    "DEFAULT_TOP_N",
    "RankedPhoto",
    "OutputCard",
    "FinalCard",
    "render_variant",
    "build_breakdown",
    "rationale_line",
    "select",
    "select_photo",
    "select_variant",
    "build_final_card",
    "cycle_variant",
    "parse_choice",
    "SelectionError",
]
