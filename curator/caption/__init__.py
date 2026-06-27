"""CaptionAgent package — length-category caption generation (no hashtags).

Produces 3 distinct, enriched captions per photo in the user's chosen length
category (short < 200 / long < 1000 / haiku), grounded in the A3 scene
description + session vibe, using Groq Llama 3.3 70B. The Groq text client is
injected/mockable — all tests run without a key, package, or network.
"""

from .agent import CaptionAgent, N_OPTIONS
from .groq_text import (
    GROQ_TEXT_MODEL,
    GroqRateLimitError,
    GroqTextGenerator,
    TextGenerator,
)
from .prompts import (
    DEFAULT_LENGTH,
    DEFAULT_PROMPTS_PATH,
    LENGTH_LABELS,
    LENGTH_MODES,
    get_prompts,
    length_spec,
    load_prompts,
    normalize_length,
)
from .result import CaptionResult, CaptionVariant

__all__ = [
    "CaptionAgent",
    "N_OPTIONS",
    "CaptionResult",
    "CaptionVariant",
    "GroqTextGenerator",
    "TextGenerator",
    "GroqRateLimitError",
    "GROQ_TEXT_MODEL",
    "load_prompts",
    "get_prompts",
    "length_spec",
    "normalize_length",
    "LENGTH_MODES",
    "LENGTH_LABELS",
    "DEFAULT_LENGTH",
    "DEFAULT_PROMPTS_PATH",
]
