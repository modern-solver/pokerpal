"""CaptionAgent package — platform-specific caption/bio/prompt/post generation.

OWNER: Agent A4 (Stage S-04). Produces the CORRECT OUTPUT TYPE per platform
(rule C-08) from the A3 BLIP-2 scene description + session context, using Groq
Llama 3.3 70B (``llama-3.3-70b-versatile``):

    post_caption  (IG/FB)         -> {caption, hashtags, char_count}
    dating_bio    (Tinder/Bumble) -> {bio, char_count}
    prompt_answer (Hinge)         -> {prompt, answer, char_count}   (rule C-10)
    linkedin_post (LinkedIn)      -> {post, hashtags, char_count}    (rule C-11)

Three tone variants per photo (PRD Section 03), with banned-phrase rejection for
dating bios (C-09), Hinge library prompt selection (C-10), LinkedIn tag policy
(C-11), and length-cap enforcement with retry+trim. The Groq text client is
injected/mockable — all tests run without a key, package, or network. A5 consumes
``CaptionResult`` (the 3 variants) alongside A3 composite scores to build ranked
output cards.
"""

from .agent import CaptionAgent, MAX_ATTEMPTS
from .groq_text import (
    GROQ_TEXT_MODEL,
    GroqRateLimitError,
    GroqTextGenerator,
    TextGenerator,
)
from .hinge import is_library_prompt, select_hinge_prompt
from .prompts import (
    DEFAULT_PROMPTS_PATH,
    OUTPUT_MODES,
    PLATFORM_OUTPUT_MODE,
    banned_phrases,
    get_prompts,
    hashtag_policy_for,
    hinge_prompt_library,
    linkedin_policy,
    load_prompts,
    max_length_for,
    output_mode_for,
    tones_for,
)
from .result import CaptionResult, CaptionVariant

__all__ = [
    "CaptionAgent",
    "MAX_ATTEMPTS",
    "CaptionResult",
    "CaptionVariant",
    "GroqTextGenerator",
    "TextGenerator",
    "GroqRateLimitError",
    "GROQ_TEXT_MODEL",
    "select_hinge_prompt",
    "is_library_prompt",
    "load_prompts",
    "get_prompts",
    "output_mode_for",
    "tones_for",
    "max_length_for",
    "hashtag_policy_for",
    "banned_phrases",
    "hinge_prompt_library",
    "linkedin_policy",
    "OUTPUT_MODES",
    "PLATFORM_OUTPUT_MODE",
    "DEFAULT_PROMPTS_PATH",
]
