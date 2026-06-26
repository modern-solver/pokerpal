"""prompts.yml loader + accessors for the CaptionAgent (Agent A4 / Stage S-04).

Loads and lightly validates the caption prompt/policy config so copy, length
caps, tone variants, the banned-phrase list, the Hinge prompt library, and the
LinkedIn tag policy all tune from YAML WITHOUT a code change (mirrors A2's
weights.py). Pure local I/O — no network, no model call, no groq import.

Output modes (PRD rule C-08):
    post_caption  -> instagram, facebook
    dating_bio    -> tinder, bumble
    prompt_answer -> hinge
    linkedin_post -> linkedin
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yaml

from ..session.schema import PLATFORMS

# Default config path: curator/config/prompts.yml relative to this file.
DEFAULT_PROMPTS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "config", "prompts.yml")
)

# The 4 output modes and which platforms map to each (PRD Section 03 table).
OUTPUT_MODES = ("post_caption", "dating_bio", "prompt_answer", "linkedin_post")

# Canonical output_mode per platform (rule C-08). The loader cross-checks the
# YAML against this so a config typo surfaces loudly.
PLATFORM_OUTPUT_MODE: Dict[str, str] = {
    "instagram": "post_caption",
    "facebook": "post_caption",
    "tinder": "dating_bio",
    "bumble": "dating_bio",
    "hinge": "prompt_answer",
    "linkedin": "linkedin_post",
}

DATING_PLATFORMS = ("tinder", "bumble")


def output_mode_for(platform: str) -> str:
    """Switch output mode based on platform (rule C-08)."""
    try:
        return PLATFORM_OUTPUT_MODE[platform]
    except KeyError as exc:
        raise ValueError(f"unknown platform: {platform!r}") from exc


def load_prompts(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate prompts.yml.

    Raises ValueError on a malformed config (missing platform, wrong/absent
    output_mode, empty tone list, missing shared blocks) so typos fail loud.
    """
    path = path or DEFAULT_PROMPTS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for key in ("architecture", "json_shapes", "banned_phrases", "hinge_prompts",
                "platforms", "linkedin"):
        if key not in data:
            raise ValueError(f"prompts.yml missing top-level block: {key!r}")

    arch = data["architecture"]
    if not arch.get("system") or not arch.get("user_template"):
        raise ValueError("prompts.yml architecture needs 'system' and 'user_template'")

    shapes = data["json_shapes"]
    for mode in OUTPUT_MODES:
        if mode not in shapes:
            raise ValueError(f"prompts.yml json_shapes missing mode: {mode!r}")

    platforms = data["platforms"]
    for platform in PLATFORMS:
        if platform not in platforms:
            raise ValueError(f"prompts.yml missing platform: {platform!r}")
        profile = platforms[platform]
        mode = profile.get("output_mode")
        expected = PLATFORM_OUTPUT_MODE[platform]
        if mode != expected:
            raise ValueError(
                f"platform {platform!r} output_mode is {mode!r}, expected {expected!r}"
            )
        if not profile.get("tones"):
            raise ValueError(f"platform {platform!r} has no tone variants")
        if not profile.get("max_length"):
            raise ValueError(f"platform {platform!r} missing max_length")

    if not data["hinge_prompts"]:
        raise ValueError("prompts.yml hinge_prompts is empty")

    return data


@lru_cache(maxsize=4)
def get_prompts(path: Optional[str] = None) -> Dict[str, Any]:
    """Cached accessor for the default (or given) prompts file."""
    return load_prompts(path)


# --- small typed accessors --------------------------------------------------
def platform_config(prompts: Dict[str, Any], platform: str) -> Dict[str, Any]:
    try:
        return prompts["platforms"][platform]
    except KeyError as exc:
        raise ValueError(f"unknown platform: {platform!r}") from exc


def tones_for(prompts: Dict[str, Any], platform: str) -> List[str]:
    """Ordered tone variants for a platform (order = /retry cycle, rule C-07)."""
    return list(platform_config(prompts, platform)["tones"])


def max_length_for(prompts: Dict[str, Any], platform: str) -> int:
    return int(platform_config(prompts, platform)["max_length"])


def hashtag_policy_for(prompts: Dict[str, Any], platform: str) -> Optional[Dict[str, int]]:
    """{'min': m, 'max': n} or None when hashtags do not apply."""
    return platform_config(prompts, platform).get("hashtags")


def banned_phrases(prompts: Dict[str, Any]) -> List[str]:
    return [str(p).lower() for p in prompts.get("banned_phrases", [])]


def hinge_prompt_library(prompts: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(prompts.get("hinge_prompts", []))


def linkedin_policy(prompts: Dict[str, Any]) -> Dict[str, Any]:
    return dict(prompts.get("linkedin", {}))
