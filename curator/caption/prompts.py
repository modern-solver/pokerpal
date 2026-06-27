"""prompts.yml loader + accessors for the CaptionAgent.

Loads and lightly validates the caption prompt config so the system/user prompt
and the per-length specs (short / long / haiku) tune from YAML WITHOUT a code
change. Pure local I/O — no network, no model call, no groq import.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_PROMPTS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "config", "prompts.yml")
)

# The caption length categories the user can pick.
LENGTH_MODES = ("short", "long", "haiku")
LENGTH_LABELS = {"short": "Short", "long": "Long", "haiku": "Haiku"}
DEFAULT_LENGTH = "short"


def load_prompts(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate prompts.yml. Raises ValueError on a malformed config."""
    path = path or DEFAULT_PROMPTS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    cap = data.get("caption")
    if not isinstance(cap, dict):
        raise ValueError("prompts.yml missing top-level 'caption' block")
    if not cap.get("system") or not cap.get("user_template"):
        raise ValueError("caption block needs 'system' and 'user_template'")
    lengths = cap.get("lengths")
    if not isinstance(lengths, dict):
        raise ValueError("caption block missing 'lengths'")
    for mode in LENGTH_MODES:
        spec = lengths.get(mode)
        if not isinstance(spec, dict):
            raise ValueError(f"caption.lengths missing mode: {mode!r}")
        if not spec.get("instruction") or not spec.get("max_chars"):
            raise ValueError(f"caption.lengths.{mode} needs 'instruction' + 'max_chars'")
    return data


@lru_cache(maxsize=4)
def get_prompts(path: Optional[str] = None) -> Dict[str, Any]:
    """Cached accessor for the default (or given) prompts file."""
    return load_prompts(path)


def normalize_length(mode: Optional[str]) -> str:
    """Coerce any input to a valid length mode (default 'short')."""
    m = (mode or "").strip().lower()
    return m if m in LENGTH_MODES else DEFAULT_LENGTH


def length_spec(prompts: Dict[str, Any], mode: Optional[str]) -> Dict[str, Any]:
    """Resolved spec for a length mode: {mode, label, max_chars, instruction}."""
    mode = normalize_length(mode)
    raw = prompts["caption"]["lengths"][mode]
    return {
        "mode": mode,
        "label": LENGTH_LABELS[mode],
        "max_chars": int(raw["max_chars"]),
        "instruction": str(raw["instruction"]).strip(),
    }


def caption_prompts(prompts: Dict[str, Any]) -> Dict[str, str]:
    """The system + user_template strings."""
    cap = prompts["caption"]
    return {
        "system": str(cap["system"]).strip(),
        "user_template": str(cap["user_template"]),
    }


def max_chars_for(prompts: Dict[str, Any], mode: Optional[str]) -> int:
    return length_spec(prompts, mode)["max_chars"]


def length_modes() -> List[str]:
    return list(LENGTH_MODES)
