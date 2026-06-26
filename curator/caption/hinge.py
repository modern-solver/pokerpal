"""Hinge prompt selection (Agent A4 / Stage S-04, rule C-10).

For Hinge the CaptionAgent does NOT invent a prompt — it SELECTS the standard
library prompt (PRD Section 06) whose mood best matches the BLIP-2 scene
description. This module is the pure, fully-testable selector: given the library
(from prompts.yml) + a scene description, it returns the chosen prompt string.

Selection is keyword overlap between the scene description and each library
entry's ``mood_keywords``; ties break by library order; nothing matches -> the
first library entry (the neutral default). The chosen prompt is always one drawn
from the library — never fabricated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def select_hinge_prompt(
    library: List[Dict[str, Any]],
    scene_description: Optional[str],
) -> str:
    """Return the best-matching library prompt string for the scene mood.

    Raises ValueError on an empty library (config error — Hinge cannot run).
    """
    if not library:
        raise ValueError("Hinge prompt library is empty")

    text = (scene_description or "").lower()
    best_idx = 0
    best_score = -1
    for idx, entry in enumerate(library):
        keywords = [str(k).lower() for k in entry.get("mood_keywords", [])]
        score = sum(1 for kw in keywords if kw and kw in text)
        if score > best_score:
            best_score = score
            best_idx = idx
    return str(library[best_idx]["prompt"])


def is_library_prompt(library: List[Dict[str, Any]], prompt: str) -> bool:
    """True iff ``prompt`` is one of the standard library prompts (C-10 guard)."""
    return any(str(e.get("prompt")) == prompt for e in library)
