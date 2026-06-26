"""Intent parsing for Curator Bot v1.1 (Stage S-01, Agent A1).

Parses free-text user input into a structured `ParsedIntent`:
  * platforms  — multi-select of the 6 supported platforms (alias-aware).
  * vibe       — free-text mood/theme (the residual text after structured tokens).
  * constraints — normalized constraint tokens (e.g. "no_location_names",
                  "max_chars:100").

Parsing is permissive: malformed / empty / None input never raises — it yields an
empty intent. Constraint extraction here is deliberately rule-based (zero API cost);
later stages may enrich it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# Aliases mapped to canonical platform keys.
_PLATFORM_ALIASES = {
    "instagram": "instagram",
    "insta": "instagram",
    "ig": "instagram",
    "facebook": "facebook",
    "fb": "facebook",
    "meta": "facebook",
    "tinder": "tinder",
    "bumble": "bumble",
    "hinge": "hinge",
    "linkedin": "linkedin",
    "li": "linkedin",
}


@dataclass
class ParsedIntent:
    platforms: List[str] = field(default_factory=list)
    vibe: Optional[str] = None
    constraints: List[str] = field(default_factory=list)


def parse_platforms(text: str) -> List[str]:
    """Extract canonical platform keys from text (de-duplicated, order-preserving)."""
    if not text:
        return []
    found: List[str] = []
    # word-boundary token scan so "li" doesn't match inside "lighting".
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    for tok in tokens:
        canonical = _PLATFORM_ALIASES.get(tok)
        if canonical and canonical not in found:
            found.append(canonical)
    return found


def parse_constraints(text: str) -> List[str]:
    """Extract normalized constraint tokens from free text.

    Recognized patterns (case-insensitive):
      * "no location names" / "no locations"     -> "no_location_names"
      * "no hashtags" / "without hashtags"        -> "no_hashtags"
      * "no emoji" / "no emojis"                  -> "no_emoji"
      * "under/below/max N chars/characters"      -> "max_chars:N"
    """
    if not text:
        return []
    low = text.lower()
    constraints: List[str] = []

    if re.search(r"\bno\s+location", low) or "no locations" in low:
        constraints.append("no_location_names")
    if re.search(r"\b(no|without)\s+hashtag", low):
        constraints.append("no_hashtags")
    if re.search(r"\b(no|without)\s+emoji", low):
        constraints.append("no_emoji")

    m = re.search(
        r"\b(?:under|below|max(?:imum)?|less than|<=?)\s*(\d{1,4})\s*(?:char|character)",
        low,
    )
    if m:
        constraints.append(f"max_chars:{int(m.group(1))}")

    # de-duplicate, preserve order
    seen = set()
    out = []
    for c in constraints:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def parse_intent(text: Optional[str]) -> ParsedIntent:
    """Parse a full user message into platforms + vibe + constraints.

    Never raises on bad input — None/empty/non-string yields an empty intent.
    The `vibe` is the residual free text after structured tokens are stripped.
    """
    if not isinstance(text, str) or not text.strip():
        return ParsedIntent()

    platforms = parse_platforms(text)
    constraints = parse_constraints(text)

    # Build a vibe by removing recognized platform tokens; leave the rest as the
    # free-text mood/theme. If nothing meaningful remains, vibe stays None.
    vibe_text = text.strip()
    # strip leading "vibe:" / "mood:" labels if present
    vibe_text = re.sub(r"^\s*(vibe|mood|theme)\s*[:\-]\s*", "", vibe_text, flags=re.I)
    # remove standalone platform alias words
    def _strip_platform_words(s: str) -> str:
        def repl(match: "re.Match[str]") -> str:
            return "" if match.group(0).lower() in _PLATFORM_ALIASES else match.group(0)

        return re.sub(r"[a-zA-Z]+", repl, s)

    residual = _strip_platform_words(vibe_text)
    residual = re.sub(r"[\s,;]+", " ", residual).strip(" ,;.-")
    vibe = residual if residual else None

    return ParsedIntent(platforms=platforms, vibe=vibe, constraints=constraints)
