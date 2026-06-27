"""CaptionAgent — length-category caption generation.

Generates THREE distinct, enriched captions for a photo in the user's chosen
length category (short < 200 / long < 1000 / haiku), grounded in the scene
description + session vibe. NEVER emits hashtags. The Groq Llama 3.3 70B text
client is INJECTED (``TextGenerator``) so every path is unit-tested with a stub —
no ``groq`` package, no ``GROQ_API_KEY``, no network. When no generator is
present the agent returns a deterministic local fallback.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ..session.schema import Session
from .groq_text import GroqRateLimitError, TextGenerator
from .prompts import caption_prompts, get_prompts, length_spec
from .result import CaptionResult, CaptionVariant

N_OPTIONS = 3


class CaptionAgent:
    """Generates 3 length-category captions (no hashtags) from a scene.

    Args:
        generator: injected Groq text client (stub in tests). None -> local
            deterministic fallback (never imports groq).
        prompts: parsed prompts.yml (defaults to the cached on-disk config).
    """

    def __init__(
        self,
        generator: Optional[TextGenerator] = None,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.generator = generator
        self.prompts = prompts or get_prompts()

    def generate(
        self,
        platform: str,
        scene_description: str,
        *,
        session: Optional[Session] = None,
        vibe: Optional[str] = None,
        constraints: Optional[Sequence[str]] = None,
        photo_index: int = 0,
        length_mode: Optional[str] = None,
    ) -> CaptionResult:
        """Generate 3 captions for one photo in the chosen length category."""
        mode = length_mode if length_mode is not None else (
            session.length_mode if session else None
        )
        spec = length_spec(self.prompts, mode)
        vibe = vibe if vibe is not None else (session.vibe if session else None)
        constraints = list(
            constraints
            if constraints is not None
            else (session.constraints if session else [])
        )

        system, user = self._build_prompt(platform, spec, scene_description, vibe, constraints)
        captions, source = self._call_model(system, user, spec, scene_description)

        result = CaptionResult(
            platform=platform,
            length_mode=spec["mode"],
            max_length=spec["max_chars"],
            scene_description=scene_description,
            photo_index=photo_index,
        )
        for i in range(N_OPTIONS):
            raw = captions[i] if i < len(captions) else ""
            text = _clean(raw) or _clean(_fallback_caption(spec, scene_description, i))
            trimmed = False
            if len(text) > spec["max_chars"]:
                text = _trim_to(text, spec["max_chars"])
                trimmed = True
            result.variants.append(
                CaptionVariant(
                    label=f"Option {i + 1}",
                    length_mode=spec["mode"],
                    text=text,
                    char_count=len(text),
                    payload={"caption": text, "char_count": len(text)},
                    trimmed=trimmed,
                    source=source,
                )
            )
        return result

    # --- prompt construction ------------------------------------------------
    def _build_prompt(
        self,
        platform: str,
        spec: Dict[str, Any],
        scene_description: str,
        vibe: Optional[str],
        constraints: List[str],
    ) -> tuple[str, str]:
        cp = caption_prompts(self.prompts)
        user = cp["user_template"].format(
            platform=platform,
            label=spec["label"],
            instruction=spec["instruction"],
            scene=scene_description or "(no scene description available)",
            vibe=vibe or "(none given)",
            constraints=", ".join(constraints) if constraints else "(none)",
        )
        return cp["system"], user

    # --- model call (injected; deterministic fallback when no generator) -----
    def _call_model(
        self, system: str, user: str, spec: Dict[str, Any], scene: str
    ) -> tuple[List[str], str]:
        """Return (captions, source). Tolerates one parse-retry, then falls back."""
        if self.generator is None:
            return self._fallback_list(spec, scene), "fallback"
        try:
            raw = self.generator.generate(system, user)
        except (GroqRateLimitError, Exception):  # noqa: BLE001 - degrade gracefully
            return self._fallback_list(spec, scene), "fallback"
        caps = _parse_captions(raw)
        if not caps:
            try:
                caps = _parse_captions(self.generator.generate(system, user))
            except Exception:  # noqa: BLE001
                caps = []
        if not caps:
            return self._fallback_list(spec, scene), "fallback"
        return caps, "groq"

    @staticmethod
    def _fallback_list(spec: Dict[str, Any], scene: str) -> List[str]:
        return [_fallback_caption(spec, scene, i) for i in range(N_OPTIONS)]


# --- module-level pure helpers ----------------------------------------------
def _fallback_caption(spec: Dict[str, Any], scene: str, i: int) -> str:
    """Deterministic, model-free caption (always within cap, no hashtags)."""
    anchor = (scene or "this moment").strip().rstrip(".")[:80]
    if spec["mode"] == "haiku":
        return f"{anchor.capitalize()}\nlight and shadow softly meet\nthe moment holds still"
    prefixes = ("", "Caught in the in-between — ", "A quiet kind of ")
    return (prefixes[i % len(prefixes)] + anchor).strip().capitalize()


def _clean(text: Any) -> str:
    """Strip hashtags, surrounding quotes, and extra whitespace from a caption."""
    s = str(text or "")
    s = re.sub(r"#\w+", "", s)          # remove any hashtags
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = s.strip().strip('"').strip("'").strip()
    # collapse 3+ newlines but keep haiku line breaks
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _parse_captions(raw: str) -> List[str]:
    """Extract a list of caption strings from a model response (fence-tolerant)."""
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    caps = data.get("captions") if isinstance(data, dict) else None
    if not isinstance(caps, list):
        return []
    return [str(c) for c in caps if str(c).strip()]


def _trim_to(text: str, max_len: int) -> str:
    """Trim text to <= max_len, preferring a word boundary, no trailing junk."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    space = cut.rfind(" ")
    if space >= max_len - 15 and space > 0:
        cut = cut[:space]
    return cut.rstrip(" ,.-")
