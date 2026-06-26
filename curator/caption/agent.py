"""CaptionAgent — platform-specific output generation (Agent A4 / Stage S-04).

Produces the CORRECT OUTPUT TYPE per platform (rule C-08) — it is NOT always a
caption:
    post_caption  (IG/FB)         -> {caption, hashtags, char_count}
    dating_bio    (Tinder/Bumble) -> {bio, char_count}
    prompt_answer (Hinge)         -> {prompt, answer, char_count}
    linkedin_post (LinkedIn)      -> {post, hashtags, char_count}

For each photo+platform it generates THREE tone variants (PRD Section 03), each:
  * built from the shared prompt architecture (SYSTEM JSON-only; USER carries
    platform / tone / BLIP-2 {blip_desc} / session vibe / constraints), with the
    per-mode JSON shape requested (prompts.yml);
  * length-enforced against the platform cap, retrying then trimming on exceed;
  * for dating bios, rejected + regenerated if any banned phrase appears (C-09);
  * for Hinge, paired with a library-selected prompt — never invented (C-10);
  * for LinkedIn, scrubbed of banned tags and given professional fallbacks (C-11).

The Groq Llama 3.3 70B client is INJECTED (``TextGenerator``) so every path is
unit-tested with a stub — no ``groq`` package, no ``GROQ_API_KEY``, no network.
Groq text models do not accept images: the model sees the BLIP-2 scene
description string (A3), never raw image bytes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ..session.schema import Session
from .groq_text import GroqRateLimitError, TextGenerator
from .hinge import is_library_prompt, select_hinge_prompt
from .prompts import (
    banned_phrases,
    get_prompts,
    hashtag_policy_for,
    hinge_prompt_library,
    linkedin_policy,
    max_length_for,
    output_mode_for,
    platform_config,
    tones_for,
)
from .result import CaptionResult, CaptionVariant

# Max regeneration attempts per variant before we accept a trimmed/cleaned result.
MAX_ATTEMPTS = 3


class CaptionAgent:
    """Generates platform-correct output variants from scene descriptions.

    Args:
        generator: injected Groq Llama 3.3 70B text client (stub in tests). When
            None, the agent runs in a deterministic local-fallback mode (used by
            tests that assert structure without any model) — it never imports groq.
        prompts: parsed prompts.yml (defaults to the cached on-disk config).
    """

    def __init__(
        self,
        generator: Optional[TextGenerator] = None,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.generator = generator
        self.prompts = prompts or get_prompts()
        self._banned = banned_phrases(self.prompts)
        self._hinge_lib = hinge_prompt_library(self.prompts)
        self._linkedin = linkedin_policy(self.prompts)

    # --- public API ---------------------------------------------------------
    def generate(
        self,
        platform: str,
        scene_description: str,
        *,
        session: Optional[Session] = None,
        vibe: Optional[str] = None,
        constraints: Optional[Sequence[str]] = None,
        photo_index: int = 0,
        tones: Optional[Sequence[str]] = None,
    ) -> CaptionResult:
        """Generate all 3 tone variants for one photo on one platform.

        ``vibe`` / ``constraints`` default to the session's when a Session is
        passed. ``tones`` overrides the configured tone order (used by A6's
        /retry tone-cycle, rule C-07) but defaults to the platform's order.
        """
        mode = output_mode_for(platform)
        max_len = max_length_for(self.prompts, platform)
        vibe = vibe if vibe is not None else (session.vibe if session else None)
        constraints = list(
            constraints
            if constraints is not None
            else (session.constraints if session else [])
        )
        tone_order = list(tones) if tones else tones_for(self.prompts, platform)

        result = CaptionResult(
            platform=platform,
            output_mode=mode,
            max_length=max_len,
            scene_description=scene_description,
            photo_index=photo_index,
        )

        # Hinge: select the library prompt ONCE per photo (rule C-10), shared by
        # all tone variants (the photo's mood does not change between tones).
        hinge_prompt: Optional[str] = None
        if mode == "prompt_answer":
            hinge_prompt = select_hinge_prompt(self._hinge_lib, scene_description)

        for tone in tone_order:
            variant = self._generate_variant(
                platform=platform,
                mode=mode,
                tone=tone,
                scene_description=scene_description,
                vibe=vibe,
                constraints=constraints,
                max_len=max_len,
                hinge_prompt=hinge_prompt,
            )
            result.variants.append(variant)

        return result

    # --- per-variant generation with retry/trim/clean -----------------------
    def _generate_variant(
        self,
        *,
        platform: str,
        mode: str,
        tone: str,
        scene_description: str,
        vibe: Optional[str],
        constraints: List[str],
        max_len: int,
        hinge_prompt: Optional[str],
    ) -> CaptionVariant:
        system, user = self._build_prompt(
            platform=platform,
            mode=mode,
            tone=tone,
            scene_description=scene_description,
            vibe=vibe,
            constraints=constraints,
            max_len=max_len,
            hinge_prompt=hinge_prompt,
        )

        last_payload: Optional[Dict[str, Any]] = None
        attempts = 0
        source = "groq"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts = attempt
            raw = self._call_model(
                system, user, mode=mode, tone=tone,
                scene_description=scene_description, hinge_prompt=hinge_prompt,
            )
            if raw.get("_fallback"):
                source = "fallback"
            payload = self._coerce_payload(raw, mode, hinge_prompt)
            last_payload = payload
            text = _primary_text(payload, mode)

            # C-09: dating-bio banned-phrase rejection -> regenerate.
            if mode == "dating_bio" and self._has_banned(text):
                continue
            # Length: prefer a model output already within cap; else retry.
            if len(text) <= max_len:
                break
        else:
            # Exhausted attempts: fall through with last_payload (cleaned+trimmed).
            payload = last_payload or {}

        return self._finalise_variant(
            platform=platform,
            mode=mode,
            tone=tone,
            payload=last_payload or {},
            max_len=max_len,
            hinge_prompt=hinge_prompt,
            attempts=attempts,
            source=source,
        )

    def _finalise_variant(
        self,
        *,
        platform: str,
        mode: str,
        tone: str,
        payload: Dict[str, Any],
        max_len: int,
        hinge_prompt: Optional[str],
        attempts: int,
        source: str,
    ) -> CaptionVariant:
        text = _primary_text(payload, mode)
        hashtags = self._finalise_hashtags(platform, mode, payload)

        # Banned-phrase scrub-of-last-resort (C-09): if a banned phrase survived
        # all retries, strip it so the cap and the no-cliche guarantee both hold.
        if mode == "dating_bio" and self._has_banned(text):
            text = self._strip_banned(text)

        # Length enforcement (final): trim primary text to the cap. For modes with
        # appended hashtags, the cap applies to the primary copy (hashtags append).
        trimmed = False
        if len(text) > max_len:
            text = _trim_to(text, max_len)
            trimmed = True

        prompt_out = hinge_prompt if mode == "prompt_answer" else None
        payload_out = _shape_payload(mode, text, hashtags, prompt_out)

        return CaptionVariant(
            tone=tone,
            output_mode=mode,
            text=text,
            char_count=len(text),
            hashtags=hashtags,
            prompt=prompt_out,
            payload=payload_out,
            attempts=attempts,
            trimmed=trimmed,
            source=source,
        )

    # --- prompt construction ------------------------------------------------
    def _build_prompt(
        self,
        *,
        platform: str,
        mode: str,
        tone: str,
        scene_description: str,
        vibe: Optional[str],
        constraints: List[str],
        max_len: int,
        hinge_prompt: Optional[str],
    ) -> tuple[str, str]:
        arch = self.prompts["architecture"]
        cfg = platform_config(self.prompts, platform)
        json_shape = self.prompts["json_shapes"][mode]

        hashtag_instruction = self._hashtag_instruction(platform, mode)
        extra_lines: List[str] = []
        guidance = cfg.get("guidance")
        if guidance:
            extra_lines.append(str(guidance).strip())
        if mode == "prompt_answer" and hinge_prompt:
            extra_lines.append(
                f'Answer this exact Hinge prompt (do not change it): "{hinge_prompt}"'
            )
        if platform == "linkedin":
            banned = self._linkedin.get("banned_hashtags", [])
            if banned:
                extra_lines.append(
                    "Never use these hashtags: "
                    + ", ".join(f"#{t}" for t in banned)
                    + ". Use professional hashtags only."
                )
            if tone == "Professional":
                extra_lines.append(
                    "Professional tone: do not use the first-person word 'I' and "
                    "no exclamation marks."
                )

        user = arch["user_template"].format(
            platform=platform,
            output_mode=mode,
            tone=tone,
            blip_desc=scene_description or "(no scene description available)",
            vibe=vibe or "(none given)",
            constraints=", ".join(constraints) if constraints else "(none)",
            max_length=max_len,
            hashtag_instruction=hashtag_instruction,
            extra="\n".join(extra_lines),
            json_shape=json_shape,
        )
        return arch["system"], user

    def _hashtag_instruction(self, platform: str, mode: str) -> str:
        policy = hashtag_policy_for(self.prompts, platform)
        if not policy:
            return "Do not include any hashtags."
        return (
            f"Include {policy['min']}-{policy['max']} relevant hashtags in the "
            f"hashtags array."
        )

    # --- model call (injected; deterministic fallback when no generator) -----
    def _call_model(
        self,
        system: str,
        user: str,
        *,
        mode: str,
        tone: str,
        scene_description: str,
        hinge_prompt: Optional[str],
    ) -> Dict[str, Any]:
        """Call the injected Groq text client and parse JSON; tolerate one retry.

        Returns a parsed dict. On a 429 or a hard failure, returns a deterministic
        local fallback payload (marked ``_fallback``) so the pipeline degrades
        gracefully rather than crashing the whole session.
        """
        if self.generator is None:
            return self._local_fallback(mode, tone, scene_description, hinge_prompt)

        raw_text = ""
        try:
            raw_text = self.generator.generate(system, user)
        except GroqRateLimitError:
            return self._local_fallback(mode, tone, scene_description, hinge_prompt)
        except Exception:
            return self._local_fallback(mode, tone, scene_description, hinge_prompt)

        parsed = _parse_json(raw_text)
        if parsed is None:
            # one tolerant retry, then local fallback (mirrors A3's parse retry).
            try:
                raw_text = self.generator.generate(system, user)
            except Exception:
                return self._local_fallback(mode, tone, scene_description, hinge_prompt)
            parsed = _parse_json(raw_text)
        if parsed is None:
            return self._local_fallback(mode, tone, scene_description, hinge_prompt)
        return parsed

    def _local_fallback(
        self,
        mode: str,
        tone: str,
        scene_description: str,
        hinge_prompt: Optional[str],
    ) -> Dict[str, Any]:
        """Deterministic, model-free payload. Conservative + always within cap."""
        anchor = (scene_description or "this moment").strip().rstrip(".")
        anchor = anchor[:60]
        out: Dict[str, Any] = {"_fallback": True}
        if mode == "post_caption":
            out["caption"] = anchor.capitalize()
            out["hashtags"] = []
        elif mode == "dating_bio":
            out["bio"] = anchor.capitalize()
        elif mode == "prompt_answer":
            out["prompt"] = hinge_prompt or ""
            out["answer"] = anchor.capitalize()
        elif mode == "linkedin_post":
            out["post"] = anchor.capitalize()
            out["hashtags"] = []
        return out

    # --- payload coercion + hashtag/banned helpers --------------------------
    def _coerce_payload(
        self, raw: Dict[str, Any], mode: str, hinge_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Normalise a parsed model payload into the canonical key for the mode."""
        out = dict(raw)
        if mode == "prompt_answer":
            # Force the library-selected prompt; the model only writes the answer.
            out["prompt"] = hinge_prompt or out.get("prompt", "")
        return out

    def _has_banned(self, text: str) -> bool:
        low = (text or "").lower()
        return any(p in low for p in self._banned)

    def _strip_banned(self, text: str) -> str:
        out = text
        for phrase in self._banned:
            out = re.sub(re.escape(phrase), "", out, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", out).strip(" ,.-")

    def _finalise_hashtags(
        self, platform: str, mode: str, payload: Dict[str, Any]
    ) -> List[str]:
        policy = hashtag_policy_for(self.prompts, platform)
        if not policy:
            return []  # dating + Hinge: never any hashtags
        tags = payload.get("hashtags") or []
        tags = [_norm_tag(t) for t in tags if str(t).strip()]

        if platform == "linkedin":
            banned = {t.lower().lstrip("#") for t in self._linkedin.get("banned_hashtags", [])}
            tags = [t for t in tags if t.lower().lstrip("#") not in banned]
            # Top up with professional fallbacks if below the minimum (C-11).
            for fb in self._linkedin.get("professional_hashtags", []):
                if len(tags) >= policy["min"]:
                    break
                ntag = _norm_tag(fb)
                if ntag.lower() not in {x.lower() for x in tags}:
                    tags.append(ntag)

        # de-dup preserving order, then cap to the platform max.
        seen: set = set()
        deduped: List[str] = []
        for t in tags:
            if t.lower() not in seen:
                seen.add(t.lower())
                deduped.append(t)
        return deduped[: policy["max"]]


# --- module-level pure helpers ----------------------------------------------
_PRIMARY_KEY = {
    "post_caption": "caption",
    "dating_bio": "bio",
    "prompt_answer": "answer",
    "linkedin_post": "post",
}


def _primary_text(payload: Dict[str, Any], mode: str) -> str:
    return str(payload.get(_PRIMARY_KEY[mode], "") or "").strip()


def _shape_payload(
    mode: str, text: str, hashtags: List[str], prompt: Optional[str]
) -> Dict[str, Any]:
    """Build the canonical mode-shaped JSON payload A5 renders."""
    if mode == "post_caption":
        return {"caption": text, "hashtags": list(hashtags), "char_count": len(text)}
    if mode == "dating_bio":
        return {"bio": text, "char_count": len(text)}
    if mode == "prompt_answer":
        return {"prompt": prompt or "", "answer": text, "char_count": len(text)}
    if mode == "linkedin_post":
        return {"post": text, "hashtags": list(hashtags), "char_count": len(text)}
    raise ValueError(f"unknown output mode: {mode!r}")


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a raw model response (fence-tolerant)."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _norm_tag(tag: Any) -> str:
    """Normalise a hashtag token to '#word' (strip spaces/extra #)."""
    s = str(tag).strip().lstrip("#").strip()
    s = re.sub(r"\s+", "", s)
    return f"#{s}" if s else ""


def _trim_to(text: str, max_len: int) -> str:
    """Trim text to <= max_len, preferring a word boundary, no trailing junk."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    # Prefer not to cut mid-word if a space is reasonably close to the end.
    space = cut.rfind(" ")
    if space >= max_len - 15 and space > 0:
        cut = cut[:space]
    return cut.rstrip(" ,.-")
