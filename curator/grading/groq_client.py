"""Groq Llama 4 Scout vision client (Agent A3 / Stage S-03).

Wraps the single structured Tier-3 grading call behind a small, injectable
``VisionGrader`` protocol so the Tier-3 logic (parse / retry / fallback) can be
tested WITHOUT the ``groq`` package, a ``GROQ_API_KEY``, or network.

The prompt is the EXACT template from the PRD (Section 02, "Grading Prompt
Template (Tier 3)"): a SYSTEM message demanding JSON-only output, and a user
message asking for a 1-10 score on four axes (composition / lighting / subject /
mood). The image is sent as a base64 data URL on the vision content part.

Rate-limit signalling: a Groq 429 surfaces as ``GroqRateLimitError`` so the
Tier-3 orchestrator can trigger the HF Serverless BLIP-2 fallback (rule G-07).
"""

from __future__ import annotations

import base64
from typing import Any, Optional, Protocol

# Groq model id for the vision model (PRD: "llama-4-scout" on GroqCloud).
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# PRD Section 02 — Grading Prompt Template (Tier 3). Kept verbatim.
GRADING_SYSTEM_PROMPT = (
    "You are a photo quality assessor. Score the image on four axes. "
    "Respond ONLY with JSON — no prose, no explanation."
)
GRADING_USER_PROMPT = (
    "Score this photo from 1-10 on each axis.\n"
    "{\n"
    '  "composition": <1-10>,  // rule of thirds, subject clarity, framing\n'
    '  "lighting": <1-10>,     // exposure, shadow detail, golden hour bonus\n'
    '  "subject": <1-10>,      // sharpness of main subject, focus accuracy\n'
    '  "mood": <1-10>          // emotional impact, colour harmony, atmosphere\n'
    "}"
)


class GroqRateLimitError(Exception):
    """Raised when Groq returns HTTP 429 (rate limit). Triggers HF fallback."""


class VisionGrader(Protocol):
    """A thing that returns the raw model text for a vision grading prompt."""

    def grade(self, image_bytes: bytes) -> str:  # pragma: no cover - protocol
        ...


def _to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


class GroqVisionGrader:
    """Calls Groq Llama 4 Scout vision and returns the raw response text.

    The Groq client is injected (``client``) so tests pass a stub. In production a
    real ``groq.Groq`` instance is created lazily from ``GROQ_API_KEY``. This class
    does the API call ONLY — JSON parsing, retry, and fallback live in ``tier3``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        model: str = GROQ_VISION_MODEL,
        max_tokens: int = 200,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError("Groq vision unavailable: GROQ_API_KEY not set")
        try:
            import groq
        except Exception as exc:  # groq not installed (sandbox)
            raise RuntimeError(f"groq package not installed: {exc}") from exc
        self._client = groq.Groq(api_key=self.api_key)
        return self._client

    def grade(self, image_bytes: bytes) -> str:
        client = self._ensure_client()
        data_url = _to_data_url(image_bytes)
        messages = [
            {"role": "system", "content": GRADING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GRADING_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
        except Exception as exc:  # normalise rate-limit -> GroqRateLimitError
            if _is_rate_limit(exc):
                raise GroqRateLimitError(str(exc)) from exc
            raise
        return completion.choices[0].message.content or ""

    def describe(self, image_bytes: bytes) -> str:
        """One-sentence factual scene description for caption grounding.

        Uses the same vision model as grading (no torch / BLIP-2 needed, which
        suits the free-tier deploy). Returns "" on any failure so captioning can
        proceed ungrounded rather than break the conversation.
        """
        try:
            client = self._ensure_client()
            data_url = _to_data_url(image_bytes)
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You describe photos in one short, factual sentence. "
                            "No preamble, no quotes."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe this photo in one short sentence: "
                                    "the subject, the setting, and the mood."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                max_tokens=60,
                temperature=0.2,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001 - description is best-effort
            return ""


def _is_rate_limit(exc: Exception) -> bool:
    """Best-effort detection of a Groq 429 across SDK versions."""
    if exc.__class__.__name__ in ("RateLimitError", "GroqRateLimitError"):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    return "429" in str(exc) or "rate limit" in str(exc).lower()
