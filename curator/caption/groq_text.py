"""Groq Llama 3.3 70B text client for captions (Agent A4 / Stage S-04).

Wraps the single caption-generation chat call behind a small, injectable
``TextGenerator`` protocol so the CaptionAgent's logic (prompt construction,
output-mode switching, banned-phrase rejection, length enforcement, Hinge prompt
selection) is fully testable WITHOUT the ``groq`` package, a ``GROQ_API_KEY``, or
network. Mirrors A3's GroqVisionGrader.

Groq's text models do NOT accept images — the model is given the BLIP-2 scene
description string (A3) as a factual anchor, never raw image bytes.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

# Groq model id for caption generation (PRD Section 03 + Provider Reference).
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"


class GroqRateLimitError(Exception):
    """Raised when Groq returns HTTP 429 (rate limit) on a caption call."""


class TextGenerator(Protocol):
    """A thing that returns raw model text for a (system, user) prompt pair."""

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - protocol
        ...


class GroqTextGenerator:
    """Calls Groq Llama 3.3 70B and returns the raw response text.

    The Groq client is injected (``client``) so tests pass a stub. In production a
    real ``groq.Groq`` instance is created lazily from ``GROQ_API_KEY`` — there is
    NO eager module-level ``import groq`` (keeps the keyless/groqless sandbox
    importable; preserves the A2 zero-API guarantee on the import path).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        model: str = GROQ_TEXT_MODEL,
        max_tokens: int = 400,
        temperature: float = 0.7,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError("Groq text unavailable: GROQ_API_KEY not set")
        try:
            import groq
        except Exception as exc:  # groq not installed (sandbox)
            raise RuntimeError(f"groq package not installed: {exc}") from exc
        self._client = groq.Groq(api_key=self.api_key)
        return self._client

    def generate(self, system: str, user: str) -> str:
        client = self._ensure_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:  # normalise rate-limit -> GroqRateLimitError
            if _is_rate_limit(exc):
                raise GroqRateLimitError(str(exc)) from exc
            raise
        return completion.choices[0].message.content or ""


def _is_rate_limit(exc: Exception) -> bool:
    """Best-effort detection of a Groq 429 across SDK versions."""
    if exc.__class__.__name__ in ("RateLimitError", "GroqRateLimitError"):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    return "429" in str(exc) or "rate limit" in str(exc).lower()
