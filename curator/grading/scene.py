"""BLIP-2 scene description (Agent A3 / Stage S-03).

Produces a 1-2 sentence scene-description string per photo. Per PRD Section 03 the
description is generated LOCALLY using HuggingFace transformers
(``Salesforce/blip-image-captioning-large``) when torch/transformers + weights are
available, and falls back to the HuggingFace Serverless Inference API when local
inference is not possible (heavy libs missing) or when Groq's rate limit forces a
fallback (rule G-07 / HF-fallback).

Everything external lives behind the small ``SceneDescriber`` protocol so the rest
of the pipeline can be tested WITHOUT torch/transformers/network or an ``HF_TOKEN``.

  * ``LocalBlip2Describer``   — wraps transformers; skips/raises cleanly if libs absent.
  * ``HFServerlessDescriber`` — wraps the HF Serverless Inference REST endpoint.
  * ``SceneDescriptionService`` — tries local first, falls back to serverless, and
    derives lightweight BLIP-2 signals (smile/activity/sunglasses/scene_context)
    that feed A2's Tier-2 signal hooks (G-09 LinkedIn suitability, dating mood, ...).

Signals (the bridge into A2's forward-compat hooks):
    scene_context  — the raw description string (Tier-2 scene_context rules read it)
    mood_score     — 0..10 heuristic smile/positivity from the description
    sunglasses     — bool, "sunglasses"/"shades" present
    text_overlay   — bool, sign/poster/text mentioned
    activity       — bool, an action verb / activity context present
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol

# PRD Section 03: local-inference scene-description model.
BLIP_LOCAL_MODEL = "Salesforce/blip-image-captioning-large"
# PRD Tier-3 fallback: BLIP-2 on HF Serverless (used on Groq 429 / parse fail).
BLIP_SERVERLESS_MODEL = "Salesforce/blip2-opt-2.7b"
HF_SERVERLESS_URL = "https://api-inference.huggingface.co/models/{model}"


# --- the mockable interface -------------------------------------------------
class SceneDescriber(Protocol):
    """A thing that turns image bytes into a scene-description string."""

    def describe(self, image_bytes: bytes) -> str:  # pragma: no cover - protocol
        ...


# --- signal derivation (pure, fully testable) -------------------------------
_POSITIVE_WORDS = (
    "smiling", "smile", "laughing", "happy", "joy", "grinning", "cheerful",
    "fun", "celebrating", "delighted",
)
_ACTIVITY_WORDS = (
    "hiking", "running", "cooking", "playing", "climbing", "surfing", "dancing",
    "traveling", "skiing", "swimming", "riding", "walking", "skateboarding",
    "working", "reading", "painting", "gardening", "fishing",
)
_SUNGLASSES_WORDS = ("sunglasses", "shades")
_TEXT_OVERLAY_WORDS = ("sign", "poster", "banner", "text", "billboard", "whiteboard",
                       "screen", "slide", "logo", "watermark")


def derive_signals(description: Optional[str]) -> Dict[str, Any]:
    """Derive Tier-2 signal hints from a BLIP-2 scene description.

    Pure function — no model, no network. Returns a dict shaped for A2's
    ``signals`` hook (mood_score / sunglasses / text_overlay / activity) plus the
    raw text under ``scene_context``. Empty/None description -> neutral signals.
    """
    if not description:
        return {}
    text = description.lower()

    positive_hits = sum(1 for w in _POSITIVE_WORDS if w in text)
    # Heuristic 0..10 mood: neutral 5, each positive cue lifts it, capped at 10.
    mood_score = min(10.0, 5.0 + 2.5 * positive_hits) if positive_hits else None

    return {
        "scene_context": description,
        "mood_score": mood_score,
        "sunglasses": any(w in text for w in _SUNGLASSES_WORDS),
        "text_overlay": any(w in text for w in _TEXT_OVERLAY_WORDS),
        "activity": any(w in text for w in _ACTIVITY_WORDS),
    }


@dataclass
class SceneResult:
    """Output of the scene-description service for one photo."""

    description: str
    source: str  # "local" | "serverless" | "none"
    signals: Dict[str, Any] = field(default_factory=dict)


# --- concrete describers (skip-gracefully when libs/keys absent) -------------
class LocalBlip2Describer:
    """BLIP scene description via local HuggingFace transformers inference.

    Lazily loads the model on first ``describe`` so importing this module is cheap
    and does NOT pull torch/transformers (keeps the A2 zero-import guarantee safe —
    nothing imports this on the Tier-1/2 path). Raises ``RuntimeError`` if the heavy
    libs are unavailable so the service can fall back.
    """

    def __init__(self, model: str = BLIP_LOCAL_MODEL) -> None:
        self.model = model
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, BlipForConditionalGeneration
        except Exception as exc:  # transformers/torch not installed (sandbox)
            raise RuntimeError(f"local BLIP-2 unavailable: {exc}") from exc
        self._processor = AutoProcessor.from_pretrained(self.model)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model)

    def describe(self, image_bytes: bytes) -> str:
        self._ensure_loaded()
        import io

        from PIL import Image  # local, already a hard dep

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        out = self._model.generate(**inputs, max_new_tokens=40)
        text = self._processor.decode(out[0], skip_special_tokens=True)
        return text.strip()


class HFServerlessDescriber:
    """BLIP scene description via the HF Serverless Inference REST API.

    Used as a fallback when local inference is unavailable, and as the Groq-429 /
    JSON-parse-fail fallback (rule G-07). The HTTP transport is injected
    (``http_post``) so tests need neither network nor an ``HF_TOKEN``. The default
    transport uses ``requests`` if present, else ``urllib`` (both stdlib-friendly).
    """

    def __init__(
        self,
        token: Optional[str] = None,
        model: str = BLIP_SERVERLESS_MODEL,
        http_post: Optional[Callable[..., Any]] = None,
        timeout: float = 10.0,
    ) -> None:
        self.token = token
        self.model = model
        self.timeout = timeout
        self._http_post = http_post or self._default_post

    def _default_post(self, url: str, *, headers: Dict[str, str], data: bytes) -> Any:
        # Prefer requests; fall back to urllib so no extra dependency is forced.
        try:
            import requests

            resp = requests.post(url, headers=headers, data=data, timeout=self.timeout)
            return resp.status_code, resp.content
        except ImportError:
            import urllib.error
            import urllib.request

            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:  # surface status for caller
                return e.code, e.read()

    def describe(self, image_bytes: bytes) -> str:
        if not self.token:
            raise RuntimeError("HF Serverless unavailable: HF_TOKEN not set")
        url = HF_SERVERLESS_URL.format(model=self.model)
        headers = {"Authorization": f"Bearer {self.token}"}
        status, content = self._http_post(url, headers=headers, data=image_bytes)
        if status != 200:
            raise RuntimeError(f"HF Serverless returned status {status}")
        import json

        payload = json.loads(content)
        # HF caption pipelines return [{"generated_text": "..."}].
        if isinstance(payload, list) and payload:
            text = payload[0].get("generated_text", "")
        elif isinstance(payload, dict):
            text = payload.get("generated_text", "")
        else:
            text = ""
        return str(text).strip()


@dataclass
class SceneDescriptionService:
    """Scene-description orchestrator: local first, HF Serverless fallback.

    ``local`` and ``serverless`` are injected ``SceneDescriber``-likes so the whole
    thing is mockable. If both are absent or fail, ``describe`` returns an empty
    description with source ``"none"`` (the pipeline degrades gracefully — Tier-2
    scene rules simply no-op, exactly as in A2).
    """

    local: Optional[SceneDescriber] = None
    serverless: Optional[SceneDescriber] = None

    def describe(self, image_bytes: bytes) -> SceneResult:
        # 1) local inference.
        if self.local is not None:
            try:
                desc = self.local.describe(image_bytes)
                return SceneResult(desc, "local", derive_signals(desc))
            except Exception:
                pass  # fall through to serverless
        # 2) HF Serverless fallback.
        if self.serverless is not None:
            try:
                desc = self.serverless.describe(image_bytes)
                return SceneResult(desc, "serverless", derive_signals(desc))
            except Exception:
                pass
        # 3) graceful no-op.
        return SceneResult("", "none", {})

    def describe_via_serverless(self, image_bytes: bytes) -> SceneResult:
        """Force the HF Serverless path (used as the Groq-429 fallback in tier3)."""
        if self.serverless is None:
            return SceneResult("", "none", {})
        try:
            desc = self.serverless.describe(image_bytes)
            return SceneResult(desc, "serverless", derive_signals(desc))
        except Exception:
            return SceneResult("", "none", {})
