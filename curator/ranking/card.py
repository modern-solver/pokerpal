"""Output-card data structures + variant rendering (Agent A5 / Stage S-05).

The RankingAgent produces, per target platform, a small ordered set of
``RankedPhoto`` (the top-N) each carrying the photo identity, the driving
composite score, a human rationale (see ``rationale.py``) and the 3 numbered
caption variants (from A4's ``CaptionResult``). The BotAgent presentation layer
renders these as ``OutputCard`` objects shown to the user; once the user picks a
photo and a variant we return a ``FinalCard`` — the post-ready output.

Nothing here calls a model or the network. Rendering turns each
``CaptionVariant.payload`` (mode-shaped dict from A4) into the display text for
the platform's output mode (rule C-08).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..caption.result import CaptionVariant


def render_variant(variant: CaptionVariant) -> str:
    """Render one caption variant from its mode-shaped payload (rule C-08).

    The payload is the canonical dict A4 produced; we never re-derive copy here,
    only present it. Returns the display string a user reads for that variant.
    """
    payload = variant.payload or {}
    mode = variant.output_mode

    if mode == "post_caption":
        caption = str(payload.get("caption", variant.text) or "")
        tags = payload.get("hashtags") or variant.hashtags
        tag_str = " ".join(tags)
        return f"{caption}\n{tag_str}".strip() if tag_str else caption.strip()

    if mode == "dating_bio":
        return str(payload.get("bio", variant.text) or "").strip()

    if mode == "prompt_answer":
        prompt = str(payload.get("prompt", variant.prompt or "") or "").strip()
        answer = str(payload.get("answer", variant.text) or "").strip()
        if prompt:
            return f"{prompt}\n{answer}".strip()
        return answer

    if mode == "linkedin_post":
        post = str(payload.get("post", variant.text) or "")
        tags = payload.get("hashtags") or variant.hashtags
        tag_str = " ".join(tags)
        return f"{post}\n{tag_str}".strip() if tag_str else post.strip()

    # Unknown mode: degrade to the primary text rather than crash.
    return str(variant.text or "").strip()


@dataclass
class RankedPhoto:
    """One photo's place in the ranking for a target platform.

    Fields:
        rank:        1-based position in the top-N (1 == best).
        photo_index: position of the source photo in the batch (joins back to the
                     GradingResult / CaptionResult).
        file_id:     Telegram file id of the photo (re-downloadable to display).
        platform:    the target platform this ranking is for.
        score:       the driving composite score (GradingResult.composite[platform]).
        rationale:   the human-readable explanation derived from the breakdown.
        breakdown:   the structured score breakdown the rationale was built from
                     (axes / fit / flags / warnings) — surfaced for tests + A6.
        variants:    the 3 numbered caption variants for this photo+platform.
        flags / warnings: copied from grading for display (no_face_detected, etc.).
    """

    rank: int
    photo_index: int
    platform: str
    score: float
    rationale: str
    file_id: Optional[str] = None
    breakdown: Dict[str, Any] = field(default_factory=dict)
    variants: List[CaptionVariant] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def rendered_variants(self) -> List[str]:
        """The numbered (1..n) display strings for each caption variant."""
        return [render_variant(v) for v in self.variants]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "photo_index": self.photo_index,
            "platform": self.platform,
            "score": self.score,
            "rationale": self.rationale,
            "file_id": self.file_id,
            "breakdown": dict(self.breakdown),
            "flags": list(self.flags),
            "warnings": list(self.warnings),
            "variants": [v.to_dict() for v in self.variants],
        }


@dataclass
class OutputCard:
    """A single ranked card the bot shows the user (photo + rationale + variants).

    ``photo`` is the RankedPhoto; ``card_index`` is the 1-based card number in the
    presented set (so a user can pick "photo 2"). ``rendered`` holds the numbered
    variant strings (1..n) for display.
    """

    card_index: int
    photo: RankedPhoto

    @property
    def file_id(self) -> Optional[str]:
        return self.photo.file_id

    @property
    def rationale(self) -> str:
        return self.photo.rationale

    @property
    def rendered_variants(self) -> List[str]:
        return self.photo.rendered_variants()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "card_index": self.card_index,
            "photo": self.photo.to_dict(),
            "rendered_variants": self.rendered_variants,
        }


@dataclass
class FinalCard:
    """The post-ready card after the user picks a photo + a caption variant.

    Fields:
        platform:    the target platform.
        file_id:     the chosen photo (by Telegram file id).
        photo_index: chosen photo's batch index.
        variant_index: 1-based number of the chosen variant.
        tone:        the chosen variant's tone name.
        output_mode: the platform output mode (rule C-08).
        rendered:    the post-ready display text for the chosen variant.
        payload:     the chosen variant's mode-shaped payload (for the caller to
                     post verbatim).
        score / rationale: carried from the ranked pick for confirmation display.
    """

    platform: str
    photo_index: int
    variant_index: int
    tone: str
    output_mode: str
    rendered: str
    payload: Dict[str, Any] = field(default_factory=dict)
    file_id: Optional[str] = None
    score: float = 0.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "photo_index": self.photo_index,
            "variant_index": self.variant_index,
            "tone": self.tone,
            "output_mode": self.output_mode,
            "rendered": self.rendered,
            "payload": dict(self.payload),
            "file_id": self.file_id,
            "score": self.score,
            "rationale": self.rationale,
        }
