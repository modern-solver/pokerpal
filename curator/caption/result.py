"""CaptionResult data structure (Agent A4 / Stage S-04).

The per-(photo, platform) caption output that A5 (ranking + output composer)
consumes to build ranked output cards. One CaptionResult carries the 3 tone
variants the spec mandates ("3 tone variants per photo"), each already shaped to
the platform's output mode and validated against its length cap + banned-phrase
policy.

Output modes (rule C-08) and the variant payload shape per mode:
    post_caption  (IG/FB)        -> {"caption", "hashtags", "char_count"}
    dating_bio    (Tinder/Bumble)-> {"bio", "char_count"}
    prompt_answer (Hinge)        -> {"prompt", "answer", "char_count"}
    linkedin_post (LinkedIn)     -> {"post", "hashtags", "char_count"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaptionVariant:
    """One tone variant for one platform.

    ``text`` is the primary copy (caption / bio / answer / post body). ``payload``
    is the full mode-shaped dict A5 renders. ``tone`` names the variant so the
    /retry tone-cycle (rule C-07) can step through them in order.
    """

    tone: str
    output_mode: str
    text: str
    char_count: int
    hashtags: List[str] = field(default_factory=list)
    prompt: Optional[str] = None  # Hinge only: the selected library prompt
    payload: Dict[str, Any] = field(default_factory=dict)
    # Provenance/diagnostics: how many regen attempts this variant took, whether a
    # length trim was applied, and the source ("groq" | "groq_retry" | "fallback").
    attempts: int = 1
    trimmed: bool = False
    source: str = "groq"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tone": self.tone,
            "output_mode": self.output_mode,
            "text": self.text,
            "char_count": self.char_count,
            "hashtags": list(self.hashtags),
            "prompt": self.prompt,
            "payload": dict(self.payload),
            "attempts": self.attempts,
            "trimmed": self.trimmed,
            "source": self.source,
        }


@dataclass
class CaptionResult:
    """All tone variants for one photo on one platform (A5 input).

    Fields:
        platform:     target platform.
        output_mode:  rule C-08 output mode for that platform.
        max_length:   the enforced character cap.
        scene_description: the BLIP-2 {blip_desc} anchor used (A3).
        variants:     ordered tone variants (variants[0] == default tone; the
                      order is the /retry cycle order, rule C-07).
        photo_index:  position of the source photo in the batch (links back to the
                      GradingResult A5 ranks).
    """

    platform: str
    output_mode: str
    max_length: int
    variants: List[CaptionVariant] = field(default_factory=list)
    scene_description: str = ""
    photo_index: int = 0
    flags: List[str] = field(default_factory=list)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    @property
    def tones(self) -> List[str]:
        return [v.tone for v in self.variants]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "output_mode": self.output_mode,
            "max_length": self.max_length,
            "scene_description": self.scene_description,
            "photo_index": self.photo_index,
            "flags": list(self.flags),
            "variants": [v.to_dict() for v in self.variants],
        }
