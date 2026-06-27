"""CaptionResult data structure.

The per-photo caption output the ranking layer consumes. One CaptionResult
carries the 3 caption options for the chosen length category (short / long /
haiku). Each option is plain caption text — NO hashtags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CaptionVariant:
    """One caption option.

    ``text`` is the caption; ``payload`` is ``{"caption", "char_count"}`` (kept so
    the presentation/selection layer has a uniform shape). ``label`` names the
    option ("Option 1") for the selection buttons / retry cycle.
    """

    label: str
    length_mode: str
    text: str
    char_count: int
    payload: Dict[str, Any] = field(default_factory=dict)
    trimmed: bool = False
    source: str = "groq"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "length_mode": self.length_mode,
            "text": self.text,
            "char_count": self.char_count,
            "payload": dict(self.payload),
            "trimmed": self.trimmed,
            "source": self.source,
        }


@dataclass
class CaptionResult:
    """The 3 caption options for one photo (ranking-layer input).

    Fields:
        platform:     target platform (light context only now).
        length_mode:  the chosen category ("short" | "long" | "haiku").
        max_length:   the enforced character cap.
        scene_description: the scene anchor used to ground the captions.
        variants:     the 3 caption options (order == the selection/retry order).
        photo_index:  position of the source photo in the batch.
    """

    platform: str
    length_mode: str
    max_length: int
    variants: List[CaptionVariant] = field(default_factory=list)
    scene_description: str = ""
    photo_index: int = 0
    flags: List[str] = field(default_factory=list)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    @property
    def labels(self) -> List[str]:
        return [v.label for v in self.variants]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "length_mode": self.length_mode,
            "max_length": self.max_length,
            "scene_description": self.scene_description,
            "photo_index": self.photo_index,
            "flags": list(self.flags),
            "variants": [v.to_dict() for v in self.variants],
        }
