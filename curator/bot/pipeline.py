"""End-to-end grading → captioning → ranking orchestration.

This is the conversational spine that was missing between the staged components:
given the downloaded photo bytes for a session it runs Tier 1+2 (local) and
Tier 3 (Groq vision, when a grader is injected), captions the top-N photos for
the target platform, and returns numbered ``OutputCard``s ready to present.

Pure and dependency-injected: the grading / caption / ranking agents and the
scene-description callable are passed in, so the whole pipeline is unit-testable
with stubs and never reaches the network on its own.
"""

from __future__ import annotations

import io
from typing import Any, Callable, List, Optional, Sequence, Tuple

from curator.grading.tier1 import extract_metadata
from curator.ranking.card import OutputCard


def run_pipeline(
    image_bytes: Sequence[bytes],
    photo_refs: Sequence[Any],
    platforms: Sequence[str],
    target: str,
    *,
    grading_agent: Any,
    caption_agent: Any,
    ranking_agent: Any,
    describe: Optional[Callable[[bytes], str]] = None,
    vibe: Optional[str] = None,
    constraints: Optional[Sequence[str]] = None,
    top_n: int = 3,
) -> Tuple[List[OutputCard], List[Any], List[Any]]:
    """Grade, caption, and rank a batch of photos for ``target``.

    Returns ``(cards, results, captions)`` so the caller can cache the grading
    results + captions for the override commands (/next, /platform, /retry).
    """
    from PIL import Image  # local import: Pillow is a runtime dep, not needed in some tests

    metadatas = [extract_metadata(Image.open(io.BytesIO(b))) for b in image_bytes]

    # Tier 1+2 (local, zero-API) across every selected platform so /platform can
    # switch later without re-grading.
    results = grading_agent.grade_batch(metadatas, platforms, photo_refs=photo_refs)
    # Tier 3 (Groq vision) — a no-op when no grader is injected (no key): the
    # composite then falls back to the Tier-2 fit, so cards still rank.
    grading_agent.grade_tier3(results, list(image_bytes), platforms)

    # Caption only the top-N photos for the target platform (saves model calls).
    top = ranking_agent.rank(target, results, None, top_n=top_n)
    captions: List[Any] = []
    for ranked in top:
        idx = ranked.photo_index
        img = image_bytes[idx] if 0 <= idx < len(image_bytes) else None
        scene = ""
        if describe is not None and img is not None:
            try:
                scene = describe(img) or ""
            except Exception:  # noqa: BLE001 - grounding is best-effort
                scene = ""
        if not scene:
            res = next((r for r in results if getattr(r, "index", None) == idx), None)
            scene = (getattr(res, "scene_description", "") or "") if res else ""
        captions.append(
            caption_agent.generate(
                target, scene, vibe=vibe, constraints=constraints, photo_index=idx
            )
        )

    cards = ranking_agent.build_cards(target, results, captions, top_n=top_n)
    return cards, results, captions
