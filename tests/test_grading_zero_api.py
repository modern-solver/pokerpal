"""ZERO-API-CALL assertion for Tiers 1+2 (Agent A2 / Stage S-02).

PRD S-02 Review & Exit gate: "Assert zero API calls in Tiers 1+2." Reviewer R
verifies this is present and exercised. We prove it two ways:

  1. NETWORK GUARD: monkeypatch socket.socket.connect (and create_connection) to
     raise on any outbound attempt, then run the full Tier 1+2 scoring path. Any
     network call would fail the test.

  2. NO MODEL IMPORT: assert that running the scoring path does not import `groq`
     or `transformers` (the only API/model clients in the stack). If the grading
     code path touched Tier 3, one of these would be imported.
"""

import socket
import sys

import pytest

from curator.grading.agent import GradingAgent
from curator.grading.tier1 import PhotoMetadata


def _meta(**kw) -> PhotoMetadata:
    return PhotoMetadata.from_dict(kw)


def _run_full_scoring_path():
    agent = GradingAgent()
    batch = [
        _meta(width=1080, height=1350, face_count=1, saturation_mean=0.6,
              largest_face_frac=0.25, phash="0000000000000001"),
        _meta(width=400, height=400, face_count=0, phash="0000000000000002"),
        _meta(width=1080, height=1080, face_count=1, saturation_mean=0.5, phash="ffffffffffffffff"),
        _meta(width=1080, height=1080, face_count=0, saturation_mean=0.1, phash="fffffffffffffffe"),
    ]
    platforms = ["instagram", "facebook", "tinder", "bumble", "hinge", "linkedin"]
    return agent.grade_batch(
        batch, platforms,
        scene_descriptions=["a beach party", None, "office headshot", None],
    )


def test_no_network_calls_in_tiers_1_and_2(monkeypatch):
    """Hard-fail if the scoring path opens any socket connection."""

    def _blocked_connect(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError("Tier 1+2 attempted a network connection (zero-API gate)")

    def _blocked_create_connection(*args, **kwargs):
        raise AssertionError("Tier 1+2 attempted socket.create_connection (zero-API gate)")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect, raising=True)
    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection, raising=True)

    results = _run_full_scoring_path()
    assert len(results) == 4
    # Sanity: scoring actually happened.
    assert all(r.platform_fit for r in results)


def test_no_groq_or_transformers_imported_by_scoring_path():
    """The Tier 1+2 path must not import any API/model client."""
    # Drop any pre-imported clients so we observe a fresh import if one occurs.
    for mod in ("groq", "transformers", "huggingface_hub"):
        sys.modules.pop(mod, None)

    _run_full_scoring_path()

    for mod in ("groq", "transformers"):
        assert mod not in sys.modules, (
            f"Tier 1+2 scoring path imported {mod!r} — that is a Tier-3/API client "
            "and must never be touched in A2."
        )


def test_grading_modules_do_not_statically_import_api_clients():
    """Static guard: the grading source must not import groq/transformers at all."""
    import curator.grading.agent as agent_mod
    import curator.grading.tier1 as tier1_mod
    import curator.grading.tier2 as tier2_mod
    import curator.grading.weights as weights_mod
    import curator.grading.result as result_mod

    import inspect

    for mod in (agent_mod, tier1_mod, tier2_mod, weights_mod, result_mod):
        src = inspect.getsource(mod)
        assert "import groq" not in src, f"{mod.__name__} imports groq"
        assert "import transformers" not in src, f"{mod.__name__} imports transformers"
        assert "from groq" not in src, f"{mod.__name__} imports from groq"
        assert "from transformers" not in src, f"{mod.__name__} imports from transformers"
