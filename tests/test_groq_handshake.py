"""Groq handshake smoke test (PRD S-00 gate: 'Groq API returns JSON').

Calls Llama 3.3 70B with 'ping' and asserts a sub-500ms response, per the PRD
roadmap step 5. SKIPS gracefully when GROQ_API_KEY is unset or the `groq` package
is not installed, so it passes-as-skip on a clean sandbox install.
"""

import os
import time

import pytest

GROQ_MODEL = "llama-3.3-70b-versatile"
LATENCY_BUDGET_S = 0.5  # PRD: < 500ms


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live Groq handshake (sandbox has no key)",
)
def test_groq_handshake_ping_under_500ms():
    groq = pytest.importorskip("groq", reason="groq package not installed")

    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])

    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
    )
    elapsed = time.perf_counter() - start

    assert completion.choices, "Groq returned no choices"
    assert completion.choices[0].message.content is not None
    assert elapsed < LATENCY_BUDGET_S, (
        f"Groq handshake took {elapsed:.3f}s, budget is {LATENCY_BUDGET_S}s"
    )
