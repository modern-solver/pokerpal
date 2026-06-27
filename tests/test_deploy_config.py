"""Deploy-config presence + well-formedness tests (Agent A6 / Stage S-06).

These assert the Railway deploy *files* are present and internally consistent.
They do NOT deploy and do NOT require any secret — the actual Railway deploy is
deferred to the credentials checkpoint (see docs/DEPLOY.md). A live-deploy smoke
test would skip gracefully without a Railway token; we keep verification here at
the config-file level so the suite stays green on a clean checkout.
"""

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name), "r", encoding="utf-8") as fh:
        return fh.read()


def test_procfile_runs_the_polling_worker():
    content = _read("Procfile")
    assert "worker:" in content
    assert "python -m curator.main" in content


def test_railway_json_is_valid_and_has_start_command():
    data = json.loads(_read("railway.json"))
    assert data["deploy"]["startCommand"] == "python -m curator.main"
    # Restart policy present so a transient failure doesn't kill the worker.
    assert data["deploy"]["restartPolicyType"] == "ON_FAILURE"


def test_python_runtime_pinned():
    assert "3.11" in _read("runtime.txt")
    assert "3.11" in _read(".python-version")


def test_deploy_doc_lists_required_env_vars_and_remaining_steps():
    doc = _read(os.path.join("docs", "DEPLOY.md"))
    for var in ("TELEGRAM_TOKEN", "GROQ_API_KEY", "HF_TOKEN"):
        assert var in doc
    # The deferred live steps must be documented.
    for step in ("BotFather", "v1.1.0", "platform_weights.yml", "session < 90s"):
        assert step in doc


def test_no_secrets_committed_in_deploy_files():
    """Guard: the committed deploy/config files contain no real secret tokens."""
    import re

    suspects = ["Procfile", "railway.json", "runtime.txt", ".env.example",
                os.path.join("docs", "DEPLOY.md")]
    # gsk_ = Groq, hf_ = HuggingFace; a Telegram token looks like <digits>:<hex>.
    patterns = [re.compile(r"gsk_[A-Za-z0-9]{20,}"),
                re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
                re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b")]
    for name in suspects:
        text = _read(name)
        for pat in patterns:
            assert not pat.search(text), f"possible secret in {name}"


@pytest.mark.skipif(
    not os.environ.get("RAILWAY_TOKEN"),
    reason="live Railway deploy is deferred to the credentials checkpoint",
)
def test_live_railway_deploy_smoke():  # pragma: no cover - skip-gracefully
    # Placeholder for the deferred live deploy verification. Intentionally skips
    # without RAILWAY_TOKEN (mirrors the Groq/BLIP-2 live smoke tests).
    raise AssertionError("live deploy smoke not run in this scoped build")
