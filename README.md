# Curator Bot v1.1

A **zero-API-cost Telegram bot** that grades user photos and generates platform-specific
captions for **6 platforms**: Instagram, Facebook, Tinder, Bumble, Hinge, and LinkedIn.

All processing uses free-tier APIs or local libraries — no variable cost at MVP scale.

- **Grading:** local OpenCV/Pillow/imagehash pre-filter + Groq Llama 4 Scout (vision)
- **Captions:** Groq Llama 3.3 70B
- **Scene description:** HuggingFace BLIP-2 (`Salesforce/blip-image-captioning-large`),
  local transformers inference with HF Serverless fallback
- **Bot framework:** python-telegram-bot

> This README documents the **S-00 scaffold** (Agent A0). Runtime logic for grading,
> captions, sessions, and ranking is filled in by downstream stages A1–A6.

---

## Architecture overview

### 3-tier grading pipeline
| Tier | What | Cost | Owner |
|---|---|---|---|
| **Tier 1 — Local pre-filter** | Resolution (Pillow), Laplacian blur (OpenCV), face count (Haar cascade), pHash dedup at distance < 10 (imagehash) | Zero (local) | A2 |
| **Tier 2 — Platform fit scoring** | Deterministic per-platform rules from `platform_weights.yml`, using only Tier-1 metadata | Zero (local) | A2 |
| **Tier 3 — Semantic grading** | Groq Llama 4 Scout vision, structured JSON 4-axis score (composition/lighting/subject/mood); HF Serverless BLIP-2 fallback on 429 | Free API | A3 |

The **API-call gate** lives between Tier 1 and Tier 3: a photo reaches Groq only if it is
not blurry, not low-res, and not a near-duplicate (and, for dating platforms, has a face).

### 6 platforms — output type per platform
| Platform | Output type | Max length | Hashtags |
|---|---|---|---|
| Instagram | Post caption | 150 chars | 5-10 |
| Facebook | Post caption | 500 chars | 3-5 |
| Tinder | Opening bio line | 100 chars | None |
| Bumble | Profile headline | 150 chars | None |
| Hinge | Prompt + answer pair | 150 chars | None |
| LinkedIn | Post copy | 300 chars | 3-5 professional |

Full scoring profiles and rules: [`docs/curator_bot_prd_v1.1.md`](docs/curator_bot_prd_v1.1.md).
Build process / agent rulebook: [`docs/dev_agents_v1.1.md`](docs/dev_agents_v1.1.md).

---

## Repository layout

```
curator/
  __init__.py            package metadata + PLATFORMS tuple
  main.py                app entry point — minimal /start bot (A0; extended by A1)
  bot/                   BotAgent — handlers, router, 6-platform selector      (A1)
  session/               SessionAgent — init/get/update/expire, SQLite fallback (A1)
  grading/               GradingAgent — Tier1 pre-filter, Tier2 fit, Tier3 Groq  (A2/A3)
  caption/               CaptionAgent — 6-platform output modes via Llama 3.3    (A4)
  ranking/               RankingAgent — weighted composite + top-N + output card (A5)
  config/
    platform_weights.yml  per-platform scoring profiles (stub; A2 fills)
    prompts.yml           caption/grading prompt templates (stub; A4 fills)
tests/                   smoke + unit tests
docs/                    durable PRD + process source-of-truth
requirements.txt
.env.example
```

---

## Setup

### 1. Virtual environment + dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** local BLIP-2 inference (Tier 3 / captions) also needs **PyTorch**. It is not
> pinned in `requirements.txt` so each environment can pick the right CPU/CUDA wheel — Agent
> A3 owns that decision. Install with e.g. `pip install torch` (CPU) when wiring up BLIP-2.

### 2. Environment variables
Copy the template and fill in real values (never commit `.env`):
```bash
cp .env.example .env
```
| Variable | Used for | Get it at |
|---|---|---|
| `TELEGRAM_TOKEN` | Telegram bot auth | @BotFather (see below) |
| `GROQ_API_KEY` | Vision grading + captions | https://console.groq.com |
| `HF_TOKEN` | BLIP-2 scene description / fallback | https://huggingface.co/settings/tokens |

### 3. Register the bot with BotFather (manual, one-time)
1. Open Telegram and message [@BotFather](https://t.me/BotFather).
2. Send `/newbot` and follow the prompts (choose a name and a unique username ending in `bot`).
3. BotFather replies with an HTTP API token — copy it into `TELEGRAM_TOKEN` in your `.env`.
4. (Optional) Set a description/about/commands via `/setdescription`, `/setcommands`.

---

## Running the bot
```bash
source .venv/bin/activate
python -m curator.main
```
With no `TELEGRAM_TOKEN` set, the entry point logs a clear warning and exits cleanly
(it does not crash) — this is expected in environments without a token. With a valid token
it starts long-polling and responds to `/start`.

---

## Running tests
```bash
source .venv/bin/activate
pytest -q
```
The test suite includes:
- `test_start_handler.py` — unit test of the `/start` handler (no live token needed).
- `test_groq_handshake.py` — live Groq Llama 3.3 70B `ping` < 500 ms; **skips** if
  `GROQ_API_KEY` is unset or `groq` is not installed.
- `test_blip2_load.py` — BLIP-2 transformers load check; **skips** if transformers/torch or
  `HF_TOKEN`/network are unavailable.

The handshake/load tests are designed to **pass-as-skip** on a clean install with no
credentials, and to actually exercise the providers once keys are present.

---

## Governance / KB hygiene

Per the build process (`docs/dev_agents_v1.1.md`), this repo keeps only durable
source-of-truth: spec docs, config schemas, `requirements.txt`, `.env.example`, the gate
checklist, and source under VCS. Generated outputs, photo batches, transient logs, model
caches, real `.env`, and scratch notebooks are **git-ignored** and must not be committed.
