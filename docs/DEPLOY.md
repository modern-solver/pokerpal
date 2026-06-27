# Curator Bot v1.1 — Deployment (Railway free tier)

> Stage S-06 (Agent A6). This doc is the durable runbook for shipping Curator Bot
> to Railway's free tier. The **config files** are committed (see "Files in this
> repo" below). The **live steps** below are deferred to the *credentials
> checkpoint* — they need real secrets the sandbox does not have, and must be run
> by an operator with a BotFather token, a GroqCloud key, and a Railway account.
>
> Nothing here commits a secret. Secrets are set in Railway's variables UI / CLI,
> never in the repo (`.env` is git-ignored; only `.env.example` is committed).

---

## What the bot runs as

A **long-polling worker** (not a web server): `python -m curator.main` opens a
Telegram long-poll loop. There is no inbound HTTP port, so no `PORT`/healthcheck
is required. On Railway this is a **worker service**.

`curator.main.main()` exits cleanly (code 0, no crash) when `TELEGRAM_TOKEN` is
unset — so a misconfigured deploy logs a clear message rather than crash-looping.

---

## Files in this repo (committed, no secrets)

| File | Purpose |
|---|---|
| `Procfile` | `worker: python -m curator.main` — the process Railway/Heroku-style buildpacks run. |
| `railway.json` | Railway build (Nixpacks, Python 3.11 + gcc for `opencv`/`imagehash` wheels) + `startCommand` + restart policy. |
| `runtime.txt` / `.python-version` | Pin Python 3.11.9 for the builder. |
| `requirements.txt` | Pinned deps (A0). NOTE: `torch` is intentionally unpinned — see "torch / BLIP-2" below. |
| `.env.example` | The variable names to set in Railway (with placeholder values only). |

---

## Required environment variables (set in Railway — NOT committed)

| Variable | Needed for | Where to get it |
|---|---|---|
| `TELEGRAM_TOKEN` | Telegram bot auth (the polling loop). **Mandatory.** | @BotFather (`/newbot`). |
| `GROQ_API_KEY` | Tier-3 vision grading (Llama 4 Scout) + caption generation (Llama 3.3 70B). | https://console.groq.com (free, no card). |
| `HF_TOKEN` | Only if BLIP-2 falls back to HF Serverless (local inference is preferred). Optional. | https://huggingface.co/settings/tokens |

Graceful degradation without keys (mirrors every prior stage):
- No `GROQ_API_KEY` → Tier-3 grading + Groq captions are skipped; the pipeline
  uses Tier-1/2 local scores and the deterministic local caption fallback.
- No `HF_TOKEN` → only matters if BLIP-2 needs the serverless fallback.
- No `TELEGRAM_TOKEN` → `main()` logs and exits 0 (no bot).

---

## torch / BLIP-2 on the free tier

`torch` is unpinned in `requirements.txt` (A0 flag) so each environment picks its
own wheel. Railway's free tier is **memory/disk constrained**; a full CPU `torch`
+ BLIP-2 download can exceed it. Two supported options:

1. **HF Serverless for scene description** (recommended on free tier): set
   `HF_TOKEN` and do not install `torch`; BLIP-2 runs via the serverless fallback.
2. **Local BLIP-2**: add a CPU `torch` wheel to `requirements.txt`
   (e.g. `torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`) and
   confirm the free-tier image has enough room. Verify before relying on it.

The BLIP-2 path skips gracefully if `torch`/`transformers` are absent, so a deploy
without `torch` still runs (it just leans on the serverless/text-signal path).

---

## Remaining manual steps (DEFERRED — credentials checkpoint)

These are NOT done in this build; do them once real secrets are available.

1. **Register the bot with BotFather.**
   - In Telegram, message @BotFather → `/newbot` → name + username → copy the token.
   - (Optional) `/setcommands` so the override commands appear in the UI:
     ```
     start - Restart and pick platforms
     help - List the override commands
     next - Show the next-ranked photo
     retry - Regenerate the caption (dating apps cycle tone in order)
     shorter - Rewrite the caption under a tighter cap
     platform - Switch the target platform and re-rank
     reset - Clear everything and start over
     ```

2. **Create the Railway service + set secrets.**
   - `railway init` (or create a project in the dashboard); link this repo.
   - Set variables (UI → Variables, or `railway variables set ...`):
     `TELEGRAM_TOKEN`, `GROQ_API_KEY`, and `HF_TOKEN` if used. Never commit them.
   - Confirm the service type is **worker** (no public domain needed).

3. **Deploy.**
   - `railway up` (or push to the linked branch for auto-deploy).
   - Watch logs for `Starting Curator Bot (polling)...`.

4. **Smoke-verify a real end-to-end session (all 6 platforms).**
   - DM the bot `/start`; pick platforms; send ~10 photos.
   - Confirm: ranked cards appear; `/next`, `/retry` (tones cycle IN ORDER on a
     dating platform), `/shorter` (caption gets shorter), `/platform <name>`
     (re-ranks), `/reset` all work live.
   - Confirm the **session timeout warning** surfaces inside the last 15 min.
   - PRD exit gate to confirm live: **session < 90s**, **all 6 platforms working**,
     **zero variable cost** (free-tier Groq + local libs).

5. **Tune `curator/config/platform_weights.yml` against real samples.**
   - Run real photo batches per platform; compare the ranking to human judgement;
     nudge the YAML weights/sub-rules (no code change needed — A2 reads the YAML).
   - Re-run `pytest -q` after tuning (weights tests live in
     `tests/test_grading_weights.py`).

6. **Confirm the model IDs against live GroqCloud** (A3 flag):
   vision `meta-llama/llama-4-scout-17b-16e-instruct`,
   text `llama-3.3-70b-versatile`. Adjust if Groq has renamed them.

7. **Tag the release.**
   ```bash
   git tag v1.1.0
   git push origin v1.1.0
   ```

---

## PRD exit-gate items still OPEN at this checkpoint

- [ ] v1.1.0 deployed live on Railway (this build = config + commands only).
- [ ] Real end-to-end session verified across all 6 platforms.
- [ ] Session completes in < 90s on live infra.
- [ ] Zero variable cost confirmed (free-tier Groq + local libs).
- [ ] `platform_weights.yml` tuned against real photo samples.
- [ ] `v1.1.0` tag pushed.

Everything else for S-06 (override commands, the T-15min timeout warning, and
this deploy config) is built, unit-tested with mocks, and committed.
