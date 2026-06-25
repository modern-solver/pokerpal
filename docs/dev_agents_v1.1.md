# Curator Bot — Development Process (v1.1)

> Durable source-of-truth extracted from `curator_bot_dev_agents_v1.1.md.pdf`
> (Governance rule #1). Retained verbatim in markdown.

**8 agents: 7 sequential stage agents (A0–A6) + 1 cross-cutting Reviewer (R).** Each stage
agent owns exactly one stage of the PRD roadmap (S-00 → S-06), and they run **in sequence**
— each consumes the previous agent's verified output. The Reviewer is not a stage; it sits
on the Review & Test gate of every stage agent and is the body that actually authorizes (or
blocks) each push to production. The runtime components the stage agents build (BotAgent,
SessionAgent, GradingAgent, CaptionAgent, RankingAgent) are *products* of these dev agents,
not the dev agents themselves.

---

## Governance rules (binding on every agent)

**1. Knowledge-base hygiene — retain only what's necessary.**
The project folder keeps only durable source-of-truth material:
- `curator_bot_prd_v1.1.pdf` (this spec)
- This agent/process document
- Config schemas: `platform_weights.yml`, `prompts.yml`, `requirements.txt`, `.env.example`
- The gate checklist (the exit gates below)
- Source code under version control

Do **not** retain in the folder: generated captions/outputs, photo test batches, transient
logs, model weight caches, `.env` with real secrets, or scratch notebooks. Those belong in
`/tmp`, `.gitignore`, or a secrets manager.

**2. Review-and-test before every push to production.**
No agent's output reaches `main`/production until its **Review & Test gate** passes. Every
iteration clears: (a) unit/functional tests green, (b) an independent review of the diff by
**Agent R**, and (c) the stage **Exit gate** from the PRD. A push that skips any of the
three is rejected. Merge authority to `main` belongs to Agent R alone — stage agents open
changes; they do not land their own work.

---

## Dependency chain

```
A0 Scaffold ─► A1 Bot+Session ─► A2 Local Grading ─► A3 Semantic Grading
                                                            │
        A6 Polish+Deploy ◄── A5 Ranking+Output ◄── A4 Caption
  R · Reviewer ── gates every push to production from A0 through A6
  (independent; holds merge authority; builds nothing)
```

---

## Agent 0 — Scaffold & Environment · Stage S-00 (0–60 min)
**Mission:** Stand up a reproducible skeleton with all external dependencies verified.
**Builds / does:** repo + venv + `requirements.txt` (`python-telegram-bot`, `groq`,
`transformers`, `Pillow`, `opencv-python`, `imagehash`, `aiohttp`); `.env` keys
(`TELEGRAM_TOKEN`, `GROQ_API_KEY`, `HF_TOKEN` — commit only `.env.example`); directory
scaffold; register bot with BotFather; Groq handshake test (Llama 3.3 `ping`, < 500 ms) and
BLIP-2 transformers load check.
**Depends on:** nothing (entry point).
**Deliverables:** runnable skeleton, dependency manifest, connectivity test logs.
**Review & Test gate:** dependency install reproducible from clean venv; handshake tests
committed as a smoke test; secrets confirmed absent from VCS.
**Exit gate (PRD):** Bot responds to `/start`; Groq API returns JSON; BLIP-2 loads without error.

## Agent 1 — Bot & Session · Stage S-01 (60–150 min)
**Mission:** Build the conversation spine and session lifecycle.
**Builds:** `BotAgent` (webhook handler, `/start`, message router, 6-platform selector);
`SessionAgent` (`init/get/update/expire` with TTL); in-memory store + SQLite fallback;
intent parsing (platform multi-select, vibe, constraints); photo-receipt handler with
progress indicator.
**Exit gate (PRD):** Session init → photo-receipt loop works; platform picker shows all 6 options.

## Agent 2 — Local Grading (Tiers 1+2) · Stage S-02 (150–240 min)
**Mission:** All zero-cost, zero-network scoring — the API-call gate lives here.
**Builds:** Tier 1 pre-filter (Pillow resolution, OpenCV Laplacian blur, imagehash dedup at
pHash distance < 10); Tier 2 `platform_weights.yml` loader + fit scorer for all 6 platforms;
dating face-gate (`face_count == 0` → cap score + warn, skip Tier 3); LinkedIn suitability flag.
**Review & Test gate:** unit test on a 10-photo batch returns correctly scored list with
flags; **assert zero API calls** in Tiers 1+2; face-gate verified on Tinder/Bumble.
**Exit gate (PRD):** All 6 platform fit scores computed locally; face-gate works; zero API calls in Tiers 1+2.

## Agent 3 — Semantic Grading (Tier 3, Groq Vision) · Stage S-03 (240–330 min)
**Mission:** Add semantic scoring only for photos that pass the local gate.
**Builds:** BLIP-2 local inference; Groq Llama 4 Scout vision call with structured JSON
grading prompt; JSON parser + single retry + HF Serverless BLIP-2 fallback on `429`;
composite aggregation of Tier 1+2+3.
**Review & Test gate:** JSON parse + retry + fallback path tested; `429` fallback verified; latency measured.
**Exit gate (PRD):** Full pipeline on 10 real photos; Groq vision < 5 s; fallback tested.

## Agent 4 — Caption (All 6 Platforms) · Stage S-04 (330–450 min)
**Mission:** Produce the correct *output type* per platform — caption vs bio vs prompt+answer vs post.
**Builds:** `prompts.yml` (6 templates: `post_caption`, `dating_bio`, `prompt_answer`,
`linkedin_post`); banned-phrase list for dating bios; Hinge 10-prompt library; Groq Llama
3.3 70B generation with tone selector; length enforcement + retry; output-mode switcher.
**Review & Test gate:** each output type asserted per platform; banned-phrase rejection
tested; Hinge returns a `{prompt, answer}` pair; length caps enforced.
**Exit gate (PRD):** All 6 platforms generate correct output type; banned phrases rejected; Hinge returns prompt+answer.

## Agent 5 — Ranking & Output Composer · Stage S-05 (450–540 min)
**Mission:** Rank, explain, and present — close the loop with the user.
**Builds:** weighted composite from `platform_weights.yml`; top-N ranker + rationale line;
BotAgent output card (photo + rationale + 3 numbered variants); user selection handler;
selection logged to session; final post-ready card.
**Exit gate (PRD):** Full session end-to-end; user sees ranked cards; selection confirmed and card returned.

## Agent 6 — Override, Polish & Deploy · Stage S-06 (540–600 min)
**Mission:** Harden, tune, and ship v1.1.0.
**Builds / does:** override commands (`/next`, `/retry`, `/shorter`, `/platform`, `/reset`);
session timeout warning at T-15 min; 5 real-session tests across all 6 platforms; tune
`platform_weights.yml`; deploy to Railway free tier; tag `v1.1.0`.
**Exit gate (PRD):** v1.1.0 live; all 6 platforms working; session < 90 s; zero cost confirmed.

## Agent R — Reviewer & Gatekeeper · Cross-cutting (all stages)
**Mission:** Be the single, independent body that authorizes every push to production. No
stage agent merges its own work — R does.
**Mandate / authority:**
- Holds merge authority to `main`/production. Stage agents open a change; only R lands it.
- Reviews **every** iteration's diff, not just the final one per stage.
- May reject a push and return it with required fixes — no override.
- Builds nothing itself (keeps independence real).

**Standing checklist applied to every push:**
1. **Tests green** — stage agent's tests pass, *and* the PRD-demanded assertions for that
   stage are present and exercised (e.g. "zero API calls in Tiers 1+2" for A2;
   "banned-phrase rejection" + "Hinge returns prompt+answer" for A4; "`429` fallback" for A3).
2. **Diff review** — reads the actual change for correctness and scope creep. Hard-rejects
   if real `.env` values, tokens, or API keys appear anywhere in the diff.
3. **Exit gate** — the PRD stage gate is met *and evidenced* (logs, test output), not asserted.
4. **KB hygiene (rule #1)** — rejects any push committing generated outputs, photo batches,
   transient logs, model-weight caches, or scratch notebooks.
5. **Regression coverage** — a downstream-found defect needs a reproducing regression test before merge.

**Deliverables:** a written verdict per push (pass / reject + rationale, attached to the
diff); the merge itself on pass; a filed defect plus required regression test on reject.

> **Note on a single-assistant setup:** independence is only as real as the separation
> behind it. Run R as a *separate* invocation — fresh context, given only the diff, the
> stage's PRD gate, and this checklist — or as a distinct subagent / human reviewer. Same
> context = self-review wearing a reviewer's hat.

---

## Handoff protocol
1. A stage agent may start only after the prior agent's **Exit gate** and **Review & Test
   gate** both pass — and "pass" means **Agent R has signed off**.
2. Each handoff carries forward: the verified artifact, its test results, **R's written
   verdict**, and any open flags.
3. If a downstream agent finds a defect in upstream work, it files back to that agent rather
   than patching around it; R requires a regression test for that defect before the fix merges.
4. Nothing merges to production — mid-stage or at stage end — without R's sign-off.
