# Curator Bot — Technical PRD v1.1

**Free API Stack + 6 Platforms.** Version 1.1 · June 2026.
Confidential — Internal Use Only.

> This is the durable source-of-truth extracted from `curator_bot_prd_v1.1.pdf`
> (Governance rule #1). Retained verbatim in markdown so downstream agents can read it.

Changes from v1.0: Free API stack replaces paid calls · Added Tinder/Bumble/Hinge/LinkedIn.

---

## 01 · What Changed from v1.0

Two focused changes. Everything else in v1.0 — the session schema, agent definitions,
information flow table, BotAgent rulebook, and deployment approach — remains unchanged.

| Change | v1.0 | v1.1 | Impact |
|---|---|---|---|
| Grading model | Claude Vision API (paid) | Groq Llama 4 Scout (free) + local OpenCV/Pillow | Variable cost → zero |
| Caption model | Claude claude-sonnet-4-6 (paid) | Groq Llama 3.3 70B (free) | Variable cost → zero |
| Platforms | Instagram, Facebook | + Tinder, Bumble, Hinge, LinkedIn | 6 scoring profiles |
| Caption type | Post caption only | Post caption / dating bio / LinkedIn post / prompt answer | Output varies by platform |

---

## 02 · Free API Stack — Grading Pipeline

The grading pipeline runs in three tiers. Tiers 1 and 2 are entirely local — zero
network calls, zero cost. Only Tier 3 (semantic scoring) touches an external API, and
that API is free within the session volume we need.

### Tier 1 — Local Pre-Filter (OpenCV + Pillow) — Zero Cost

Runs first, before any API call. Filters out technically unpostable images. Pure Python
libraries, no network, no rate limits.

| Check | Library | Method | Threshold | Action on fail |
|---|---|---|---|---|
| Resolution | Pillow | `img.size[0] * img.size[1]` | < 0.5MP | Flag `low_res=True` |
| Blur/sharpness | OpenCV | Laplacian variance | < 50 | Flag `blurry=True` |
| Aspect ratio | Pillow | width / height ratio | — | Store for platform scoring (tag only) |
| Saturation | Pillow | ImageStat on HSV S channel | — | Store mean (tag only) |
| Brightness | Pillow | ImageStat on V channel | — | Store mean (tag only) |
| Face presence | OpenCV | Haar cascade (`haarcascade_frontalface`) | — | Store `face_count` |
| Perceptual hash | imagehash | `average_hash(img)` | pHash distance < 10 = near-dupe | Suppress lower-scored dupe |

### Tier 2 — Platform Fit Scoring (Local Rules) — Zero Cost

Deterministic rules applied after pre-filter. Assigns a `platform_fit` score per target
platform using only the Tier 1 metadata. No model call needed. Each platform's fit score
is computed from `platform_weights.yml` so weights tune without a code change (Section 05).

### Tier 3 — Semantic Grading (Groq Llama 4 Scout Vision) — Free API

Runs only on photos that pass Tier 1 (not blurry, not low-res) and are not near-duplicates.
Sends image + a structured grading prompt to Groq's free tier using Llama 4 Scout, a
vision-capable open-source model on Groq's LPU hardware.

**Groq Free Tier — Current Limits (as of June 2026)**

| Model | RPM | RPD | TPM | Vision? | Free? |
|---|---|---|---|---|---|
| llama-4-scout | 30 | 1,000 | 6,000 | Yes | Yes — no credit card |
| llama-3.3-70b-versatile | 30 | 1,000 | 12,000 | No | Yes — used for captions |
| llama-3.1-8b-instant | 30 | 14,400 | 6,000 | No | Yes — fast fallback |

At 20 photos/session with ~50% passing pre-filter (~10 vision calls), 1,000 RPD supports
~100 sessions/day — sufficient for MVP.

**Grading Prompt Template (Tier 3)**

```
SYSTEM: You are a photo quality assessor. Score the image on four axes. Respond ONLY
with JSON — no prose, no explanation.
PROMPT: Score this photo from 1-10 on each axis.
{
  "composition": <1-10>,  // rule of thirds, subject clarity, framing
  "lighting":    <1-10>,  // exposure, shadow detail, golden hour bonus
  "subject":     <1-10>,  // sharpness of main subject, focus accuracy
  "mood":        <1-10>   // emotional impact, colour harmony, atmosphere
}
```

JSON-only output keeps tokens minimal (~50 tokens/response). The 4-axis score is combined
with Tier 2 `platform_fit` in the RankingAgent weighted composite.

**Fallback: Hugging Face Serverless Inference API.** If Groq rate limit is hit (429),
GradingAgent falls back to HF Serverless using BLIP-2 (`Salesforce/blip2-opt-2.7b`) for
scene description; it parses subject/mood signals from the text and fills missing axes
with Tier 2 estimates.

---

## 03 · Free API Stack — Caption Pipeline

Caption generation uses Groq's free tier with Llama 3.3 70B (~394 tokens/s on LPU). The
model receives the Tier 3 scene description, session context, and a tone template, and
returns three caption variants.

**Caption Prompt Architecture**

```
SYSTEM: You are a social media caption writer. Output ONLY valid JSON. No preamble.
No explanation.
USER:
  Platform: {platform}
  Tone requested: {tone}   // witty | heartfelt | minimal | bio | prompt_answer | professional
  Scene description: {blip_desc}
  Session vibe: {vibe}
  Constraints: {constraints}   // e.g. "no location names", "under 100 chars"
  Generate exactly this JSON:
  { "caption": "", "hashtags": ["", "", ...], "char_count": }
```

**Scene Description Source.** `{blip_desc}` is generated locally using BLIP-2
(`Salesforce/blip-image-captioning-large`) via the HuggingFace transformers library (local
inference if hardware allows, else HF Serverless fallback). Gives the caption model a
factual scene anchor without sending the raw image to Groq (Groq text models don't accept
images).

**Two-Stage Caption Flow**

| Stage | Tool | Input | Output | Cost |
|---|---|---|---|---|
| 1. Scene description | BLIP-2 (local or HF Serverless) | Raw image | 1-2 sentence scene description | Free |
| 2. Caption generation | Groq Llama 3.3 70B (free tier) | Scene desc + tone template + session ctx | 3 tone variants per photo | Free |

**Platform-Specific Caption Types**

| Platform | Output type | Max length | Tone variants | Hashtags? |
|---|---|---|---|---|
| Instagram | Post caption | 150 chars (MVP) | Witty / Heartfelt / Minimal | 5-10, appended |
| Facebook | Post caption | 500 chars | Warm / Storytelling / Minimal | 3-5, appended |
| Tinder | Opening bio line | 100 chars | Playful / Confident / Mysterious | None |
| Bumble | Profile headline | 150 chars | Approachable / Adventurous / Witty | None |
| Hinge | Prompt + answer pair | 150 chars | Funny / Heartfelt / Curiosity-bait | None |
| LinkedIn | Post copy | 300 chars | Professional / Insight / Achievement | 3-5 professional |

---

## 04 · API Provider Reference Table

Every provider has a genuinely free tier — no credit card required to start.

| Provider | Use in Curator | Free limits | Key env var | Signup |
|---|---|---|---|---|
| Groq (GroqCloud) | Semantic grading (Llama 4 Scout vision) + Caption gen (Llama 3.3 70B) | 30 RPM / 1,000 RPD / 14,400 RPD (smaller models). No CC. | `GROQ_API_KEY` | console.groq.com |
| Hugging Face Serverless | BLIP-2 scene description (fallback) + grading fallback on Groq 429 | Rate-limited, no hard daily cap. Free HF account. | `HF_TOKEN` | huggingface.co |
| OpenCV (local) | Blur, face detection, sharpness | No API — `pip install opencv-python`. Local. | None | pip install |
| Pillow (local) | Resolution, saturation, brightness, aspect ratio, histogram | No API — `pip install Pillow`. Local. | None | pip install |
| imagehash (local) | Perceptual hash for near-dupe detection | No API — `pip install imagehash`. Local. | None | pip install |

**Note on Gemini (Not Used in v1.1).** Gemini 2.5 Flash free tier was slashed to ~20 RPD
(too low); Gemini 3 Flash is preview-only. Groq Llama 4 Scout gives equivalent vision
quality at 1,000 RPD free. Gemini remains a viable paid upgrade path.

---

## 05 · 6-Platform Scoring Profiles

Each platform has a distinct profile in `platform_weights.yml`: weight allocation across
grading axes, platform-fit sub-rules, and the expected CaptionAgent output format.

### INSTAGRAM — weights: Composition 25% · Lighting 25% · Technical 15% · Platform fit 35%
1. Aspect ratio: 4:5 portrait +4, 1:1 square +2, 9:16 story +3, 16:9 landscape -3.
2. Saturation > 0.55: +2. Under 0.25: -1. Warm tones (hue 15-45 dominant): +1.
3. Face detected: +3 (IG engagement premium highest of all platforms).
4. Horizon level (within 3°): +1.
5. Text overlay or watermark detected: -4.
6. Caption output: post caption, 150 char MVP target, 5-10 hashtags appended.

### FACEBOOK — weights: Composition 20% · Lighting 20% · Technical 15% · Platform fit 45%
1. Aspect ratio: 16:9 landscape +4, 4:3 +3, 1:1 +1, portrait neutral.
2. Scene complexity > 0.6 (multi-subject, context-rich background): +2.
3. Text overlay present and legible: +1 (FB link preview compatible).
4. Face detected: +1 (lower premium than IG).
5. Group photo (2+ faces): +2 (FB skews social/community).
6. Caption output: post caption, 500 chars, conversational tone, 3-5 hashtags.

### TINDER — weights: Composition 15% · Lighting 20% · Technical 15% · Platform fit 50%
1. MANDATORY: face detected. If `face_count == 0`: score capped at 3/10 regardless of other axes.
2. Face size > 20% of frame area: +4 (close-up face = #1 Tinder success signal).
3. Aspect ratio: 9:16 portrait +4, 4:5 +3, 1:1 neutral, landscape -3.
4. Smile/positive expression (BLIP-2 mood score > 7): +2.
5. Solo subject (`face_count == 1`): +2. Group (3+ faces): -2 for primary pick.
6. Sunglasses covering eyes: -3 (face clarity required).
7. Heavy filter (saturation > 0.85, extreme warmth/coolness): -2.
8. Caption output: bio opening line, 100 chars max, Playful/Confident/Mysterious. No hashtags.

### BUMBLE — weights: Composition 15% · Lighting 20% · Technical 15% · Platform fit 50%
1. MANDATORY: face detected. Cap at 3/10 if `face_count == 0`.
2. Face visible and unobstructed: +3 (Bumble skews quality-conscious).
3. Activity context present (BLIP-2 detects person doing something): +3.
4. Aspect ratio: 9:16 +4, 4:5 +3. Landscape: -2.
5. Authentic/candid feel (low saturation edit, natural light): +1 vs over-edited.
6. Solo or 1+1 (`face_count <= 2`): preferred. Group 3+: -1.
7. Caption output: profile headline, 150 chars, Approachable/Adventurous/Witty. No hashtags.

### HINGE — weights: Composition 20% · Lighting 20% · Technical 15% · Platform fit 45%
1. Face detected preferred but not mandatory (activity/context shots can anchor prompts).
2. Activity/lifestyle context detected (BLIP-2): +4 (Hinge is personality-first).
3. Travel/outdoor scene: +2. Food/drink scene: +1. Generic indoor: neutral.
4. Candid or action shot preferred over posed: BLIP mood variety > single expression.
5. Aspect ratio: 4:5 +3, 1:1 +2, 9:16 +2.
6. Caption output: prompt + answer pair. Prompts from Hinge standard list.
   Tones: Funny/Heartfelt/Curiosity-bait. 150 chars. No hashtags.

### LINKEDIN — weights: Composition 30% · Lighting 25% · Technical 20% · Platform fit 25%
1. Face detected + centred + well-lit: +4 (professional headshot premium).
2. Clean/neutral background (low texture variance behind subject): +2.
3. Saturation moderate (0.3-0.55): +1. Over-saturated -2, desaturated -1.
4. Aspect ratio: 1:1 +3, 4:5 +2. Landscape neutral. Extreme portrait -1.
5. Heavy filter, party context, or low-light setting: -4.
6. Text overlay (slides, whiteboards, conference signage): +2 for content posts.
7. Caption output: LinkedIn post copy, 300 chars, Professional/Insight/Achievement,
   3-5 professional hashtags.

---

## 06 · Platform Output Types

The CaptionAgent does not always produce a traditional photo caption. The output type is
determined by the target platform and shapes the prompt template used.

### Dating App Context (Tinder, Bumble, Hinge)

Dating apps have no photo captions in the traditional sense — the photo is matched with
profile text (bio, headline, prompt answer). The CaptionAgent generates this profile text
using the selected photo as visual context, NOT text beneath the photo.

| Platform | Text location | Character of output | Example |
|---|---|---|---|
| Tinder | Bio (below photos) | Opening line that hooks. Personality > resume. No clichés. | "Equally good at ordering wine and assembling IKEA furniture" |
| Bumble | Profile bio | Approachable, hinting at activity/lifestyle shown in photo. | "Perpetually planning my next trip. Currently: pretending to wo…" |
| Hinge | Prompt + answer card | Specific Hinge prompt + answer that pairs with the photo's mood. | Prompt: "The way to my heart is…" Answer: "Finding good ligh…" |

**Standard Hinge Prompts Library (injected into CaptionAgent)**
- Two truths and a lie
- The way to my heart is...
- I'm looking for...
- This year I really want to...
- My most controversial opinion is...
- Don't hate me if I...
- I go crazy for...
- Best travel story
- A life goal of mine...
- I know the best spot in town for...

### LinkedIn Context

LinkedIn photos are professional headshots (profile) or supporting visuals for
thought-leadership posts. The CaptionAgent generates post copy as a standalone paragraph
that could accompany the photo — not a photo caption. Tone: Professional/Insight/
Achievement. No exclamation marks on Professional tone. Achievement tone: quantify if
BLIP-2 detects an award, certificate, or event banner.

---

## 07 · Updated Agent Rulebooks (Delta from v1.0)

Only rules that change in v1.1 are listed. All other v1.0 rules remain in force.

### GradingAgent — New and Changed Rules
- **G-06 (NEW):** Always run Tier 1 (Pillow + OpenCV) before any API call. The API-call
  gate is: not blurry AND not low_res AND not duplicate_suppressed. Ordering is mandatory.
- **G-07 (NEW):** Use Groq Llama 4 Scout for Tier 3. Send a single structured JSON prompt;
  parse the JSON. If parse fails, retry once. On second failure, fall back to HF Serverless
  BLIP-2 scene description and estimate scores from text.
- **G-08 (NEW):** For Tinder/Bumble: if `face_count == 0` after Tier 1, set
  `composite_score = 2.0` and attach flag `no_face_detected`. Do not send to Tier 3.
  Surface: "This photo has no visible face — not recommended for dating platforms."
- **G-09 (NEW):** For LinkedIn: if BLIP-2 scene description contains party, beach, outdoor
  festival, heavy filter → `suitability_flag = linkedin_inappropriate`. Still score, surface flag.
- **G-05 (UPDATED):** Dedup suppression uses pHash distance < 10 (was < 8). Relaxed
  threshold reduces false-positives on portrait series (same pose, slightly different
  expression should NOT be suppressed for dating apps).

### CaptionAgent — New and Changed Rules
- **C-08 (NEW):** Switch `output_mode` by platform: `post_caption` (IG/FB), `dating_bio`
  (Tinder/Bumble), `prompt_answer` (Hinge), `linkedin_post` (LinkedIn). Load corresponding
  template from `prompts.yml`.
- **C-09 (NEW):** For dating platforms: never generate generic bio clichés. Banned phrases
  in `prompts.yml` include: "love to laugh", "fluent in sarcasm", "partner in crime",
  "loves adventures", "looking for my person". Retry if any banned phrase appears.
- **C-10 (NEW):** For Hinge: select the prompt from the standard library (Section 06) that
  best matches the BLIP-2 scene description mood. Do not invent custom prompts. Output
  format: `{'prompt': '', 'answer': ''}`.
- **C-11 (NEW):** For LinkedIn: avoid first-person casual language. No "I" in Professional
  tone. Achievement tone may use first-person sparingly. Never use #hustle, #blessed,
  #grind — professional tags only (e.g. #leadership, #innovation, #[industry]).
- **C-07 (UPDATED):** On `/retry` for dating platforms, cycle through tone variants in
  order (Playful → Confident → Mysterious for Tinder) rather than regenerating the same
  tone with a different seed.

---

## 08 · Updated Cost Model

The v1.1 stack has zero variable cost per session at MVP scale.

| Component | v1.0 cost | v1.1 cost | Tool | Free limit |
|---|---|---|---|---|
| Pre-filter (blur, resolution, hash) | $0 (local) | $0 (local) | OpenCV + Pillow | Unlimited |
| Platform fit scoring | $0 (local) | $0 (local) | Pillow + YAML rules | Unlimited |
| Semantic grading (per photo) | ~$0.00024 (Claude) | $0 | Groq Llama 4 Scout | 1,000 RPD |
| Scene description (per photo) | Bundled in Claude call | $0 | BLIP-2 local or HF Serverless | Unlimited / rate-limited |
| Caption generation (per session) | ~$0.00054 (3 photos × Claude) | $0 | Groq Llama 3.3 70B | 1,000 RPD |
| Total per session (12 img, 3 top) | $0.003–0.006 | $0 | — | — |
| Daily capacity (free tier) | Unlimited (pay per call) | ~100 sessions | Groq RPD ceiling | 1,000 calls/day |

**Scale trigger:** when sessions exceed ~80/day consistently, add Groq Developer tier
(no monthly fee, just add card, 10× limits). Marginal cost ≈ $0.001/session — still 3-6×
cheaper than the v1.0 Claude stack.

---

## 09 · Updated Roadmap in Minutes

| Stage | Title | Window | Exit gate |
|---|---|---|---|
| S-00 | Environment + Scaffold | 0–60 min | Bot responds to /start; Groq API returns JSON; BLIP-2 loads without error. |
| S-01 | BotAgent + Session Init | 60–150 min | Session init → photo receipt loop works; platform picker shows all 6 options. |
| S-02 | GradingAgent — Local Tiers 1+2 | 150–240 min | All 6 platform fit scores computed locally; face-gate works; zero API calls in Tiers 1+2. |
| S-03 | GradingAgent — Tier 3 Groq Vision | 240–330 min | Full pipeline on 10 real photos; Groq vision < 5s; fallback tested. |
| S-04 | CaptionAgent — All 6 Platforms | 330–450 min | All 6 platforms generate correct output type; banned phrases rejected; Hinge returns prompt+answer. |
| S-05 | RankingAgent + Output Composer | 450–540 min | Full session end-to-end; user sees ranked cards; selection confirmed and card returned. |
| S-06 | Override Commands + Polish + Deploy | 540–600 min | v1.1.0 live; all 6 platforms working; session < 90s; zero cost confirmed. |

**Total: 600 minutes (10 hours)** to a working, zero-cost MVP supporting 6 platforms.
