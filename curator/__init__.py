"""Curator Bot v1.1 — zero-API-cost Telegram photo curation bot.

Grades user photos and generates platform-specific captions for 6 platforms
(Instagram, Facebook, Tinder, Bumble, Hinge, LinkedIn).

Package layout (filled in by downstream stage agents A1-A6):
    bot/      BotAgent      — Telegram handlers, router, 6-platform selector   (A1)
    session/  SessionAgent  — init/get/update/expire, in-mem + SQLite fallback (A1)
    grading/  GradingAgent  — Tier1 local pre-filter, Tier2 fit, Tier3 Groq    (A2/A3)
    caption/  CaptionAgent  — 6-platform output modes via Groq Llama 3.3 70B   (A4)
    ranking/  RankingAgent  — weighted composite + top-N + output card         (A5)
    config/                 — platform_weights.yml, prompts.yml                (A2/A4)
"""

__version__ = "1.1.0-dev"

# Supported platforms (PRD Section 05/06). Used by BotAgent selector and scorers.
PLATFORMS = ("instagram", "facebook", "tinder", "bumble", "hinge", "linkedin")
