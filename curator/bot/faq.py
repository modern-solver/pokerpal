"""User-facing help, capability Q&A, and the canonical command list.

Single source of truth for:
  * ``BOT_COMMANDS`` — the command menu, used for both the in-chat /help guide
    and Telegram ``setMyCommands`` registration (the BotFather command list).
  * ``USER_GUIDE`` / ``ABOUT_TEXT`` — the /help walkthrough and /about summary.
  * ``answer_question`` — a deterministic, zero-cost FAQ responder so the bot can
    answer a basic array of questions about its functionality, limitations, and
    capabilities without spending an API call.

Keeping this here (a leaf module, importing nothing from the bot) avoids
duplicating the command list across the agent, the help text, and deployment.
"""

from __future__ import annotations

from typing import List, Tuple

# --- canonical command list (source of truth) -------------------------------
# (name, short description). Descriptions are kept <= 256 chars and names
# lowercase a-z/0-9/_ so they are valid for Telegram setMyCommands / BotFather.
BOT_COMMANDS: List[Tuple[str, str]] = [
    ("start", "Restart and pick the platforms you're posting to"),
    ("help", "How to use me — the full walkthrough"),
    ("about", "What I can do, and my limits"),
    ("length", "Pick a caption style: short / long / haiku"),
    ("done", "Finish sending photos — grade & rank them"),
    ("next", "Show the next-ranked photo"),
    ("retry", "Regenerate the current caption (dating apps cycle tone)"),
    ("shorter", "Rewrite the current caption shorter"),
    ("platform", "Switch the target platform and re-rank"),
    ("reset", "Clear everything and start over"),
]

PLATFORMS_LINE = "Instagram, Facebook, Tinder, Bumble, Hinge, and LinkedIn"


def command_reference() -> str:
    """Formatted quick command list (used in /start and /help)."""
    lines = ["Commands:"]
    lines += [f"  /{name} — {desc}" for name, desc in BOT_COMMANDS]
    return "\n".join(lines)


# --- the full user guide (/help) --------------------------------------------
USER_GUIDE = (
    "How to get the best out of Curator Bot\n"
    "\n"
    "I grade your photos and write platform-specific text for "
    f"{PLATFORMS_LINE}.\n"
    "\n"
    "Step by step:\n"
    "  1. /start, then tap to pick one or more platforms (multi-select), then Done.\n"
    "  2. Pick a caption style: Short (<200), Long (<1000), or Haiku.\n"
    "  3. (Optional) Tell me the vibe or any constraints in plain text — e.g. "
    "\"adventurous, no location names\".\n"
    "  4. Send your photos (a batch is best — I compare them and rank the strongest).\n"
    "  5. Say \"done\" (or /done). I grade, rank, and send your best photos, each with "
    "3 caption options as buttons.\n"
    "  6. Tap a caption button under a photo (or reply with a number) — I return the "
    "post-ready caption.\n"
    "\n"
    "While reviewing, use:\n"
    "  /length — change the caption style (short / long / haiku)\n"
    "  /next — see the next-ranked photo\n"
    "  /retry — cycle to a different caption option\n"
    "  /shorter — make the current caption tighter\n"
    "  /platform <name> — switch platform (e.g. /platform tinder) and re-rank\n"
    "  /reset — clear everything and start over\n"
    "\n"
    "Captions are always clean — no hashtags. Ask me anything about how I work, or "
    "send /about for capabilities and limits."
)

# --- about / capabilities + limits (/about) ---------------------------------
ABOUT_TEXT = (
    "About Curator Bot v1.1\n"
    "\n"
    "What I do:\n"
    f"  • Grade your photos for {PLATFORMS_LINE}.\n"
    "  • Score each photo in 3 tiers: a local quality pre-filter (sharpness, "
    "resolution, faces, duplicates), platform-fit rules, then an AI vision read "
    "(composition, lighting, subject, mood).\n"
    "  • Rank your batch and write 3 caption options in your chosen style "
    "(short / long / haiku) — never any hashtags.\n"
    "\n"
    "My limits:\n"
    "  • Photos only — I don't handle video.\n"
    "  • Tinder and Bumble need a visible face; a no-face photo is flagged and "
    "scored low for dating.\n"
    "  • I suggest — I don't post for you, and I can't guarantee matches or likes.\n"
    "  • I read the image and your text; I don't browse the web or know who's in a "
    "photo.\n"
    "  • Free-tier AI has rate limits, so very large batches may be graded in parts.\n"
    "\n"
    "Privacy: I keep photo references for your session only (auto-expires), and I "
    "never store the generated captions on the server. Send /help for the walkthrough."
)


# --- capability Q&A (deterministic, zero-cost) ------------------------------
# Each entry: (keyword triggers, answer). First entry whose triggers best match
# the user's text wins; ``answer_question`` always returns something useful.
_FAQ: List[Tuple[Tuple[str, ...], str]] = [
    (
        ("platform", "instagram", "tinder", "bumble", "hinge", "linkedin",
         "facebook", "which app", "what app"),
        f"I support {PLATFORMS_LINE}. I rank your photos per platform and write "
        "captions in the style you pick — short, long, or haiku — always without "
        "hashtags.",
    ),
    (
        ("grade", "grading", "score", "scoring", "rank", "ranking", "rate",
         "how do you decide", "how does it work", "how do you work"),
        "I grade in 3 tiers: (1) a local quality check — sharpness, resolution, "
        "faces, duplicates; (2) platform-fit rules tuned per app; (3) an AI vision "
        "read scoring composition, lighting, subject, and mood. I combine those into "
        "a per-platform score, then rank your batch.",
    ),
    (
        ("caption", "bio", "hashtag", "length", "short", "long", "haiku", "write", "text"),
        "I write 3 caption options per photo in the style you choose: Short "
        "(<200 chars), Long (<1000), or Haiku. No hashtags, ever. Use /length to "
        "change the style, /retry for a different option, or /shorter to tighten it.",
    ),
    (
        ("cost", "free", "price", "pricing", "pay", "subscription", "money"),
        "I'm free to use — I run on free-tier AI and local processing, so there's no "
        "per-photo cost.",
    ),
    (
        ("limit", "limitation", "can't", "cannot", "cant", "weakness", "drawback",
         "what can you not"),
        "Limits: photos only (no video); Tinder/Bumble need a visible face; I "
        "suggest but don't post for you and can't promise matches; I read the image "
        "and your text but don't browse the web. Large batches may be graded in "
        "parts due to free-tier rate limits.",
    ),
    (
        ("privacy", "data", "store", "stored", "save", "keep my", "delete",
         "gdpr", "secure"),
        "Privacy: I keep photo references only for your active session, which "
        "auto-expires, and I don't store generated captions on the server. Use "
        "/reset to clear your session at any time.",
    ),
    (
        ("face", "selfie", "no face", "portrait"),
        "Faces matter most for dating: Tinder and Bumble require a visible face, and "
        "a clear close-up face is the top signal. A no-face photo gets flagged and "
        "scored low for those apps (but can still work for Instagram, Hinge, or "
        "LinkedIn).",
    ),
    (
        ("start", "begin", "how to use", "how do i", "get started", "use you",
         "instruction", "guide", "tutorial"),
        "Quick start: send /start, pick your platforms, optionally tell me the vibe "
        "or constraints, then send your photos and say \"done\". I'll grade, rank, "
        "and give you 3 caption options per top photo. Full walkthrough: /help.",
    ),
    (
        ("command", "commands", "what can you do", "capabilities", "features",
         "what do you do", "help"),
        None,  # filled below with command_reference()
    ),
]


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def looks_like_question(text: str) -> bool:
    """Heuristic: is this free text a question about the bot (vs. an intent)?

    True if it ends with '?', or opens with a question word, or mentions the bot's
    capabilities/limits. Kept conservative so real intent text ("witty, beachy")
    still flows to intent parsing.
    """
    t = _normalize(text)
    if not t:
        return False
    if t.endswith("?"):
        return True
    openers = (
        "what", "how", "can you", "do you", "are you", "who", "why", "when",
        "where", "which", "tell me", "explain", "is there", "could you",
    )
    if t.startswith(openers):
        return True
    topic_words = (
        "capabilit", "limitation", "command", "feature", "help me", "how do you",
        "what do you do", "what can you", "privacy", "free?", "cost", "support",
    )
    return any(w in t for w in topic_words)


def answer_question(text: str) -> str:
    """Return a helpful answer to a capability/usage/limits question.

    Deterministic and zero-cost: scores the text against the curated FAQ and
    returns the best match, falling back to a capabilities overview. Always
    returns a non-empty, useful string.
    """
    t = _normalize(text)
    best_idx = -1
    best_score = 0
    for idx, (triggers, _answer) in enumerate(_FAQ):
        score = sum(1 for kw in triggers if kw in t)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx >= 0 and best_score > 0:
        answer = _FAQ[best_idx][1]
        if answer is None:  # the "commands / what can you do" entry
            return (
                "I grade your photos and write platform-specific captions for "
                f"{PLATFORMS_LINE}.\n\n" + command_reference() +
                "\n\nSend /help for the full walkthrough or /about for my limits."
            )
        return answer

    # No keyword hit — give the overview rather than a dead end.
    return (
        "I grade your photos and write platform-specific captions for "
        f"{PLATFORMS_LINE}. Ask me about platforms, grading, captions, privacy, or "
        "my limits — or send /help to get started and /about for capabilities."
    )
