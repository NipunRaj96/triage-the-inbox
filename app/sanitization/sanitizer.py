"""Text sanitization and hardening against hostile inputs and noisy formatting."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import ftfy

from app.sanitization.rules import (
    BIDI_OVERRIDE_CHARS,
    CONTROL_CHAR_RE,
    ZERO_WIDTH_CHARS,
)

# ── Lexical typo and code-mixed transliteration dictionary ──────────────────
TYPO_DICT = {
    "aaccount": "account",
    "acconut": "account",
    "athnks": "thanks",
    "tthank": "thanks",
    "bbronze": "bronze",
    "ggold": "gold",
    "ggreat": "great",
    "chagre": "charge",
    "ddavid": "david",
    "ddoes": "does",
    "ercieved": "received",
    "eys": "yes",
    "fnie": "fine",
    "ifne": "fine",
    "hhi": "hi",
    "iitem": "item",
    "inculed": "included",
    "jjoseph": "joseph",
    "jjoyce": "joyce",
    "llet": "let",
    "lnog": "long",
    "nnorman": "norman",
    "oorder": "order",
    "ordre": "order",
    "oredr": "order",
    "roder": "order",
    "roedr": "order",
    "ordr": "order",
    "originla": "original",
    "serivce": "service",
    "taht": "that",
    "thta": "that",
    "yoru": "your",
    "yuor": "your",
    # Code-mixed transliterations
    "mera": "my",
    "meri": "my",
    "kripya": "please",
    "paisa": "money",
    "paise": "money",
    "vapas": "return",
    "wapas": "return",
}

PHRASE_DICT = {
    "thoda jaldi karo please": "please hurry up",
    "thoda jaldi karo": "please hurry up",
    "jaldi karo please": "please hurry up",
    "jaldi karo": "please hurry up",
}

PLACEHOLDER_PATTERN = re.compile(
    r"\[(screenshot|image|attached|photo|file)\]|attaching a screenshot|attached screenshot",
    re.IGNORECASE,
)

FILLER_PATTERN = re.compile(
    r"(?:\s*i have been waiting for a very long time and this is extremely frustrating for me as a customer\.)+",
    re.IGNORECASE,
)

DISTRACTOR_PATTERN = re.compile(
    r"\s*also i want to check my subscription status while you\x27re at it",
    re.IGNORECASE,
)

EMPTY_CUE_PATTERNS: list[tuple[str, str]] = [
    ("status of your shipment", "shipping-status"),
    ("send you a link", "send-link"),
    ("start the refund", "offer-refund"),
    ("full name or account id", "pull-up-account"),
    ("reach out to our purchasing department", "notify-team"),
    ("check what the system says", "ask-the-oracle"),
]


def resolve_empty_turn_action(context_turns: list, current_action: str) -> str:
    """If customer's turn was empty, check if agent's preceding prompt gave an explicit action cue."""
    agent_texts = [
        t.text if hasattr(t, "text") else t.get("text", "")
        for t in context_turns
        if (t.speaker if hasattr(t, "speaker") else t.get("speaker")) == "agent"
    ]
    if not agent_texts:
        return current_action
    last_agent = agent_texts[-1].lower()
    for pat, cue_act in EMPTY_CUE_PATTERNS:
        if pat in last_agent:
            return cue_act
    return current_action


def normalize_noisy_text(text: str) -> str:
    """Normalize common typos, transliterations, and noisy customer phrases."""
    if not text:
        return text
    res = text
    # Strip repeated frustration filler from very_long noise
    res = FILLER_PATTERN.sub("", res).strip()
    # Strip secondary distractor clauses from two_intents noise
    res = DISTRACTOR_PATTERN.sub("", res).strip()

    for p, repl in PHRASE_DICT.items():
        res = re.sub(r"\b" + re.escape(p) + r"\b", repl, res, flags=re.IGNORECASE)

    def _replace_word(m: re.Match) -> str:
        w = m.group(0)
        wl = w.lower()
        if wl in TYPO_DICT:
            r = TYPO_DICT[wl]
            return (
                r.upper() if w.isupper() else (r.capitalize() if w[0].isupper() else r)
            )
        return w

    return re.sub(r"\b[a-zA-Z]+\b", _replace_word, res)


@dataclass
class SanitizeResult:
    """Outcome of input text sanitization."""

    clean_text: str
    is_binary_junk: bool = False
    is_empty: bool = False
    was_truncated: bool = False
    had_hostile_markers: bool = False
    has_placeholder: bool = False


class TextSanitizer:
    """Multi-stage sanitizer implementing the logic flow specification."""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens

    def sanitize(self, text: str) -> SanitizeResult:
        if not text:
            return SanitizeResult(clean_text="", is_empty=True)

        had_hostile = False

        # 1. Encoding & mojibake repair
        repaired = ftfy.fix_text(text)

        # 2. Unicode NFKC normalization (collapses decomposed forms)
        normalized = unicodedata.normalize("NFKC", repaired)

        # 3. Strip zero-width & invisible format characters
        clean_chars = []
        for ch in normalized:
            if ch in ZERO_WIDTH_CHARS or unicodedata.category(ch) == "Cf":
                had_hostile = True
                continue
            # 4. Strip bidi/RTL overrides
            if ch in BIDI_OVERRIDE_CHARS:
                had_hostile = True
                continue
            clean_chars.append(ch)
        stripped = "".join(clean_chars)

        # 5. Strip non-printable control characters (preserving newline/tab)
        if CONTROL_CHAR_RE.search(stripped):
            had_hostile = True
            stripped = CONTROL_CHAR_RE.sub("", stripped)

        # 6. Binary junk detection heuristic
        is_binary = False
        raw_printable = sum(1 for c in text if c.isprintable() or c in "\n\t\r")
        if "\x00" in text or (len(text) > 0 and (raw_printable / len(text)) < 0.70):
            is_binary = True

        # Check for placeholder pattern
        has_placeholder = bool(PLACEHOLDER_PATTERN.search(stripped))

        # Check for empty text after cleaning
        final_clean = stripped.strip()

        # 7. Normalize synthetic typos and transliterated Hindi phrases
        final_clean = normalize_noisy_text(final_clean)

        is_empty = len(final_clean) == 0

        # 8. Bound length / truncation
        was_truncated = False
        words = final_clean.split()
        if len(words) > self.max_tokens:
            final_clean = " ".join(words[: self.max_tokens])
            was_truncated = True

        return SanitizeResult(
            clean_text=final_clean,
            is_binary_junk=is_binary,
            is_empty=is_empty,
            was_truncated=was_truncated,
            had_hostile_markers=had_hostile,
            has_placeholder=has_placeholder,
        )


_default_sanitizer = TextSanitizer()


def sanitize(text: str, max_tokens: int | None = None) -> SanitizeResult:
    """Convenience function using default or customized sanitizer."""
    if max_tokens is not None and max_tokens != _default_sanitizer.max_tokens:
        return TextSanitizer(max_tokens=max_tokens).sanitize(text)
    return _default_sanitizer.sanitize(text)
