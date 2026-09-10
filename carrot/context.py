"""Conversation-derived context for Slice 2.

Owns all turn-scanning derivation: question tracking, play frame,
repair support, and lexical topics. Pure and deterministic: no I/O,
no emotional-state access, no floats. The classifier (Phase 2) and
behavior read these flags instead of re-scanning raw turns.

Cue sets below intentionally mirror the classifier's cue tuples
(which stay classifier-owned for event decisions). This module must
not import from ``classifier`` or ``carrot`` to keep the dependency
direction one-way.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .conversation import Turn

# A bare acknowledgment is not a real question, even with "?".
FILLER_BEFORE_QUESTION = frozenset({
    "yeah", "ok", "okay", "right", "huh", "hm", "hmm", "mm",
    "sure", "yes", "no",
})

# Play cues mirror the classifier's humor/play set.
PLAY_CUES = ("lol", "lmao", "haha", "\U0001F602", "\U0001F923",
             "kidding", "teasing", "joke", "playful")

# Serious cues mirror the classifier's insult set: a harsh turn closes
# the play frame even if an older turn opened it.
SERIOUS_CUES = ("useless", "stupid", "dumb", "idiot", "hate you",
                "shut up", "worthless", "pathetic", "trash")

# Tease cues mirror the classifier's humor set for harshness anchoring.
TEASE_CUES = ("lol", "lmao", "haha", "\U0001F602", "\U0001F923",
              " funny", "joke")

# Hedge cues are NOT evaluated here. Whether the live message is a hedge
# is decided by the classifier (current text) and behavior (user_text);
# this flag is only the harsh-side precondition. (No hedge set is kept
# in this module.)
REPAIR_WINDOW = 5
TOPIC_CAP = 5
MAX_TOKEN_LEN = 20

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "it",
    "that", "this", "i", "you", "we", "for", "with", "at", "my",
    "your", "me", "be", "are", "was", "so", "but", "just", "like",
    "as", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "not", "hey", "hi", "hello", "oh",
    "yeah", "ok", "okay", "lol", "haha", "im", "ive", "dont", "cant",
    "its", "us", "our", "they", "them", "he", "she", "what", "when",
    "how", "why", "there", "here", "out", "now",
    "about", "after", "again", "against", "because", "before", "being",
    "between", "doing", "down", "during", "each", "from", "further",
    "having", "hers", "herself", "himself", "into", "itself", "more",
    "most", "other", "over", "same", "some", "such", "than", "their",
    "theirs", "themselves", "then", "these", "they", "those", "through",
    "under", "until", "very", "were", "where", "which", "while",
    "yours", "yourself", "need", "needs",
})

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def tokenize(text: str) -> tuple:
    """Normalize one text into tokens.

    Lowercase, alphabetic-only, length >= 2, stopword-filtered,
    order-preserving with duplicates removed. Pure.
    """
    seen: list = []
    for raw in _TOKEN_RE.findall(str(text or "").lower()):
        token = raw[:MAX_TOKEN_LEN]
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.append(token)
    return tuple(seen)


def _text_of(turn) -> str:
    if isinstance(turn, dict):
        return str(turn.get("text", turn.get("content", "")))
    return str(getattr(turn, "text", ""))


def _role_of(turn) -> str:
    if isinstance(turn, dict):
        return str(turn.get("role", ""))
    return str(getattr(turn, "role", ""))


def _has_any(haystack: str, cues) -> bool:
    return any(cue in haystack for cue in cues)


@dataclass(frozen=True)
class ConversationContext:
    turns: tuple = ()
    pending_question: bool = False
    playful_frame: bool = False
    recent_harshness: bool = False
    recent_topics: tuple = ()


def _pending_question(turns: Sequence) -> bool:
    if not turns:
        return False
    last = turns[-1]
    if _role_of(last) != "carrot":
        return False
    text = _text_of(last).strip()
    if not text.endswith("?"):
        return False
    before = text[:-1].strip().lower()
    return bool(before) and before not in FILLER_BEFORE_QUESTION


def _playful_frame(turns: Sequence) -> bool:
    # Newest user turn bearing a cue wins; Carrot turns never vote.
    for turn in reversed(list(turns)):
        if _role_of(turn) != "user":
            continue
        lowered = _text_of(turn).lower()
        if _has_any(lowered, PLAY_CUES):
            return True
        if _has_any(lowered, SERIOUS_CUES):
            return False
    return False


def _recent_harshness(turns: Sequence) -> bool:
    # Precondition only, not a verdict: harsh/tease support for a repair
    # lives in the last 5 user turns. The hedge half is checked against
    # the live message by the classifier, not here.
    user_turns = [t for t in turns if _role_of(t) == "user"]
    recent = user_turns[-REPAIR_WINDOW:]
    return any(_has_any(_text_of(t).lower(), SERIOUS_CUES + TEASE_CUES)
               for t in recent)


def _recent_topics(turns: Sequence) -> tuple:
    seen: list = []
    for turn in turns:
        if _role_of(turn) != "user":
            continue
        for token in tokenize(_text_of(turn)):
            if token in seen:
                continue
            seen.append(token)
            if len(seen) >= TOPIC_CAP:
                return tuple(seen)
    return tuple(seen)


def derive_context(turns: Sequence) -> ConversationContext:
    """Derive read-only context flags from a turn sequence."""
    seq = tuple(turns or ())
    return ConversationContext(
        turns=seq,
        pending_question=_pending_question(seq),
        playful_frame=_playful_frame(seq),
        recent_harshness=_recent_harshness(seq),
        recent_topics=_recent_topics(seq),
    )
