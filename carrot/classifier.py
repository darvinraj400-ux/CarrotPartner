"""Interaction interpretation.

Slice 2: the deterministic classifier reads a pre-derived
``ConversationContext`` (see ``context.py``) instead of scanning raw
turns. Cue detection on the live message stays here; all history
scanning lives in ``context.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .context import ConversationContext

EVENT_TYPES = ("compliment", "insult", "apology", "joke", "affection",
               "repair", "achievement", "neutral", "return")


@dataclass(frozen=True)
class InteractionEvent:
    type: str
    intensity: float  # 0..1
    reason: str


class InteractionClassifier(Protocol):
    def classify(self, text: str,
                 context: ConversationContext) -> InteractionEvent:
        ...


_INSULT_CUES = ("useless", "stupid", "dumb", "idiot", "hate you",
                "shut up", "worthless", "pathetic", "trash")
_APOLOGY_CUES = ("sorry", "apologiz", "apologis", "my bad", "forgive me")
_REPAIR_CUES = ("kidding", "just joking", "didn't mean", "didnt mean",
                "don't mean", "dont mean", "just teasing", "only teasing",
                "that came out wrong")
_COMPLIMENT_CUES = ("cute", "smart", "great", "awesome", "amazing",
                    "well done", "good job", "proud of you", "thank you",
                    "thanks", "nice work")
_AFFECTION_CUES = ("love you", "miss you", "hug", "xoxo", "<3")
_ACHIEVE_CUES = ("fixed it", "fixed that", "did it", "finished",
                 "got the job", "passed", "i won", "finally fixed")
_JOKE_CUES = ("lol", "lmao", "haha", "😂", "🤣", " funny", "joke")
_LAUGHTER_EMOJI = ("😂", "🤣")


def _has_any(haystack: str, cues) -> bool:
    return any(cue in haystack for cue in cues)


def _intensity(text: str, strong: bool = False) -> float:
    value = 0.5
    if "!!" in text or "??" in text:
        value += 0.15
    words = [w.strip("!?.,").strip() for w in text.split()]
    if any(len(w) >= 3 and w.isupper() and w.isalpha() for w in words):
        value += 0.1
    repeats = sum(text.count(e) for e in _LAUGHTER_EMOJI)
    if repeats >= 3:
        value += 0.2
    elif repeats >= 2:
        value += 0.1
    if strong:
        value += 0.1
    return max(0.2, min(1.0, round(value, 3)))


class DeterministicClassifier:
    """Rule-and-cue interpreter over the live message + context flags."""

    def classify(self, text: str,
                 context: ConversationContext) -> InteractionEvent:
        raw = str(text or "")
        lowered = raw.lower().strip()
        if not lowered:
            return InteractionEvent("neutral", 0.0, "empty input")

        if _has_any(lowered, _REPAIR_CUES):
            if context.recent_harshness:
                return InteractionEvent(
                    "repair", _intensity(raw),
                    "hedge with recent harsh/tease context")
            return InteractionEvent("neutral", 0.3, "unanchored hedge")

        if _has_any(lowered, _APOLOGY_CUES):
            return InteractionEvent("apology", _intensity(raw),
                                    "explicit apology cue")

        if _has_any(lowered, _INSULT_CUES):
            if (_has_any(lowered, _LAUGHTER_EMOJI)
                    and context.playful_frame):
                return InteractionEvent(
                    "joke", _intensity(raw),
                    "hostile cue softened by laughter in a playful frame")
            return InteractionEvent("insult", _intensity(raw, strong=True),
                                    "hostile cue without a play frame")

        if _has_any(lowered, _COMPLIMENT_CUES):
            return InteractionEvent("compliment", _intensity(raw),
                                    "praise cue")

        if _has_any(lowered, _AFFECTION_CUES):
            return InteractionEvent("affection", _intensity(raw),
                                    "affection cue")

        if _has_any(lowered, _ACHIEVE_CUES):
            return InteractionEvent("achievement", _intensity(raw),
                                    "shared-win cue")

        if _has_any(lowered, _JOKE_CUES):
            return InteractionEvent("joke", _intensity(raw),
                                    "humor cue")

        if lowered.endswith("?"):
            return InteractionEvent("neutral", 0.4, "plain question")
        return InteractionEvent("neutral", 0.3, "no cue matched")
