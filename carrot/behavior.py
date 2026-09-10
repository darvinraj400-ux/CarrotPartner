"""Behavior selection: how internal state is expressed.

Deterministic threshold table over (emotions, energy, personality,
recent causes, current context). Personality acts HERE -- as gates on
expression, not as multipliers on feelings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .classifier import _INSULT_CUES, _REPAIR_CUES
from .context import ConversationContext

# Warmth shifts the affection gate: higher warmth lowers the bar.
WARMTH_WEIGHT = 0.10

# Safety rails for modulated thresholds: emotion values are in [0, 1];
# thresholds are meaningful anywhere in (0, 1); these bounds prevent
# pathological 0.0/1.0 thresholds without restricting modulation.
THRESHOLD_SAFETY_MIN = 0.05
THRESHOLD_SAFETY_MAX = 0.95

# Patience shifts the frustration gates: higher patience raises the bar,
# so it takes more anger to read as annoyed or frustrated.
PATIENCE_WEIGHT = 0.10


def _patience_threshold(base: float, patience: float) -> float:
    return max(THRESHOLD_SAFETY_MIN,
               min(THRESHOLD_SAFETY_MAX,
                   base + (patience - 0.5) * PATIENCE_WEIGHT))

BEHAVIORS = ("normal", "playful", "excited", "sad", "annoyed", "sulking",
             "quiet", "shy", "affectionate", "embarrassed", "curious",
             "proud", "frustrated")

# Proud trigger sets: high pride also takes pride in shared jokes and
# affection, not only achievements and praise. Below PROUD_MIN_PRIDE
# the gate cannot fire at all.
PROUD_MIN_PRIDE = 0.3
PROUD_HIGH_PRIDE_CUTOFF = 0.7
BASE_PROUD_EVENTS = (
    "achievement",
    "compliment",
)
PROUD_HIGH_PRIDE_EVENTS = (
    "achievement",
    "compliment",
    "joke",
    "affection",
)


@dataclass(frozen=True)
class Behavior:
    name: str
    band: str  # low | med | high


def _band(strength: float) -> str:
    if strength >= 0.7:
        return "high"
    if strength >= 0.45:
        return "med"
    return "low"


def _contains_any(text, cues) -> bool:
    return any(cue in str(text or "").lower() for cue in cues)


def _user_answered(user_text: str) -> bool:
    lowered = str(user_text or "").strip().lower()
    if not lowered or lowered.endswith("?"):
        return False
    if _contains_any(lowered, _REPAIR_CUES):
        return False
    return not _contains_any(lowered, _INSULT_CUES)


class BehaviorSelector:
    def __init__(self, thresholds: dict):
        self._t = dict(thresholds or {})

    @classmethod
    def load(cls, path: str) -> "BehaviorSelector":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def _get(self, key: str, default: float) -> float:
        try:
            return float(self._t.get(key, default))
        except (TypeError, ValueError):
            return default

    def _affection_threshold(self, personality) -> float:
        base = self._get("affectionate", 0.65)
        adjusted = base - (float(personality.warmth) - 0.5) * WARMTH_WEIGHT
        return max(THRESHOLD_SAFETY_MIN,
                   min(THRESHOLD_SAFETY_MAX, adjusted))

    def pick(self, state, personality, context: ConversationContext,
             user_text: str) -> Behavior:
        e = state.emotions
        t = self._get
        causes = state.recent_causes
        last_event = causes[-1].get("event") if causes else None

        emb = e["embarrassment"]
        if emb >= t("embarrassed", 0.5):
            if personality.shyness >= 0.4:
                return Behavior("shy", _band(emb))
            return Behavior("embarrassed", _band(emb))

        hurt, anger = e["hurt"], e["anger"]
        if hurt >= t("sulk_hurt", 0.55) and anger >= t("sulk_anger", 0.3):
            peak = max(hurt, anger)
            if personality.stoicism >= 0.5:
                return Behavior("sulking", _band(peak))
            return Behavior("sad", _band(peak))

        if e["sadness"] >= t("sad", 0.55):
            return Behavior("sad", _band(e["sadness"]))

        if anger >= _patience_threshold(t("annoyed", 0.55),
                                          personality.patience):
            return Behavior("annoyed", _band(anger))
        if anger >= _patience_threshold(t("frustrated_anger", 0.4),
                                        personality.patience
                                        ) and state.energy < t("low_energy", 0.35):
            return Behavior("frustrated", _band(anger))

        exc, hap, aff = e["excitement"], e["happiness"], e["affection"]
        if exc >= t("playful_excitement", 0.55) and personality.playfulness >= 0.5:
            return Behavior("playful", _band(exc))
        if (context.playful_frame and personality.playfulness >= 0.5
                and (exc >= t("playful_excitement", 0.55)
                     or hap >= t("happy", 0.65))):
            return Behavior("playful", _band(max(exc, hap)))
        if context.recent_harshness and _contains_any(user_text,
                                                      _REPAIR_CUES):
            if aff >= self._affection_threshold(personality) and hap >= 0.5:
                return Behavior("affectionate", _band(aff))
            return Behavior("normal", "low")
        if aff >= self._affection_threshold(personality) and hap >= 0.5:
            return Behavior("affectionate", _band(aff))
        if exc >= t("excited", 0.6):
            return Behavior("excited", _band(exc))
        if hap >= t("happy", 0.65):
            if personality.playfulness >= 0.5:
                return Behavior("playful", _band(hap))
            return Behavior("excited", _band(hap))

        if personality.pride < PROUD_MIN_PRIDE:
            allowed_events = ()  # too humble: gate cannot fire
        elif personality.pride >= PROUD_HIGH_PRIDE_CUTOFF:
            allowed_events = PROUD_HIGH_PRIDE_EVENTS
        else:
            allowed_events = BASE_PROUD_EVENTS
        if (last_event in allowed_events
                and hap >= t("proud_happiness", 0.45)):
            return Behavior("proud", _band(hap))

        if state.energy < t("low_energy", 0.35):
            return Behavior("quiet", _band(1.0 - state.energy))

        # A question from the user invites curiosity. This reads the live
        # message (not context) so Carrot's own "yeah?" can never
        # self-trigger it, including on the very first turn.
        if context.pending_question and _user_answered(user_text):
            return Behavior("curious", "low")
        if str(user_text or "").strip().endswith("?"):
            return Behavior("curious", "low")

        return Behavior("normal", "low")
