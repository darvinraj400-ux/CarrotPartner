"""Behavior selection: how internal state is expressed.

Deterministic threshold table over (emotions, energy, personality,
recent causes, current context). Personality acts HERE -- as gates on
expression, not as multipliers on feelings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

BEHAVIORS = ("normal", "playful", "excited", "sad", "annoyed", "sulking",
             "quiet", "shy", "affectionate", "embarrassed", "curious",
             "proud", "frustrated")


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

    def pick(self, state, personality, context=None,
             user_text: str = "") -> Behavior:
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

        if anger >= t("annoyed", 0.55):
            return Behavior("annoyed", _band(anger))
        if anger >= t("frustrated_anger", 0.4) and state.energy < t("low_energy", 0.35):
            return Behavior("frustrated", _band(anger))

        exc, hap, aff = e["excitement"], e["happiness"], e["affection"]
        if exc >= t("playful_excitement", 0.55) and personality.playfulness >= 0.5:
            return Behavior("playful", _band(exc))
        if aff >= t("affectionate", 0.65) and hap >= 0.5:
            return Behavior("affectionate", _band(aff))
        if exc >= t("excited", 0.6):
            return Behavior("excited", _band(exc))
        if hap >= t("happy", 0.65):
            if personality.playfulness >= 0.5:
                return Behavior("playful", _band(hap))
            return Behavior("excited", _band(hap))

        if (last_event in ("achievement", "compliment")
                and hap >= t("proud_happiness", 0.45)
                and personality.pride >= 0.5):
            return Behavior("proud", _band(hap))

        if state.energy < t("low_energy", 0.35):
            return Behavior("quiet", _band(1.0 - state.energy))

        # A question from the user invites curiosity. This reads the live
        # message (not context) so Carrot's own "yeah?" can never
        # self-trigger it, including on the very first turn.
        if str(user_text or "").strip().endswith("?"):
            return Behavior("curious", "low")

        return Behavior("normal", "low")
