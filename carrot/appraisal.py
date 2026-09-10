"""Emotional appraisal: event -> base delta intent.

The table is personality-free and lives in config. Personality enters
in exactly one justified place: ``sensitivity`` scales how strongly an
insult lands as hurt. Everything else is scaled only by the event's
own intensity. Appraisal never touches state; it returns an intent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .classifier import EVENT_TYPES
from .emotions import EMOTIONS


@dataclass(frozen=True)
class AppraisalIntent:
    event_type: str
    deltas: dict = field(default_factory=dict)
    reason: str = ""


class Appraisal:
    def __init__(self, table: dict):
        self._table = {}
        for event, deltas in (table or {}).items():
            if event not in EVENT_TYPES:
                raise ValueError(f"unknown event type: {event!r}")
            clean = {}
            for key, value in (deltas or {}).items():
                if key not in EMOTIONS and key != "energy":
                    raise ValueError(f"bad appraisal key: {key!r}")
                clean[key] = round(float(value), 4)
            self._table[event] = clean

    @classmethod
    def load(cls, path: str) -> "Appraisal":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def appraise(self, event, personality) -> AppraisalIntent:
        base = self._table.get(event.type, {})
        strength = max(0.0, min(1.0, float(event.intensity)))
        deltas = {k: round(v * strength, 4) for k, v in base.items()}
        if event.type == "insult" and "hurt" in deltas:
            # The single justified sensitivity gate: same insult,
            # thinner skin hurts more. Factor spans 0.6..1.4.
            factor = 0.6 + 0.8 * float(personality.sensitivity)
            deltas["hurt"] = round(deltas["hurt"] * factor, 4)
        return AppraisalIntent(event_type=event.type, deltas=deltas,
                               reason=event.reason)
