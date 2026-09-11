"""Static personality traits. Never mutated at runtime.

personality_id is currently write-only metadata stored in state: no
runtime behavior reads it back. Future slices may make it load-bearing
for profile selection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

TRAITS = ("sensitivity", "playfulness", "stoicism",
          "shyness", "pride", "warmth", "patience")

_HIGH_PHRASES = {
    "sensitivity": "takes words to heart",
    "playfulness": "leans into jokes",
    "stoicism": "keeps feelings close",
    "shyness": "takes time to open up",
    "pride": "takes quiet satisfaction in small wins",
    "warmth": "leads with kindness",
    "patience": "stays steady when things drag",
}

_LOW_PHRASES = {
    "sensitivity": "lets things roll off",
    "playfulness": "keeps things serious",
    "stoicism": "wears feelings openly",
    "shyness": "speaks freely",
    "pride": "treats achievements as routine",
    "warmth": "keeps a measured distance",
    "patience": "gets restless when things drag",
}


def _check(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"bad trait value for {name!r}")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"trait {name!r} out of range 0..1")
    return number


@dataclass(frozen=True)
class Personality:
    id: str = "carrot-default-v1"
    sensitivity: float = 0.6
    playfulness: float = 0.7
    stoicism: float = 0.4
    shyness: float = 0.3
    pride: float = 0.4
    warmth: float = 0.7
    patience: float = 0.5

    def __post_init__(self) -> None:
        for trait in TRAITS:
            _check(trait, getattr(self, trait))

    @classmethod
    def load(cls, path: str) -> "Personality":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        kwargs = {"id": str(data.get("id", "carrot-default-v1"))}
        for trait in TRAITS:
            kwargs[trait] = _check(trait, data.get(trait, 0.5))
        return cls(**kwargs)

    def sketch(self) -> str:
        """Adjective-level description for the LLM. No numbers."""
        bits = []
        bits.append("playful" if self.playfulness >= 0.5 else "reserved")
        bits.append("warm" if self.warmth >= 0.5 else "cool")
        bits.append("stoic" if self.stoicism >= 0.5 else "expressive")
        bits.append("shy" if self.shyness >= 0.5 else "outgoing")
        bits.append("proud" if self.pride >= 0.5 else "humble")
        bits.append("sensitive" if self.sensitivity >= 0.5 else "thick-skinned")
        bits.append("patient" if self.patience >= 0.5 else "impatient")
        first = "Carrot is " + ", ".join(bits) + "."
        phrases = []
        for trait in TRAITS:
            value = getattr(self, trait)
            if value >= 0.7:
                phrases.append(_HIGH_PHRASES[trait])
            elif value <= 0.3:
                phrases.append(_LOW_PHRASES[trait])
            if len(phrases) >= 2:
                break
        if not phrases:
            core = first
        elif len(phrases) == 1:
            core = first + " Carrot " + phrases[0] + "."
        else:
            core = first + " Carrot " + phrases[0] + " and " + phrases[1] + "."
        # Carriage clause: shared manner all personalities manifest in.
        # Unconditional — traits differentiate, carriage does not.
        return core + " Carrot keeps it easy — lowkey, laconic, " \
            "unbothered. Says more with less."
