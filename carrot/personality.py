"""Static personality traits. Never mutated at runtime in Slice 1."""
from __future__ import annotations

import json
from dataclasses import dataclass

TRAITS = ("sensitivity", "playfulness", "stoicism",
          "shyness", "pride", "warmth")


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
        return "Carrot is " + ", ".join(bits) + "."
