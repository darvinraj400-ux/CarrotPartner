"""Avatar expression mapping — pure, no tkinter, no state, no I/O.

Contract:
- The avatar is a read-only expression layer over existing state.
- This module contains only the mapping from (mood, behavior, band,
  closeness band) to a visual Expression. Phase 2 adds the window.
- No imports of carrot.carrot, carrot.cli, carrot.llm. No process_turn.
  No generate. No after()/after_idle(). No file writes.
- Slice 12 is visual presence only; voice (Slice 13) attaches via
  expression_for, not via the window.
"""
from __future__ import annotations

from dataclasses import dataclass

from .mood import MOODS

VISUAL_STATES = (
    "content", "happy", "tender", "sad", "tense", "shy", "curious",
)
INTENSITY_BANDS = ("low", "medium", "high")


@dataclass(frozen=True)
class Expression:
    name: str            # one of VISUAL_STATES
    intensity_band: str  # one of INTENSITY_BANDS
    palette_key: str     # equals name for now; a stable hook for Phase 2
    presence_cue: str | None  # None or "warm"

    def __post_init__(self) -> None:
        if self.name not in VISUAL_STATES:
            raise ValueError(f"bad visual state: {self.name!r}")
        if self.intensity_band not in INTENSITY_BANDS:
            raise ValueError(
                f"bad intensity band: {self.intensity_band!r}")
        if self.presence_cue is not None and self.presence_cue != "warm":
            raise ValueError(
                f"bad presence cue: {self.presence_cue!r}")
        if not isinstance(self.palette_key, str) or not self.palette_key:
            raise ValueError("palette_key must be a non-empty string")


def expression_for(
    mood: str,
    behavior_name: str,
    band: str,
    closeness_band: str | None = None,
) -> Expression:
    """Map (mood, behavior_name, band, closeness_band) to an Expression.

    Behavior-name overrides (behavior wins over mood):
      - behavior_name in ("shy", "embarrassed") and mood != "low" -> "shy"
      - behavior_name == "curious" and mood != "bright"           -> "curious"

    Mood-driven (no override applies):
      - mood == "bright"  -> "happy"
      - mood == "low"     -> "sad"
      - mood == "tense"   -> "tense"
      - mood == "tender"  -> "tender"

    Fallback:
      - otherwise -> "content"

    intensity_band is passed through (normalized against INTENSITY_BANDS;
    unknown band -> "medium").

    presence_cue is "warm" iff closeness_band == "high", else None.

    Pure: no I/O, no clock, no state reads, deterministic.
    """
    clean_mood = mood if isinstance(mood, str) and mood in MOODS \
        else "content"
    clean_behavior = behavior_name \
        if isinstance(behavior_name, str) else ""
    clean_band = band if band in INTENSITY_BANDS else "medium"
    if clean_behavior in ("shy", "embarrassed") and clean_mood != "low":
        name = "shy"
    elif clean_behavior == "curious" and clean_mood != "bright":
        name = "curious"
    elif clean_mood == "bright":
        name = "happy"
    elif clean_mood == "low":
        name = "sad"
    elif clean_mood == "tense":
        name = "tense"
    elif clean_mood == "tender":
        name = "tender"
    else:
        name = "content"
    cue = "warm" if closeness_band == "high" else None
    return Expression(name=name, intensity_band=clean_band,
                      palette_key=name, presence_cue=cue)
