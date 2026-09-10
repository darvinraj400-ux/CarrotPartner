"""Derived mood: a slow background label, never a second emotion system."""
from __future__ import annotations

MOODS = ("content", "low", "tender", "tense", "bright")

_MARGIN = 0.05


def _candidate(emotions: dict) -> tuple[str, float]:
    get = lambda k: float(emotions.get(k, 0.0))
    sadness, hurt = get("sadness"), get("hurt")
    anger = get("anger")
    affection, happiness = get("affection"), get("happiness")
    excitement = get("excitement")
    if sadness >= 0.55 or hurt >= 0.6:
        return "low", max(sadness, hurt)
    if anger >= 0.55:
        return "tense", anger
    if affection >= 0.6 and happiness >= 0.55:
        return "tender", min(affection, happiness)
    if happiness >= 0.65 or excitement >= 0.6:
        return "bright", max(happiness, excitement)
    return "content", 0.0


def derive(emotions: dict, prev_mood: str = "content") -> str:
    """Map emotions to a mood label with a small hysteresis margin.

    Entering a charged mood happens as soon as its threshold is met.
    Leaving one requires signals to fall clearly below the trigger, so
    mood reads as a slow background state rather than flickering.
    """
    cand, _ = _candidate(emotions)
    if cand != "content" or prev_mood not in MOODS or prev_mood == "content":
        return cand
    get = lambda k: float(emotions.get(k, 0.0))
    if prev_mood == "low":
        if get("sadness") >= 0.55 - _MARGIN or get("hurt") >= 0.6 - _MARGIN:
            return "low"
    elif prev_mood == "tense":
        if get("anger") >= 0.55 - _MARGIN:
            return "tense"
    elif prev_mood == "tender":
        if get("affection") >= 0.6 - _MARGIN and get("happiness") >= 0.55 - _MARGIN:
            return "tender"
    elif prev_mood == "bright":
        if get("happiness") >= 0.65 - _MARGIN or get("excitement") >= 0.6 - _MARGIN:
            return "bright"
    return "content"
