"""Initiative decision function (Slice 9, Phase 1: pure logic only).

Decides whether Carrot should initiate, and why. No store, no CLI, no
Carrot wiring, no actuation: callers compute a decision from state they
already hold. snapshot and relationship are accepted but ignored; the
parameter slots document the Slice 10/11 seams.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .drives import DriveState
from .environment import EnvironmentSnapshot

REASONS = ("long-absence", "high-social-drive", "session-open",
           "user-returned")
ACTIVE_REASONS = ("long-absence", "high-social-drive", "session-open")

_DEFAULT_COOLDOWNS = {
    "long-absence": 43200.0,       # 12h
    "high-social-drive": 14400.0,  # 4h
    "session-open": 14400.0,       # 4h
    "user-returned": 7200.0,       # 2h (reserved; unused in Slice 9)
}


@dataclass(frozen=True)
class InitiateDecision:
    reason: str  # one of REASONS


@dataclass(frozen=True)
class InitiativeConfig:
    long_absence_threshold_s: float   # default 21600 (6h; mirrors return)
    high_social_drive_threshold: float  # default 0.6
    cooldowns: Mapping[str, float]    # reason -> seconds

    @classmethod
    def load(cls, path: str | Path) -> "InitiativeConfig":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise ValueError(f"bad initiative config: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("initiative config must be an object")
        try:
            long_absence = float(data.get("long_absence_threshold_s",
                                          21600.0))
        except (TypeError, ValueError):
            raise ValueError("bad long_absence_threshold_s")
        if long_absence < 0:
            raise ValueError("long_absence_threshold_s must be non-negative")
        try:
            social = float(data.get("high_social_drive_threshold", 0.6))
        except (TypeError, ValueError):
            raise ValueError("bad high_social_drive_threshold")
        if not 0.0 <= social <= 1.0:
            raise ValueError("high_social_drive_threshold out of range 0..1")
        raw_cooldowns = data.get("cooldowns", {})
        if not isinstance(raw_cooldowns, dict):
            raise ValueError("cooldowns must be an object")
        cooldowns = dict(_DEFAULT_COOLDOWNS)
        try:
            for key, value in raw_cooldowns.items():
                number = float(value)
                if number < 0:
                    raise ValueError(f"bad cooldown for {key!r}")
                cooldowns[str(key)] = number
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"bad cooldowns: {exc}") from exc
        return cls(long_absence_threshold_s=long_absence,
                   high_social_drive_threshold=social,
                   cooldowns=MappingProxyType(cooldowns))


def should_initiate(
    drives: DriveState,
    elapsed_s: float,
    snapshot: EnvironmentSnapshot,
    per_reason_last_fire: Mapping[str, float],
    now: float,
    cfg: InitiativeConfig,
    relationship: float | None = None,
) -> InitiateDecision | None:
    """Decide whether Carrot should initiate, and why.

    Priority order (highest first):
        1. "long-absence"      if elapsed_s >= cfg.long_absence_threshold_s
        2. "high-social-drive" if drives.social >= cfg.high_social_drive_threshold
        3. "session-open"      always qualifies (fallback)

    A qualifying reason is returned only if its cooldown has elapsed:
        now - per_reason_last_fire.get(reason, 0.0) >= cfg.cooldowns.get(reason, 0.0)

    If no qualifying reason has an expired cooldown, returns None.
    "user-returned" is reserved and never returned here.
    snapshot and relationship are unused in Slice 9 (typed for future slices).
    Pure: no I/O, no clock, no mutation of inputs.
    """
    del snapshot, relationship  # accepted but ignored in Slice 9
    try:
        elapsed = float(elapsed_s)
    except (TypeError, ValueError):
        elapsed = 0.0
    now = float(now)
    fired = per_reason_last_fire or {}
    for reason in ACTIVE_REASONS:
        if reason == "long-absence":
            qualifies = elapsed >= cfg.long_absence_threshold_s
        elif reason == "high-social-drive":
            qualifies = (float(drives.social)
                         >= cfg.high_social_drive_threshold)
        else:  # "session-open" always qualifies as the fallback
            qualifies = True
        if not qualifies:
            continue
        try:
            last = float(fired.get(reason, 0.0))
        except (TypeError, ValueError):
            last = 0.0
        cooldown = float(cfg.cooldowns.get(reason, 0.0))
        if now - last >= cooldown:
            return InitiateDecision(reason)
    return None
