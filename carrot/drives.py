"""Internal drives data layer (Slice 5, Phase 1: persistence only).

Drives are deprivation-accumulating needs (social, novelty): they rise
with elapsed time and fall when satisfied. This module owns storage
and pure update math only: no process_turn wiring, no behavior gates,
no prompt logic, and no emotional-state access (it must never import
``emotions``).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

DRIVES = ("social", "novelty")
DRIVE_SCHEMA_VERSION = 1
DRIVE_FLOOR = 0.0
DRIVE_CEIL = 1.0


def clamp_level(value: float) -> float:
    """Clamp to [DRIVE_FLOOR, DRIVE_CEIL]. Non-numeric → 0.0."""
    try:
        return max(DRIVE_FLOOR, min(DRIVE_CEIL, float(value)))
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class DriveState:
    social: float = 0.0
    novelty: float = 0.0

    def to_dict(self) -> dict:
        return {"social": self.social, "novelty": self.novelty}

    @classmethod
    def from_dict(cls, data: dict) -> "DriveState":
        if not isinstance(data, dict):
            raise ValueError("drive state must be an object")
        return cls(
            social=clamp_level(data.get("social", 0.0)),
            novelty=clamp_level(data.get("novelty", 0.0)),
        )


def update_rise(state: DriveState, elapsed_s: float,
                rise_per_hour: Mapping[str, float]) -> DriveState:
    """Linear-capped rise per drive. Non-positive elapsed → identity."""
    elapsed = float(elapsed_s)
    if elapsed <= 0:
        return state
    rates = rise_per_hour or {}
    levels = {}
    for name in DRIVES:
        delta = float(rates.get(name, 0.0)) * elapsed / 3600.0
        levels[name] = clamp_level(getattr(state, name) + delta)
    return DriveState(**levels)


def apply_satisfaction(
        state: DriveState, event_type: str,
        presence_satisfaction: Mapping[str, float],
        event_satisfaction: Mapping[str, Mapping[str, float]]
) -> DriveState:
    """Subtract presence (always) then event deltas. Clamped at floor."""
    present = presence_satisfaction or {}
    per_event = (event_satisfaction or {}).get(event_type, {}) or {}
    levels = {}
    for name in DRIVES:
        delta = float(present.get(name, 0.0))
        delta += float(per_event.get(name, 0.0))
        levels[name] = clamp_level(getattr(state, name) + delta)
    return DriveState(**levels)


def _atomic_write_json(path: Path, data: dict) -> None:
    # Deliberate duplication of the MemoryStore pattern: drive
    # persistence must not depend on state_store.py or memory.py, so the
    # three files (and their quarantine/version schemes) stay independent.
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class DriveStore:
    def __init__(self, path: Path,
                 time_fn: Callable[[], float] = time.time):
        self.path = Path(path)
        self._time_fn = time_fn or time.time

    def _quarantine(self) -> None:
        try:
            dest = self.path.parent / (
                f"quarantine-drives-{int(self._time_fn())}.json")
            os.replace(self.path, dest)
        except OSError:
            pass

    def load(self) -> DriveState:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return DriveState()
        except (UnicodeDecodeError, OSError):
            self._quarantine()
            return DriveState()
        try:
            data = json.loads(raw)
        except ValueError:
            self._quarantine()
            return DriveState()
        if (not isinstance(data, dict) or "version" not in data
                or "social" not in data or "novelty" not in data):
            self._quarantine()
            return DriveState()
        try:
            version = int(data["version"])
        except (TypeError, ValueError):
            self._quarantine()
            return DriveState()
        if version != DRIVE_SCHEMA_VERSION:
            return DriveState()
        try:
            return DriveState.from_dict(data)
        except (ValueError, TypeError, AttributeError):
            self._quarantine()
            return DriveState()

    def save(self, state: DriveState) -> None:
        if not isinstance(state, DriveState):
            raise ValueError("can only save DriveState objects")
        _atomic_write_json(
            self.path,
            {"version": DRIVE_SCHEMA_VERSION, **state.to_dict()})


@dataclass(frozen=True)
class DrivesConfig:
    rise_per_hour: Mapping[str, float]
    pull_threshold: float
    presence_satisfaction: Mapping[str, float]
    event_satisfaction: Mapping[str, Mapping[str, float]]

    @classmethod
    def load(cls, path) -> "DrivesConfig":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise ValueError(f"bad drives config: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("drives config must be an object")
        rise = data.get("rise_per_hour", {})
        if not isinstance(rise, dict):
            raise ValueError("rise_per_hour must be an object")
        try:
            rates = {str(k): float(v) for k, v in rise.items()}
        except (TypeError, ValueError):
            raise ValueError("bad rise_per_hour rates")
        try:
            threshold = float(data.get("pull_threshold", 0.6))
        except (TypeError, ValueError):
            raise ValueError("bad pull_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("pull_threshold out of range 0..1")
        presence = data.get("presence_satisfaction", {})
        events = data.get("event_satisfaction", {})
        if not isinstance(presence, dict) or not isinstance(events, dict):
            raise ValueError("satisfaction tables must be objects")
        try:
            clean_events = {str(k): {str(d): float(x) for d, x in v.items()}
                            if isinstance(v, dict) else dict(v)
                            for k, v in events.items()}
        except (TypeError, ValueError, AttributeError):
            raise ValueError("bad event_satisfaction rows")
        return cls(rise_per_hour=rates, pull_threshold=threshold,
                   presence_satisfaction=dict(presence),
                   event_satisfaction=clean_events)


def advance(state: DriveState, elapsed_s: float, event_type: str,
            cfg: DrivesConfig) -> DriveState:
    """Rise first, then apply satisfaction for event_type. Pure."""
    risen = update_rise(state, elapsed_s, cfg.rise_per_hour)
    return apply_satisfaction(risen, event_type, cfg.presence_satisfaction,
                              cfg.event_satisfaction)


def drive_pull(state: DriveState, cfg: DrivesConfig):
    """Highest drive at-or-above threshold; ties go by DRIVES order."""
    best = None
    best_level = 0.0
    for name in DRIVES:
        level = float(getattr(state, name))
        if level >= cfg.pull_threshold and (best is None
                                            or level > best_level):
            best, best_level = name, level
    return best
