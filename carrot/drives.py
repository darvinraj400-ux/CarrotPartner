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
