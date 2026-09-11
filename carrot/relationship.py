"""Relationship state — event-driven, two dimensions, no decay.

Contract:
- Two dimensions: trust, closeness. Both [0,1], clamped, no drift.
- Evolution: event-driven only (apply_event). No wall-clock decay.
- Not a score: dimensions are separate, with separate semantics.
- No project imports: pure math over passed-in floats and strings.
- Slice 11 persists dimensions + per-dimension updated_ts only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

DIMENSIONS = ("trust", "closeness")
RELATIONSHIP_SCHEMA_VERSION = 1


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _check_level(name: str, value) -> float:
    if isinstance(value, bool):
        raise ValueError(f"bad relationship level for {name!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"bad relationship level for {name!r}")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"relationship level {name!r} out of range 0..1")
    return number


def _check_ts(name: str, value) -> float:
    if isinstance(value, bool):
        raise ValueError(f"bad relationship timestamp for {name!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"bad relationship timestamp for {name!r}")
    if number < 0:
        raise ValueError(
            f"relationship timestamp {name!r} must be non-negative")
    return number


@dataclass(frozen=True)
class RelationshipState:
    trust: float = 0.5
    closeness: float = 0.5
    updated_ts: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trust = _check_level("trust", self.trust)
        closeness = _check_level("closeness", self.closeness)
        raw = self.updated_ts if self.updated_ts is not None else {}
        if not isinstance(raw, Mapping):
            raise ValueError("updated_ts must be a mapping")
        clean = {}
        for key, value in raw.items():
            if key not in DIMENSIONS:
                raise ValueError(f"bad updated_ts key: {key!r}")
            clean[key] = _check_ts(key, value)
        object.__setattr__(self, "trust", trust)
        object.__setattr__(self, "closeness", closeness)
        object.__setattr__(self, "updated_ts",
                           MappingProxyType(dict(clean)))

    def to_dict(self) -> dict:
        return {
            "trust": self.trust,
            "closeness": self.closeness,
            "updated_ts": dict(self.updated_ts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RelationshipState":
        if not isinstance(data, dict):
            raise ValueError("relationship state must be an object")
        return cls(
            trust=data.get("trust", 0.5),
            closeness=data.get("closeness", 0.5),
            updated_ts=data.get("updated_ts", {}),
        )


@dataclass(frozen=True)
class RelationshipConfig:
    event_deltas: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        raw = self.event_deltas if self.event_deltas is not None else {}
        if not isinstance(raw, Mapping):
            raise ValueError("event_deltas must be an object")
        clean = {}
        for event, row in raw.items():
            if not isinstance(event, str):
                raise ValueError("event_deltas keys must be strings")
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"event_deltas row for {event!r} must be an object")
            clean_row = {}
            for dim, delta in row.items():
                if dim not in DIMENSIONS:
                    raise ValueError(
                        f"bad event_deltas dimension: {dim!r}")
                if isinstance(delta, bool):
                    raise ValueError(
                        f"bad event_deltas value for {event!r}/{dim!r}")
                try:
                    clean_row[dim] = float(delta)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"bad event_deltas value for {event!r}/{dim!r}")
            clean[event] = MappingProxyType(dict(clean_row))
        object.__setattr__(self, "event_deltas",
                           MappingProxyType(dict(clean)))

    @classmethod
    def load(cls, path: str | Path) -> "RelationshipConfig":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise ValueError(f"bad relationship config: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("relationship config must be an object")
        deltas = data.get("event_deltas", {})
        if not isinstance(deltas, dict):
            raise ValueError("event_deltas must be an object")
        return cls(event_deltas=deltas)


def apply_event(
    state: RelationshipState,
    event_type: str,
    intensity: float,
    now: float,
    cfg: RelationshipConfig,
) -> RelationshipState:
    """Apply one classified event's deltas to the relationship state.

    - event_type absent from cfg.event_deltas → return state unchanged.
    - Each delta is multiplied by intensity (clamped [0,1] defensively).
    - New values clamped to [0,1].
    - updated_ts[dim] = float(now) for dimensions whose delta was non-zero.
    - Pure: no I/O, no clock read, no mutation of inputs.
    """
    row = (cfg.event_deltas or {}).get(str(event_type), None)
    if not row:
        return state
    try:
        strength = float(intensity)
    except (TypeError, ValueError):
        strength = 0.0
    strength = _clamp(strength)
    if strength <= 0.0:
        return state
    moment = float(now)
    levels = {"trust": state.trust, "closeness": state.closeness}
    stamps = dict(state.updated_ts)
    touched = False
    for dim in DIMENSIONS:
        delta = float(row.get(dim, 0.0))
        if delta == 0.0:
            continue
        levels[dim] = _clamp(levels[dim] + delta * strength)
        stamps[dim] = moment
        touched = True
    if not touched:
        return state
    return RelationshipState(trust=levels["trust"],
                             closeness=levels["closeness"],
                             updated_ts=stamps)
