"""Relationship state — event-driven, two dimensions, no decay.

Contract:
- Two dimensions: trust, closeness. Both [0,1], clamped, no drift.
- Evolution: event-driven only (apply_event). No wall-clock decay.
- Not a score: dimensions are separate, with separate semantics.
- No project imports: pure math over passed-in floats and strings.
- Slice 11 persists dimensions + per-dimension updated_ts only.

Persistence: state/relationship.json (schema v1, ≤4 KB, atomic write,
own quarantine prefix). Load is fail-soft: malformed file → quarantine +
defaults; bad scalar → clamp or default; bad updated_ts entry → skipped.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

DIMENSIONS = ("trust", "closeness")
RELATIONSHIP_SCHEMA_VERSION = 1
MAX_RELATIONSHIP_BYTES = 4096


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


def _atomic_write_json(path: Path, data: dict) -> None:
    # Deliberate duplication of the sibling-store pattern: relationship
    # persistence must not depend on any sibling module, so the five
    # files (and their quarantine/version schemes) stay independent.
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


def _coerce_level(value, default: float = 0.5) -> float:
    # Load-time fail-soft: bool/non-numeric → default, out-of-range → clamp.
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return _clamp(number)


class RelationshipStore:
    """Load/save RelationshipState. No mutation helpers: the orchestrator
    loads, calls apply_event, and saves."""

    def __init__(self, path: Path,
                 time_fn: Callable[[], float] = time.time):
        self.path = Path(path)
        self._time_fn = time_fn or time.time

    def _quarantine(self) -> None:
        try:
            dest = self.path.parent / (
                f"quarantine-relationship-{int(self._time_fn())}.json")
            os.replace(self.path, dest)
        except OSError:
            pass

    def load(self) -> RelationshipState:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return RelationshipState()
        except (UnicodeDecodeError, OSError):
            self._quarantine()
            return RelationshipState()
        try:
            data = json.loads(raw)
        except ValueError:
            self._quarantine()
            return RelationshipState()
        if not isinstance(data, dict) or "version" not in data:
            self._quarantine()
            return RelationshipState()
        try:
            version = int(data["version"])
        except (TypeError, ValueError):
            self._quarantine()
            return RelationshipState()
        if version != RELATIONSHIP_SCHEMA_VERSION:
            return RelationshipState()
        trust = _coerce_level(data.get("trust", 0.5))
        closeness = _coerce_level(data.get("closeness", 0.5))
        raw_ts = data.get("updated_ts", {})
        stamps = {}
        if isinstance(raw_ts, Mapping):
            for key, value in raw_ts.items():
                if key not in DIMENSIONS:
                    continue
                if isinstance(value, bool):
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if number < 0:
                    continue
                stamps[key] = number
        try:
            return RelationshipState(trust=trust, closeness=closeness,
                                     updated_ts=stamps)
        except ValueError:
            return RelationshipState()

    def save(self, state: RelationshipState) -> None:
        if not isinstance(state, RelationshipState):
            raise TypeError("can only save RelationshipState objects")
        payload = {"version": RELATIONSHIP_SCHEMA_VERSION,
                   "trust": float(state.trust),
                   "closeness": float(state.closeness),
                   "updated_ts": dict(state.updated_ts)}
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2)
                   + "\n").encode("utf-8")
        if len(encoded) > MAX_RELATIONSHIP_BYTES:
            raise ValueError("relationship state exceeds size cap")
        _atomic_write_json(self.path, payload)
