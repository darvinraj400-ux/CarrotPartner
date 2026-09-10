"""Environment perception data layer (Slice 6, Phase 1: types only).

Read-only OS observation vocabulary: raw observations, allowlist
mapping, frozen snapshots, config loading, and a scripted mock probe.
No process_turn wiring, no prompt logic, no behavior gates, and no OS
calls anywhere in this module (in particular, no ctypes: only the
future carrot/environment_win32.py may import it).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RawObservation:
    process_name: str | None  # user-controlled; lowercase app/process name
    idle: bool


@dataclass(frozen=True)
class EnvironmentSnapshot:
    app_noun: str | None  # allowlisted category noun; None if unmapped
    idle: bool


EMPTY_RAW = RawObservation(process_name=None, idle=False)
EMPTY_SNAPSHOT = EnvironmentSnapshot(app_noun=None, idle=False)


class EnvironmentProbe(Protocol):
    def observe(self) -> RawObservation:
        ...


@dataclass(frozen=True)
class EnvironmentConfig:
    allowed_apps: Mapping[str, str]  # process_name (lowercase) -> noun
    idle_threshold_s: float

    @classmethod
    def load(cls, path: str | Path) -> "EnvironmentConfig":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise ValueError(f"bad environment config: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("environment config must be an object")
        apps = data.get("allowed_apps", {})
        if not isinstance(apps, dict):
            raise ValueError("allowed_apps must be an object")
        try:
            clean = {str(k).lower(): str(v).lower()
                     for k, v in apps.items()}
        except (TypeError, ValueError, AttributeError):
            raise ValueError("bad allowed_apps entries")
        try:
            idle = float(data.get("idle_threshold_s", 300.0))
        except (TypeError, ValueError):
            raise ValueError("bad idle_threshold_s")
        if idle < 0:
            raise ValueError("idle_threshold_s must be non-negative")
        return cls(allowed_apps=MappingProxyType(clean),
                   idle_threshold_s=idle)


def map_observation(raw: RawObservation,
                    config: EnvironmentConfig) -> EnvironmentSnapshot:
    """Map a raw observation via the config allowlist.

    Unknown or missing process names yield app_noun None (allowlist or
    nothing, never passthrough). idle passes through unchanged.
    """
    name = (raw.process_name or "").strip().lower()
    noun = config.allowed_apps.get(name) if name else None
    return EnvironmentSnapshot(app_noun=noun, idle=bool(raw.idle))


class MockEnvironmentProbe:
    """Scripted probe. Returns observations in sequence; holds the last
    one once the sequence is exhausted. No OS access."""

    def __init__(self,
                 observations: Sequence[RawObservation] = ()) -> None:
        self._script = tuple(observations or ())
        self._index = 0

    def observe(self) -> RawObservation:
        if not self._script:
            return EMPTY_RAW
        result = self._script[min(self._index, len(self._script) - 1)]
        if self._index < len(self._script) - 1:
            self._index += 1
        return result
