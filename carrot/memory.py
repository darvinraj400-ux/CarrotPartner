"""Long-term memory data layer (Slice 3, Phase 1: persistence only).

Memories are curated user-content facts/preferences. This module owns
storage: no extraction, no retrieval ranking, no prompt logic, and no
emotional-state access (it must never import ``emotions``).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

MEMORY_SCHEMA_VERSION = 1
MAX_MEMORIES = 50
MAX_MEMORY_TEXT_CHARS = 140
MAX_FILE_BYTES = 16 * 1024
KINDS = ("fact", "preference")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class Memory:
    text: str
    kind: str
    importance: float
    created_ts: float
    last_recalled_ts: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("memory text must be a string")
        if len(self.text) > MAX_MEMORY_TEXT_CHARS:
            raise ValueError(
                f"memory text over {MAX_MEMORY_TEXT_CHARS} chars")
        if self.kind not in KINDS:
            raise ValueError(f"unknown memory kind: {self.kind!r}")
        try:
            importance = _clamp01(self.importance)
            created = float(self.created_ts)
            recalled = float(self.last_recalled_ts)
        except (TypeError, ValueError):
            raise ValueError("bad memory importance/timestamps")
        object.__setattr__(self, "importance", importance)
        object.__setattr__(self, "created_ts", created)
        object.__setattr__(self, "last_recalled_ts", recalled)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "kind": self.kind,
            "importance": self.importance,
            "created_ts": self.created_ts,
            "last_recalled_ts": self.last_recalled_ts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        if not isinstance(data, dict):
            raise ValueError("memory entry must be an object")
        return cls(
            text=data.get("text"),
            kind=data.get("kind"),
            importance=data.get("importance"),
            created_ts=data.get("created_ts"),
            last_recalled_ts=data.get("last_recalled_ts"),
        )


def _atomic_write_json(path: Path, data: dict) -> None:
    # Deliberate ~12-line duplication of the StateStore pattern:
    # memory persistence must not depend on state_store.py, so the two
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


class MemoryStore:
    def __init__(self, path: Path,
                 time_fn: Callable[[], float] = time.time):
        self.path = Path(path)
        self._time_fn = time_fn or time.time

    def _quarantine(self) -> None:
        try:
            dest = self.path.parent / (
                f"quarantine-memories-{int(self._time_fn())}.json")
            os.replace(self.path, dest)
        except OSError:
            pass

    def load(self) -> tuple:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ()
        except (UnicodeDecodeError, OSError):
            self._quarantine()
            return ()
        try:
            data = json.loads(raw)
        except ValueError:
            self._quarantine()
            return ()
        if (not isinstance(data, dict) or "version" not in data
                or "memories" not in data):
            self._quarantine()
            return ()
        try:
            version = int(data["version"])
        except (TypeError, ValueError):
            self._quarantine()
            return ()
        if version != MEMORY_SCHEMA_VERSION:
            return ()
        if not isinstance(data["memories"], list):
            self._quarantine()
            return ()
        out = []
        for entry in data["memories"]:
            try:
                out.append(Memory.from_dict(entry))
            except (ValueError, TypeError, AttributeError, KeyError):
                continue
        return tuple(out)

    def save(self, memories: Sequence) -> None:
        items = list(memories or ())
        if len(items) > MAX_MEMORIES:
            raise ValueError(f"over memory cap of {MAX_MEMORIES}")
        for item in items:
            if not isinstance(item, Memory):
                raise ValueError("can only save Memory objects")
        _atomic_write_json(
            self.path,
            {"version": MEMORY_SCHEMA_VERSION,
             "memories": [m.to_dict() for m in items]})

    def add(self, memory: Memory) -> bool:
        if not isinstance(memory, Memory):
            raise TypeError("can only add Memory objects")
        current = list(self.load())
        if len(current) >= MAX_MEMORIES:
            return False
        current.append(memory)
        self.save(current)
        return True

    def remove_matching(self, substring: str) -> int:
        needle = str(substring or "").strip().lower()
        if not needle:
            return 0
        current = list(self.load())
        kept = [m for m in current if needle not in m.text.lower()]
        removed = len(current) - len(kept)
        if removed:
            self.save(kept)
        return removed

    def all(self) -> tuple:
        return self.load()
