"""Long-term memory data layer (Slice 3, Phase 1: persistence only).

Memories are curated user-content facts/preferences. This module owns
storage: no extraction, no retrieval ranking, no prompt logic, and no
emotional-state access (it must never import ``emotions``).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

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


@dataclass(frozen=True)
class MemoryCandidate:
    text: str
    kind: str


class Extractor(Protocol):
    def extract(self, user_text: str) -> tuple:
        ...


# Coarse turn-level rejection: hostile turns yield zero candidates.
_HOSTILE_CUES = ("you are useless", "you're useless", "youre useless",
                 "i hate you", "shut up", "fuck you", "screw you",
                 "stupid", "idiot", "moron", "dumbass")

# Ordered (pattern, kind). Explicit-memory patterns first, then specific
# identity patterns, then preferences, then the generic possessive LAST:
# "my favorite color is blue" must reach the preference rule before the
# generic "my ... is ..." rule can claim it as a fact. Non-greedy `A`
# groups keep compound statements narrow ("my dog is funny" -> A="dog").
_ORDERED_PATTERNS = tuple(
    (re.compile(pattern), kind)
    for pattern, kind in [
        (r"remember (that )?(?P<X>.+)", "fact"),
        (r"don't forget (that )?(?P<X>.+)", "fact"),
        (r"note (that )?(?P<X>.+)", "fact"),
        (r"my name is (?P<X>.+)", "fact"),
        (r"call me (?P<X>.+)", "fact"),
        (r"i'?m (?P<X>.+) years old", "fact"),
        (r"my birthday is (?P<X>.+)", "fact"),
        (r"i live in (?P<X>.+)", "fact"),
        (r"i'?m from (?P<X>.+)", "fact"),
        (r"i work (at|for|as) (?P<X>.+)", "fact"),
        (r"i have a (?P<X>.+)", "fact"),
        (r"i'?m allergic to (?P<X>.+)", "fact"),
        (r"my favorite (?P<A>.+?) is (?P<X>.+)", "preference"),
        (r"i (really )?(love|like|prefer) (?P<X>.+)", "preference"),
        (r"i (really )?(hate|dislike) (?P<X>.+)", "preference"),
        (r"my (?P<A>.+?) is (?P<X>.+)", "fact"),
    ]
)

_SENTENCE_SPLIT_RE = re.compile(r"[.!?…;\n]+")
_TRAILING_PUNCT = ".,!?:;"


class DeterministicExtractor:
    """Rule-based candidate extraction. Pure: no I/O, no state."""

    def extract(self, user_text: str) -> tuple:
        lowered = str(user_text or "").lower()
        if not lowered.strip():
            return ()
        if any(cue in lowered for cue in _HOSTILE_CUES):
            return ()
        found: list = []
        seen = set()
        for segment in _SENTENCE_SPLIT_RE.split(lowered):
            candidate = self._match_segment(segment.strip())
            if candidate is not None and candidate.text not in seen:
                seen.add(candidate.text)
                found.append(candidate)
        return tuple(found)

    @staticmethod
    def _match_segment(segment: str):
        if not segment:
            return None
        for pattern, kind in _ORDERED_PATTERNS:
            match = pattern.search(segment)
            if not match:
                continue
            text = match.group(0).strip().rstrip(_TRAILING_PUNCT).strip()
            if not text or len(text) > MAX_MEMORY_TEXT_CHARS:
                return None
            return MemoryCandidate(text=text, kind=kind)
        return None
