"""Episodic memory data layer — bounded, deterministic, derived.

Contract:
- Episodes remember that an interaction happened, not what was said.
  No verbatim turn text is ever stored. This is enforced by
  test_no_verbatim_turn_text_in_new_stores (field names) and, from
  Phase 2 onward, by the runtime content pin.
- Episodes are per-companion. The store path resolves through
  companion_paths; the tuning config is global.
- This module holds the record type, the store, and the config loader.
  Derivation and retrieval land in Phase 2 and Phase 3.
- No imports of carrot.carrot, carrot.cli, carrot.llm. No subprocess,
  ctypes, threads, or schedulers. Stdlib plus carrot.mood only.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .mood import MOODS

EPISODE_SCHEMA_VERSION = 1
MAX_EPISODES = 50
# 16 KB: episodes are a bounded list of small records. 50 full records
# serialize to ~8.5 KB, so 4 KB (the sibling-store convention for
# tiny-record stores) would bind before the count cap. 16 KB matches
# the figure declared in memory.py and leaves comfortable margin while
# keeping the count cap (MAX_EPISODES) as the intended binding limit.
MAX_FILE_BYTES = 16384
MAX_TOPIC_TOKENS = 10
JOIN_WINDOW_S = 1800.0


def _check_ts(name: str, value) -> float:
    if isinstance(value, bool):
        raise ValueError(f"bad episode timestamp for {name!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"bad episode timestamp for {name!r}")
    if number < 0:
        raise ValueError(
            f"episode timestamp {name!r} must be non-negative")
    return number


@dataclass(frozen=True)
class Episode:
    first_seen_ts: float
    last_seen_ts: float
    turn_count: int
    topic_tokens: tuple = field(default_factory=tuple)
    mood_label: str = "content"
    last_behavior: str | None = None

    def __post_init__(self) -> None:
        first = _check_ts("first_seen_ts", self.first_seen_ts)
        last = _check_ts("last_seen_ts", self.last_seen_ts)
        if last < first:
            raise ValueError(
                "episode last_seen_ts < first_seen_ts")
        if isinstance(self.turn_count, bool) or \
                not isinstance(self.turn_count, int):
            raise ValueError("bad episode turn_count")
        turns = self.turn_count
        if turns < 1:
            raise ValueError("bad episode turn_count")
        tokens = self.topic_tokens if self.topic_tokens is not None else ()
        if not isinstance(tokens, (tuple, list)):
            raise ValueError("bad episode topic_tokens")
        if len(tokens) > MAX_TOPIC_TOKENS:
            raise ValueError("bad episode topic_tokens")
        for token in tokens:
            if not isinstance(token, str) or not token:
                raise ValueError("bad episode topic_tokens")
        if self.mood_label not in MOODS:
            raise ValueError(
                f"bad episode mood_label: {self.mood_label!r}")
        if self.last_behavior is not None and (
                not isinstance(self.last_behavior, str)
                or not self.last_behavior):
            raise ValueError("bad episode last_behavior")
        object.__setattr__(self, "first_seen_ts", first)
        object.__setattr__(self, "last_seen_ts", last)
        object.__setattr__(self, "turn_count", turns)
        object.__setattr__(self, "topic_tokens", tuple(tokens))

    def to_dict(self) -> dict:
        return {
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
            "turn_count": self.turn_count,
            "topic_tokens": list(self.topic_tokens),
            "mood_label": self.mood_label,
            "last_behavior": self.last_behavior,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        if not isinstance(data, dict):
            raise ValueError("episode entry must be an object")
        known = ("first_seen_ts", "last_seen_ts", "turn_count",
                 "topic_tokens", "mood_label", "last_behavior")
        return cls(**{key: data[key] for key in known if key in data})


@dataclass(frozen=True)
class EpisodesConfig:
    join_window_s: float = 1800.0
    max_episodes: int = 50
    max_topic_tokens: int = 10
    retrieval_min_overlap: int = 2
    recall_k: int = 1

    def __post_init__(self) -> None:
        try:
            window = float(self.join_window_s)
        except (TypeError, ValueError):
            raise ValueError("bad join_window_s")
        if isinstance(self.join_window_s, bool) or window <= 0:
            raise ValueError("bad join_window_s")
        clean_ints = {}
        for key in ("max_episodes", "max_topic_tokens",
                    "retrieval_min_overlap", "recall_k"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"bad {key}")
            if value < 1:
                raise ValueError(f"bad {key}")
            clean_ints[key] = value
        object.__setattr__(self, "join_window_s", window)
        for key, value in clean_ints.items():
            object.__setattr__(self, key, value)

    @classmethod
    def load(cls, path: str | Path) -> "EpisodesConfig":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise ValueError(f"bad episodes config: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("episodes config must be an object")
        kwargs = {}
        for key in ("join_window_s", "max_episodes", "max_topic_tokens",
                    "retrieval_min_overlap", "recall_k"):
            if key in data:
                kwargs[key] = data[key]
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc


def _atomic_write_json(path: Path, data: dict) -> None:
    # Deliberate duplication of the sibling-store pattern: episode
    # persistence must not depend on any sibling module, so the files
    # (and their quarantine/version schemes) stay independent.
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


class EpisodeStore:
    def __init__(self, path: Path | str,
                 time_fn: Callable[[], float] = time.time):
        self.path = Path(path)
        self._time_fn = time_fn or time.time

    def _quarantine(self) -> None:
        try:
            dest = self.path.parent / (
                f"quarantine-episodes-{int(self._time_fn())}.json")
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
        try:
            return self._parse(data)
        except Exception:
            self._quarantine()
            return ()

    def _parse(self, data: dict) -> tuple:
        if not isinstance(data, dict):
            self._quarantine()
            return ()
        if "version" not in data or "episodes" not in data:
            self._quarantine()
            return ()
        try:
            version = int(data["version"])
        except (TypeError, ValueError):
            self._quarantine()
            return ()
        if version != EPISODE_SCHEMA_VERSION:
            return ()
        entries = data["episodes"]
        if not isinstance(entries, list):
            self._quarantine()
            return ()
        out = []
        order = []
        for index, entry in enumerate(entries):
            try:
                out.append(Episode.from_dict(entry))
                order.append(index)
            except Exception:
                continue
        if len(out) > MAX_EPISODES:
            ranked = sorted(zip(out, order),
                            key=lambda pair: (-pair[0].last_seen_ts,
                                              -pair[0].first_seen_ts,
                                              pair[1]))
            keep = {id(item) for item, _ in ranked[:MAX_EPISODES]}
            out = [item for item in out if id(item) in keep]
        return tuple(out)

    def save(self, episodes: Sequence) -> None:
        items = list(episodes or ())
        for item in items:
            if not isinstance(item, Episode):
                raise ValueError("can only save Episode objects")
        if len(items) > MAX_EPISODES:
            raise ValueError(f"over episode cap of {MAX_EPISODES}")
        payload = {"version": EPISODE_SCHEMA_VERSION,
                   "episodes": [e.to_dict() for e in items]}
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2)
                   + "\n").encode("utf-8")
        if len(encoded) > MAX_FILE_BYTES:
            raise ValueError("episode state exceeds size cap")
        _atomic_write_json(self.path, payload)

    def add(self, episode: Episode) -> bool:
        if not isinstance(episode, Episode):
            raise TypeError("can only add Episode objects")
        current = list(self.load())
        if len(current) >= MAX_EPISODES:
            return False
        current.append(episode)
        self.save(current)
        return True

    def remove_matching(self, substring: str) -> int:
        if not isinstance(substring, str):
            return 0
        needle = substring.strip().lower()
        if not needle:
            return 0
        current = list(self.load())
        kept = [e for e in current
                if needle not in " ".join(e.topic_tokens).lower()]
        removed = len(current) - len(kept)
        if removed:
            self.save(kept)
        return removed

    def all(self) -> tuple:
        return self.load()
