"""Persistent consent data layer (Slice 8, Phase 1: storage only).

PermissionState holds per-tool persistent grants plus a monotonic
policy revision. Mutation helpers (set/clear) bump the revision;
raw save() writes exactly what it is given. No orchestration, no
prompt logic, no project imports.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

PERMISSIONS_SCHEMA_VERSION = 1
MAX_PERMISSIONS_BYTES = 4096


@dataclass(frozen=True)
class PermissionState:
    grants: Mapping[str, bool]  # tool name -> persistent consent
    policy_rev: int = 0

    def __post_init__(self) -> None:
        grants = self.grants or {}
        if not isinstance(grants, Mapping):
            raise ValueError("grants must be a mapping")
        clean = {}
        for key, value in grants.items():
            if not isinstance(key, str):
                raise ValueError("grant keys must be strings")
            if not isinstance(value, bool):
                raise ValueError("grant values must be bools")
            clean[key] = value
        rev = self.policy_rev
        if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
            raise ValueError("policy_rev must be a non-negative int")
        object.__setattr__(self, "grants",
                           MappingProxyType(dict(clean)))
        object.__setattr__(self, "policy_rev", rev)


def _atomic_write_json(path: Path, data: dict) -> None:
    # Deliberate duplication of the sibling-store pattern: consent
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


class PermissionStore:
    def __init__(self, path: Path,
                 time_fn: Callable[[], float] = time.time):
        self.path = Path(path)
        self._time_fn = time_fn or time.time

    def _quarantine(self) -> None:
        try:
            dest = self.path.parent / (
                f"quarantine-permissions-{int(self._time_fn())}.json")
            os.replace(self.path, dest)
        except OSError:
            pass

    def load(self) -> PermissionState:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return PermissionState(grants={}, policy_rev=0)
        except (UnicodeDecodeError, OSError):
            self._quarantine()
            return PermissionState(grants={}, policy_rev=0)
        try:
            data = json.loads(raw)
        except ValueError:
            self._quarantine()
            return PermissionState(grants={}, policy_rev=0)
        if (not isinstance(data, dict) or "version" not in data
                or "grants" not in data):
            self._quarantine()
            return PermissionState(grants={}, policy_rev=0)
        try:
            version = int(data["version"])
        except (TypeError, ValueError):
            self._quarantine()
            return PermissionState(grants={}, policy_rev=0)
        if version != PERMISSIONS_SCHEMA_VERSION:
            return PermissionState(grants={}, policy_rev=0)
        raw_grants = data["grants"]
        if not isinstance(raw_grants, dict):
            self._quarantine()
            return PermissionState(grants={}, policy_rev=0)
        clean = {k: v for k, v in raw_grants.items()
                 if isinstance(k, str) and isinstance(v, bool)}
        rev = data.get("policy_rev", 0)
        if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
            rev = 0
        return PermissionState(grants=clean, policy_rev=rev)

    def save(self, state: PermissionState) -> None:
        if not isinstance(state, PermissionState):
            raise ValueError("can only save PermissionState objects")
        payload = {"version": PERMISSIONS_SCHEMA_VERSION,
                   "grants": dict(state.grants),
                   "policy_rev": state.policy_rev}
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2)
                   + "\n").encode("utf-8")
        if len(encoded) > MAX_PERMISSIONS_BYTES:
            raise ValueError("permission state exceeds size cap")
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(encoded)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def set_grant(self, tool: str, allowed: bool) -> PermissionState:
        if not isinstance(tool, str) or not tool:
            raise ValueError("tool must be a non-empty string")
        if not isinstance(allowed, bool):
            raise ValueError("allowed must be a bool")
        current = self.load()
        merged = dict(current.grants)
        merged[tool] = allowed
        state = PermissionState(grants=merged,
                                policy_rev=current.policy_rev + 1)
        self.save(state)
        return state

    def clear_grant(self, tool: str) -> PermissionState:
        current = self.load()
        merged = {k: v for k, v in current.grants.items() if k != tool}
        state = PermissionState(grants=merged,
                                policy_rev=current.policy_rev + 1)
        self.save(state)
        return state

    def clear_all(self) -> PermissionState:
        current = self.load()
        state = PermissionState(grants={},
                                policy_rev=current.policy_rev + 1)
        self.save(state)
        return state
