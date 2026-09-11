"""Companion value type, name validation, and path construction.

Contract:
- A companion is a named Carrot instance with its own per-companion
  stores (state, memories, drives, relationship, initiative, social)
  and its own personality file.
- Global stores (permissions, tools, environment) are shared across
  companions — they are user/machine policy, not character.
- The default companion is named "carrot" and uses the flat layout:
  state/state.json, config/personality.json. Every other companion
  uses state/<name>/state.json and config/companions/<name>.json.
- No migration: the flat layout for "carrot" is permanent.

Boundaries:
- No orchestrator (carrot.carrot), CLI (carrot.cli), or LLM (carrot.llm)
  imports. No process_turn. No generate.
- No subprocess, ctypes, threads, or schedulers.
- No .json string literals for companion-specific config files — the
  config/companions/ directory name is a literal, not a file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .paths import _paths

DEFAULT_COMPANION = "carrot"
_COMPANION_NAME_RE = re.compile(r"^[a-z0-9-]{1,32}$")


def validate_name(name: str) -> str:
    """Return name if it matches ^[a-z0-9-]{1,32}$, else raise ValueError."""
    if not isinstance(name, str) or not _COMPANION_NAME_RE.match(name):
        raise ValueError(f"bad companion name: {name!r}")
    return name


def companion_paths(root: Path, name: str) -> dict:
    """Per-companion layout dict. Pure: no I/O, no writes, deterministic."""
    validate_name(name)
    base = _paths(root)
    if name == DEFAULT_COMPANION:
        return dict(base)
    r = Path(root)
    state = r / "state" / name
    out = dict(base)
    out.update({
        "state_dir": state,
        "state_json": state / "state.json",
        "memories_json": state / "memories.json",
        "drives_json": state / "drives.json",
        "relationship_json": state / "relationship.json",
        "initiative_json": state / "initiative.json",
        "social_dir": state / "social",
        "channel_dir": state / "channel",
        "episodes_json": state / "episodes.json",
        "personality_json": r / "config" / "companions" / f"{name}.json",
        "companion_config_dir": r / "config" / "companions",
    })
    return out


@dataclass(frozen=True)
class Companion:
    name: str
    personality_id: str
    state_dir: Path
    config_dir: Path

    def __post_init__(self) -> None:
        validate_name(self.name)


def _default_personality_id(root: Path) -> str:
    try:
        with open(Path(root) / "config" / "personality.json", "r",
                   encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, ValueError, UnicodeDecodeError, OSError):
        return "carrot-default-v1"
    if not isinstance(data, dict):
        return "carrot-default-v1"
    ident = data.get("id", "carrot-default-v1")
    return str(ident) if ident else "carrot-default-v1"


def list_companions(root: Path) -> tuple:
    """Return all discoverable companions under root.

    Always includes the default companion ("carrot") first, regardless
    of whether config/companions/carrot file exists — the default uses
    config/personality.json. Then scans config/companions/*.json
    (sorted by filename stem), one Companion per parseable file;
    malformed files and invalid stems are skipped silently.

    Pure: reads only; no filesystem writes.
    """
    r = Path(root)
    paths = companion_paths(r, DEFAULT_COMPANION)
    found = [Companion(name=DEFAULT_COMPANION,
                       personality_id=_default_personality_id(r),
                       state_dir=paths["state_dir"],
                       config_dir=paths["config_dir"])]
    companions_dir = r / "config" / "companions"
    try:
        files = sorted(
            (p for p in companions_dir.iterdir()
             if p.is_file() and p.suffix == ".json"),
            key=lambda p: p.stem)
    except OSError:
        return tuple(found)
    for path in files:
        try:
            name = validate_name(path.stem)
        except ValueError:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (ValueError, UnicodeDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        ident = data.get("id", name)
        paths = companion_paths(r, name)
        found.append(Companion(name=name,
                               personality_id=str(ident) if ident else name,
                               state_dir=paths["state_dir"],
                               config_dir=paths["config_dir"]))
    return tuple(found)
