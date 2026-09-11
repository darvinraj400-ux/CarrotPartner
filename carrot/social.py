"""Social channel adapter — thin platform-tagged wrapper over the file
channel (Slice 14).

Contract:
- Social is a Channel: text in (receive), text out (send). Nothing else.
- This module holds the platform-tagged adapter, the config loader, and
  a read-only status view. It reuses carrot.channel.FileChannel for
  the machinery; it does not duplicate it.
- No orchestrator, CLI, or LLM contact. No subprocess, ctypes, threads,
  or schedulers. Manual-mode file bridge only.

Credential pattern (for future slices):
- config/social stores handles only (committed, no secrets).
- state/social/<platform>/credentials (gitignored via state/) is the
  home for any future token/secret. Slice 15 stores nothing.

Scope:
- Plain-text send/receive only. Threads, mentions, media attachments,
  and rate-limit backoff are out of scope. send(text) is the whole
  send-side contract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .channel import Channel, FileChannel, MockChannel

PLATFORMS = ("instagram",)


@dataclass(frozen=True)
class SocialConfig:
    platforms: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        raw = self.platforms if self.platforms is not None else {}
        if not isinstance(raw, Mapping):
            raise ValueError("platforms must be an object")
        clean = {}
        for name, inner in raw.items():
            if not isinstance(name, str):
                raise ValueError("platform names must be strings")
            if not isinstance(inner, Mapping):
                raise ValueError(
                    f"platform {name!r} must be an object")
            clean[name] = MappingProxyType(dict(inner))
        object.__setattr__(self, "platforms",
                           MappingProxyType(dict(clean)))

    @classmethod
    def load(cls, path: str | Path) -> "SocialConfig":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise ValueError(f"bad social config: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("social config must be an object")
        platforms = data.get("platforms", {})
        if not isinstance(platforms, dict):
            raise ValueError("platforms must be an object")
        return cls(platforms=platforms)


class SocialChannel(FileChannel):
    """FileChannel with a platform tag. base_dir is the platform root
    (the same directory FileChannel would receive)."""

    def __init__(self, platform: str, base_dir: Path | str) -> None:
        super().__init__(base_dir)
        self.platform = str(platform)


class MockSocialChannel(MockChannel):
    def __init__(self, platform: str = "instagram",
                 incoming: Sequence[str] = ()) -> None:
        super().__init__(incoming)
        self.platform = str(platform)


def social_status(config: SocialConfig, root: Path) -> str:
    """Read-only status view, one line per configured platform.

    "<platform>: <enabled|disabled> pending=<N> sent=<N>".
    Missing directories count as 0. Empty config renders the
    no-platforms sentinel. No mutation.
    """
    base = Path(root)
    rows = dict(config.platforms or {})
    if not rows:
        return "(no social platforms configured)"
    lines = []
    for platform, inner in rows.items():
        inner = inner if isinstance(inner, Mapping) else {}
        enabled = inner.get("enabled", True)
        label = "enabled" if enabled else "disabled"
        raw_dir = inner.get("inbox_dir")
        if isinstance(raw_dir, str) and raw_dir:
            path = Path(raw_dir)
            if not path.is_absolute():
                path = base / path
        else:
            path = base / "state" / "social" / str(platform)
        try:
            pending = sum(1 for p in (path / "inbox").glob("*.msg")
                          if p.is_file())
        except OSError:
            pending = 0
        try:
            sent = sum(1 for p in (path / "outbox").glob("*.msg")
                       if p.is_file())
        except OSError:
            sent = 0
        lines.append(f"{platform}: {label} pending={pending} sent={sent}")
    return "\n".join(lines)
