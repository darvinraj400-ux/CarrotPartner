"""PC tool value types, parser, registry, and config (Slice 7, Phase 1:
data layer only). No execution, no permission decisions, no orchestrator
wiring. No subprocess-family imports here yet (the Phase 0 exception for
this module exists but is not exercised until execution lands).
"""
from __future__ import annotations

import json
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

TOOLS = ("read_file", "open_url")
DEFAULT_MAX_FILE_BYTES = 4096

_FENCE_PREFIX = "```tool:"
_CLOSING_FENCE = "```"


@dataclass(frozen=True)
class ToolRequest:
    name: str
    args: Mapping[str, str]  # arg name -> value; values are strings for now


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str  # preview text; <= DEFAULT_MAX_FILE_BYTES for read_file
    error: str | None = None


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args_preview: str  # <=80 chars, sanitized preview
    ok: bool
    allowed: bool
    ts: float


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str | None = None


def parse_tool_request(text: str) -> ToolRequest | None:
    """Parse the first fenced tool block from LLM output text.

    Grammar: ```tool:NAME newline {"arg": "value"} newline ```.
    Marker and NAME are case-sensitive; NAME must be in TOOLS; the body
    must be a JSON object of string->string. Anything else -> None.
    Only the first block is considered; the rest is ignored as text.
    """
    src = str(text or "")
    start = src.find(_FENCE_PREFIX)
    if start < 0:
        return None
    first_nl = src.find("\n", start)
    if first_nl < 0:
        return None
    name = src[start + len(_FENCE_PREFIX):first_nl].strip()
    if not name or name not in TOOLS:
        return None
    end = src.find(_CLOSING_FENCE, first_nl + 1)
    if end < 0:
        return None
    try:
        data = json.loads(src[first_nl + 1:end])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return None
    return ToolRequest(name=name, args=dict(data))


@dataclass(frozen=True)
class ToolsConfig:
    enabled: Mapping[str, bool]  # tool name -> enabled
    read_file_dirs: tuple[str, ...]  # resolved absolute dirs
    open_url_domains: tuple[str, ...]  # lowercased, scheme-less
    max_file_bytes: int
    max_iterations: int  # tool loop cap
    levels: Mapping[str, int]  # tool name -> 0..4; unlisted tools read as 4

    @classmethod
    def load(cls, path: str | Path) -> "ToolsConfig":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise ValueError(f"bad tools config: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("tools config must be an object")
        enabled = data.get(
            "enabled", {tool: True for tool in TOOLS})
        if not isinstance(enabled, dict):
            raise ValueError("enabled must be an object")
        try:
            clean_enabled = {}
            for key, flag in enabled.items():
                if not isinstance(flag, bool):
                    raise ValueError(f"bad enabled flag for {key!r}")
                clean_enabled[str(key)] = flag
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"bad enabled table: {exc}") from exc
        raw_dirs = data.get("read_file_dirs", [])
        if not isinstance(raw_dirs, list):
            raise ValueError("read_file_dirs must be a list")
        dirs = []
        for entry in raw_dirs:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("read_file_dirs must hold non-empty strings")
            dirs.append(os.path.abspath(os.path.expanduser(entry)))
        raw_domains = data.get("open_url_domains", [])
        if not isinstance(raw_domains, list):
            raise ValueError("open_url_domains must be a list")
        domains = []
        for entry in raw_domains:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    "open_url_domains must hold non-empty strings")
            domains.append(entry.strip().lower())
        try:
            max_bytes = data.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES)
            if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
                raise ValueError("bad max_file_bytes")
            if max_bytes < 0:
                raise ValueError("max_file_bytes must be non-negative")
            max_iter = data.get("max_iterations", 2)
            if isinstance(max_iter, bool) or not isinstance(max_iter, int):
                raise ValueError("bad max_iterations")
            if max_iter not in (1, 2, 3):
                raise ValueError("max_iterations must be 1, 2, or 3")
            raw_levels = data.get("levels", {})
            if not isinstance(raw_levels, dict):
                raise ValueError("levels must be an object")
            clean_levels = {}
            for key, level in raw_levels.items():
                if (isinstance(level, bool) or not isinstance(level, int)
                        or level not in (0, 1, 2, 3, 4)):
                    raise ValueError(f"bad level for {key!r}")
                clean_levels[str(key)] = level
            for tool in TOOLS:
                clean_levels.setdefault(tool, 4)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(str(exc)) from exc
        return cls(enabled=clean_enabled, read_file_dirs=tuple(dirs),
                   open_url_domains=tuple(domains),
                   max_file_bytes=max_bytes, max_iterations=max_iter,
                   levels=MappingProxyType(clean_levels))


# Slice 8: signature grows only by defaulted parameters
# (persistent_grants=None). No breaking changes across phases.
def decide(
    request: ToolRequest,
    config: ToolsConfig,
    session_grants: Mapping[str, bool],
) -> Decision:
    """Coarse permission check at the tool-category level, then
    fine-grained argument validation per tool (Phase 3).

    Deny unknown tools, config-disabled tools, and anything without an
    explicit session grant (absent and revoked both read as "no", so the
    LLM cannot distinguish them).
    """
    if request.name not in TOOLS:
        return Decision(False, "unknown tool")
    if config.enabled.get(request.name) is not True:
        return Decision(False, "disabled")
    grants = session_grants or {}
    if grants.get(request.name) is not True:
        return Decision(False, "not permitted")

    # Fine-grained checks -- NEW:
    if request.name == "read_file":
        return _decide_read_file(request.args, config)
    if request.name == "open_url":
        return _decide_open_url(request.args, config)
    return Decision(True, None)  # future tools default to coarse-only


def _decide_read_file(args, config: ToolsConfig) -> Decision:
    path = (args or {}).get("path")
    if not isinstance(path, str) or not path.strip():
        return Decision(False, "missing path")
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        return Decision(False, "path must be absolute")
    resolved = os.path.realpath(expanded)
    norm_resolved = os.path.normcase(resolved)
    for allowed in config.read_file_dirs:
        norm_allowed = os.path.normcase(allowed)
        if norm_resolved == norm_allowed or norm_resolved.startswith(
                norm_allowed + os.sep):
            return Decision(True, None)
    return Decision(False, "path not allowed")


def _decide_open_url(args, config: ToolsConfig) -> Decision:
    url = (args or {}).get("url")
    if not isinstance(url, str) or not url.strip():
        return Decision(False, "missing url")
    try:
        parsed = urllib.parse.urlparse(url.strip())
    except ValueError:
        return Decision(False, "invalid url")
    if parsed.scheme.lower() not in ("http", "https"):
        return Decision(False, "unsupported scheme")
    host = parsed.hostname or ""
    if not host:
        return Decision(False, "invalid url")
    if host not in config.open_url_domains:
        return Decision(False, "domain not allowed")
    return Decision(True, None)


def execute(
    request: ToolRequest,
    config: ToolsConfig,
    *,
    launcher: Callable[[str], bool] | None = None,
) -> ToolResult:
    """Perform the tool action. Caller must have already obtained
    Decision(allowed=True) from decide() -- execute does not re-check.
    Never raises; all errors return as ToolResult(ok=False, error=...).
    """
    if request.name == "read_file":
        return _execute_read_file(request.args, config)
    if request.name == "open_url":
        return _execute_open_url(request.args, launcher)
    return ToolResult(ok=False, output="", error="unknown tool")


def _execute_read_file(args, config: ToolsConfig) -> ToolResult:
    path = (args or {}).get("path", "")
    try:
        resolved = os.path.realpath(os.path.expanduser(path))
        with open(resolved, "rb") as fh:
            raw = fh.read(config.max_file_bytes + 1)
        text = raw[:config.max_file_bytes].decode("utf-8", errors="replace")
        return ToolResult(ok=True, output=text, error=None)
    except Exception:
        return ToolResult(ok=False, output="", error="read failed")


def _execute_open_url(args, launcher) -> ToolResult:
    url = (args or {}).get("url", "")
    try:
        if launcher is None:
            import webbrowser

            launcher = webbrowser.open
        if launcher(url):
            return ToolResult(ok=True, output="opened", error=None)
        return ToolResult(ok=False, output="", error="launch failed")
    except Exception:
        return ToolResult(ok=False, output="", error="launch failed")
