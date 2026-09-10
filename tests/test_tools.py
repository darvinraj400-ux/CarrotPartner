"""Slice 7 Phase-1 tests: tools data layer (types, parser, config)."""
import ast
import dataclasses
import json
import os
import sys

import pytest

from carrot.tools import (
    TOOLS,
    Decision,
    ToolCall,
    ToolRequest,
    ToolResult,
    ToolsConfig,
    decide,
    execute,
    parse_tool_request,
)
from tests.conftest import ROOT
from tests.test_scenarios import make_carrot


def test_value_types_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ToolRequest("read_file", {}).name = "x"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ToolResult(True, "out").ok = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        ToolCall("read_file", "p", True, True, 1.0).tool = "x"
    with pytest.raises(dataclasses.FrozenInstanceError):
        Decision(True).allowed = False


def test_parse_valid_read_file():
    assert parse_tool_request(
        '```tool:read_file\n{"path": "notes.txt"}\n```') == \
        ToolRequest("read_file", {"path": "notes.txt"})


def test_parse_valid_open_url():
    assert parse_tool_request(
        '```tool:open_url\n{"url": "https://example.com"}\n```') == \
        ToolRequest("open_url", {"url": "https://example.com"})


def test_parse_empty_args():
    assert parse_tool_request("```tool:read_file\n{}\n```") == \
        ToolRequest("read_file", {})


def test_parse_surrounding_text_ignored():
    assert parse_tool_request(
        'Sure thing!\n```tool:read_file\n{"path": "a.txt"}\n```\n'
        "Anything else?") == ToolRequest("read_file", {"path": "a.txt"})


def test_parse_first_block_wins():
    assert parse_tool_request(
        '```tool:read_file\n{"path": "a.txt"}\n```\n'
        '```tool:open_url\n{"url": "https://example.com"}\n```') == \
        ToolRequest("read_file", {"path": "a.txt"})


def test_parse_unknown_tool():
    assert parse_tool_request("```tool:rm_rf\n{}\n```") is None


def test_parse_malformed_json():
    assert parse_tool_request("```tool:read_file\n{not json}\n```") is None


def test_parse_non_string_value():
    assert parse_tool_request('```tool:read_file\n{"path": 5}\n```') is None


def test_parse_non_dict_json():
    assert parse_tool_request("```tool:read_file\n[1, 2, 3]\n```") is None


def test_parse_unclosed_fence():
    assert parse_tool_request('```tool:read_file\n{"path": "a"}') is None


def test_parse_no_fence():
    assert parse_tool_request("Just plain text") is None


def test_parse_empty_string():
    assert parse_tool_request("") is None


def test_parse_wrong_fence_prefix():
    assert parse_tool_request("```toolread_file\n{}\n```") is None


def test_parse_case_sensitive():
    assert parse_tool_request("```TOOL:read_file\n{}\n```") is None


def test_config_loads_valid():
    cfg = ToolsConfig.load(str(ROOT / "config" / "tools.json"))
    assert cfg.enabled == {"read_file": True, "open_url": True}
    assert cfg.read_file_dirs == ()
    assert cfg.open_url_domains == ()
    assert cfg.max_file_bytes == 4096
    assert cfg.max_iterations == 2
    assert set(TOOLS) == {"read_file", "open_url"}


def _write_config(tmp_path, data, name="tools.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ToolsConfig.load(str(tmp_path / "nope.json"))


def test_config_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        ToolsConfig.load(str(path))


def test_config_missing_enabled(tmp_path):
    data = {"read_file_dirs": [], "open_url_domains": [],
            "max_file_bytes": 4096, "max_iterations": 2}
    cfg = ToolsConfig.load(_write_config(tmp_path, data))
    assert cfg.enabled == {"read_file": True, "open_url": True}


def test_config_missing_max_iterations(tmp_path):
    data = {"enabled": {"read_file": True, "open_url": True},
            "read_file_dirs": [], "open_url_domains": [],
            "max_file_bytes": 4096}
    assert ToolsConfig.load(
        _write_config(tmp_path, data)).max_iterations == 2


def test_config_max_iterations_out_of_range(tmp_path):
    base = {"enabled": {"read_file": True, "open_url": True},
            "read_file_dirs": [], "open_url_domains": [],
            "max_file_bytes": 4096}
    with pytest.raises(ValueError):
        ToolsConfig.load(_write_config(
            tmp_path, dict(base, max_iterations=0)))
    with pytest.raises(ValueError):
        ToolsConfig.load(_write_config(
            tmp_path, dict(base, max_iterations=4)))


def test_config_dir_resolution(tmp_path):
    data = {"enabled": {"read_file": True, "open_url": True},
            "read_file_dirs": ["~/notes"], "open_url_domains": [],
            "max_file_bytes": 4096, "max_iterations": 2}
    cfg = ToolsConfig.load(_write_config(tmp_path, data))
    assert cfg.read_file_dirs == (
        os.path.abspath(os.path.expanduser("~/notes")),)


def test_config_domain_lowercasing(tmp_path):
    data = {"enabled": {"read_file": True, "open_url": True},
            "read_file_dirs": [],
            "open_url_domains": ["Example.COM", "TEST.org"],
            "max_file_bytes": 4096, "max_iterations": 2}
    assert ToolsConfig.load(
        _write_config(tmp_path, data)).open_url_domains == \
        ("example.com", "test.org")


def test_config_frozen(tmp_path):
    cfg = ToolsConfig.load(_write_config(tmp_path, {
        "enabled": {"read_file": True, "open_url": True},
        "read_file_dirs": [], "open_url_domains": [],
        "max_file_bytes": 4096, "max_iterations": 2}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.max_iterations = 3


def test_tools_module_boundary():
    banned = {"emotions", "drives", "memory", "context", "environment",
              "carrot", "classifier", "appraisal", "behavior",
              "personality", "llm", "subprocess"}
    offenders = []
    tree = ast.parse((ROOT / "carrot" / "tools.py").read_text(
        encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in banned:
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if node.level == 0 and mod in banned:
                offenders.append(node.module)
            elif node.level > 0 and mod in banned:
                offenders.append(mod)
    assert offenders == []


def _tools_cfg(enabled=None):
    return ToolsConfig(
        enabled={"read_file": True, "open_url": True}
        if enabled is None else dict(enabled),
        read_file_dirs=(), open_url_domains=(),
        max_file_bytes=4096, max_iterations=2, levels={})


def test_decide_unknown_tool():
    assert decide(ToolRequest("rm_rf", {}), _tools_cfg(), {}) == \
        Decision(False, "unknown tool")


def test_decide_config_disabled_overrides_grant():
    cfg = _tools_cfg(enabled={"read_file": False, "open_url": True})
    assert decide(ToolRequest("read_file", {"path": "a.txt"}), cfg,
                  {"read_file": True}) == Decision(False, "disabled")


def test_decide_no_session_grant():
    assert decide(ToolRequest("read_file", {"path": "a.txt"}),
                  _tools_cfg(), {}) == Decision(False, "not permitted")


def test_decide_session_revoke():
    assert decide(ToolRequest("read_file", {"path": "a.txt"}),
                  _tools_cfg(), {"read_file": False}) == \
        Decision(False, "not permitted")


def test_decide_disabled_beats_missing_grant():
    cfg = _tools_cfg(enabled={"read_file": False, "open_url": True})
    assert decide(ToolRequest("read_file", {}), cfg, {}) == \
        Decision(False, "disabled")


def test_decide_allow_with_grant(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("read_file", {"path": str(notes / "a.txt")}),
                  cfg, {"read_file": True}) == Decision(True, None)


def test_decide_allow_both_tools(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    grants = {"read_file": True, "open_url": True}
    assert decide(ToolRequest("read_file", {"path": str(notes / "a.txt")}),
                  cfg, grants) == Decision(True, None)
    assert decide(ToolRequest("open_url", {"url": "https://example.com/"}),
                  cfg, grants) == Decision(True, None)


def test_decide_does_not_mutate_grants():
    grants = {"read_file": True}
    decide(ToolRequest("read_file", {}), _tools_cfg(), grants)
    assert grants == {"read_file": True}


def test_decide_does_not_mutate_config():
    cfg = _tools_cfg()
    snapshot = (dict(cfg.enabled), cfg.read_file_dirs,
                cfg.open_url_domains, cfg.max_file_bytes,
                cfg.max_iterations)
    decide(ToolRequest("read_file", {}), cfg, {"read_file": True})
    assert (dict(cfg.enabled), cfg.read_file_dirs, cfg.open_url_domains,
            cfg.max_file_bytes, cfg.max_iterations) == snapshot


def test_decide_deterministic(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    first = decide(ToolRequest("read_file", {"path": str(notes / "a.txt")}),
                   cfg, {"read_file": True})
    second = decide(ToolRequest("read_file", {"path": str(notes / "a.txt")}),
                    cfg, {"read_file": True})
    assert first == second == Decision(True, None)


def test_session_grants_initialised_empty():
    assert make_carrot().session_grants == {}


def test_session_grants_not_persisted():
    first = make_carrot()
    first.session_grants["read_file"] = True
    assert make_carrot().session_grants == {}


def test_session_grants_not_in_state():
    assert "session_grants" not in make_carrot().state.to_dict()


def test_decide_config_checked_before_session():
    cfg = _tools_cfg(enabled={"read_file": False, "open_url": True})
    assert decide(ToolRequest("read_file", {}), cfg, {}) == \
        Decision(False, "disabled")


def test_decide_unknown_checked_before_config():
    cfg = _tools_cfg(enabled={"rm_rf": False, "read_file": True})
    assert decide(ToolRequest("rm_rf", {}), cfg, {}) == \
        Decision(False, "unknown tool")


def test_decide_inspects_args_for_known_tools(tmp_path):
    # Phase 2's "decide ignores args" boundary has moved: a valid grant
    # still passes coarse checks, but fine-grained validation now denies
    # off-allowlist paths.
    cfg, notes = _allow_cfg(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir(exist_ok=True)
    assert decide(ToolRequest("read_file",
                              {"path": str(outside / "passwd")}),
                  cfg, {"read_file": True}) == \
        Decision(False, "path not allowed")


def _allow_cfg(tmp_path, sub="notes", domains=("example.com",),
               max_bytes=4096):
    notes = tmp_path / sub
    notes.mkdir(exist_ok=True)
    return ToolsConfig(
        enabled={"read_file": True, "open_url": True},
        read_file_dirs=(str(notes),),
        open_url_domains=tuple(domains),
        max_file_bytes=max_bytes, max_iterations=2, levels={}), notes


def _granted():
    return {"read_file": True, "open_url": True}


def test_read_file_valid_inside_dir(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("read_file", {"path": str(notes / "a.txt")}),
                  cfg, _granted()) == Decision(True, None)


def test_read_file_path_equals_dir(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("read_file", {"path": str(notes)}),
                  cfg, _granted()) == Decision(True, None)


def test_read_file_outside_dir(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir(exist_ok=True)
    assert decide(ToolRequest("read_file", {"path": str(outside / "x.txt")}),
                  cfg, _granted()) == Decision(False, "path not allowed")


def test_read_file_traversal_denied(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    sneaky = str(notes / ".." / "secrets" / "a.txt")
    assert decide(ToolRequest("read_file", {"path": sneaky}),
                  cfg, _granted()) == Decision(False, "path not allowed")


def test_read_file_prefix_boundary(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    rival = tmp_path / "notes-secret"
    rival.mkdir(exist_ok=True)
    assert decide(ToolRequest("read_file", {"path": str(rival / "a.txt")}),
                  cfg, _granted()) == Decision(False, "path not allowed")


def test_read_file_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    cfg, notes = _allow_cfg(tmp_path)
    link = notes / "link"
    try:
        os.symlink(str(outside), str(link), target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert decide(ToolRequest("read_file", {"path": str(link / "a.txt")}),
                  cfg, _granted()) == Decision(False, "path not allowed")


def test_read_file_missing_path(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("read_file", {}),
                  cfg, _granted()) == Decision(False, "missing path")


def test_read_file_empty_path(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("read_file", {"path": ""}),
                  cfg, _granted()) == Decision(False, "missing path")


def test_read_file_relative_path(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("read_file", {"path": "notes/a.txt"}),
                  cfg, _granted()) == Decision(False, "path must be absolute")


def test_read_file_tilde_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cfg, notes = _allow_cfg(tmp_path)
    (tmp_path / "notes").mkdir(exist_ok=True)
    cfg_home = ToolsConfig(
        enabled={"read_file": True, "open_url": True},
        read_file_dirs=(str(tmp_path),),
        open_url_domains=(), max_file_bytes=4096, max_iterations=2,
        levels={})
    assert decide(ToolRequest("read_file", {"path": "~/notes/a.txt"}),
                  cfg_home, _granted()) == Decision(True, None)


@pytest.mark.skipif(sys.platform != "win32", reason="windows-only")
def test_read_file_case_insensitive_windows(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    weird = str(notes).upper() + os.sep + "A.TXT"
    assert decide(ToolRequest("read_file", {"path": weird}),
                  cfg, _granted()) == Decision(True, None)


def test_read_file_empty_allowlist(tmp_path):
    cfg = _tools_cfg()
    assert decide(ToolRequest("read_file", {"path": str(tmp_path / "a.txt")}),
                  cfg, _granted()) == Decision(False, "path not allowed")


def test_open_url_valid_http(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(
        ToolRequest("open_url", {"url": "http://example.com/page"}),
        cfg, _granted()) == Decision(True, None)


def test_open_url_valid_https(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(
        ToolRequest("open_url", {"url": "https://example.com/page"}),
        cfg, _granted()) == Decision(True, None)


def test_open_url_domain_not_allowed(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("open_url", {"url": "https://example.org/"}),
                  cfg, _granted()) == Decision(False, "domain not allowed")


def test_open_url_subdomain_not_wildcard(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(
        ToolRequest("open_url", {"url": "https://www.example.com/"}),
        cfg, _granted()) == Decision(False, "domain not allowed")


def test_open_url_file_scheme(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("open_url", {"url": "file:///etc/passwd"}),
                  cfg, _granted()) == Decision(False, "unsupported scheme")


def test_open_url_javascript_scheme(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("open_url", {"url": "javascript:alert(1)"}),
                  cfg, _granted()) == Decision(False, "unsupported scheme")


def test_open_url_no_scheme(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("open_url", {"url": "example.com"}),
                  cfg, _granted()) == Decision(False, "unsupported scheme")


def test_open_url_missing_url(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("open_url", {}),
                  cfg, _granted()) == Decision(False, "missing url")


def test_open_url_empty_url(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("open_url", {"url": ""}),
                  cfg, _granted()) == Decision(False, "missing url")


def test_open_url_port_stripped(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(
        ToolRequest("open_url", {"url": "http://example.com:8080/page"}),
        cfg, _granted()) == Decision(True, None)


def test_open_url_uppercase_host(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert decide(ToolRequest("open_url", {"url": "http://EXAMPLE.COM/"}),
                  cfg, _granted()) == Decision(True, None)


def test_execute_read_file_valid(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    target = notes / "a.txt"
    target.write_text("hello world", encoding="utf-8")
    result = execute(ToolRequest("read_file", {"path": str(target)}), cfg)
    assert result == ToolResult(ok=True, output="hello world", error=None)


def test_execute_read_file_byte_cap(tmp_path):
    cfg, notes = _allow_cfg(tmp_path, max_bytes=10)
    target = notes / "big.txt"
    target.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    result = execute(ToolRequest("read_file", {"path": str(target)}), cfg)
    assert result.ok is True
    assert result.output == "abcdefghij"


def test_execute_read_file_missing(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    result = execute(
        ToolRequest("read_file", {"path": str(notes / "nope.txt")}), cfg)
    assert result == ToolResult(ok=False, output="", error="read failed")


def test_execute_read_file_non_utf8(tmp_path):
    cfg, notes = _allow_cfg(tmp_path)
    target = notes / "raw.bin"
    target.write_bytes(b"\xff\xfehello")
    result = execute(ToolRequest("read_file", {"path": str(target)}), cfg)
    assert result.ok is True
    assert "�" in result.output
    assert result.output.endswith("hello")


def test_execute_open_url_fake_launcher(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    calls = []
    launcher = lambda url: calls.append(url) or True  # noqa: E731
    result = execute(
        ToolRequest("open_url", {"url": "https://example.com/"}), cfg,
        launcher=launcher)
    assert calls == ["https://example.com/"]
    assert result == ToolResult(ok=True, output="opened", error=None)


def test_execute_open_url_launcher_false(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    result = execute(
        ToolRequest("open_url", {"url": "https://example.com/"}), cfg,
        launcher=lambda url: False)
    assert result == ToolResult(ok=False, output="", error="launch failed")


def test_execute_open_url_launcher_raises(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)

    def boom(url):
        raise RuntimeError("no browser here")

    result = execute(
        ToolRequest("open_url", {"url": "https://example.com/"}), cfg,
        launcher=boom)
    assert result == ToolResult(ok=False, output="", error="launch failed")


def test_execute_never_touches_real_launcher():
    assert "webbrowser" not in sys.modules
    cfg = ToolsConfig(
        enabled={"read_file": True, "open_url": True},
        read_file_dirs=(), open_url_domains=("example.com",),
        max_file_bytes=4096, max_iterations=2, levels={})
    result = execute(
        ToolRequest("open_url", {"url": "https://example.com/"}), cfg,
        launcher=lambda url: True)
    assert result.ok is True
    assert "webbrowser" not in sys.modules


def test_execute_unknown_tool(tmp_path):
    cfg, _notes = _allow_cfg(tmp_path)
    assert execute(ToolRequest("rm_rf", {}), cfg) == \
        ToolResult(ok=False, output="", error="unknown tool")


def _levels_base(**overrides):
    base = {"enabled": {"read_file": True, "open_url": True},
            "levels": {"read_file": 1, "open_url": 2},
            "read_file_dirs": [], "open_url_domains": [],
            "max_file_bytes": 4096, "max_iterations": 2}
    base.update(overrides)
    return base


def test_levels_loads_valid():
    cfg = ToolsConfig.load(str(ROOT / "config" / "tools.json"))
    assert cfg.levels["read_file"] == 1
    assert cfg.levels["open_url"] == 2


def test_levels_missing_defaults_all_four(tmp_path):
    data = {"enabled": {"read_file": True, "open_url": True},
            "read_file_dirs": [], "open_url_domains": [],
            "max_file_bytes": 4096, "max_iterations": 2}
    path = tmp_path / "tools.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    cfg = ToolsConfig.load(str(path))
    assert cfg.levels["read_file"] == 4
    assert cfg.levels["open_url"] == 4


def test_levels_out_of_range(tmp_path):
    for bad in (5, -1):
        data = _levels_base(levels={"read_file": bad, "open_url": 2})
        path = tmp_path / f"bad-{bad}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError):
            ToolsConfig.load(str(path))


def test_levels_non_int(tmp_path):
    data = _levels_base(levels={"read_file": "1", "open_url": 2})
    path = tmp_path / "badtype.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        ToolsConfig.load(str(path))


def test_levels_frozen(tmp_path):
    cfg = ToolsConfig.load(_write_config(tmp_path, _levels_base()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.levels = {}
    with pytest.raises(TypeError):
        cfg.levels["read_file"] = 9
