"""Slice 7 Phase-1 tests: tools data layer (types, parser, config)."""
import ast
import dataclasses
import json
import os

import pytest

from carrot.tools import (
    TOOLS,
    Decision,
    ToolCall,
    ToolRequest,
    ToolResult,
    ToolsConfig,
    decide,
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
        max_file_bytes=4096, max_iterations=2)


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


def test_decide_allow_with_grant():
    assert decide(ToolRequest("read_file", {"path": "a.txt"}),
                  _tools_cfg(), {"read_file": True}) == \
        Decision(True, None)


def test_decide_allow_both_tools():
    cfg, grants = _tools_cfg(), {"read_file": True, "open_url": True}
    assert decide(ToolRequest("read_file", {}), cfg, grants) == \
        Decision(True, None)
    assert decide(ToolRequest("open_url", {"url": "https://x.example"}),
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


def test_decide_deterministic():
    cfg = _tools_cfg()
    first = decide(ToolRequest("read_file", {"path": "a.txt"}), cfg,
                   {"read_file": True})
    second = decide(ToolRequest("read_file", {"path": "a.txt"}), cfg,
                    {"read_file": True})
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


def test_decide_ignores_args():
    assert decide(ToolRequest("read_file", {"path": "/etc/passwd"}),
                  _tools_cfg(), {"read_file": True}) == \
        Decision(True, None)
