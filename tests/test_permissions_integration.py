"""Slice 8 Phase-3 tests: Carrot permission integration.

No new gates, no prompt changes: permission state loads at construction
and flows into step 6's decide() call only.
"""
import json

from carrot.permissions import PermissionStore
from carrot.tools import Decision, ToolRequest, ToolsConfig, decide
from tests.test_scenarios import make_carrot


def _perm_carrot(tmp_path, grants=None, replies=None, name="permissions.json"):
    store = PermissionStore(tmp_path / name)
    for tool, flag in (grants or {}).items():
        store.set_grant(tool, flag)
    c = make_carrot(permission_store=store)
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    c.tools_config = ToolsConfig(
        enabled={"read_file": True, "open_url": True},
        read_file_dirs=(str(notes),),
        open_url_domains=("example.com",),
        max_file_bytes=4096, max_iterations=2,
        levels={"read_file": 1, "open_url": 2})
    if replies is not None:
        calls = {"n": 0, "histories": []}

        def generate(*, system, history, user):
            calls["n"] += 1
            calls["histories"].append(list(history))
            return replies[min(calls["n"] - 1, len(replies) - 1)]

        c.llm.generate = generate
        c.calls = calls
    return c


def _request(path):
    return "```tool:read_file\n" + json.dumps({"path": path}) + "\n```"


def test_fresh_loads_empty(tmp_path):
    c = _perm_carrot(tmp_path)
    assert c.permission_state.grants == {}
    assert c.permission_state.policy_rev == 0


def test_existing_store_loaded(tmp_path):
    path = tmp_path / "permissions.json"
    path.write_text(json.dumps({"version": 1,
                                "grants": {"read_file": True},
                                "policy_rev": 3}),
                    encoding="utf-8")
    c = make_carrot(permission_store=PermissionStore(path))
    assert c.permission_state.grants["read_file"] is True
    assert c.permission_state.policy_rev == 3
    assert c.permission_store.path == path


def test_persistent_grant_allows_level1(tmp_path):
    target = tmp_path / "notes" / "a.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text("hello", encoding="utf-8")
    c = _perm_carrot(tmp_path, grants={"read_file": True},
                     replies=[_request(str(target)), "Here it is."])
    c.process_turn("please read the file")
    assert c.last_tool_calls[-1].allowed is True
    assert c.last_tool_calls[-1].ok is True


def test_no_grant_denies(tmp_path):
    c = _perm_carrot(tmp_path,
                     replies=[_request(str(tmp_path / "a.txt")),
                              "I can't do that."])
    c.process_turn("please read the file")
    assert c.last_tool_calls[-1].allowed is False
    assert "(not permitted: not permitted)" in c.calls["histories"][1][-1][
        "content"]


def test_level3_session_only_decide(tmp_path):
    # Direct decide() check (not through process_turn): no TOOLS change
    # needed to exercise the level-3 tier rule.
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    cfg = ToolsConfig(
        enabled={"read_file": True, "open_url": True},
        read_file_dirs=(str(notes),), open_url_domains=(),
        max_file_bytes=4096, max_iterations=2,
        levels={"read_file": 3, "open_url": 2})
    req = ToolRequest("read_file", {"path": str(notes / "a.txt")})
    assert decide(req, cfg, {"read_file": True}, {}) == \
        Decision(False, "requires persistent grant")


def test_level3_persistent_decide(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    cfg = ToolsConfig(
        enabled={"read_file": True, "open_url": True},
        read_file_dirs=(str(notes),), open_url_domains=(),
        max_file_bytes=4096, max_iterations=2,
        levels={"read_file": 3, "open_url": 2})
    req = ToolRequest("read_file", {"path": str(notes / "a.txt")})
    assert decide(req, cfg, {}, {"read_file": True}) == \
        Decision(True, None)


def test_revoke_wins_decide(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    cfg = ToolsConfig(
        enabled={"read_file": True, "open_url": True},
        read_file_dirs=(str(notes),), open_url_domains=(),
        max_file_bytes=4096, max_iterations=2,
        levels={"read_file": 1, "open_url": 2})
    req = ToolRequest("read_file", {"path": str(notes / "a.txt")})
    assert decide(req, cfg, {"read_file": False},
                  {"read_file": True}) == Decision(False, "not permitted")


def test_deterministic_pair(tmp_path):
    target = tmp_path / "notes" / "a.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text("hello", encoding="utf-8")
    req = _request(str(target))
    first = _perm_carrot(tmp_path, grants={"read_file": True},
                         replies=[req, "done."])
    second = _perm_carrot(tmp_path, grants={"read_file": True},
                          replies=[req, "done."])
    first.process_turn("please read the file")
    second.process_turn("please read the file")
    assert list(first.last_tool_calls) == list(second.last_tool_calls)
    assert first.state.to_dict() == second.state.to_dict()


def test_permission_state_not_in_state_dict(tmp_path):
    c = _perm_carrot(tmp_path, grants={"read_file": True})
    c.process_turn("hello there")
    assert "permissions" not in c.state.to_dict()
    assert "grants" not in c.state.to_dict()


def test_permission_state_unmutated_by_turn(tmp_path):
    c = _perm_carrot(tmp_path, grants={"read_file": True})
    before = dict(c.permission_state.grants)
    c.process_turn("hello there")
    assert dict(c.permission_state.grants) == before


def test_no_writes_on_turn(tmp_path):
    c = _perm_carrot(tmp_path, grants={"read_file": True})
    path = tmp_path / "permissions.json"
    mtime = path.stat().st_mtime_ns
    c.process_turn("hello there")
    assert path.stat().st_mtime_ns == mtime


def test_construction_writes_nothing(tmp_path):
    path = tmp_path / "fresh.json"
    make_carrot(permission_store=PermissionStore(path))
    assert not path.exists()
