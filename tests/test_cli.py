"""Phase-5 tests: /ctx handler, last_context retention, help text."""
import json
import os
import re
import sys

import pytest

from carrot.cli import (
    HELP_TEXT,
    NO_CONTEXT_YET,
    NO_MEMORIES_YET,
    _initiative_system_prompt,
    _parse_away_args,
    _run_away_mode,
    build_carrot,
    format_ctx,
    format_dbg,
    format_drives,
    format_env,
    format_calls,
    format_initiative,
    format_permissions,
    format_tools,
    handle_allow,
    handle_deny,
    handle_revoke,
    handle_revoke_all,
    format_mem_debug,
    format_memories,
    format_relationship,
    handle_ctx,
    handle_forget,
    handle_remember,
    main,
)
from carrot.context import derive_context
from carrot.avatar import AVATAR_UNAVAILABLE_MSG
from carrot.state_store import StateStore
from carrot.voice import VOICE_EMPTY_MSG
from carrot.channel import CHANNEL_EMPTY_MSG, FileChannel
from carrot.drives import DriveState, DriveStore, DrivesConfig
from carrot.tools import ToolsConfig, ToolRequest, decide
from carrot.permissions import PermissionStore
from carrot.environment import (
    EMPTY_SNAPSHOT,
    MockEnvironmentProbe,
    RawObservation,
)
from carrot.initiative import ACTIVE_REASONS
from carrot.memory import MAX_MEMORIES, Memory, MemoryStore
from carrot.relationship import RelationshipState, RelationshipStore
from tests.conftest import ROOT
from tests.test_scenarios import FakeTime, make_carrot


def _mem_carrot(tmp_path):
    return make_carrot(
        memory_store=MemoryStore(str(tmp_path / "mem.json")))


def test_ctx_before_any_turn():
    c = make_carrot()
    assert c.last_context is None
    assert handle_ctx(c) == NO_CONTEXT_YET


def test_last_context_matches_derive():
    c = make_carrot()
    c.process_turn("talking about gardening")
    # First turn was interpreted against an empty pre-turn context.
    assert c.last_context == derive_context(())
    assert "turns: 0" in handle_ctx(c)
    c.process_turn("my guitar needs tuning")
    prior = tuple(c.buffer.recent()[:-2])
    assert c.last_context == derive_context(prior)
    out = handle_ctx(c)
    assert "pending_question: False" in out
    assert "playful_frame: False" in out
    assert "recent_harshness: False" in out
    assert "recent_topics: [talking, gardening]" in out
    assert "turns: 2" in out


def test_ctx_shows_flags_and_recent_turns():
    c = make_carrot()
    c.process_turn("that joke was funny lol")
    assert "playful_frame: False" in handle_ctx(c)  # pre-turn was empty
    c.process_turn("more jokes please")
    out = handle_ctx(c)
    assert "playful_frame: True" in out
    assert "turns: 2" in out
    assert "user: that joke was funny lol" in out
    assert re.search(r"\d\.\d", out) is None
    assert "happiness" not in out


def test_ctx_does_not_mutate():
    c = make_carrot()
    c.process_turn("hello there?")
    before_state = c.state.to_dict()
    before_buffer = list(c.buffer.recent())
    handle_ctx(c)
    format_ctx(c.last_context)
    assert c.state.to_dict() == before_state
    assert list(c.buffer.recent()) == before_buffer


def test_help_lists_ctx():
    assert "/ctx" in HELP_TEXT
    assert "/dbg" in HELP_TEXT
    assert "/quit" in HELP_TEXT


def _seed(carrot, *texts):
    for text in texts:
        carrot.memory_store.add(Memory(
            text=text, kind="fact", importance=0.5,
            created_ts=1000.0, last_recalled_ts=1000.0))


def test_memories_empty(tmp_path):
    assert format_memories(_mem_carrot(tmp_path).memory_store) == \
        NO_MEMORIES_YET


def test_memories_lists_kind_and_text(tmp_path):
    c = _mem_carrot(tmp_path)
    _seed(c, "my dog is funny", "I love ramen", "my cat is grumpy")
    lines = format_memories(c.memory_store).split("\n")
    assert lines == ["[fact] my dog is funny",
                     "[fact] I love ramen",
                     "[fact] my cat is grumpy"]
    out = format_memories(c.memory_store)
    assert re.search(r"\d\.\d", out) is None
    assert "importance" not in out
    assert "created" not in out


def test_memories_no_mutation(tmp_path):
    c = _mem_carrot(tmp_path)
    _seed(c, "my dog is funny")
    path = tmp_path / "mem.json"
    before = c.memory_store.all()
    mtime = path.stat().st_mtime_ns
    format_memories(c.memory_store)
    assert c.memory_store.all() == before
    assert path.stat().st_mtime_ns == mtime


def test_mem_empty(tmp_path):
    assert format_mem_debug(_mem_carrot(tmp_path).memory_store) == \
        NO_MEMORIES_YET


def test_mem_shows_debug_detail(tmp_path):
    c = _mem_carrot(tmp_path)
    _seed(c, "my dog is funny", "I love ramen", "my cat is grumpy")
    lines = format_mem_debug(c.memory_store).split("\n")
    assert len(lines) == 3
    for line in lines:
        assert "importance=" in line
        assert "created=" in line
        assert "recalled=" in line


def test_mem_no_mutation(tmp_path):
    c = _mem_carrot(tmp_path)
    _seed(c, "my dog is funny")
    path = tmp_path / "mem.json"
    before = c.memory_store.all()
    mtime = path.stat().st_mtime_ns
    format_mem_debug(c.memory_store)
    assert c.memory_store.all() == before
    assert path.stat().st_mtime_ns == mtime


def test_remember_valid(tmp_path):
    c = _mem_carrot(tmp_path)
    assert handle_remember(c, "my dog is a corgi") == "remembered"
    mems = c.memory_store.all()
    assert len(mems) == 1
    assert mems[0].text == "my dog is a corgi"
    assert mems[0].kind == "fact"
    assert mems[0].importance == 0.8
    assert mems[0].created_ts != 0
    assert mems[0].last_recalled_ts != 0


def test_remember_empty(tmp_path):
    c = _mem_carrot(tmp_path)
    assert handle_remember(c, "") == "usage: /remember <text>"
    assert c.memory_store.all() == ()


def test_remember_whitespace(tmp_path):
    c = _mem_carrot(tmp_path)
    assert handle_remember(c, "    ") == "usage: /remember <text>"
    assert c.memory_store.all() == ()


def test_remember_too_long(tmp_path):
    c = _mem_carrot(tmp_path)
    assert handle_remember(c, "x" * 141) == "too long"
    assert c.memory_store.all() == ()


def test_remember_duplicate(tmp_path):
    c = _mem_carrot(tmp_path)
    assert handle_remember(c, "My dog is funny") == "remembered"
    assert handle_remember(c, "my DOG is FUNNY") == "already remembered"
    assert len(c.memory_store.all()) == 1


def test_remember_cap_full(tmp_path):
    c = _mem_carrot(tmp_path)
    for i in range(MAX_MEMORIES):
        c.memory_store.add(Memory(
            text=f"fact number {i}", kind="fact", importance=0.5,
            created_ts=float(i), last_recalled_ts=float(i)))
    assert handle_remember(c, "one more") == "memory full"
    assert len(c.memory_store.all()) == MAX_MEMORIES


def test_forget_matching(tmp_path):
    c = _mem_carrot(tmp_path)
    _seed(c, "my dog is funny", "my dog likes snow", "I love ramen")
    assert handle_forget(c, "dog") == "forgot 2"
    assert [m.text for m in c.memory_store.all()] == ["I love ramen"]


def test_forget_case_insensitive(tmp_path):
    c = _mem_carrot(tmp_path)
    _seed(c, "my dog is a corgi")
    assert handle_forget(c, "CORGI") == "forgot 1"
    assert c.memory_store.all() == ()


def test_forget_no_match(tmp_path):
    c = _mem_carrot(tmp_path)
    _seed(c, "I love ramen")
    assert handle_forget(c, "zzzz") == "no matches"
    assert len(c.memory_store.all()) == 1


def test_forget_empty_arg(tmp_path):
    c = _mem_carrot(tmp_path)
    _seed(c, "I love ramen")
    assert handle_forget(c, "") == "usage: /forget <substring>"
    assert handle_forget(c, "   ") == "usage: /forget <substring>"
    assert len(c.memory_store.all()) == 1


def test_help_lists_memory_commands():
    for cmd in ("/memories", "/remember", "/forget", "/mem"):
        assert cmd in HELP_TEXT


def test_cli_memory_round_trip(tmp_path):
    c = _mem_carrot(tmp_path)
    assert handle_remember(c, "my dog is a corgi") == "remembered"
    assert "[fact] my dog is a corgi" in format_memories(c.memory_store)
    assert handle_forget(c, "corgi") == "forgot 1"
    assert format_memories(c.memory_store) == NO_MEMORIES_YET


def _drive_cfg():
    return DrivesConfig(rise_per_hour={}, pull_threshold=0.6,
                        presence_satisfaction={}, event_satisfaction={})


def _drive_carrot_cli(tmp_path, social=0.0, novelty=0.0):
    store = DriveStore(tmp_path / "d.json")
    store.save(DriveState(social=social, novelty=novelty))
    return make_carrot(drive_store=store)


def test_drives_empty(tmp_path):
    out = format_drives(_drive_carrot_cli(tmp_path).drive_store,
                        _drive_cfg())
    assert "social: low" in out
    assert "novelty: low" in out
    assert "pull: none" in out
    assert re.search(r"\d", out) is None


def test_drives_high_social(tmp_path):
    out = format_drives(_drive_carrot_cli(tmp_path, social=0.8).drive_store,
                        _drive_cfg())
    assert "social: high" in out
    assert "pull: social" in out


def test_drives_high_novelty(tmp_path):
    out = format_drives(_drive_carrot_cli(tmp_path, novelty=0.8).drive_store,
                        _drive_cfg())
    assert "novelty: high" in out
    assert "pull: novelty" in out


def test_drives_boundary_exact_threshold(tmp_path):
    out = format_drives(_drive_carrot_cli(tmp_path, social=0.6).drive_store,
                        _drive_cfg())
    assert "social: high" in out
    assert "pull: social" in out


def test_drives_medium(tmp_path):
    out = format_drives(_drive_carrot_cli(tmp_path, social=0.4).drive_store,
                        _drive_cfg())
    assert "social: medium" in out
    assert "pull: none" in out


def test_drives_low(tmp_path):
    out = format_drives(_drive_carrot_cli(tmp_path, social=0.2,
                                          novelty=0.1).drive_store,
                        _drive_cfg())
    assert "social: low" in out
    assert "novelty: low" in out
    assert "pull: none" in out


def test_drives_no_mutation(tmp_path):
    c = _drive_carrot_cli(tmp_path, social=0.8, novelty=0.2)
    path = tmp_path / "d.json"
    before = c.drive_store.load()
    mtime = path.stat().st_mtime_ns
    format_drives(c.drive_store, _drive_cfg())
    assert c.drive_store.load() == before
    assert path.stat().st_mtime_ns == mtime


def test_drives_no_digits(tmp_path):
    for social, novelty in ((0.0, 0.0), (0.8, 0.2), (1.0, 1.0)):
        out = format_drives(
            _drive_carrot_cli(tmp_path, social, novelty).drive_store,
            _drive_cfg())
        assert re.search(r"\d", out) is None, out


def test_dbg_includes_drives_floats(tmp_path):
    c = _drive_carrot_cli(tmp_path, social=0.8, novelty=0.2)
    c.process_turn("hello there")
    out = format_dbg(c)
    assert "social=" in out
    assert "novelty=" in out


def test_dbg_still_shows_existing_content(tmp_path):
    c = _drive_carrot_cli(tmp_path)
    c.process_turn("hello there")
    out = format_dbg(c)
    assert "mood=" in out
    assert "behavior=" in out
    assert "happiness" in out


def test_help_lists_drives():
    assert "/drives" in HELP_TEXT


def _staged_root(tmp_path, monkeypatch):
    import shutil

    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    for name in ("personality.json", "appraisal.json", "behavior.json",
                 "decay.json", "environment.json"):
        shutil.copy(ROOT / "config" / name, tmp_path / "config" / name)
    monkeypatch.setattr("carrot.cli.ROOT", tmp_path)
    return tmp_path


def test_launch_increments_session_count(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    assert build_carrot(environment_probe=MockEnvironmentProbe()) \
        .state.session_count == 1


def test_launch_again_increments(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    build_carrot(environment_probe=MockEnvironmentProbe())
    assert build_carrot(environment_probe=MockEnvironmentProbe()) \
        .state.session_count == 2


def test_first_seen_set_once(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    assert build_carrot(time_fn=lambda: 1000.0,
                        environment_probe=MockEnvironmentProbe()) \
        .state.first_seen_ts == 1000.0
    assert build_carrot(time_fn=lambda: 2000.0,
                        environment_probe=MockEnvironmentProbe()) \
        .state.first_seen_ts == 1000.0


def test_last_seen_untouched_by_launch(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    first = build_carrot(time_fn=lambda: 1000.0,
                         environment_probe=MockEnvironmentProbe())
    first.process_turn("hello there")
    first.store.save(first.state)
    seen = first.state.last_seen
    relaunched = build_carrot(time_fn=lambda: 9999.0,
                              environment_probe=MockEnvironmentProbe())
    assert relaunched.state.last_seen == seen
    assert relaunched.state.session_count == 2


class _CountingProbe:
    def __init__(self, observation):
        self._observation = observation
        self.calls = 0

    def observe(self):
        self.calls += 1
        return self._observation


def _env_carrot_cli(observations):
    c = make_carrot()
    c.environment_probe = MockEnvironmentProbe(observations)
    return c


def test_env_fresh_state():
    c = make_carrot()
    out = format_env(c)
    assert "app: unknown" in out
    assert "idle: False" in out
    assert "observed: no" in out


def test_env_with_app():
    c = _env_carrot_cli([RawObservation("cmd.exe", False)])
    c.process_turn("hello there")
    out = format_env(c)
    assert "app: terminal" in out
    assert "observed: yes" in out


def test_env_with_idle():
    c = _env_carrot_cli([RawObservation(None, True)])
    c.process_turn("hello there")
    out = format_env(c)
    assert "idle: True" in out
    assert "observed: yes" in out


def test_env_no_digits():
    c = _env_carrot_cli([RawObservation("cmd.exe", True)])
    c.process_turn("hello there")
    assert re.search(r"\d", format_env(c)) is None
    fresh = format_env(make_carrot())
    assert re.search(r"\d", fresh) is None


def test_env_no_mutation(tmp_path):
    path = tmp_path / "d.json"
    store = DriveStore(path)
    store.save(DriveState(social=0.8, novelty=0.2))
    c = make_carrot(drive_store=store)
    c.process_turn("hello there")
    before_state = c.state.to_dict()
    before_drives = store.load()
    mtime = path.stat().st_mtime_ns
    format_env(c)
    assert c.state.to_dict() == before_state
    assert store.load() == before_drives
    assert path.stat().st_mtime_ns == mtime


def test_env_does_not_call_probe():
    probe = _CountingProbe(RawObservation("cmd.exe", False))
    c = make_carrot()
    c.environment_probe = probe
    c.process_turn("hello there")
    assert probe.calls == 1
    format_env(c)
    assert probe.calls == 1


def test_dbg_includes_environment_line():
    c = _env_carrot_cli([RawObservation("cmd.exe", False)])
    c.process_turn("hello there")
    out = format_dbg(c)
    assert "environment: app=" in out
    assert "idle=" in out


def test_dbg_shows_env_and_existing_content():
    c = make_carrot()
    c.process_turn("hello there")
    out = format_dbg(c)
    assert "mood=" in out
    assert "behavior=" in out
    assert "happiness" in out
    assert "social=" in out


def test_help_lists_env():
    assert "/env" in HELP_TEXT


def test_build_carrot_with_explicit_mock(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    carrot = build_carrot(environment_probe=MockEnvironmentProbe())
    assert carrot.last_environment == EMPTY_SNAPSHOT
    assert format_env(carrot) == "app: unknown\nidle: False\nobserved: no"


@pytest.mark.skipif(sys.platform != "win32", reason="windows-only")
def test_build_carrot_windows_real_probe(tmp_path, monkeypatch):
    from carrot.environment_win32 import WindowsEnvironmentProbe

    _staged_root(tmp_path, monkeypatch)
    carrot = build_carrot()
    assert isinstance(carrot.environment_probe, WindowsEnvironmentProbe)


@pytest.mark.skipif(sys.platform == "win32", reason="non-windows-only")
def test_build_carrot_non_windows_mock(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    carrot = build_carrot()
    assert isinstance(carrot.environment_probe, MockEnvironmentProbe)


def _tool_cli_carrot(tmp_path=None):
    c = make_carrot()
    c.tools_config = ToolsConfig(
        enabled={"read_file": True, "open_url": True},
        read_file_dirs=(str(tmp_path),) if tmp_path is not None else (),
        open_url_domains=("example.com",),
        max_file_bytes=4096, max_iterations=2,
        # Explicit levels: Slice 8 makes unlisted tools deny-by-default.
        levels={"read_file": 1, "open_url": 2})
    return c
    return c


def _scripted_tool_carrot(tmp_path, replies):
    c = _tool_cli_carrot(tmp_path)
    calls = {"n": 0}

    def generate(*, system, history, user):
        calls["n"] += 1
        return replies[min(calls["n"] - 1, len(replies) - 1)]

    c.llm.generate = generate
    c.calls = calls
    return c


def test_tools_fresh_state():
    out = format_tools(make_carrot())
    assert "read_file: enabled (session: none)" in out
    assert "open_url: enabled (session: none)" in out


def test_tools_after_allow():
    c = make_carrot()
    assert handle_allow(c, "read_file") == "granted: read_file (session)"
    assert "read_file: enabled (session: granted)" in format_tools(c)


def test_tools_after_deny():
    c = make_carrot()
    assert handle_deny(c, "open_url") == "denied: open_url"
    assert "open_url: enabled (session: denied)" in format_tools(c)


def test_tools_deterministic_order():
    out = format_tools(make_carrot())
    assert out.index("read_file:") < out.index("open_url:")


def test_tools_no_mutation():
    c = _tool_cli_carrot()
    before_state = c.state.to_dict()
    before_grants = dict(c.session_grants)
    format_tools(c)
    assert c.state.to_dict() == before_state
    assert c.session_grants == before_grants


def test_tools_no_digits():
    assert re.search(r"\d", format_tools(make_carrot())) is None


def test_calls_empty():
    assert format_calls(make_carrot()) == "(no tool calls yet)"


def test_calls_allowed_ok_entry(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    req = "```tool:read_file\n" + json.dumps({"path": str(target)}) + "\n```"
    c = _scripted_tool_carrot(tmp_path, [req, "Here it is."])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    out = format_calls(c)
    assert "read_file allowed=yes ok=yes args=path=" in out


def test_calls_denied_entry(tmp_path):
    req = "```tool:read_file\n" + json.dumps({"path": "x.txt"}) + "\n```"
    c = _scripted_tool_carrot(tmp_path, [req, "I can't."])
    c.process_turn("please read the file")
    out = format_calls(c)
    assert "read_file allowed=no ok=no args=path=" in out


def test_calls_no_mutation(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    req = "```tool:read_file\n" + json.dumps({"path": str(target)}) + "\n```"
    c = _scripted_tool_carrot(tmp_path, [req, "done."])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    before_state = c.state.to_dict()
    before_grants = dict(c.session_grants)
    format_calls(c)
    assert c.state.to_dict() == before_state
    assert c.session_grants == before_grants


def test_allow_valid_tool():
    c = make_carrot()
    assert handle_allow(c, "read_file") == "granted: read_file (session)"
    assert c.session_grants["read_file"] is True


def test_allow_unknown_tool():
    c = make_carrot()
    assert handle_allow(c, "rm_rf") == "unknown tool: rm_rf"
    assert c.session_grants == {}


def test_allow_empty_arg():
    c = make_carrot()
    assert handle_allow(c, "") == "usage: /allow <tool> [always]"
    assert handle_allow(c, "   ") == "usage: /allow <tool> [always]"
    assert c.session_grants == {}


def test_allow_already_granted():
    c = make_carrot()
    assert handle_allow(c, "read_file") == "granted: read_file (session)"
    assert handle_allow(c, "read_file") == "already granted (session)"


def test_deny_valid_tool():
    c = make_carrot()
    assert handle_deny(c, "read_file") == "denied: read_file"
    assert c.session_grants["read_file"] is False


def test_deny_unknown_tool():
    c = make_carrot()
    assert handle_deny(c, "rm_rf") == "unknown tool: rm_rf"
    assert c.session_grants == {}


def test_deny_empty_arg():
    c = make_carrot()
    assert handle_deny(c, "") == "usage: /deny <tool>"
    assert handle_deny(c, "   ") == "usage: /deny <tool>"
    assert c.session_grants == {}


def test_deny_idempotent():
    c = make_carrot()
    assert handle_deny(c, "read_file") == "denied: read_file"
    assert handle_deny(c, "read_file") == "denied: read_file"


def test_help_lists_all_four_tool_commands():
    for cmd in ("/tools", "/calls", "/allow", "/deny"):
        assert cmd in HELP_TEXT


def test_help_first_lines_unchanged():
    assert "  /dbg inner state (dev only) | /ctx context (dev only) | " \
        in HELP_TEXT
    assert "  /memories list memories | /mem memory details (dev only) | " \
        in HELP_TEXT


def _perm_carrot(tmp_path):
    store = PermissionStore(tmp_path / "permissions.json")
    return make_carrot(permission_store=store)


def test_permissions_fresh_state():
    out = format_permissions(make_carrot())
    assert "read_file: level=1 enabled=yes session=none persistent=none" \
        in out
    assert "open_url: level=2 enabled=yes session=none persistent=none" \
        in out


def test_permissions_after_session_grant():
    c = make_carrot()
    handle_allow(c, "read_file")
    out = format_permissions(c)
    assert "read_file: level=1 enabled=yes session=granted " \
        "persistent=none" in out


def test_permissions_after_always_grant(tmp_path):
    c = _perm_carrot(tmp_path)
    assert handle_allow(c, "read_file always") == \
        "granted: read_file (always)"
    out = format_permissions(c)
    assert "session=granted" in out
    assert "persistent=granted" in out


def test_permissions_after_revoke(tmp_path):
    c = _perm_carrot(tmp_path)
    handle_allow(c, "read_file always")
    assert handle_revoke(c, "read_file") == "revoked: read_file"
    out = format_permissions(c)
    assert "read_file: level=1 enabled=yes session=denied " \
        "persistent=none" in out


def test_permissions_deterministic(tmp_path):
    first = _perm_carrot(tmp_path)
    handle_allow(first, "read_file always")
    second = _perm_carrot(tmp_path)
    handle_allow(second, "read_file always")
    assert format_permissions(first) == format_permissions(second)


def test_permissions_no_mutation(tmp_path):
    c = _perm_carrot(tmp_path)
    handle_allow(c, "read_file always")
    before_state = c.state.to_dict()
    before_grants = dict(c.permission_state.grants)
    path = tmp_path / "permissions.json"
    mtime = path.stat().st_mtime_ns
    format_permissions(c)
    assert c.state.to_dict() == before_state
    assert dict(c.permission_state.grants) == before_grants
    assert path.stat().st_mtime_ns == mtime


def test_permissions_digits_present():
    out = format_permissions(make_carrot())
    assert re.search(r"\d", out) is not None


def test_allow_session_mode(tmp_path):
    c = _perm_carrot(tmp_path)
    assert handle_allow(c, "read_file") == "granted: read_file (session)"
    assert c.session_grants["read_file"] is True
    assert c.permission_state.grants.get("read_file") is None


def test_allow_session_idempotent():
    c = make_carrot()
    handle_allow(c, "read_file")
    assert handle_allow(c, "read_file") == "already granted (session)"


def test_allow_always_mode(tmp_path):
    c = _perm_carrot(tmp_path)
    assert handle_allow(c, "read_file always") == \
        "granted: read_file (always)"
    assert c.session_grants["read_file"] is True
    assert c.permission_state.grants["read_file"] is True


def test_allow_always_persists(tmp_path):
    first = _perm_carrot(tmp_path)
    handle_allow(first, "read_file always")
    second = make_carrot(
        permission_store=PermissionStore(tmp_path / "permissions.json"))
    assert second.permission_state.grants["read_file"] is True


def test_allow_always_idempotent(tmp_path):
    c = _perm_carrot(tmp_path)
    handle_allow(c, "read_file always")
    assert handle_allow(c, "read_file always") == "already granted (always)"


def test_allow_upgrade_session_to_always(tmp_path):
    c = _perm_carrot(tmp_path)
    assert handle_allow(c, "read_file") == "granted: read_file (session)"
    assert handle_allow(c, "read_file always") == \
        "granted: read_file (always)"


def test_allow_unknown_tool_extended():
    c = make_carrot()
    assert handle_allow(c, "rm_rf") == "unknown tool: rm_rf"


def test_allow_extra_arg():
    c = make_carrot()
    assert handle_allow(c, "read_file forever") == \
        "usage: /allow <tool> [always]"
    assert c.session_grants == {}


def test_allow_too_many_args():
    c = make_carrot()
    assert handle_allow(c, "read_file always forever") == \
        "usage: /allow <tool> [always]"
    assert c.session_grants == {}


def test_revoke_full(tmp_path):
    c = _perm_carrot(tmp_path)
    handle_allow(c, "read_file always")
    assert handle_revoke(c, "read_file") == "revoked: read_file"
    assert c.session_grants["read_file"] is False
    assert "read_file" not in c.permission_state.grants


def test_revoke_unknown_tool():
    c = make_carrot()
    assert handle_revoke(c, "rm_rf") == "unknown tool: rm_rf"


def test_revoke_empty_arg():
    c = make_carrot()
    assert handle_revoke(c, "") == "usage: /revoke <tool>"
    assert handle_revoke(c, "   ") == "usage: /revoke <tool>"


def test_revoke_idempotent(tmp_path):
    c = _perm_carrot(tmp_path)
    assert handle_revoke(c, "read_file") == "revoked: read_file"
    assert handle_revoke(c, "read_file") == "revoked: read_file"


def test_revoke_revokes_access(tmp_path):
    c = _perm_carrot(tmp_path)
    c.session_grants["read_file"] = True
    handle_revoke(c, "read_file")
    req = ToolRequest("read_file", {"path": str(tmp_path / "a.txt")})
    assert decide(req, c.tools_config, c.session_grants,
                  c.permission_state.grants).allowed is False


def test_revoke_all_clears_everything(tmp_path):
    c = _perm_carrot(tmp_path)
    handle_allow(c, "read_file always")
    c.session_grants["open_url"] = True
    assert handle_revoke_all(c) == "revoked all"
    assert c.session_grants == {}
    assert dict(c.permission_state.grants) == {}


def test_revoke_all_idempotent(tmp_path):
    c = _perm_carrot(tmp_path)
    assert handle_revoke_all(c) == "revoked all"
    assert handle_revoke_all(c) == "revoked all"


def test_revoke_all_persists(tmp_path):
    first = _perm_carrot(tmp_path)
    handle_allow(first, "read_file always")
    handle_revoke_all(first)
    second = make_carrot(
        permission_store=PermissionStore(tmp_path / "permissions.json"))
    assert dict(second.permission_state.grants) == {}


def test_help_lists_new_permission_commands():
    for cmd in ("/permissions", "/revoke", "/revoke-all"):
        assert cmd in HELP_TEXT
    assert "/allow" in HELP_TEXT


def _away_carrot(drive_social=0.0, last_seen_delta=0.0, clock=None):
    clock = clock or FakeTime()
    c = make_carrot(start=clock)
    if drive_social:
        c.drive_store.save(DriveState(social=drive_social, novelty=0.0))
    if last_seen_delta:
        c.state.set_last_seen(clock() - last_seen_delta)
    return c, clock


def test_initiative_fresh_state():
    c, _clock = _away_carrot()
    out = format_initiative(c)
    assert out.splitlines()[0] == "decision: session-open"
    assert "session-open: ready" in out


def test_initiative_long_absence():
    c, _clock = _away_carrot(last_seen_delta=25000)
    out = format_initiative(c)
    assert out.splitlines()[0] == "decision: long-absence"


def test_initiative_cooldown_shown():
    c, clock = _away_carrot()
    c.initiative_store.record_fire("session-open", clock() - 100)
    out = format_initiative(c)
    assert "session-open: cooldown(14300)" in out
    assert out.splitlines()[0] == "decision: none"


def test_initiative_all_on_cooldown():
    c, clock = _away_carrot()
    for reason in ("long-absence", "high-social-drive", "session-open"):
        c.initiative_store.record_fire(reason, clock())
    out = format_initiative(c)
    assert out.splitlines()[0] == "decision: none"


def test_initiative_no_mutation(tmp_path):
    c, _clock = _away_carrot()
    path = tmp_path / "probe.json"
    c.drive_store.save(DriveState(social=0.1, novelty=0.0))
    import shutil

    probe_path = tmp_path / "probe-drives.json"
    shutil.copy(c.drive_store.path, probe_path)
    before_state = c.state.to_dict()
    mtime = probe_path.stat().st_mtime_ns
    format_initiative(c)
    out = format_initiative(c)
    assert c.state.to_dict() == before_state
    assert probe_path.stat().st_mtime_ns == mtime
    assert "decision: session-open" in out


def test_initiative_deterministic():
    first, _ = _away_carrot(last_seen_delta=20000)
    second, _ = _away_carrot(last_seen_delta=20000)
    assert format_initiative(first) == format_initiative(second)


def test_initiative_digits_allowed():
    c, clock = _away_carrot()
    c.initiative_store.record_fire("session-open", clock() - 100)
    assert re.search(r"\d", format_initiative(c)) is not None


def test_parse_away_no_flag():
    assert _parse_away_args([]) == (False, 3)


def test_parse_away_alone():
    assert _parse_away_args(["--away"]) == (True, 3)


def test_parse_away_count():
    assert _parse_away_args(["--away", "5"]) == (True, 5)


def test_parse_away_zero_clamped():
    assert _parse_away_args(["--away", "0"]) == (True, 1)


def test_parse_away_large_clamped():
    assert _parse_away_args(["--away", "99"]) == (True, 10)


def test_parse_away_unparseable():
    assert _parse_away_args(["--away", "xyz"]) == (True, 3)


def test_away_empty_eligibility(capsys):
    c, _clock = _away_carrot()
    _run_away_mode(c, 3)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert dict(c.initiative_store.load().per_reason_last_fire) == \
        {"session-open": c._now()}


def test_away_breaks_on_second_none(capsys):
    c, _clock = _away_carrot()
    _run_away_mode(c, 3)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert dict(c.initiative_store.load().per_reason_last_fire) == \
        {"session-open": c._now()}


def test_away_two_messages_then_none(capsys):
    c, _clock = _away_carrot(drive_social=0.9, last_seen_delta=25000)
    c.initiative_store.record_fire("session-open", c._now())
    _run_away_mode(c, 5)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
    fired = dict(c.initiative_store.load().per_reason_last_fire)
    assert set(fired) == {"long-absence", "high-social-drive",
                          "session-open"}


def test_away_bounded_at_max_n(capsys):
    c, clock = _away_carrot(drive_social=0.9, last_seen_delta=25000)
    _run_away_mode(c, 2)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2


def test_away_no_process_turn_or_buffer_write(monkeypatch):
    c, _clock = _away_carrot()
    before_buffer = list(c.buffer.recent())
    before_state = c.state.to_dict()
    calls = []
    monkeypatch.setattr(c, "process_turn",
                        lambda *a, **k: calls.append((a, k)))
    _run_away_mode(c, 2)
    assert calls == []
    assert list(c.buffer.recent()) == before_buffer
    assert c.state.to_dict() == before_state


def test_away_llm_generate_counts():
    c, _clock = _away_carrot()
    counts = {"n": 0}
    orig = c.llm.generate

    def counting_generate(*, system, history, user):
        counts["n"] += 1
        return orig(system=system, history=history, user=user)

    c.llm.generate = counting_generate
    _run_away_mode(c, 3)
    assert counts["n"] == 1


def test_away_no_drive_rise_saved(tmp_path):
    import shutil

    c, _clock = _away_carrot(drive_social=0.9)
    probe_path = tmp_path / "probe-drives.json"
    shutil.copy(c.drive_store.path, probe_path)
    mtime = probe_path.stat().st_mtime_ns
    before = probe_path.read_bytes()
    _run_away_mode(c, 1)
    assert probe_path.read_bytes() == before
    assert probe_path.stat().st_mtime_ns == mtime


def test_away_static_strings_and_leakage():
    from carrot.cli import _INITIATIVE_REASON_LINES

    banned = ["granted", "grant", "denied", "deny", "consent", "policy",
              "revoke", "permission", "permissions"]
    for reason in ("long-absence", "high-social-drive", "session-open"):
        line = _INITIATIVE_REASON_LINES[reason]
        assert re.search(r"\d", line) is None
        for word in banned:
            assert re.search(rf"\b{word}\b", line.lower()) is None, word
    c, _clock = _away_carrot()
    prompt = c.__class__  # placeholder never used; see below
    del prompt
    import carrot.cli as _cli

    line = _cli._INITIATIVE_REASON_LINES["long-absence"]
    assert "away for a while" in line


def test_away_deterministic(capsys):
    def _run_once():
        carrot, _ = _away_carrot(drive_social=0.9,
                                 last_seen_delta=25000)
        _run_away_mode(carrot, 3)
        return (capsys.readouterr().out,
                dict(carrot.initiative_store.load().per_reason_last_fire))

    first_out, first_fires = _run_once()
    second_out, second_fires = _run_once()
    assert first_out == second_out
    assert first_fires.keys() == second_fires.keys()


def test_help_lists_initiative():
    assert "/initiative" in HELP_TEXT


def test_main_no_flag_enters_repl(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()
                                                           ).throw(EOFError))
    monkeypatch.setattr("carrot.cli.build_carrot", lambda: make_carrot())
    from carrot.cli import main

    main([])
    out = capsys.readouterr().out
    assert "Carrot (slice 1)" in out


def test_main_away_flag_runs_away(monkeypatch, capsys):
    monkeypatch.setattr("carrot.cli.build_carrot", lambda: make_carrot())
    from carrot.cli import main

    main(["--away"])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "Carrot (slice 1)" not in capsys.readouterr().out


# --- Slice 10 Phase 4: /ctx relevance line tests ---


def _relevance_carrot(observations, *texts):
    c = make_carrot()
    c.environment_probe = MockEnvironmentProbe(observations)
    _seed(c, *texts)
    return c


def test_ctx_relevance_no_after_neutral_turn():
    # Fresh instance, empty probe: boost cannot fire. (A turn is
    # required: pre-turn /ctx returns the frozen NO_CONTEXT_YET
    # sentinel, pinned by test_ctx_before_any_turn.)
    c = make_carrot()
    c.process_turn("hello there")
    assert c.last_boost_effect is False
    assert "relevance: no" in handle_ctx(c)


def test_ctx_relevance_yes_when_boost_changed_topk():
    c = _relevance_carrot([RawObservation("chrome.exe", False)],
                          "i love pizza", "my browser tabs are many")
    c.process_turn("hello there")
    assert c.last_boost_effect is True
    assert "relevance: yes" in handle_ctx(c)


def test_ctx_relevance_keeps_prior_content():
    c = _relevance_carrot([RawObservation("chrome.exe", False)],
                          "i love pizza", "my browser tabs are many")
    c.process_turn("hello there")
    out = handle_ctx(c)
    assert "pending_question: False" in out
    assert "playful_frame: False" in out
    assert "recent_harshness: False" in out
    assert "turns: 0" in out
    assert "relevance: yes" in out


def test_ctx_relevance_is_last_line():
    c = _relevance_carrot([RawObservation("chrome.exe", False)],
                          "i love pizza", "my browser tabs are many")
    c.process_turn("hello there")
    assert handle_ctx(c).splitlines()[-1] == "relevance: yes"
    plain = make_carrot()
    plain.process_turn("hello there")
    assert handle_ctx(plain).splitlines()[-1] == "relevance: no"


def test_ctx_relevance_no_mutation(tmp_path):
    c = _relevance_carrot([RawObservation("chrome.exe", False)],
                          "i love pizza", "my browser tabs are many")
    c.process_turn("hello there")
    before_state = c.state.to_dict()
    mtime_before = os.stat(c.memory_store.path).st_mtime_ns
    handle_ctx(c)
    format_ctx(c.last_context, boost_effect=c.last_boost_effect)
    assert c.state.to_dict() == before_state
    assert os.stat(c.memory_store.path).st_mtime_ns == mtime_before


def test_ctx_relevance_line_has_no_digits():
    c = _relevance_carrot([RawObservation("chrome.exe", False)],
                          "i love pizza", "my browser tabs are many")
    c.process_turn("hello there")
    assert handle_ctx(c).splitlines()[-1] in ("relevance: yes",
                                              "relevance: no")


# --- Slice 11 Phase 6: /relationship tests ---


def _rel_carrot(tmp_path, trust, closeness, name="rel.json"):
    store = RelationshipStore(tmp_path / name)
    store.save(RelationshipState(trust=trust, closeness=closeness,
                                 updated_ts={}))
    return make_carrot(relationship_store=store)


def test_relationship_fresh_state():
    assert format_relationship(make_carrot()) == \
        "trust: medium\ncloseness: medium"


def test_relationship_band_low(tmp_path):
    out = format_relationship(_rel_carrot(tmp_path, 0.2, 0.3))
    assert out == "trust: low\ncloseness: low"


def test_relationship_band_high(tmp_path):
    out = format_relationship(_rel_carrot(tmp_path, 0.8, 0.9))
    assert out == "trust: high\ncloseness: high"


def test_relationship_boundary_low_medium(tmp_path):
    assert format_relationship(
        _rel_carrot(tmp_path, 0.4, 0.5, "a.json")).splitlines()[0] == \
        "trust: medium"
    assert format_relationship(
        _rel_carrot(tmp_path, 0.399, 0.5, "b.json")).splitlines()[0] == \
        "trust: low"


def test_relationship_boundary_medium_high(tmp_path):
    assert format_relationship(
        _rel_carrot(tmp_path, 0.5, 0.7, "a.json")).splitlines()[1] == \
        "closeness: high"
    assert format_relationship(
        _rel_carrot(tmp_path, 0.5, 0.699, "b.json")).splitlines()[1] == \
        "closeness: medium"


def test_relationship_no_digits(tmp_path):
    for trust, closeness in ((0.5, 0.5), (0.2, 0.3), (0.8, 0.9),
                             (0.4, 0.7), (0.0, 1.0)):
        out = format_relationship(
            _rel_carrot(tmp_path, trust, closeness,
                        f"r-{trust}-{closeness}.json"))
        assert re.search(r"\d", out) is None, (trust, closeness)


def test_relationship_no_floats(tmp_path):
    out = format_relationship(_rel_carrot(tmp_path, 0.83, 0.91))
    assert "." not in out


def test_relationship_line_order(tmp_path):
    out = format_relationship(_rel_carrot(tmp_path, 0.2, 0.9))
    assert out.index("trust:") < out.index("closeness:")


def test_relationship_no_mutation(tmp_path):
    path = tmp_path / "rel.json"
    store = RelationshipStore(path)
    store.save(RelationshipState(trust=0.6, closeness=0.6,
                                 updated_ts={}))
    c = make_carrot(relationship_store=store)
    c.process_turn("You're cute, thanks!")
    before_state = c.state.to_dict()
    mtime_before = os.stat(path).st_mtime_ns
    format_relationship(c)
    assert c.state.to_dict() == before_state
    assert os.stat(path).st_mtime_ns == mtime_before


def test_relationship_deterministic(tmp_path):
    first = _rel_carrot(tmp_path, 0.62, 0.38, "a.json")
    second = _rel_carrot(tmp_path, 0.62, 0.38, "b.json")
    assert format_relationship(first) == format_relationship(second)
    assert format_relationship(first) == format_relationship(first)


def test_dbg_includes_relationship_line():
    out = format_dbg(make_carrot())
    assert "relationship: trust=" in out
    assert "closeness=" in out


def test_dbg_relationship_shows_floats():
    out = format_dbg(make_carrot())
    assert re.search(
        r"relationship: trust=\d+\.\d+ closeness=\d+\.\d+", out) is not None


def test_dbg_existing_content_intact():
    out = format_dbg(make_carrot())
    assert "mood=" in out
    assert "behavior=" in out
    assert "social=" in out
    assert "novelty=" in out
    assert "environment:" in out


def test_help_lists_relationship():
    assert "/relationship" in HELP_TEXT


# --- Slice 12 Phase 3: /avatar dispatch tests ---


def _run_main_scripted(monkeypatch, lines, factory=None):
    it = iter(lines)

    def fake_input(*a, **k):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("carrot.cli.build_carrot",
                        factory or (lambda: make_carrot()))
    from carrot.cli import main

    main([])


def test_avatar_dispatch_calls_launcher_once(monkeypatch):
    from tests.conftest import ROOT as _ROOT

    calls = []
    monkeypatch.setattr("carrot.cli.launch_avatar",
                        lambda p: calls.append(p) or "sentinel-opened")

    _run_main_scripted(monkeypatch, ["/avatar"])
    assert len(calls) == 1
    assert calls[0] == _ROOT / "state" / "state.json"


def test_avatar_dispatch_prints_return(monkeypatch, capsys):
    monkeypatch.setattr("carrot.cli.launch_avatar",
                        lambda p: "sentinel-opened")
    _run_main_scripted(monkeypatch, ["/avatar"])
    assert "sentinel-opened" in capsys.readouterr().out


def test_avatar_dispatch_no_window_in_tests(monkeypatch):
    monkeypatch.setattr("carrot.cli.launch_avatar", lambda p: "ok")
    _run_main_scripted(monkeypatch, ["/avatar"])
    assert "tkinter" not in sys.modules


def test_avatar_dispatch_degrade_message(monkeypatch, capsys):
    monkeypatch.setattr("carrot.cli.launch_avatar",
                        lambda p: AVATAR_UNAVAILABLE_MSG)
    _run_main_scripted(monkeypatch, ["/avatar"])
    assert AVATAR_UNAVAILABLE_MSG in capsys.readouterr().out


def test_avatar_dispatch_no_state_change(monkeypatch, tmp_path):
    c = make_carrot(memory_store=MemoryStore(str(tmp_path / "m.json")))
    c.process_turn("hello there")
    before_state = c.state.to_dict()
    mtime_before = os.stat(c.memory_store.path).st_mtime_ns
    monkeypatch.setattr("carrot.cli.launch_avatar", lambda p: "ok")
    _run_main_scripted(monkeypatch, ["/avatar"], factory=lambda: c)
    assert c.state.to_dict() == before_state
    assert os.stat(c.memory_store.path).st_mtime_ns == mtime_before


def test_commands_still_dispatch_after_avatar(monkeypatch, capsys):
    monkeypatch.setattr("carrot.cli.launch_avatar", lambda p: "ok")
    _run_main_scripted(monkeypatch, ["/dbg", "/ctx", "/relationship",
                                     "/avatar"])
    out = capsys.readouterr().out
    assert "mood=" in out
    assert "trust: medium" in out
    assert "ok" in out


def test_help_lists_avatar():
    assert "/avatar" in HELP_TEXT


# --- Slice 13 Phase 2: /voice dispatch tests ---


def _voice_carrot(tmp_path):
    c = make_carrot()
    c.store = StateStore(str(tmp_path / "state.json"))
    return c


def test_voice_dispatch_empty(monkeypatch, capsys, tmp_path):
    from carrot.voice import MockSpeechRecognizer, MockSpeechSynthesizer

    c = _voice_carrot(tmp_path)
    monkeypatch.setattr("carrot.cli.MockSpeechRecognizer",
                        lambda: MockSpeechRecognizer([]))
    monkeypatch.setattr("carrot.cli.MockSpeechSynthesizer",
                        MockSpeechSynthesizer)
    before_buffer = list(c.buffer.recent())
    _run_main_scripted(monkeypatch, ["/voice"], factory=lambda: c)
    out = capsys.readouterr().out
    assert VOICE_EMPTY_MSG in out
    assert list(c.buffer.recent()) == before_buffer


def test_voice_dispatch_non_empty(monkeypatch, capsys, tmp_path):
    from carrot.voice import MockSpeechRecognizer, MockSpeechSynthesizer

    c = _voice_carrot(tmp_path)
    monkeypatch.setattr("carrot.cli.MockSpeechRecognizer",
                        lambda: MockSpeechRecognizer(["hello there"]))
    monkeypatch.setattr("carrot.cli.MockSpeechSynthesizer",
                        MockSpeechSynthesizer)
    _run_main_scripted(monkeypatch, ["/voice"], factory=lambda: c)
    out = capsys.readouterr().out
    assert "heard: hello there" in out
    assert "carrot: " in out
    texts = [t.text for t in c.buffer.recent()]
    assert "hello there" in texts


def test_voice_dispatch_saves_state(monkeypatch, tmp_path):
    from carrot.voice import MockSpeechRecognizer, MockSpeechSynthesizer

    path = tmp_path / "state.json"
    c = make_carrot()
    c.store = StateStore(str(path))
    assert not path.exists()
    monkeypatch.setattr("carrot.cli.MockSpeechRecognizer",
                        lambda: MockSpeechRecognizer(["hello"]))
    monkeypatch.setattr("carrot.cli.MockSpeechSynthesizer",
                        MockSpeechSynthesizer)
    _run_main_scripted(monkeypatch, ["/voice"], factory=lambda: c)
    assert path.exists()


def test_voice_dispatch_no_audio_device(monkeypatch, tmp_path):
    from carrot.voice import MockSpeechRecognizer, MockSpeechSynthesizer

    c = _voice_carrot(tmp_path)
    monkeypatch.setattr("carrot.cli.MockSpeechRecognizer",
                        MockSpeechRecognizer)
    monkeypatch.setattr("carrot.cli.MockSpeechSynthesizer",
                        MockSpeechSynthesizer)
    _run_main_scripted(monkeypatch, ["/voice"], factory=lambda: c)
    assert "tkinter" not in sys.modules


def test_voice_dispatch_empty_touches_no_llm(monkeypatch, tmp_path):
    from carrot.voice import MockSpeechRecognizer, MockSpeechSynthesizer

    c = _voice_carrot(tmp_path)
    monkeypatch.setattr("carrot.cli.MockSpeechRecognizer",
                        lambda: MockSpeechRecognizer([]))
    monkeypatch.setattr("carrot.cli.MockSpeechSynthesizer",
                        MockSpeechSynthesizer)
    before = c.llm._counter
    _run_main_scripted(monkeypatch, ["/voice"], factory=lambda: c)
    assert c.llm._counter == before


def test_help_lists_voice():
    assert "/voice" in HELP_TEXT


# --- Slice 14 Phase 2: /drain + --drain tests ---


def _drain_carrot(tmp_path, name="state.json"):
    c = make_carrot()
    c.store = StateStore(str(tmp_path / name))
    return c


def _queue_msg(base, seq, text):
    inbox = base / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / f"{seq:06d}-aabbccdd.msg").write_text(text, encoding="utf-8")


def test_drain_dispatch_empty(monkeypatch, capsys, tmp_path):
    base = tmp_path / "ch"
    monkeypatch.setattr("carrot.cli.FileChannel", lambda p: FileChannel(base))
    c = _drain_carrot(tmp_path)
    _run_main_scripted(monkeypatch, ["/drain"], factory=lambda: c)
    assert CHANNEL_EMPTY_MSG in capsys.readouterr().out


def test_drain_dispatch_queued(monkeypatch, capsys, tmp_path):
    base = tmp_path / "ch"
    _queue_msg(base, 1, "hello there")
    monkeypatch.setattr("carrot.cli.FileChannel", lambda p: FileChannel(base))
    c = _drain_carrot(tmp_path)
    _run_main_scripted(monkeypatch, ["/drain"], factory=lambda: c)
    out = capsys.readouterr().out
    assert "drained: 1" in out
    assert len(list((base / "outbox").glob("*.msg"))) == 1
    assert len(list((base / "inbox" / "done").glob("*.msg"))) == 1


def test_drain_dispatch_no_window_or_audio(monkeypatch, tmp_path):
    base = tmp_path / "ch"
    monkeypatch.setattr("carrot.cli.FileChannel", lambda p: FileChannel(base))
    _run_main_scripted(monkeypatch, ["/drain"])
    assert "tkinter" not in sys.modules


def test_drain_dispatch_saves_state(monkeypatch, tmp_path):
    base = tmp_path / "ch"
    _queue_msg(base, 1, "hello")
    monkeypatch.setattr("carrot.cli.FileChannel", lambda p: FileChannel(base))
    path = tmp_path / "state.json"
    c = _drain_carrot(tmp_path)
    assert not path.exists()
    _run_main_scripted(monkeypatch, ["/drain"], factory=lambda: c)
    assert path.exists()


def test_parse_drain_args():
    from carrot.cli import _parse_drain_args

    assert _parse_drain_args([]) == (False, None)
    assert _parse_drain_args(["--away"]) == (False, None)
    assert _parse_drain_args(["--drain", "some/dir"]) == (True, "some/dir")
    assert _parse_drain_args(["--drain"]) == (True, None)


def test_away_and_drain_mutually_exclusive(monkeypatch, capsys, tmp_path):
    called = []
    monkeypatch.setattr("carrot.cli.build_carrot",
                        lambda: called.append(1) or make_carrot())
    from carrot.cli import main

    main(["--away", "--drain", str(tmp_path)])
    out = capsys.readouterr().out
    assert "usage: --away [N] | --drain <dir>" in out
    assert called == []


def test_drain_flag_missing_arg(monkeypatch, capsys):
    monkeypatch.setattr("carrot.cli.build_carrot", lambda: make_carrot())
    from carrot.cli import main

    main(["--drain"])
    assert "usage: --drain <dir>" in capsys.readouterr().out


def test_drain_flag_real_inbox(monkeypatch, capsys, tmp_path):
    base = tmp_path / "ch"
    _queue_msg(base, 1, "hello there")
    c = _drain_carrot(tmp_path)
    monkeypatch.setattr("carrot.cli.build_carrot", lambda: c)
    from carrot.cli import main

    main(["--drain", str(base)])
    out = capsys.readouterr().out
    assert "drained: 1" in out
    assert len(list((base / "outbox").glob("*.msg"))) == 1


def test_help_lists_drain():
    assert "/drain" in HELP_TEXT


# --- Slice 15 Phase 2: /social + --social tests ---


def _social_cfg(inbox_dir):
    from carrot.social import SocialConfig

    return SocialConfig(platforms={"instagram": {"handle": "",
                                                "enabled": True,
                                                "inbox_dir": inbox_dir}})


def test_social_dispatch_empty_config(monkeypatch, capsys):
    class _MissingCfg:
        @classmethod
        def load(cls, path):
            raise FileNotFoundError("nope")

    monkeypatch.setattr("carrot.cli.SocialConfig", _MissingCfg)
    _run_main_scripted(monkeypatch, ["/social"])
    assert "(no social config)" in capsys.readouterr().out


def test_social_dispatch_valid_empty(monkeypatch, capsys):
    _run_main_scripted(monkeypatch, ["/social"])
    assert "instagram: enabled pending=0 sent=0" in capsys.readouterr().out


def test_social_dispatch_counts(monkeypatch, capsys, tmp_path):
    from carrot import cli as _cli

    base = tmp_path / "soc" / "ig"
    (base / "inbox").mkdir(parents=True)
    (base / "outbox").mkdir(parents=True)
    (base / "inbox" / "000001-aa.msg").write_text("a", encoding="utf-8")
    (base / "inbox" / "000002-bb.msg").write_text("b", encoding="utf-8")
    (base / "outbox" / "000001-cc.msg").write_text("c", encoding="utf-8")
    monkeypatch.setattr(_cli, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "social.json").write_text(
        '{"platforms": {"instagram": {"handle": "", "enabled": true, '
        '"inbox_dir": "soc/ig"}}}', encoding="utf-8")
    _run_main_scripted(monkeypatch, ["/social"])
    assert "instagram: enabled pending=2 sent=1" in capsys.readouterr().out


def test_social_dispatch_no_mutation(monkeypatch, tmp_path):
    path = tmp_path / "state.json"
    c = make_carrot()
    c.store = StateStore(str(path))
    c.store.save(c.state)
    before = (os.stat(path).st_mtime_ns, path.read_bytes())
    _run_main_scripted(monkeypatch, ["/social"], factory=lambda: c)
    assert (os.stat(path).st_mtime_ns, path.read_bytes()) == before


def test_parse_social_args():
    from carrot.cli import _parse_social_args

    assert _parse_social_args([]) == (False, None)
    assert _parse_social_args(["--social", "instagram"]) == \
        (True, "instagram")
    assert _parse_social_args(["--social"]) == (True, None)


def test_social_flag_missing_arg(monkeypatch, capsys):
    monkeypatch.setattr("carrot.cli.build_carrot", lambda: make_carrot())
    from carrot.cli import main

    main(["--social"])
    assert "usage: --social <platform>" in capsys.readouterr().out


def test_social_unknown_platform(monkeypatch, capsys):
    monkeypatch.setattr("carrot.cli.build_carrot", lambda: make_carrot())
    from carrot.cli import main

    main(["--social", "twitter"])
    assert "unknown platform: twitter" in capsys.readouterr().out


def test_social_flag_valid_drain(monkeypatch, capsys, tmp_path):
    base = tmp_path / "soc" / "ig"
    (base / "inbox").mkdir(parents=True)
    (base / "inbox" / "000001-aa.msg").write_text("hello there",
                                                  encoding="utf-8")
    cfg = _social_cfg(str(base))

    class _StubCfg:
        @classmethod
        def load(cls, path):
            return cfg

    monkeypatch.setattr("carrot.cli.SocialConfig", _StubCfg)
    c = make_carrot()
    c.store = StateStore(str(tmp_path / "state.json"))
    monkeypatch.setattr("carrot.cli.build_carrot", lambda: c)
    from carrot.cli import main

    main(["--social", "instagram"])
    out = capsys.readouterr().out
    assert "drained: 1" in out
    assert len(list((base / "outbox").glob("*.msg"))) == 1


def test_social_flag_empty_inbox(monkeypatch, capsys, tmp_path):
    base = tmp_path / "soc" / "ig"
    (base / "inbox").mkdir(parents=True)
    cfg = _social_cfg(str(base))

    class _StubCfg:
        @classmethod
        def load(cls, path):
            return cfg

    monkeypatch.setattr("carrot.cli.SocialConfig", _StubCfg)
    c = make_carrot()
    c.store = StateStore(str(tmp_path / "state.json"))
    monkeypatch.setattr("carrot.cli.build_carrot", lambda: c)
    from carrot.cli import main

    main(["--social", "instagram"])
    assert CHANNEL_EMPTY_MSG in capsys.readouterr().out


def test_social_drain_mutually_exclusive(monkeypatch, capsys, tmp_path):
    called = []
    monkeypatch.setattr("carrot.cli.build_carrot",
                        lambda: called.append(1) or make_carrot())
    from carrot.cli import main

    main(["--social", "instagram", "--drain", str(tmp_path)])
    out = capsys.readouterr().out
    assert "usage: --away [N] | --drain <dir> | --social <platform>" in out
    assert called == []


def test_social_away_mutually_exclusive(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("carrot.cli.build_carrot",
                        lambda: called.append(1) or make_carrot())
    from carrot.cli import main

    main(["--away", "--social", "instagram"])
    out = capsys.readouterr().out
    assert "usage: --away [N] | --drain <dir> | --social <platform>" in out
    assert called == []


def test_help_lists_social():
    assert "/social" in HELP_TEXT


# --- Slice 16 Phase 3: --companion + cross-companion tests ---


def test_parse_companion_args():
    from carrot.cli import _parse_companion_args

    assert _parse_companion_args([]) == (False, None)
    assert _parse_companion_args(["--companion", "ada"]) == (True, "ada")
    assert _parse_companion_args(["--companion"]) == (True, None)


def test_companion_flag_missing_arg(monkeypatch, capsys):
    from carrot.cli import main

    main(["--companion"])
    assert "usage: --companion <name>" in capsys.readouterr().out


def test_companion_flag_invalid_name(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("carrot.cli.build_carrot",
                        lambda **kw: called.append(kw))
    from carrot.cli import main

    main(["--companion", "Ada"])
    assert "unknown companion: Ada" in capsys.readouterr().out
    assert called == []


def test_companion_flag_unknown_no_config(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("carrot.cli.build_carrot",
                        lambda **kw: called.append(kw))
    from carrot.cli import main

    main(["--companion", "ada"])
    assert "unknown companion: ada" in capsys.readouterr().out
    assert called == []


def _run_argv(monkeypatch, argv):
    monkeypatch.setattr(
        "builtins.input",
        lambda *a, **k: (_ for _ in ()).throw(EOFError))
    from carrot.cli import main

    main(argv)


def test_companion_flag_builds_named(monkeypatch, tmp_path):
    (tmp_path / "config" / "companions").mkdir(parents=True)
    (tmp_path / "config" / "companions" / "ada.json").write_text(
        '{"id": "ada-v1"}', encoding="utf-8")
    monkeypatch.setattr("carrot.cli.ROOT", tmp_path)
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return make_carrot()

    monkeypatch.setattr("carrot.cli.build_carrot", fake_build)
    _run_argv(monkeypatch, ["--companion", "ada"])
    assert calls == [{"companion": "ada"}]


def test_companion_away_mutually_exclusive(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("carrot.cli.build_carrot",
                        lambda **kw: called.append(kw))
    from carrot.cli import main

    main(["--companion", "ada", "--away"])
    out = capsys.readouterr().out
    assert "usage: --away [N] | --drain <dir> | --social <platform> " \
        "| --companion <name>" in out
    assert called == []


def test_companion_drain_mutually_exclusive(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("carrot.cli.build_carrot",
                        lambda **kw: called.append(kw))
    from carrot.cli import main

    main(["--companion", "ada", "--drain", "x"])
    out = capsys.readouterr().out
    assert "usage: --away [N] | --drain <dir> | --social <platform> " \
        "| --companion <name>" in out
    assert called == []


def test_companion_social_mutually_exclusive(monkeypatch, capsys):
    called = []
    monkeypatch.setattr("carrot.cli.build_carrot",
                        lambda **kw: called.append(kw))
    from carrot.cli import main

    main(["--companion", "ada", "--social", "instagram"])
    out = capsys.readouterr().out
    assert "usage: --away [N] | --drain <dir> | --social <platform> " \
        "| --companion <name>" in out
    assert called == []


def _ada_repl(tmp_path, monkeypatch, lines):
    import json

    from tests.test_scenarios import make_companion

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "social.json").write_text(
        json.dumps({"platforms": {"instagram": {"handle": "",
                                                "enabled": True,
                                                "inbox_dir":
                                                "state/social/instagram"}}}),
        encoding="utf-8")
    ada = make_companion("ada", tmp_path)
    ada.store = StateStore(str(tmp_path / "state-ada.json"))
    assert ada.companion == "ada"
    monkeypatch.setattr("carrot.cli.ROOT", tmp_path)
    _run_main_scripted(monkeypatch, lines, factory=lambda: ada)
    return ada


def test_social_uses_active_companion_dir(monkeypatch, capsys, tmp_path):
    inbox = tmp_path / "state" / "ada" / "social" / "instagram" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "000001-aa.msg").write_text("hi", encoding="utf-8")
    _ada_repl(tmp_path, monkeypatch, ["/social"])
    assert "pending=1" in capsys.readouterr().out


def test_drain_uses_active_companion_channel(monkeypatch, capsys, tmp_path):
    inbox = tmp_path / "state" / "ada" / "channel" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "000001-aa.msg").write_text("hello", encoding="utf-8")
    _ada_repl(tmp_path, monkeypatch, ["/drain"])
    out = capsys.readouterr().out
    assert "drained: 1" in out
    assert len(list((tmp_path / "state" / "ada" / "channel" / "outbox")
                    .glob("*.msg"))) == 1
    assert not (tmp_path / "state" / "channel").exists()


def test_avatar_uses_active_companion_state(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("carrot.cli.launch_avatar",
                        lambda p: calls.append(p) or "ok")
    _ada_repl(tmp_path, monkeypatch, ["/avatar"])
    assert calls == [tmp_path / "state" / "ada" / "state.json"]


def test_build_default_companion_attr(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    from carrot.cli import build_carrot

    assert build_carrot().companion == "carrot"


def test_build_named_companion_attr(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    (tmp_path / "config" / "companions").mkdir(exist_ok=True)
    (tmp_path / "config" / "companions" / "ada.json").write_text(
        '{"id": "ada-v1"}', encoding="utf-8")
    from carrot.cli import build_carrot

    assert build_carrot(companion="ada").companion == "ada"


def test_make_carrot_companion_default():
    assert make_carrot().companion == "carrot"


def test_help_lists_companions():
    assert "/companions" in HELP_TEXT
