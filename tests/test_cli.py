"""Phase-5 tests: /ctx handler, last_context retention, help text."""
import re

from carrot.cli import (
    HELP_TEXT,
    NO_CONTEXT_YET,
    NO_MEMORIES_YET,
    build_carrot,
    format_ctx,
    format_dbg,
    format_drives,
    format_env,
    format_mem_debug,
    format_memories,
    handle_ctx,
    handle_forget,
    handle_remember,
)
from carrot.context import derive_context
from carrot.drives import DriveState, DriveStore, DrivesConfig
from carrot.environment import MockEnvironmentProbe, RawObservation
from carrot.memory import MAX_MEMORIES, Memory, MemoryStore
from tests.conftest import ROOT
from tests.test_scenarios import make_carrot


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
                 "decay.json"):
        shutil.copy(ROOT / "config" / name, tmp_path / "config" / name)
    monkeypatch.setattr("carrot.cli.ROOT", tmp_path)
    return tmp_path


def test_launch_increments_session_count(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    assert build_carrot().state.session_count == 1


def test_launch_again_increments(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    build_carrot()
    assert build_carrot().state.session_count == 2


def test_first_seen_set_once(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    assert build_carrot(time_fn=lambda: 1000.0).state.first_seen_ts == 1000.0
    assert build_carrot(time_fn=lambda: 2000.0).state.first_seen_ts == 1000.0


def test_last_seen_untouched_by_launch(tmp_path, monkeypatch):
    _staged_root(tmp_path, monkeypatch)
    first = build_carrot(time_fn=lambda: 1000.0)
    first.process_turn("hello there")
    first.store.save(first.state)
    seen = first.state.last_seen
    relaunched = build_carrot(time_fn=lambda: 9999.0)
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
