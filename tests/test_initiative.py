"""Slice 9 Phase-1 tests: initiative decision function (pure logic)."""
import ast
import dataclasses
import json
import sys

import pytest

from carrot.drives import DriveState
from carrot.environment import EMPTY_SNAPSHOT, EnvironmentSnapshot
from carrot.initiative import (
    ACTIVE_REASONS,
    REASONS,
    InitiateDecision,
    InitiativeConfig,
    InitiativeState,
    InitiativeStore,
    should_initiate,
)
from tests.conftest import ROOT

NOW = 1_000_000.0


def _cfg():
    return InitiativeConfig.load(str(ROOT / "config" / "initiative.json"))


def _decide(drives=None, elapsed=0.0, fired=None, now=NOW, cfg=None):
    return should_initiate(
        drives if drives is not None else DriveState(),
        elapsed,
        EMPTY_SNAPSHOT,
        fired if fired is not None else {},
        now,
        cfg if cfg is not None else _cfg(),
    )


# ---- config loader ----

def test_config_loads_valid():
    cfg = _cfg()
    assert cfg.long_absence_threshold_s == 21600
    assert cfg.high_social_drive_threshold == 0.6
    assert cfg.cooldowns["long-absence"] == 43200.0
    assert cfg.cooldowns["high-social-drive"] == 14400.0
    assert cfg.cooldowns["session-open"] == 14400.0
    assert cfg.cooldowns["user-returned"] == 7200.0


def test_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        InitiativeConfig.load(str(tmp_path / "nope.json"))


def test_config_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        InitiativeConfig.load(str(path))


def test_config_negative_absence_threshold(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"long_absence_threshold_s": -5}),
                    encoding="utf-8")
    with pytest.raises(ValueError):
        InitiativeConfig.load(str(path))


def test_config_social_threshold_out_of_range(tmp_path):
    for bad in (-0.1, 1.5):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"high_social_drive_threshold": bad}),
                        encoding="utf-8")
        with pytest.raises(ValueError):
            InitiativeConfig.load(str(path))


def test_config_missing_cooldowns_default(tmp_path):
    path = tmp_path / "nocd.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    cfg = InitiativeConfig.load(str(path))
    assert cfg.cooldowns["long-absence"] == 43200.0
    assert cfg.cooldowns["session-open"] == 14400.0


def test_config_partial_cooldown_keys(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"cooldowns": {"session-open": 60.0}}),
                    encoding="utf-8")
    cfg = InitiativeConfig.load(str(path))
    assert cfg.cooldowns["session-open"] == 60.0
    assert cfg.cooldowns["long-absence"] == 43200.0


def test_config_frozen(tmp_path):
    cfg = _cfg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.pull_threshold = 0.9
    with pytest.raises(TypeError):
        cfg.cooldowns["session-open"] = 1.0


def test_reasons_shape():
    assert "user-returned" in REASONS
    assert "user-returned" not in ACTIVE_REASONS
    assert tuple(ACTIVE_REASONS) == ("long-absence", "high-social-drive",
                                     "session-open")


# ---- decision matrix ----

def test_long_absence_fires_at_threshold():
    assert _decide(elapsed=21600) == InitiateDecision("long-absence")


def test_below_absence_no_drive_falls_back():
    assert _decide(elapsed=21599,
                   drives=DriveState(social=0.0)) == \
        InitiateDecision("session-open")


def test_high_social_drive_fires_at_threshold():
    assert _decide(elapsed=0,
                   drives=DriveState(social=0.6)) == \
        InitiateDecision("high-social-drive")


def test_priority_long_absence_first():
    assert _decide(elapsed=99999,
                   drives=DriveState(social=0.9)) == \
        InitiateDecision("long-absence")


def test_fallback_session_open():
    assert _decide(elapsed=0,
                   drives=DriveState(social=0.0)) == \
        InitiateDecision("session-open")


def test_cooldown_falls_through_to_next():
    assert _decide(elapsed=99999,
                   drives=DriveState(social=0.9),
                   fired={"long-absence": NOW - 100}) == \
        InitiateDecision("high-social-drive")


def test_all_cooldowns_active_returns_none():
    fired = {"long-absence": NOW - 100, "high-social-drive": NOW - 100,
             "session-open": NOW - 100}
    assert _decide(elapsed=99999, drives=DriveState(social=0.9),
                   fired=fired) is None


def test_cooldown_boundary_inclusive():
    assert _decide(elapsed=0, drives=DriveState(social=0.0),
                   fired={"session-open": NOW - 14400.0}) == \
        InitiateDecision("session-open")


def test_cooldown_one_second_short():
    assert _decide(elapsed=0, drives=DriveState(social=0.0),
                   fired={"session-open": NOW - 14399.0}) is None


# ---- purity / seam ----

def test_relationship_none_identical_to_default():
    first = should_initiate(DriveState(social=0.9), 99999, EMPTY_SNAPSHOT,
                            {}, NOW, _cfg(), relationship=None)
    second = should_initiate(DriveState(social=0.9), 99999, EMPTY_SNAPSHOT,
                             {}, NOW, _cfg())
    assert first == second == InitiateDecision("long-absence")


def test_deterministic():
    args = (DriveState(social=0.9), 99999, EMPTY_SNAPSHOT, {}, NOW, _cfg())
    assert should_initiate(*args) == should_initiate(*args)


def test_inputs_unchanged():
    fired = {"session-open": NOW - 100}
    snapshot = EnvironmentSnapshot(app_noun="terminal", idle=True)
    before_fired = dict(fired)
    should_initiate(DriveState(), 0, snapshot, fired, NOW, _cfg())
    assert fired == before_fired


def test_ignores_snapshot():
    fired, cfg = {}, _cfg()
    first = should_initiate(DriveState(), 0,
                            EnvironmentSnapshot(app_noun="terminal",
                                                idle=True),
                            fired, NOW, cfg)
    second = should_initiate(DriveState(), 0, EMPTY_SNAPSHOT, fired, NOW,
                             cfg)
    assert first == second == InitiateDecision("session-open")


def _istore(tmp_path, name="initiative.json"):
    return InitiativeStore(tmp_path / name,
                           time_fn=lambda: 1700000000.0)


def test_state_frozen():
    state = InitiativeState(per_reason_last_fire={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.policy_rev = 9


def test_state_mapping_immutable():
    state = InitiativeState(per_reason_last_fire={"session-open": 1.0})
    with pytest.raises(TypeError):
        state.per_reason_last_fire["session-open"] = 2.0


def test_state_rejects_bad_key():
    with pytest.raises(ValueError):
        InitiativeState(per_reason_last_fire={1: 2.0})


def test_state_rejects_bad_value():
    with pytest.raises(ValueError):
        InitiativeState(per_reason_last_fire={"x": -1.0})
    with pytest.raises(ValueError):
        InitiativeState(per_reason_last_fire={"x": "y"})


def test_store_missing_file(tmp_path):
    loaded = _istore(tmp_path).load()
    assert dict(loaded.per_reason_last_fire) == {}
    assert not (tmp_path / "initiative.json").exists()


def test_store_round_trip(tmp_path):
    from carrot.initiative import InitiativeState

    store = _istore(tmp_path)
    store.save(InitiativeState(
        per_reason_last_fire={"session-open": 100.0}))
    assert store.load() == InitiativeState(
        per_reason_last_fire={"session-open": 100.0})


def test_store_corrupt_json(tmp_path):
    p = tmp_path / "initiative.json"
    p.write_text("{not json", encoding="utf-8")
    assert dict(_istore(tmp_path).load().per_reason_last_fire) == {}
    assert not p.exists()
    assert len(list(tmp_path.glob("quarantine-initiative-*.json"))) == 1


def test_store_unparseable_version(tmp_path):
    p = tmp_path / "initiative.json"
    p.write_text('{"version": "x", "per_reason_last_fire": {}}',
                 encoding="utf-8")
    assert dict(_istore(tmp_path).load().per_reason_last_fire) == {}
    assert len(list(tmp_path.glob("quarantine-initiative-*.json"))) == 1


def test_store_unknown_version(tmp_path):
    p = tmp_path / "initiative.json"
    p.write_text('{"version": 2, "per_reason_last_fire":'
                 ' {"session-open": 5.0}}', encoding="utf-8")
    assert dict(_istore(tmp_path).load().per_reason_last_fire) == {}
    assert list(tmp_path.glob("quarantine-initiative-*.json")) == []
    assert p.exists()


def test_store_wrong_top_level_shape(tmp_path):
    p = tmp_path / "initiative.json"
    p.write_text('{"version": 1, "per_reason_last_fire": "not a dict"}',
                 encoding="utf-8")
    assert dict(_istore(tmp_path).load().per_reason_last_fire) == {}
    assert len(list(tmp_path.glob("quarantine-initiative-*.json"))) == 1


def test_store_per_entry_fail_soft(tmp_path):
    p = tmp_path / "initiative.json"
    p.write_text('{"version": 1, "per_reason_last_fire":'
                 ' {"long-absence": 100.0, "bad": "yes", "neg": -1.0}}',
                 encoding="utf-8")
    loaded = _istore(tmp_path).load()
    assert dict(loaded.per_reason_last_fire) == {"long-absence": 100.0}
    assert list(tmp_path.glob("quarantine-initiative-*.json")) == []


def test_store_missing_field(tmp_path):
    p = tmp_path / "initiative.json"
    p.write_text('{"version": 1}', encoding="utf-8")
    assert dict(_istore(tmp_path).load().per_reason_last_fire) == {}


def test_record_fire_first(tmp_path):
    store = _istore(tmp_path)
    state = store.record_fire("long-absence", 1000.0)
    assert dict(state.per_reason_last_fire) == {"long-absence": 1000.0}
    assert dict(store.load().per_reason_last_fire) == \
        {"long-absence": 1000.0}


def test_record_fire_same_reason_overwrites(tmp_path):
    store = _istore(tmp_path)
    store.record_fire("long-absence", 1000.0)
    state = store.record_fire("long-absence", 2000.0)
    assert dict(state.per_reason_last_fire) == {"long-absence": 2000.0}


def test_record_fire_different_reason_accumulates(tmp_path):
    store = _istore(tmp_path)
    store.record_fire("long-absence", 1000.0)
    state = store.record_fire("session-open", 2000.0)
    assert dict(state.per_reason_last_fire) == {"long-absence": 1000.0,
                                                "session-open": 2000.0}


def test_record_fire_returns_persisted_state(tmp_path):
    store = _istore(tmp_path)
    state = store.record_fire("session-open", 1000.0)
    assert state == store.load()


def test_save_size_cap(tmp_path):
    store = _istore(tmp_path)
    huge = {f"reason-{i}": float(i) for i in range(2000)}
    with pytest.raises(ValueError):
        store.save(InitiativeState(per_reason_last_fire=huge))
    assert not (tmp_path / "initiative.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_initiative_module_still_clean():
    tree = ast.parse((ROOT / "carrot" / "initiative.py").read_text(
        encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"carrot"}
    assert mods <= allowed, mods
    src = (ROOT / "carrot" / "initiative.py").read_text(encoding="utf-8")
    assert "from .emotions import" not in src
    assert "from .permissions import" not in src
