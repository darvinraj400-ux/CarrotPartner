"""Slice 5 Phase-1 tests: drives data layer (persistence + pure math)."""
import ast
import dataclasses
import json
import sys

import pytest

from carrot.drives import (
    DRIVE_CEIL,
    DRIVE_FLOOR,
    DRIVE_SCHEMA_VERSION,
    DriveState,
    DriveStore,
    DrivesConfig,
    advance,
    apply_satisfaction,
    clamp_level,
    drive_pull,
    update_rise,
)
from tests.conftest import ROOT

PRESENCE = {"social": -0.01}
EVENTS = {
    "compliment": {"social": -0.15},
    "neutral": {"social": -0.02},
}


def _store(tmp_path, name="drives.json"):
    return DriveStore(tmp_path / name, time_fn=lambda: 1700000000.0)


def test_defaults_zero():
    assert DriveState() == DriveState(social=0.0, novelty=0.0)


def test_round_trip(tmp_path):
    store = _store(tmp_path)
    store.save(DriveState(social=0.7, novelty=0.25))
    assert store.load() == DriveState(social=0.7, novelty=0.25)


def test_missing_file_loads_zeros(tmp_path):
    assert _store(tmp_path).load() == DriveState()


def test_corrupt_json_quarantined(tmp_path):
    p = tmp_path / "drives.json"
    p.write_text("{not json", encoding="utf-8")
    assert _store(tmp_path).load() == DriveState()
    assert not p.exists()
    assert len(list(tmp_path.glob("quarantine-drives-*.json"))) == 1


def test_unparseable_version_quarantined(tmp_path):
    p = tmp_path / "drives.json"
    p.write_text('{"version": "x", "social": 0.5, "novelty": 0.5}',
                 encoding="utf-8")
    assert _store(tmp_path).load() == DriveState()
    assert len(list(tmp_path.glob("quarantine-drives-*.json"))) == 1


def test_unknown_version_no_quarantine(tmp_path):
    p = tmp_path / "drives.json"
    p.write_text('{"version": 2, "social": 0.5, "novelty": 0.5}',
                 encoding="utf-8")
    assert _store(tmp_path).load() == DriveState()
    assert list(tmp_path.glob("quarantine-drives-*.json")) == []
    assert p.exists()


def test_missing_top_level_keys_quarantined(tmp_path):
    p = tmp_path / "drives.json"
    p.write_text('{"version": 1}', encoding="utf-8")
    assert _store(tmp_path).load() == DriveState()
    assert len(list(tmp_path.glob("quarantine-drives-*.json"))) == 1


def test_invalid_field_drops_to_default(tmp_path):
    p = tmp_path / "drives.json"
    p.write_text('{"version": 1, "social": "x", "novelty": 0.5}',
                 encoding="utf-8")
    loaded = _store(tmp_path).load()
    assert loaded.social == 0.0
    assert loaded.novelty == 0.5
    assert list(tmp_path.glob("quarantine-drives-*.json")) == []


def test_out_of_range_fields_clamped_on_load(tmp_path):
    p = tmp_path / "drives.json"
    p.write_text('{"version": 1, "social": 1.5, "novelty": -0.3}',
                 encoding="utf-8")
    loaded = _store(tmp_path).load()
    assert loaded.social == 1.0
    assert loaded.novelty == 0.0


def test_atomic_save_leaves_no_tmp(tmp_path):
    store = _store(tmp_path)
    store.save(DriveState(social=0.3, novelty=0.4))
    names = [p.name for p in tmp_path.iterdir()]
    assert "drives.json" in names
    assert not [n for n in names if n.endswith(".tmp")]


def test_deterministic_with_fixed_time_fn(tmp_path):
    first = DriveStore(tmp_path / "a.json", time_fn=lambda: 42.0)
    second = DriveStore(tmp_path / "b.json", time_fn=lambda: 42.0)
    first.save(DriveState(social=0.3, novelty=0.4))
    second.save(DriveState(social=0.3, novelty=0.4))
    assert (tmp_path / "a.json").read_bytes() == \
        (tmp_path / "b.json").read_bytes()


def test_clamp_level():
    assert clamp_level(-0.5) == DRIVE_FLOOR == 0.0
    assert clamp_level(0.0) == 0.0
    assert clamp_level(0.6) == 0.6
    assert clamp_level(1.0) == DRIVE_CEIL == 1.0
    assert clamp_level(2.0) == 1.0
    assert clamp_level("nonsense") == 0.0
    assert clamp_level(None) == 0.0


def test_update_rise_zero_elapsed_identity():
    state = DriveState(social=0.4, novelty=0.2)
    assert update_rise(state, 0, {"social": 0.05, "novelty": 0.03}) is state
    assert update_rise(state, -10, {"social": 0.05, "novelty": 0.03}) is state


def test_update_rise_positive_elapsed():
    got = update_rise(DriveState(), 3600.0,
                      {"social": 0.05, "novelty": 0.03})
    assert got.social == 0.05
    assert got.novelty == 0.03


def test_update_rise_capped():
    got = update_rise(DriveState(social=0.98, novelty=0.0), 36000.0,
                      {"social": 0.05, "novelty": 0.03})
    assert got.social == 1.0
    assert got.novelty == 0.3


def test_update_rise_missing_rate_defaults_zero():
    got = update_rise(DriveState(), 3600.0, {})
    assert got == DriveState()


def test_satisfaction_presence_only():
    state = DriveState(social=0.5, novelty=0.5)
    assert apply_satisfaction(state, "compliment", {}, {}) == state
    got = apply_satisfaction(state, "compliment", PRESENCE, {})
    assert got.social == 0.49
    assert got.novelty == 0.5


def test_satisfaction_event_adds_to_presence():
    state = DriveState(social=0.5, novelty=0.5)
    got = apply_satisfaction(state, "compliment", PRESENCE, EVENTS)
    assert got.social == 0.5 - 0.01 - 0.15
    assert got.novelty == 0.5


def test_satisfaction_floor_clamp():
    state = DriveState(social=0.05, novelty=0.5)
    got = apply_satisfaction(state, "compliment",
                             {"social": -0.1}, {"compliment": {"social": -0.2}})
    assert got.social == 0.0


def test_satisfaction_unknown_event():
    state = DriveState(social=0.5, novelty=0.5)
    got = apply_satisfaction(state, "mystery", PRESENCE, EVENTS)
    assert got.social == 0.49
    assert got.novelty == 0.5


def test_drives_module_boundary():
    tree = ast.parse((ROOT / "carrot" / "drives.py").read_text(
        encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    assert "emotions" not in mods, mods
    assert mods <= set(sys.stdlib_module_names) | {"carrot"}, mods
    src = (ROOT / "carrot" / "drives.py").read_text(encoding="utf-8")
    assert "from .emotions import" not in src
    assert "from carrot.emotions import" not in src


def test_drive_state_frozen_and_hashable():
    import pytest

    state = DriveState(social=0.5, novelty=0.5)
    hash(state)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.social = 0.9


def test_save_shape_has_version_and_levels(tmp_path):
    store = _store(tmp_path)
    store.save(DriveState(social=0.3, novelty=0.4))
    data = json.loads((tmp_path / "drives.json").read_text(
        encoding="utf-8"))
    assert data["version"] == DRIVE_SCHEMA_VERSION == 1
    assert data["social"] == 0.3
    assert data["novelty"] == 0.4
    assert set(data) == {"version", "social", "novelty"}


def _cfg(**overrides):
    base = {
        "rise_per_hour": {"social": 0.05, "novelty": 0.03},
        "pull_threshold": 0.6,
        "presence_satisfaction": {"social": -0.01},
        "event_satisfaction": {
            "compliment": {"social": -0.15},
            "neutral": {"social": -0.02},
            "insult": {},
        },
    }
    base.update(overrides)
    return base


def _write_config(tmp_path, data, name="drives.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_config_loads_valid():
    cfg = DrivesConfig.load(str(ROOT / "config" / "drives.json"))
    assert cfg.pull_threshold == 0.6
    assert cfg.rise_per_hour["social"] == 0.05
    assert cfg.event_satisfaction["insult"] == {}


def test_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DrivesConfig.load(str(tmp_path / "nope.json"))


def test_config_malformed_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        DrivesConfig.load(str(path))


def test_config_missing_threshold_defaults(tmp_path):
    data = _cfg()
    del data["pull_threshold"]
    assert DrivesConfig.load(_write_config(tmp_path, data)).pull_threshold \
        == 0.6


def test_config_invalid_threshold_raises(tmp_path):
    with pytest.raises(ValueError):
        DrivesConfig.load(_write_config(tmp_path, _cfg(pull_threshold=1.5)))


def test_config_missing_presence_defaults(tmp_path):
    data = _cfg()
    del data["presence_satisfaction"]
    assert DrivesConfig.load(
        _write_config(tmp_path, data)).presence_satisfaction == {}


def test_config_frozen():
    cfg = DrivesConfig.load(str(ROOT / "config" / "drives.json"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.pull_threshold = 0.9


def test_advance_rise_then_satisfaction(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    got = advance(DriveState(), 7200.0, "compliment", cfg)
    assert got.social == pytest.approx(0.0)
    assert got.novelty == pytest.approx(0.06)


def test_advance_is_pure(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    before = DriveState(social=0.4, novelty=0.2)
    advance(before, 3600.0, "compliment", cfg)
    assert before == DriveState(social=0.4, novelty=0.2)


def test_advance_unknown_event(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    got = advance(DriveState(), 3600.0, "something_random", cfg)
    assert got.social == pytest.approx(0.05 - 0.01)
    assert got.novelty == pytest.approx(0.03)


def test_advance_zero_elapsed_neutral(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    got = advance(DriveState(), 0, "neutral", cfg)
    assert got.social == 0.0
    assert got.novelty == 0.0


def test_advance_zero_elapsed_insult(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    got = advance(DriveState(), 0, "insult", cfg)
    assert got.social == 0.0
    assert got.novelty == 0.0


def test_pull_both_below(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    assert drive_pull(DriveState(social=0.4, novelty=0.3), cfg) is None


def test_pull_social_above(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    assert drive_pull(DriveState(social=0.7, novelty=0.3), cfg) == "social"


def test_pull_novelty_above(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    assert drive_pull(DriveState(social=0.3, novelty=0.7), cfg) == "novelty"


def test_pull_tie_goes_by_drives_order(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    assert drive_pull(DriveState(social=0.7, novelty=0.7), cfg) == "social"


def test_pull_boundary_inclusive(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    assert drive_pull(DriveState(social=0.6, novelty=0.0), cfg) == "social"


def test_pull_just_below(tmp_path):
    cfg = DrivesConfig.load(_write_config(tmp_path, _cfg()))
    assert drive_pull(DriveState(social=0.59, novelty=0.0), cfg) is None
