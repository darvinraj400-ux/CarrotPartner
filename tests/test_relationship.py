"""Slice 11 Phase-1 tests: relationship data layer, config, apply_event."""
import json

import pytest

from carrot.relationship import (
    DIMENSIONS,
    RelationshipConfig,
    RelationshipState,
    RelationshipStore,
    apply_event,
)
from tests.conftest import ROOT


def _cfg():
    return RelationshipConfig.load(str(ROOT / "config" / "relationship.json"))


# ---- RelationshipState ----


def test_state_defaults():
    s = RelationshipState()
    assert s.trust == 0.5
    assert s.closeness == 0.5
    assert dict(s.updated_ts) == {}


def test_state_frozen():
    s = RelationshipState()
    with pytest.raises(Exception):
        s.trust = 0.9
    with pytest.raises(Exception):
        s.closeness = 0.9


def test_state_rejects_bad_trust():
    with pytest.raises(ValueError):
        RelationshipState(trust=1.5)
    with pytest.raises(ValueError):
        RelationshipState(trust=-0.1)
    with pytest.raises(ValueError):
        RelationshipState(trust=True)


def test_state_rejects_bad_closeness():
    with pytest.raises(ValueError):
        RelationshipState(closeness=1.5)
    with pytest.raises(ValueError):
        RelationshipState(closeness=-0.1)
    with pytest.raises(ValueError):
        RelationshipState(closeness=False)


def test_state_rejects_bad_updated_ts_keys():
    with pytest.raises(ValueError):
        RelationshipState(updated_ts={"unknown": 1.0})


def test_state_rejects_bad_updated_ts_values():
    with pytest.raises(ValueError):
        RelationshipState(updated_ts={"trust": -1.0})
    with pytest.raises(ValueError):
        RelationshipState(updated_ts={"trust": "x"})
    with pytest.raises(ValueError):
        RelationshipState(updated_ts={"trust": True})


def test_state_updated_ts_immutable():
    s = RelationshipState(
        updated_ts={"trust": 1.0, "closeness": 2.0})
    with pytest.raises(TypeError):
        s.updated_ts["trust"] = 2.0


def test_state_coerces_int_to_float():
    s = RelationshipState(trust=1, closeness=0)
    assert s.trust == 1.0
    assert isinstance(s.trust, float)
    assert s.closeness == 0.0


# ---- RelationshipConfig.load ----


def test_config_loads_valid():
    cfg = _cfg()
    assert cfg.event_deltas["compliment"]["trust"] == 0.03
    assert set(cfg.event_deltas) >= {"compliment", "affection", "apology",
                                     "repair", "insult", "joke",
                                     "achievement", "return"}
    assert "neutral" not in cfg.event_deltas


def test_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        RelationshipConfig.load(str(tmp_path / "nope.json"))


def test_config_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        RelationshipConfig.load(str(path))


def test_config_missing_event_deltas(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    cfg = RelationshipConfig.load(str(path))
    assert dict(cfg.event_deltas) == {}


def test_config_bad_inner_key(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"event_deltas": {"compliment": {"happiness": 0.1}}}',
                    encoding="utf-8")
    with pytest.raises(ValueError):
        RelationshipConfig.load(str(path))


def test_config_bad_inner_value_type(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"event_deltas": {"compliment": {"trust": "x"}}}',
                    encoding="utf-8")
    with pytest.raises(ValueError):
        RelationshipConfig.load(str(path))


def test_config_frozen():
    cfg = _cfg()
    with pytest.raises(Exception):
        cfg.event_deltas = {}


# ---- apply_event: positive ----


def test_apply_event_both_dims():
    s = apply_event(RelationshipState(), "compliment", 1.0, 1000.0, _cfg())
    assert s.trust == pytest.approx(0.53)
    assert s.closeness == pytest.approx(0.52)


def test_apply_event_one_dim():
    s = apply_event(RelationshipState(), "apology", 1.0, 1000.0, _cfg())
    assert s.trust == pytest.approx(0.56)
    assert s.closeness == 0.50


def test_apply_event_intensity_scaling():
    s = apply_event(RelationshipState(), "compliment", 0.5, 1000.0, _cfg())
    assert s.trust == pytest.approx(0.515)
    assert s.closeness == pytest.approx(0.51)


def test_apply_event_zero_intensity_unchanged():
    s0 = RelationshipState()
    s1 = apply_event(s0, "compliment", 0.0, 1000.0, _cfg())
    assert s1 == s0
    assert dict(s1.updated_ts) == {}


def test_apply_event_missing_row_unchanged():
    s0 = RelationshipState()
    assert apply_event(s0, "neutral", 1.0, 1000.0, _cfg()) == s0


def test_apply_event_unknown_type_unchanged():
    s0 = RelationshipState()
    assert apply_event(s0, "made-up-event", 1.0, 1000.0, _cfg()) == s0


# ---- apply_event: clamping & edges ----


def test_apply_event_clamps_high():
    s0 = RelationshipState(trust=0.99)
    s1 = apply_event(s0, "compliment", 1.0, 1000.0, _cfg())
    assert s1.trust == 1.0


def test_apply_event_clamps_low():
    s0 = RelationshipState(trust=0.01)
    s1 = apply_event(s0, "insult", 1.0, 1000.0, _cfg())
    assert s1.trust == 0.0


def test_apply_event_negative_closeness():
    s1 = apply_event(RelationshipState(), "insult", 1.0, 1000.0, _cfg())
    assert s1.closeness == pytest.approx(0.47)


def test_apply_event_zero_delta_dim_untouched():
    # Apology's row carries trust only: closeness keeps value and stamp.
    s0 = RelationshipState(updated_ts={"closeness": 500.0})
    s1 = apply_event(s0, "apology", 1.0, 1000.0, _cfg())
    assert s1.closeness == 0.5
    assert s1.updated_ts["closeness"] == 500.0


# ---- apply_event: updated_ts ----


def test_apply_event_sets_touched_dims_only():
    s1 = apply_event(RelationshipState(), "compliment", 1.0, 1000.0,
                     _cfg())
    assert dict(s1.updated_ts) == {"trust": 1000.0, "closeness": 1000.0}


def test_apply_event_single_dim_stamp():
    s1 = apply_event(RelationshipState(), "apology", 1.0, 1000.0, _cfg())
    assert dict(s1.updated_ts) == {"trust": 1000.0}


def test_apply_event_preserves_untouched_stamps():
    s0 = RelationshipState(updated_ts={"closeness": 500.0})
    s1 = apply_event(s0, "apology", 1.0, 1000.0, _cfg())
    assert dict(s1.updated_ts) == {"trust": 1000.0, "closeness": 500.0}


def test_apply_event_zero_intensity_no_stamp():
    s0 = RelationshipState(updated_ts={"trust": 500.0})
    s1 = apply_event(s0, "compliment", 0.0, 1000.0, _cfg())
    assert dict(s1.updated_ts) == {"trust": 500.0}


# ---- purity & determinism ----


def test_apply_event_does_not_mutate_input():
    s0 = RelationshipState()
    before = s0.to_dict()
    s1 = apply_event(s0, "compliment", 1.0, 1000.0, _cfg())
    assert s0.to_dict() == before
    assert s1 is not s0


def test_apply_event_deterministic():
    cfg = _cfg()
    first = apply_event(RelationshipState(), "affection", 0.7, 1000.0,
                        cfg)
    second = apply_event(RelationshipState(), "affection", 0.7, 1000.0,
                         cfg)
    assert first == second
    assert DIMENSIONS == ("trust", "closeness")


# --- Slice 11 Phase 2: RelationshipStore tests ---


def _store(tmp_path, name="relationship.json"):
    return RelationshipStore(tmp_path / name)


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_store_missing_file_returns_defaults(tmp_path):
    store = _store(tmp_path)
    state = store.load()
    assert state == RelationshipState()
    assert not (tmp_path / "relationship.json").exists()


def test_store_round_trip(tmp_path):
    store = _store(tmp_path)
    state = RelationshipState(trust=0.7, closeness=0.3,
                              updated_ts={"trust": 100.0})
    store.save(state)
    loaded = store.load()
    assert loaded.trust == 0.7
    assert loaded.closeness == 0.3
    assert dict(loaded.updated_ts) == {"trust": 100.0}


def test_store_save_type_check(tmp_path):
    with pytest.raises(TypeError):
        _store(tmp_path).save("not a state")


def test_store_malformed_json_quarantines(tmp_path):
    path = tmp_path / "relationship.json"
    path.write_text("{not json", encoding="utf-8")
    state = _store(tmp_path).load()
    assert state == RelationshipState()
    assert not path.exists()
    assert len(list(tmp_path.glob("quarantine-relationship-*.json"))) == 1


def test_store_top_level_non_object_quarantines(tmp_path):
    path = tmp_path / "relationship.json"
    _write(path, [1, 2, 3])
    assert _store(tmp_path).load() == RelationshipState()
    assert not path.exists()
    assert len(list(tmp_path.glob("quarantine-relationship-*.json"))) == 1


def test_store_unparseable_version_quarantines(tmp_path):
    path = tmp_path / "relationship.json"
    _write(path, {"version": "x", "trust": 0.7})
    assert _store(tmp_path).load() == RelationshipState()
    assert not path.exists()
    assert len(list(tmp_path.glob("quarantine-relationship-*.json"))) == 1


def test_store_unknown_version_no_quarantine(tmp_path):
    path = tmp_path / "relationship.json"
    _write(path, {"version": 2, "trust": 0.7})
    assert _store(tmp_path).load() == RelationshipState()
    assert path.exists()
    assert list(tmp_path.glob("quarantine-relationship-*.json")) == []


def test_store_non_numeric_trust_defaults(tmp_path):
    path = tmp_path / "relationship.json"
    _write(path, {"version": 1, "trust": "high", "closeness": 0.3})
    state = _store(tmp_path).load()
    assert state.trust == 0.5
    assert state.closeness == 0.3
    assert path.exists()  # no quarantine over a bad scalar


def test_store_out_of_range_trust_clamped(tmp_path):
    path = tmp_path / "relationship.json"
    _write(path, {"version": 1, "trust": 1.5, "closeness": 0.3})
    assert _store(tmp_path).load().trust == 1.0
    _write(path, {"version": 1, "trust": -0.1, "closeness": 0.3})
    assert _store(tmp_path).load().trust == 0.0
    assert list(tmp_path.glob("quarantine-relationship-*.json")) == []


def test_store_bool_trust_rejected(tmp_path):
    path = tmp_path / "relationship.json"
    _write(path, {"version": 1, "trust": True, "closeness": 0.3})
    state = _store(tmp_path).load()
    assert state.trust == 0.5
    assert state.closeness == 0.3


def test_store_bad_updated_ts_key_skipped(tmp_path):
    path = tmp_path / "relationship.json"
    _write(path, {"version": 1, "trust": 0.7,
                  "updated_ts": {"trust": 100.0, "unknown": 200.0}})
    state = _store(tmp_path).load()
    assert dict(state.updated_ts) == {"trust": 100.0}


def test_store_bad_updated_ts_value_skipped(tmp_path):
    path = tmp_path / "relationship.json"
    _write(path, {"version": 1, "trust": 0.7,
                  "updated_ts": {"trust": "x", "closeness": 300.0}})
    state = _store(tmp_path).load()
    assert dict(state.updated_ts) == {"closeness": 300.0}


def test_store_non_mapping_updated_ts_empty(tmp_path):
    path = tmp_path / "relationship.json"
    _write(path, {"version": 1, "trust": 0.7,
                  "updated_ts": "not a mapping"})
    assert dict(_store(tmp_path).load().updated_ts) == {}


def test_store_missing_updated_ts_empty(tmp_path):
    path = tmp_path / "relationship.json"
    _write(path, {"version": 1, "trust": 0.7, "closeness": 0.3})
    assert dict(_store(tmp_path).load().updated_ts) == {}


def test_store_size_cap_enforced(tmp_path, monkeypatch):
    # No constructible RelationshipState can exceed 4 KB (updated_ts is
    # capped to two float-valued dims), so shrink the cap to prove the
    # enforcement path raises before any write occurs.
    import carrot.relationship as rel

    assert rel.MAX_RELATIONSHIP_BYTES == 4096
    monkeypatch.setattr(rel, "MAX_RELATIONSHIP_BYTES", 10)
    with pytest.raises(ValueError):
        _store(tmp_path).save(RelationshipState())
    assert not (tmp_path / "relationship.json").exists()
