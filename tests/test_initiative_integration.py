"""Slice 9 Phase-3 tests: Carrot initiative wiring (construction only).

No should_initiate calls, no per-turn state, no prompt changes: the
store and config are loaded and retained, nothing more.
"""
import ast
import json

from carrot.initiative import InitiativeStore
from tests.conftest import ROOT
from tests.test_scenarios import make_carrot


def test_fresh_loads_empty():
    c = make_carrot()
    assert dict(c.initiative_store.load().per_reason_last_fire) == {}


def test_existing_store_loaded(tmp_path):
    path = tmp_path / "initiative.json"
    path.write_text(json.dumps({"version": 1, "per_reason_last_fire":
                                {"session-open": 500.0}}),
                    encoding="utf-8")
    c = make_carrot(initiative_store=InitiativeStore(path))
    loaded = c.initiative_store.load()
    assert dict(loaded.per_reason_last_fire) == {"session-open": 500.0}


def test_config_defaults_from_file():
    c = make_carrot()
    assert c.initiative_config.long_absence_threshold_s == 21600.0


def test_store_injectable_config_default(tmp_path):
    from carrot.initiative import InitiativeConfig

    path = tmp_path / "custom.json"
    c = make_carrot(initiative_store=InitiativeStore(path))
    assert c.initiative_store.path == path
    assert c.initiative_config == InitiativeConfig.load(
        str(ROOT / "config" / "initiative.json"))


def test_process_turn_does_not_touch_store(tmp_path):
    c = make_carrot(
        initiative_store=InitiativeStore(tmp_path / "initiative.json"))
    c.process_turn("hello there")
    path = tmp_path / "initiative.json"
    assert not path.exists()
    c.initiative_store.save(c.initiative_store.load())
    mtime = path.stat().st_mtime_ns
    c.process_turn("hello again")
    assert path.stat().st_mtime_ns == mtime


def test_no_should_initiate_call_in_carrot():
    tree = ast.parse((ROOT / "carrot" / "carrot.py").read_text(
        encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "should_initiate"]
    assert calls == []


def test_no_last_initiative_attribute():
    assert hasattr(make_carrot(), "last_initiative") is False


def test_state_dict_unchanged():
    c = make_carrot()
    c.process_turn("hello there")
    assert "initiative" not in c.state.to_dict()
    assert "per_reason_last_fire" not in c.state.to_dict()


def test_construction_writes_nothing(tmp_path):
    path = tmp_path / "fresh.json"
    make_carrot(initiative_store=InitiativeStore(path))
    assert not path.exists()


def test_make_carrot_isolates_store_paths():
    first = make_carrot()
    second = make_carrot()
    assert first.initiative_store.path != second.initiative_store.path
