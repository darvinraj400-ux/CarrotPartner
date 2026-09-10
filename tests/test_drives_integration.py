"""Slice 5 Phase-3 tests: drives wiring in process_turn (no gate/prompt
changes here — behavior and prompt identity are asserted, not assumed)."""
import ast

from carrot.drives import DriveState, DriveStore
from tests.conftest import ROOT
from tests.test_classifier import StubSemanticClassifier
from tests.test_scenarios import FakeTime, make_carrot
from carrot.classifier import InteractionEvent


def _seeded_store(tmp_path, social, novelty, name="d.json"):
    store = DriveStore(tmp_path / name)
    store.save(DriveState(social=social, novelty=novelty))
    return store


def test_rise_on_turn_with_gap():
    clock = FakeTime()
    c = make_carrot(start=clock)
    c.process_turn("hey")
    clock.advance(7200)
    c.process_turn("You are USELESS!!!")
    loaded = c.drive_store.load()
    assert abs(loaded.social - 0.09) < 1e-9
    assert abs(loaded.novelty - 0.06) < 1e-9


def test_satisfaction_on_classified_event():
    clock = FakeTime()
    c = make_carrot(start=clock)
    c.process_turn("hey")
    clock.advance(18000)
    c.process_turn("You're cute, thanks!")
    loaded = c.drive_store.load()
    assert abs(loaded.social - (0.25 - 0.01 - 0.15)) < 1e-9
    assert abs(loaded.novelty - 0.15) < 1e-9


def test_drives_persist_across_instances(tmp_path):
    clock = FakeTime()
    shared = DriveStore(tmp_path / "shared.json")
    first = make_carrot(start=clock, drive_store=shared)
    first.process_turn("my name is Alice")
    snapshot = shared.load().to_dict()
    second = make_carrot(drive_store=shared)
    assert second.drive_store.load().to_dict() == snapshot


def test_last_drive_pull_set():
    clock = FakeTime()
    c = make_carrot(start=clock)
    c.process_turn("hey")
    clock.advance(13 * 3600)
    c.process_turn("hey there")
    assert c.last_drive_pull == "social"


def test_last_drive_pull_none_when_low():
    c = make_carrot()
    c.process_turn("hey")
    c.clock.advance(5)
    c.process_turn("hey there")
    assert c.last_drive_pull is None


def test_no_behavior_change_from_drives(tmp_path):
    high = _seeded_store(tmp_path, 1.0, 1.0, "high.json")
    low = _seeded_store(tmp_path, 0.0, 0.0, "low.json")
    a = make_carrot(drive_store=high)
    b = make_carrot(drive_store=low)
    a.process_turn("hello there")
    b.process_turn("hello there")
    assert a.last_behavior.name == b.last_behavior.name == "normal"
    for carrot in (a, b):
        carrot.state.apply_deltas({"excitement": 0.6}, cause="test setup")
    a.process_turn("hey")
    b.process_turn("hey")
    assert a.last_behavior.name == b.last_behavior.name == "playful"


def test_no_prompt_change_from_drives(tmp_path):
    high = _seeded_store(tmp_path, 0.9, 0.9, "high.json")
    low = _seeded_store(tmp_path, 0.0, 0.0, "low.json")
    a = make_carrot(drive_store=high)
    b = make_carrot(drive_store=low)
    a.process_turn("hello there")
    b.process_turn("hello there")
    assert a.llm.last_system == b.llm.last_system


def test_no_emotion_coupling_from_drives(tmp_path):
    high = _seeded_store(tmp_path, 0.9, 0.9, "high.json")
    low = _seeded_store(tmp_path, 0.0, 0.0, "low.json")
    a = make_carrot(drive_store=high)
    b = make_carrot(drive_store=low)
    a.process_turn("hello there")
    b.process_turn("hello there")
    assert a.state.to_dict() == b.state.to_dict()


def test_no_process_turn_outside_user_loop():
    callers = set()
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute)
                        and func.attr == "process_turn"):
                    callers.add(path.name)
    assert callers == {"cli.py"}


def test_state_schema_has_no_drives_key():
    c = make_carrot()
    c.process_turn("my name is Alice")
    assert "drives" not in c.state.to_dict()


def test_unknown_event_applies_presence_only():
    clock = FakeTime()
    c = make_carrot(start=clock)
    c.process_turn("hey")
    clock.advance(3600)
    c.classifier = StubSemanticClassifier(
        InteractionEvent("mystery", 0.5, "stub"))
    c.process_turn("blah")
    loaded = c.drive_store.load()
    assert abs(loaded.social - (0.05 - 0.01)) < 1e-9
    assert abs(loaded.novelty - 0.03) < 1e-9


def test_zero_elapsed_rise_is_zero():
    c = make_carrot()
    c.process_turn("hey")
    c.process_turn("You're cute, thanks!")
    loaded = c.drive_store.load()
    assert loaded.social == 0.0
    assert loaded.novelty == 0.0
