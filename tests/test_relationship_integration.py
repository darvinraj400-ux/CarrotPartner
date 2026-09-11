"""Slice 11 Phase-3 tests: Carrot relationship wiring, step 4c, bands."""
import pytest

from carrot.relationship import (
    RelationshipConfig,
    RelationshipState,
    RelationshipStore,
)
from tests.conftest import ROOT
from tests.test_scenarios import FakeTime, make_carrot


def _seeded_store(tmp_path, trust, closeness, name="rel.json"):
    store = RelationshipStore(tmp_path / name)
    store.save(RelationshipState(trust=trust, closeness=closeness,
                                 updated_ts={}))
    return store


# ---- construction ----


def test_fresh_carrot_loads_default_relationship():
    c = make_carrot()
    assert c.last_relationship.trust == 0.5
    assert c.last_relationship.closeness == 0.5


def test_existing_store_is_loaded(tmp_path):
    store = _seeded_store(tmp_path, 0.7, 0.6)
    c = make_carrot(relationship_store=store)
    assert c.last_relationship.trust == 0.7
    assert c.last_relationship.closeness == 0.6


def test_band_initialises_to_none():
    assert make_carrot().last_relationship_band is None


def test_store_and_config_injectable(tmp_path):
    from carrot.appraisal import Appraisal
    from carrot.behavior import BehaviorSelector
    from carrot.carrot import Carrot
    from carrot.classifier import DeterministicClassifier
    from carrot.conversation import ConversationBuffer
    from carrot.drives import DriveStore, DrivesConfig
    from carrot.emotions import EmotionState
    from carrot.environment import EnvironmentConfig
    from carrot.initiative import InitiativeConfig, InitiativeStore
    from carrot.llm import MockLLM
    from carrot.memory import MemoryStore
    from carrot.permissions import PermissionStore
    from carrot.personality import Personality
    from carrot.tools import ToolsConfig

    clock = FakeTime()
    custom_cfg_path = tmp_path / "rel-config.json"
    custom_cfg_path.write_text(
        '{"event_deltas": {"compliment": {"trust": 0.5}}}',
        encoding="utf-8")
    custom_store = RelationshipStore(tmp_path / "custom-rel.json",
                                     time_fn=clock)
    c = Carrot(
        state=EmotionState(), personality=Personality(),
        classifier=DeterministicClassifier(),
        appraisal=Appraisal.load(str(ROOT / "config" / "appraisal.json")),
        selector=BehaviorSelector.load(
            str(ROOT / "config" / "behavior.json")),
        llm=MockLLM(), decay_cfg={}, buffer=ConversationBuffer(),
        memory_store=MemoryStore(tmp_path / "m.json", time_fn=clock),
        drive_store=DriveStore(tmp_path / "d.json", time_fn=clock),
        drives_config=DrivesConfig.load(
            str(ROOT / "config" / "drives.json")),
        environment_config=EnvironmentConfig.load(
            str(ROOT / "config" / "environment.json")),
        tools_config=ToolsConfig.load(str(ROOT / "config" / "tools.json")),
        permission_store=PermissionStore(tmp_path / "p.json",
                                         time_fn=clock),
        initiative_config=InitiativeConfig.load(
            str(ROOT / "config" / "initiative.json")),
        initiative_store=InitiativeStore(tmp_path / "i.json",
                                         time_fn=clock),
        relationship_config=RelationshipConfig.load(
            str(custom_cfg_path)),
        relationship_store=custom_store,
        time_fn=clock,
    )
    assert c.relationship_store.path == custom_store.path
    assert c.relationship_config.event_deltas["compliment"]["trust"] == 0.5
    c.process_turn("You're cute, thanks!")
    assert c.last_relationship.trust == pytest.approx(
        0.5 + 0.5 * c.last_event.intensity)


# ---- step 4c: event-driven evolution ----


def test_compliment_moves_trust():
    c = make_carrot()
    c.process_turn("You're cute, thanks!")
    assert c.last_event.type == "compliment"
    assert c.last_event.intensity > 0  # guards against vacuous scaling
    assert c.last_relationship.trust == pytest.approx(
        0.5 + 0.03 * c.last_event.intensity)


def test_insult_moves_trust_and_closeness_down():
    c = make_carrot()
    c.process_turn("You are USELESS!!!")
    assert c.last_event.type == "insult"
    assert c.last_relationship.trust == pytest.approx(
        0.5 - 0.08 * c.last_event.intensity)
    assert c.last_relationship.closeness == pytest.approx(
        0.5 - 0.03 * c.last_event.intensity)


def test_neutral_event_is_noop_identity():
    c = make_carrot()
    prior = c.last_relationship
    c.process_turn("hello there")
    assert c.last_event.type == "neutral"
    assert c.last_relationship is prior


def test_changed_state_persisted(tmp_path):
    store = RelationshipStore(tmp_path / "rel.json")
    c = make_carrot(relationship_store=store)
    c.process_turn("You're cute, thanks!")
    loaded = store.load()
    assert loaded.trust == pytest.approx(c.last_relationship.trust)
    assert dict(loaded.updated_ts) == dict(c.last_relationship.updated_ts)


# ---- step 4c: no-op behavior ----


def test_neutral_writes_nothing(tmp_path):
    path = tmp_path / "rel.json"
    c = make_carrot(relationship_store=RelationshipStore(path))
    c.process_turn("hello there")
    assert not path.exists()


def test_insult_writes(tmp_path):
    path = tmp_path / "rel.json"
    c = make_carrot(relationship_store=RelationshipStore(path))
    c.process_turn("You are USELESS!!!")
    assert path.exists()
    assert RelationshipStore(path).load().trust < 0.5


# ---- band computation ----


def test_band_none_at_default(tmp_path):
    c = make_carrot(
        relationship_store=RelationshipStore(tmp_path / "rel.json"))
    c.process_turn("hello there")
    assert c.last_relationship_band is None


def test_band_high_when_closeness_crosses(tmp_path):
    store = _seeded_store(tmp_path, 0.5, 0.69)
    c = make_carrot(relationship_store=store)
    c.process_turn("i love you")
    assert c.last_event.type == "affection"
    assert c.last_relationship.closeness >= 0.7
    assert c.last_relationship_band == "high"


def test_band_persists_across_neutral(tmp_path):
    store = _seeded_store(tmp_path, 0.5, 0.69)
    c = make_carrot(relationship_store=store)
    c.process_turn("i love you")
    assert c.last_relationship_band == "high"
    c.process_turn("hello there")
    assert c.last_relationship_band == "high"


def test_band_clears_below_threshold(tmp_path):
    store = _seeded_store(tmp_path, 0.5, 0.70)
    c = make_carrot(relationship_store=store)
    c.process_turn("You are USELESS!!!")
    assert c.last_relationship.closeness < 0.7
    assert c.last_relationship_band is None


# ---- isolation / determinism / boundary ----


def test_no_relationship_keys_in_state_dict():
    c = make_carrot()
    c.process_turn("You're cute, thanks!")
    d = c.state.to_dict()
    assert "trust" not in d
    assert "closeness" not in d
    assert "relationship" not in d


def test_make_carrot_isolates_relationship_store():
    first = make_carrot()
    second = make_carrot()
    first.process_turn("You are USELESS!!!")
    assert first.last_relationship.trust < 0.5
    assert second.last_relationship == RelationshipState()


def test_construction_writes_nothing(tmp_path):
    path = tmp_path / "fresh-rel.json"
    make_carrot(relationship_store=RelationshipStore(path))
    assert not path.exists()


def test_relationship_deterministic(tmp_path):
    first = make_carrot(
        relationship_store=_seeded_store(tmp_path, 0.6, 0.6, "a.json"))
    second = make_carrot(
        relationship_store=_seeded_store(tmp_path, 0.6, 0.6, "b.json"))
    first.process_turn("You're cute, thanks!")
    second.process_turn("You're cute, thanks!")
    assert (first.last_relationship.to_dict()
            == second.last_relationship.to_dict())
