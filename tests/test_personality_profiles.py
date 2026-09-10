"""Slice 4 Phase-6 tests: alternate profile fixture + divergence.

No production code changes: these tests prove the current architecture
already supports multiple profiles through Personality.load.
"""
from carrot.behavior import BehaviorSelector
from carrot.context import derive_context
from carrot.emotions import EmotionState
from carrot.personality import Personality
from tests.conftest import ROOT

ALT_PATH = str(ROOT / "tests" / "fixtures" / "personality-alt.json")


def _blank():
    return derive_context(())


def _alt():
    return Personality.load(ALT_PATH)


def test_alternate_profile_loads():
    alt = _alt()
    assert alt.id == "carrot-test-alt-v1"
    assert (alt.sensitivity, alt.playfulness, alt.stoicism, alt.shyness,
            alt.pride, alt.warmth, alt.patience) == \
        (0.9, 0.2, 0.8, 0.7, 0.3, 0.3, 0.9)


def test_sketches_diverge():
    default_sketch = Personality().sketch()
    alt_sketch = _alt().sketch()
    assert default_sketch and alt_sketch
    assert default_sketch != alt_sketch


def test_shy_path_diverges():
    sel = BehaviorSelector({})
    state = EmotionState(emotions={"embarrassment": 0.7})
    assert sel.pick(state, Personality(), _blank(), "").name == "embarrassed"
    assert sel.pick(state, _alt(), _blank(), "").name == "shy"


def test_stoic_path_diverges():
    sel = BehaviorSelector({})
    state = EmotionState(emotions={"hurt": 0.65, "anger": 0.4})
    assert sel.pick(state, Personality(), _blank(), "").name == "sad"
    assert sel.pick(state, _alt(), _blank(), "").name == "sulking"


def test_patience_path_diverges():
    sel = BehaviorSelector({})
    low = EmotionState(emotions={"anger": 0.38}, energy=0.2)
    assert sel.pick(low, Personality(), _blank(), "").name != "frustrated"
    assert sel.pick(low, _alt(), _blank(), "").name != "frustrated"
    mid = EmotionState(emotions={"anger": 0.42}, energy=0.2)
    assert sel.pick(mid, Personality(), _blank(), "").name == "frustrated"
    assert sel.pick(mid, _alt(), _blank(), "").name != "frustrated"


def test_proud_floor_holds_for_alternate():
    sel = BehaviorSelector({})
    state = EmotionState(emotions={"happiness": 0.6},
                         recent_causes=[{"event": "joke"}])
    assert sel.pick(state, Personality(), _blank(), "").name != "proud"
    assert sel.pick(state, _alt(), _blank(), "").name != "proud"


def test_divergence_is_deterministic():
    sel = BehaviorSelector({})
    state = EmotionState(emotions={"hurt": 0.65, "anger": 0.4})
    first = (sel.pick(state, Personality(), _blank(), "").name,
             sel.pick(state, _alt(), _blank(), "").name)
    second = (sel.pick(state, Personality(), _blank(), "").name,
              sel.pick(state, _alt(), _blank(), "").name)
    assert first == second == ("sad", "sulking")
