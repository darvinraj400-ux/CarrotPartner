"""Stage-1 tests: EmotionState sole ownership, clamping, decay, bounded causes."""
import json

import pytest

from carrot.emotions import (
    EMOTIONS,
    RECENT_CAUSES_CAP,
    RESTING,
    RESTING_ENERGY,
    EmotionState,
    clamp,
)
from tests.conftest import ROOT

DECAY = {
    "baselines": {"happiness": 0.5, "hurt": 0.05, "energy": 0.7,
                  "excitement": 0.2},
    "half_lives": {"happiness": 3600.0, "hurt": 10800.0,
                   "excitement": 600.0, "energy": 7200.0},
}


def test_seven_emotions_only():
    assert set(EMOTIONS) == {"happiness", "sadness", "anger", "hurt",
                             "affection", "excitement", "embarrassment"}
    assert "energy" not in EMOTIONS


def test_fresh_state_starts_at_rest():
    s = EmotionState()
    assert s.emotions == RESTING
    assert s.energy == RESTING_ENERGY


def test_resting_matches_decay_baselines():
    cfg = json.loads((ROOT / "config" / "decay.json").read_text(
        encoding="utf-8"))
    assert cfg["baselines"] == {**RESTING, "energy": RESTING_ENERGY}


def test_clamp_bounds():
    s = EmotionState(emotions={"hurt": 0.9})
    s.apply_deltas({"hurt": 0.5}, cause="overflow")
    assert s.emotions["hurt"] == 1.0
    s.apply_deltas({"hurt": -2.0}, cause="underflow")
    assert s.emotions["hurt"] == 0.0


def test_unknown_keys_rejected():
    s = EmotionState()
    with pytest.raises(ValueError):
        EmotionState(emotions={"loneliness": 0.5})
    with pytest.raises(ValueError):
        s.apply_deltas({"loneliness": 0.1}, cause="x")
    with pytest.raises(ValueError):
        s.apply_deltas({"hurt": "a lot"}, cause="x")


def test_recent_causes_bounded():
    s = EmotionState()
    for i in range(RECENT_CAUSES_CAP + 10):
        s.apply_deltas({"happiness": 0.0}, cause=f"event-{i}")
    assert len(s.recent_causes) == RECENT_CAUSES_CAP
    assert s.recent_causes[0]["event"] == "event-10"
    # Over-long history trimmed on load too.
    data = s.to_dict()
    data["recent_causes"] = [{"event": f"old-{i}"} for i in range(100)]
    s2 = EmotionState.from_dict(data)
    assert len(s2.recent_causes) == RECENT_CAUSES_CAP


def test_decay_moves_toward_baseline():
    s = EmotionState(emotions={"happiness": 0.9, "hurt": 0.8,
                               "excitement": 0.9}, energy=0.9)
    before = s.emotions
    s.apply_decay(600.0, DECAY)  # one excitement half-life
    after = s.emotions
    assert after["excitement"] == pytest.approx(0.55, abs=1e-9)
    assert after["happiness"] < before["happiness"]
    assert after["happiness"] > 0.5
    assert after["hurt"] < before["hurt"]
    assert s.energy < 0.9 and s.energy > 0.7


def test_decay_zero_or_negative_elapsed_is_noop():
    s = EmotionState(emotions={"hurt": 0.8})
    snap = s.emotions
    s.apply_decay(0, DECAY)
    s.apply_decay(-5, DECAY)
    assert s.emotions == snap


def test_round_trip():
    s = EmotionState(emotions={"hurt": 0.6}, energy=0.4, mood="low",
                     last_seen=123.0,
                     recent_causes=[{"event": "insult", "reason": "r"}])
    s2 = EmotionState.from_dict(s.to_dict())
    assert s2.to_dict() == s.to_dict()
