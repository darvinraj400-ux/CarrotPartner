"""Stage-1 tests: atomic persistence, corrupt-file safety, mood derivation."""
import json

from carrot.emotions import EmotionState
from carrot.mood import derive
from carrot.state_store import StateStore


def test_save_load_round_trip(tmp_path):
    store = StateStore(str(tmp_path / "state.json"))
    s = EmotionState(emotions={"hurt": 0.6}, energy=0.4, mood="low",
                     last_seen=42.0)
    store.save(s)
    assert store.load().to_dict() == s.to_dict()


def test_missing_file_gives_fresh_state(tmp_path):
    assert StateStore(str(tmp_path / "nope.json")).load().to_dict() == \
        EmotionState().to_dict()


def test_corrupt_file_gives_fresh_state(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{not json", encoding="utf-8")
    assert StateStore(str(p)).load().to_dict() == EmotionState().to_dict()


def test_save_is_atomic_and_utf8(tmp_path):
    store = StateStore(str(tmp_path / "sub" / "state.json"))
    s = EmotionState(recent_causes=[{"event": "x", "reason": "emoji 😂 intact"}])
    store.save(s)
    raw = (tmp_path / "sub" / "state.json").read_text(encoding="utf-8")
    assert "😂" in raw
    assert json.loads(raw)["recent_causes"][0]["reason"] == "emoji 😂 intact"


def test_unknown_version_gives_fresh_state():
    loaded = EmotionState.from_dict({"version": 2,
                                     "emotions": {"hurt": 0.9},
                                     "energy": 0.1})
    assert loaded.to_dict() == EmotionState().to_dict()


def test_corrupt_file_quarantined_then_fresh(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{not json", encoding="utf-8")
    loaded = StateStore(str(p)).load()
    assert loaded.to_dict() == EmotionState().to_dict()
    assert not p.exists()
    assert len(list(tmp_path.glob("quarantine-state-*.json"))) == 1


def test_invalid_schema_quarantined_then_fresh(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"version": 1, "emotions": {"loneliness": 0.5}}',
                 encoding="utf-8")
    loaded = StateStore(str(p)).load()
    assert loaded.to_dict() == EmotionState().to_dict()
    assert len(list(tmp_path.glob("quarantine-state-*.json"))) == 1


def test_mood_derivation():
    assert derive({"sadness": 0.7}, "content") == "low"
    assert derive({"hurt": 0.8}, "content") == "low"
    assert derive({"anger": 0.7}, "content") == "tense"
    assert derive({"affection": 0.7, "happiness": 0.6}, "content") == "tender"
    assert derive({"happiness": 0.8}, "content") == "bright"
    assert derive({"happiness": 0.5, "affection": 0.5}, "content") == "content"


def test_mood_hysteresis_keeps_background_state():
    # Just below trigger: stays in charged mood instead of flickering.
    assert derive({"sadness": 0.52}, "low") == "low"
    assert derive({"sadness": 0.52}, "content") == "content"
    # Clearly recovered: returns to content.
    assert derive({"sadness": 0.1, "hurt": 0.05}, "low") == "content"
