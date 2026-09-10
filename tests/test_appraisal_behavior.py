"""Stage-3 tests: appraisal intents and behavior selection."""
import pytest

from carrot.appraisal import Appraisal
from carrot.behavior import BEHAVIORS, BehaviorSelector
from carrot.classifier import InteractionEvent
from carrot.emotions import EmotionState
from carrot.personality import Personality

TABLE = {
    "insult": {"hurt": 0.25, "anger": 0.18},
    "apology": {"hurt": -0.2, "anger": -0.22},
    "neutral": {},
}

OPEN = Personality(id="open", sensitivity=0.2, playfulness=0.7,
                   stoicism=0.1, shyness=0.2, pride=0.4, warmth=0.7)
STOIC = Personality(id="stoic", sensitivity=0.9, playfulness=0.3,
                    stoicism=0.9, shyness=0.5, pride=0.5, warmth=0.4)


def test_no_worried_behavior():
    assert "worried" not in BEHAVIORS


def test_appraisal_scales_with_intensity_only():
    ap = Appraisal(TABLE)
    event = InteractionEvent("apology", 0.5, "explicit apology cue")
    intent = ap.appraise(event, OPEN)
    assert intent.deltas["hurt"] == pytest.approx(-0.1)
    assert intent.reason == "explicit apology cue"
    # Appraisal must not touch state itself.
    s = EmotionState(emotions={"hurt": 0.6})
    assert s.emotions["hurt"] == 0.6


def test_sensitivity_gate_only_touches_insult_hurt():
    ap = Appraisal(TABLE)
    event = InteractionEvent("insult", 1.0, "hostile cue")
    thin = ap.appraise(event, STOIC).deltas  # sensitivity 0.9
    thick = ap.appraise(event, OPEN).deltas  # sensitivity 0.2
    assert thin["hurt"] > thick["hurt"]
    assert thin["anger"] == thick["anger"]  # anger ungated


def test_same_state_diverges_by_personality():
    sel = BehaviorSelector({})
    s = EmotionState(emotions={"hurt": 0.65, "anger": 0.4})
    assert sel.pick(s, STOIC).name == "sulking"
    assert sel.pick(s, OPEN).name == "sad"


def test_behavior_paths():
    sel = BehaviorSelector({})
    p = Personality()
    assert sel.pick(EmotionState(emotions={"anger": 0.7}), p).name == "annoyed"
    assert sel.pick(EmotionState(emotions={"sadness": 0.7}), p).name == "sad"
    shy = Personality(shyness=0.8)
    assert sel.pick(EmotionState(emotions={"embarrassment": 0.7}),
                    shy).name == "shy"
    assert sel.pick(EmotionState(emotions={"embarrassment": 0.7}),
                    p).name == "embarrassed"
    assert sel.pick(EmotionState(energy=0.1), p).name == "quiet"
    proud = Personality(pride=0.8)
    s = EmotionState(emotions={"happiness": 0.6},
                     recent_causes=[{"event": "achievement"}])
    assert sel.pick(s, proud).name == "proud"
    q = sel.pick(EmotionState(), p, user_text="are you there?")
    assert q.name == "curious"
    # Carrot's own trailing "?" must never self-trigger curiosity.
    own = sel.pick(EmotionState(), p,
                   context=[{"role": "carrot", "text": "yeah?"}],
                   user_text="anyway.")
    assert own.name == "normal"
    assert sel.pick(EmotionState(), p).name == "normal"
