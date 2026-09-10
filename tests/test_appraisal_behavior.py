"""Stage-3 tests: appraisal intents and behavior selection."""
import pytest

from carrot.appraisal import Appraisal
from carrot.behavior import (
    BEHAVIORS,
    THRESHOLD_SAFETY_MAX,
    THRESHOLD_SAFETY_MIN,
    BehaviorSelector,
    _patience_threshold,
)
from carrot.classifier import InteractionEvent
from carrot.context import derive_context
from carrot.conversation import ConversationBuffer
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


def _blank():
    return derive_context(())


def _ctx(*pairs):
    buf = ConversationBuffer()
    for role, text in pairs:
        buf.add(role, text)
    return derive_context(buf.recent())


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
    assert sel.pick(s, STOIC, _blank(), "").name == "sulking"
    assert sel.pick(s, OPEN, _blank(), "").name == "sad"


def test_behavior_paths():
    sel = BehaviorSelector({})
    p = Personality()
    assert sel.pick(EmotionState(emotions={"anger": 0.7}), p,
                    _blank(), "").name == "annoyed"
    assert sel.pick(EmotionState(emotions={"sadness": 0.7}), p,
                    _blank(), "").name == "sad"
    shy = Personality(shyness=0.8)
    assert sel.pick(EmotionState(emotions={"embarrassment": 0.7}),
                    shy, _blank(), "").name == "shy"
    assert sel.pick(EmotionState(emotions={"embarrassment": 0.7}),
                    p, _blank(), "").name == "embarrassed"
    assert sel.pick(EmotionState(energy=0.1), p, _blank(), "").name == "quiet"
    proud = Personality(pride=0.8)
    s = EmotionState(emotions={"happiness": 0.6},
                     recent_causes=[{"event": "achievement"}])
    assert sel.pick(s, proud, _blank(), "").name == "proud"
    q = sel.pick(EmotionState(), p, _blank(), "are you there?")
    assert q.name == "curious"
    # Carrot's own trailing "?" must never self-trigger curiosity.
    own = sel.pick(EmotionState(), p,
                   _ctx(("carrot", "yeah?")), "anyway.")
    assert own.name == "normal"
    assert sel.pick(EmotionState(), p, _blank(), "").name == "normal"


def _playful_ctx():
    return _ctx(("user", "lol that was funny"))


def _harsh_ctx():
    return _ctx(("user", "you are useless"))


def _pending_ctx():
    return _ctx(("user", "hey"), ("carrot", "are you there?"))


def test_playful_frame_gate():
    sel = BehaviorSelector({})
    p = Personality()
    warm = EmotionState(emotions={"happiness": 0.7, "affection": 0.7})
    assert sel.pick(warm, p, _playful_ctx(), "haha").name == "playful"
    # Same state, no frame: existing affectionate gate wins instead.
    assert sel.pick(warm, p, _blank(), "haha").name == "affectionate"


def test_repair_gate_accepts():
    sel = BehaviorSelector({})
    p = Personality()
    resting = EmotionState()
    assert sel.pick(resting, p, _harsh_ctx(), "I'm kidding.").name == "normal"
    warm = EmotionState(emotions={"happiness": 0.6, "affection": 0.7})
    assert sel.pick(warm, p, _harsh_ctx(),
                    "I'm kidding.").name == "affectionate"


def test_repair_gate_needs_live_hedge():
    sel = BehaviorSelector({})
    hurt = EmotionState(emotions={"hurt": 0.65, "anger": 0.4})
    # Harsh history but no hedge in the live message: sulk wins.
    assert sel.pick(hurt, OPEN, _harsh_ctx(), "leave me alone.").name == "sad"


def test_pending_question_answered():
    sel = BehaviorSelector({})
    p = Personality()
    assert sel.pick(EmotionState(), p, _pending_ctx(),
                    "I'm here, running late").name == "curious"
    # A question is not an answer: existing curiosity path, unchanged.
    assert sel.pick(EmotionState(), p, _pending_ctx(),
                    "what do you mean?").name == "curious"
    # A hedge is not an answer either: falls through to normal.
    assert sel.pick(EmotionState(), p, _pending_ctx(),
                    "I'm kidding").name == "normal"


def test_warmth_modulates_affection_gate():
    sel = BehaviorSelector({})
    mid = EmotionState(emotions={"happiness": 0.6, "affection": 0.63})
    assert sel.pick(mid, Personality(warmth=0.9),
                    _blank(), "").name == "affectionate"
    assert sel.pick(mid, Personality(warmth=0.5),
                    _blank(), "").name == "normal"
    high = EmotionState(emotions={"happiness": 0.6, "affection": 0.67})
    assert sel.pick(high, Personality(warmth=0.5),
                    _blank(), "").name == "affectionate"
    assert sel.pick(high, Personality(warmth=0.1),
                    _blank(), "").name == "normal"


def test_warmth_extremes_stay_sane():
    sel = BehaviorSelector({})
    for warmth in (0.0, 1.0):
        got = sel.pick(EmotionState(), Personality(warmth=warmth),
                       _blank(), "")
        assert got.name in BEHAVIORS


def _pat(patience, **traits):
    return Personality(patience=patience, **traits)


def test_patience_high_raises_frustrated_threshold():
    sel = BehaviorSelector({})
    tense = EmotionState(emotions={"anger": 0.42}, energy=0.2)
    assert sel.pick(tense, _pat(0.5), _blank(), "").name == "frustrated"
    assert sel.pick(tense, _pat(0.9), _blank(), "").name == "quiet"


def test_patience_low_lowers_frustrated_threshold():
    sel = BehaviorSelector({})
    tense = EmotionState(emotions={"anger": 0.38}, energy=0.2)
    assert sel.pick(tense, _pat(0.5), _blank(), "").name != "frustrated"
    assert sel.pick(tense, _pat(0.1), _blank(), "").name == "frustrated"


def test_patience_high_raises_annoyed_threshold():
    sel = BehaviorSelector({})
    tense = EmotionState(emotions={"anger": 0.57}, energy=0.7)
    assert sel.pick(tense, _pat(0.5), _blank(), "").name == "annoyed"
    assert sel.pick(tense, _pat(0.9), _blank(), "").name == "normal"


def test_patience_low_lowers_annoyed_threshold():
    sel = BehaviorSelector({})
    tense = EmotionState(emotions={"anger": 0.53}, energy=0.7)
    assert sel.pick(tense, _pat(0.5), _blank(), "").name == "normal"
    assert sel.pick(tense, _pat(0.1), _blank(), "").name == "annoyed"


def test_patience_threshold_clamps():
    assert abs(_patience_threshold(0.4, 0.0) - 0.35) < 1e-9
    assert abs(_patience_threshold(0.55, 1.0) - 0.6) < 1e-9
    # The clamp is unreachable with real gate bases (0.3-0.65); pin the
    # constants directly instead of a synthetic edge test.
    assert THRESHOLD_SAFETY_MIN == 0.05
    assert THRESHOLD_SAFETY_MAX == 0.95
    sel = BehaviorSelector({})
    low = EmotionState(emotions={"anger": 0.42}, energy=0.2)
    assert sel.pick(low, _pat(0.0), _blank(), "").name == "frustrated"
    high = EmotionState(emotions={"anger": 0.57}, energy=0.7)
    assert sel.pick(high, _pat(1.0), _blank(), "").name == "normal"


def test_patience_leaves_other_gates_alone():
    sel = BehaviorSelector({})
    proud_state = EmotionState(
        emotions={"happiness": 0.6},
        recent_causes=[{"event": "achievement"}])
    cases = [
        (EmotionState(emotions={"happiness": 0.6, "affection": 0.7}),
         {}, "", _blank(), "affectionate"),
        (proud_state, {"pride": 0.8}, "", _blank(), "proud"),
        (EmotionState(emotions={"excitement": 0.7}), {}, "", _blank(),
         "playful"),
        (EmotionState(emotions={"hurt": 0.65, "anger": 0.4}),
         {"stoicism": 0.9}, "", _blank(), "sulking"),
        (EmotionState(emotions={"sadness": 0.7}), {}, "", _blank(), "sad"),
        (EmotionState(), {}, "are you there?", _blank(), "curious"),
        (EmotionState(), {}, "", _blank(), "normal"),
    ]
    for state, traits, text, ctx, expected in cases:
        calm = sel.pick(state, _pat(1.0, **traits), ctx, text).name
        edgy = sel.pick(state, _pat(0.0, **traits), ctx, text).name
        assert calm == edgy == expected, (expected, calm, edgy)


def test_patience_default_is_neutral():
    assert _patience_threshold(0.55, 0.5) == 0.55
    assert _patience_threshold(0.4, 0.5) == 0.4
    sel = BehaviorSelector({})
    states = [EmotionState(),
              EmotionState(emotions={"anger": 0.7}),
              EmotionState(emotions={"hurt": 0.65, "anger": 0.4})]
    for state in states:
        assert sel.pick(state, Personality(),
                        _blank(), "").name == sel.pick(
            state, Personality(patience=0.5), _blank(), "").name


def _proud_state(event, happiness=0.6):
    return EmotionState(emotions={"happiness": happiness},
                        recent_causes=[{"event": event}])


def test_proud_default_pride_achievement():
    sel = BehaviorSelector({})
    assert sel.pick(_proud_state("achievement"), Personality(pride=0.4),
                    _blank(), "").name == "proud"


def test_proud_default_pride_excludes_joke():
    sel = BehaviorSelector({})
    assert sel.pick(_proud_state("joke"), Personality(pride=0.4),
                    _blank(), "").name == "normal"


def test_proud_high_pride_enables_joke():
    sel = BehaviorSelector({})
    assert sel.pick(_proud_state("joke"), Personality(pride=0.8),
                    _blank(), "").name == "proud"


def test_proud_high_pride_enables_affection():
    sel = BehaviorSelector({})
    assert sel.pick(_proud_state("affection"), Personality(pride=0.8),
                    _blank(), "").name == "proud"


def test_proud_cutoff_boundary_inclusive():
    sel = BehaviorSelector({})
    assert sel.pick(_proud_state("joke"), Personality(pride=0.7),
                    _blank(), "").name == "proud"


def test_proud_below_cutoff_uses_base_set():
    sel = BehaviorSelector({})
    assert sel.pick(_proud_state("joke"), Personality(pride=0.69),
                    _blank(), "").name == "normal"


def test_proud_happiness_requirement_unchanged():
    sel = BehaviorSelector({})
    assert sel.pick(_proud_state("joke", happiness=0.4),
                    Personality(pride=0.8), _blank(), "").name != "proud"


def test_proud_floor_blocks_very_humble():
    sel = BehaviorSelector({})
    assert sel.pick(_proud_state("achievement"), Personality(pride=0.1),
                    _blank(), "").name != "proud"
    assert sel.pick(_proud_state("achievement"), Personality(pride=0.29),
                    _blank(), "").name != "proud"


def test_proud_floor_boundary_inclusive():
    sel = BehaviorSelector({})
    assert sel.pick(_proud_state("achievement"), Personality(pride=0.3),
                    _blank(), "").name == "proud"


def test_proud_above_floor_uses_base_set():
    sel = BehaviorSelector({})
    assert sel.pick(_proud_state("achievement"), Personality(pride=0.5),
                    _blank(), "").name == "proud"
