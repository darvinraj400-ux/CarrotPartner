"""Stage-2 tests: personality loading, conversation buffer, classifier."""
import pytest

from carrot.classifier import DeterministicClassifier, InteractionEvent
from carrot.context import derive_context
from carrot.conversation import ConversationBuffer, Turn
from carrot.personality import Personality


class StubSemanticClassifier:
    """Test-only stand-in proving the protocol is caller-agnostic."""

    def __init__(self, event: InteractionEvent):
        self._event = event
        self.seen = []

    def classify(self, text, context):
        self.seen.append((text, context))
        return self._event


def test_personality_load_and_sketch(tmp_path):
    p = tmp_path / "p.json"
    p.write_text('{"id": "t", "sensitivity": 0.9, "playfulness": 0.1,'
                 ' "stoicism": 0.8, "shyness": 0.2, "pride": 0.3,'
                 ' "warmth": 0.4}', encoding="utf-8")
    loaded = Personality.load(str(p))
    assert loaded.sensitivity == 0.9
    assert "stoic" in loaded.sketch()
    assert "0.9" not in loaded.sketch()  # never leak numbers
    bad = tmp_path / "bad.json"
    bad.write_text('{"sensitivity": 5}', encoding="utf-8")
    with pytest.raises(ValueError):
        Personality.load(str(bad))


def test_sketch_vocabulary_pinned():
    # Any sketch wording change must be a deliberate diff, not an
    # accident: pin the exact output for carrot-default-v1.
    assert Personality().sketch() == (
        "Carrot is playful, warm, expressive, outgoing, humble, "
        "sensitive.")


def _ctx(*texts):
    buf = ConversationBuffer()
    for t in texts:
        buf.add("user", t)
    return derive_context(buf.recent())


def test_buffer_verbatim_and_bounded():
    buf = ConversationBuffer(capacity=3)
    buf.add("user", "NO WAY 😭😭😭")
    buf.add("carrot", "ikr 😂")
    buf.add("user", "a")
    buf.add("user", "b")
    assert len(buf) == 3
    assert buf.recent()[0] == Turn(role="carrot", text="ikr 😂")
    with pytest.raises(ValueError):
        buf.add("system", "x")


def test_classifier_basic_types():
    clf = DeterministicClassifier()
    assert clf.classify("You're cute, thanks!", _ctx()).type == "compliment"
    assert clf.classify("You are useless.", _ctx()).type == "insult"
    assert clf.classify("Sorry about that.", _ctx()).type == "apology"
    assert clf.classify("I love you", _ctx()).type == "affection"
    assert clf.classify("I finally fixed that thing!", _ctx()).type == "achievement"
    assert clf.classify("that joke was funny lol", _ctx()).type == "joke"
    assert clf.classify("   ", _ctx()).type == "neutral"


def test_context_disambiguates_tease_vs_insult():
    clf = DeterministicClassifier()
    cold = clf.classify("You're useless.", _ctx("what is 2+2?"))
    assert cold.type == "insult"
    playful = clf.classify("You're useless 😂",
                           _ctx("that joke was funny lol", "haha nice"))
    assert playful.type == "joke"
    # Same text, cold history: stays an insult even with the emoji.
    assert clf.classify("You're useless 😂",
                        _ctx("what is 2+2?")).type == "insult"


def test_repair_needs_anchored_context():
    clf = DeterministicClassifier()
    anchored = clf.classify("I'm kidding.",
                            _ctx("You're useless 😂"))
    assert anchored.type == "repair"
    assert clf.classify("I'm kidding.", _ctx()).type == "neutral"


def test_intensity_boosters_and_emoji_preserved():
    clf = DeterministicClassifier()
    calm = clf.classify("You are useless.", _ctx())
    loud = clf.classify("You are USELESS!!!", _ctx())
    assert loud.intensity > calm.intensity
    assert 0.2 <= loud.intensity <= 1.0


def _frame(playful: bool):
    buf = ConversationBuffer()
    buf.add("user", "lol that was funny" if playful else "the report is due")
    return derive_context(buf.recent())


def test_hedge_needs_recent_harshness():
    clf = DeterministicClassifier()
    assert clf.classify("I'm kidding.",
                        _ctx("you are useless")).type == "repair"
    assert clf.classify("I'm kidding.",
                        _ctx("what is 2+2?")).type == "neutral"


def test_playful_frame_splits_tease_vs_insult():
    clf = DeterministicClassifier()
    assert clf.classify("You're useless \U0001F602",
                        _frame(True)).type == "joke"
    assert clf.classify("You're useless \U0001F602",
                        _frame(False)).type == "insult"


def test_carrot_only_play_cue_does_not_open_frame():
    clf = DeterministicClassifier()
    buf = ConversationBuffer()
    buf.add("user", "what is 2+2?")
    buf.add("carrot", "haha nice \U0001F602")
    ctx = derive_context(buf.recent())
    assert ctx.playful_frame is False
    assert clf.classify("You're useless \U0001F602", ctx).type == "insult"


def test_classifier_protocol_conformance():
    import inspect

    from carrot.classifier import InteractionClassifier

    for cls in (DeterministicClassifier, StubSemanticClassifier):
        assert callable(getattr(cls, "classify", None))
        params = list(inspect.signature(cls.classify).parameters)
        assert params[1:] == ["text", "context"], params
    assert InteractionClassifier is not None
