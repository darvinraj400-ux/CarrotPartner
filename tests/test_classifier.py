"""Stage-2 tests: personality loading, conversation buffer, classifier."""
import pytest

from carrot.classifier import DeterministicClassifier
from carrot.conversation import ConversationBuffer, Turn
from carrot.personality import Personality


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


def _ctx(*texts):
    buf = ConversationBuffer()
    for t in texts:
        buf.add("user", t)
    return buf.recent()


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
    assert clf.classify("You're cute, thanks!", []).type == "compliment"
    assert clf.classify("You are useless.", []).type == "insult"
    assert clf.classify("Sorry about that.", []).type == "apology"
    assert clf.classify("I love you", []).type == "affection"
    assert clf.classify("I finally fixed that thing!", []).type == "achievement"
    assert clf.classify("that joke was funny lol", []).type == "joke"
    assert clf.classify("   ", []).type == "neutral"


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
    assert clf.classify("I'm kidding.", []).type == "neutral"


def test_intensity_boosters_and_emoji_preserved():
    clf = DeterministicClassifier()
    calm = clf.classify("You are useless.", [])
    loud = clf.classify("You are USELESS!!!", [])
    assert loud.intensity > calm.intensity
    assert 0.2 <= loud.intensity <= 1.0
