"""Slice 3 Phase-2 tests: deterministic extractor (+ stub protocol)."""
import inspect

from carrot.memory import DeterministicExtractor, Extractor, MemoryCandidate


class StubExtractor:
    """Test-only stand-in proving the Extractor protocol is pluggable."""

    def __init__(self, candidates=()):
        self._candidates = tuple(candidates)
        self.seen = []

    def extract(self, user_text):
        self.seen.append(user_text)
        return self._candidates


def _extract(text):
    return DeterministicExtractor().extract(text)


def _one(text):
    got = _extract(text)
    assert len(got) == 1
    return got[0]


def test_name_is_fact():
    assert _one("my name is Alice") == MemoryCandidate("my name is alice",
                                                       "fact")


def test_call_me_is_fact():
    assert _one("call me Al") == MemoryCandidate("call me al", "fact")


def test_birthday_preserves_digits():
    assert _one("my birthday is May 3") == MemoryCandidate(
        "my birthday is may 3", "fact")


def test_allergy_is_fact():
    assert _one("I'm allergic to peanuts") == MemoryCandidate(
        "i'm allergic to peanuts", "fact")


def test_love_is_preference():
    assert _one("I love pizza") == MemoryCandidate("i love pizza",
                                                   "preference")


def test_hate_is_preference():
    assert _one("I hate mondays") == MemoryCandidate("i hate mondays",
                                                     "preference")


def test_favorite_is_preference_not_generic_fact():
    assert _one("my favorite color is blue") == MemoryCandidate(
        "my favorite color is blue", "preference")


def test_remember_is_fact():
    assert _one("remember that my sister's wedding is in June") == \
        MemoryCandidate("remember that my sister's wedding is in june",
                        "fact")


def test_generic_possessive_is_fact():
    assert _one("my dog is funny") == MemoryCandidate("my dog is funny",
                                                      "fact")


def test_hostile_turn_rejected():
    assert _extract("you are useless") == ()


def test_hostile_inside_longer_turn_rejected():
    assert _extract("my useless brother yelled at me") == ()


def test_hate_you_beats_preference_pattern():
    assert _extract("I hate you") == ()


def test_hostile_stack_rejected():
    assert _extract("shut up stupid") == ()


def test_no_pattern_no_candidate():
    assert _extract("lol") == ()


def test_intra_turn_dedup():
    assert _extract("my name is Alice. My name is Alice.") == (
        MemoryCandidate("my name is alice", "fact"),)


def test_overlong_candidate_dropped():
    assert _extract("my name is " + "x" * 200) == ()


def test_trailing_punctuation_stripped():
    assert _one("My name is Alice.") == MemoryCandidate("my name is alice",
                                                        "fact")


def test_empty_string():
    assert _extract("") == ()
    assert _extract("   ") == ()


def test_stub_satisfies_protocol():
    for cls in (DeterministicExtractor, StubExtractor):
        assert callable(getattr(cls, "extract", None))
        assert list(inspect.signature(cls.extract).parameters) == \
            ["self", "user_text"]
    assert Extractor is not None
    stub = StubExtractor((MemoryCandidate("x", "fact"),))
    assert stub.extract("anything") == (MemoryCandidate("x", "fact"),)
    assert stub.seen == ["anything"]


def test_extractor_is_deterministic_and_pure():
    first = _extract("My name is Alice. I love pizza.")
    second = _extract("My name is Alice. I love pizza.")
    assert first == second == (MemoryCandidate("my name is alice", "fact"),
                               MemoryCandidate("i love pizza", "preference"))
