"""Slice 13 Phase-1 tests: voice protocols and mocks (pure, headless)."""
from carrot.voice import (
    VOICE_EMPTY_MSG,
    MockSpeechRecognizer,
    MockSpeechSynthesizer,
)


# ---- constant ----


def test_voice_empty_msg():
    assert VOICE_EMPTY_MSG == "(heard nothing)"


# ---- protocol conformance (duck-typed) ----


def test_recognizer_has_listen():
    rec = MockSpeechRecognizer(["hi"])
    assert callable(rec.listen)
    assert isinstance(rec.listen(), str)


def test_synthesizer_has_speak():
    synth = MockSpeechSynthesizer()
    assert callable(synth.speak)
    assert synth.speak("hi") is None


def test_synthesizer_returns_none():
    assert MockSpeechSynthesizer().speak("hi") is None


# ---- MockSpeechRecognizer ----


def test_recognizer_empty_default_queue():
    assert MockSpeechRecognizer().listen() == ""


def test_recognizer_single_transcript():
    assert MockSpeechRecognizer(["hello"]).listen() == "hello"


def test_recognizer_multiple_transcripts():
    rec = MockSpeechRecognizer(["a", "b", "c"])
    assert [rec.listen(), rec.listen(), rec.listen()] == ["a", "b", "c"]


def test_recognizer_exhausted_returns_empty():
    rec = MockSpeechRecognizer(["only"])
    assert rec.listen() == "only"
    assert rec.listen() == ""


def test_recognizer_listen_count():
    rec = MockSpeechRecognizer(["a"])
    rec.listen()
    rec.listen()
    assert rec.listen_count == 2


def test_recognizer_non_string_coerced():
    assert MockSpeechRecognizer([123]).listen() == "123"


def test_recognizer_none_coalesced_to_empty():
    # A null transcript means "nothing recognized" — same as an empty
    # queue, so the caller's empty-guard handles both uniformly.
    assert MockSpeechRecognizer([None]).listen() == ""


# ---- MockSpeechSynthesizer ----


def test_synthesizer_spoken_starts_empty():
    assert MockSpeechSynthesizer().spoken == []


def test_synthesizer_speak_appends():
    synth = MockSpeechSynthesizer()
    synth.speak("hi")
    assert synth.spoken == ["hi"]


def test_synthesizer_preserves_order():
    synth = MockSpeechSynthesizer()
    synth.speak("a")
    synth.speak("b")
    assert synth.spoken == ["a", "b"]


def test_synthesizer_none_coerced():
    synth = MockSpeechSynthesizer()
    synth.speak(None)
    assert synth.spoken == [""]


def test_synthesizer_returns_none():
    assert MockSpeechSynthesizer().speak("hi") is None


# ---- determinism ----


def test_voice_mocks_deterministic():
    first = MockSpeechRecognizer(["x", "y"])
    second = MockSpeechRecognizer(["x", "y"])
    assert [first.listen(), first.listen()] == \
        [second.listen(), second.listen()]
    assert first.listen_count == second.listen_count == 2
