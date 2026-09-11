"""Slice 13 Phase-1 tests: voice protocols and mocks (pure, headless)."""
from carrot.cli import run_voice_turn
from carrot.voice import (
    VOICE_EMPTY_MSG,
    MockSpeechRecognizer,
    MockSpeechSynthesizer,
)
from tests.test_scenarios import make_carrot


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


# --- Slice 13 Phase 2: run_voice_turn tests ---


def test_run_voice_turn_empty_returns_pair():
    c = make_carrot()
    synth = MockSpeechSynthesizer()
    assert run_voice_turn(c, MockSpeechRecognizer([]),
                          synth) == ("", "")
    assert synth.spoken == []
    assert len(c.buffer) == 0


def test_run_voice_turn_whitespace_returns_pair():
    c = make_carrot()
    synth = MockSpeechSynthesizer()
    assert run_voice_turn(c, MockSpeechRecognizer(["   "]),
                          synth) == ("", "")
    assert synth.spoken == []
    assert len(c.buffer) == 0


def test_run_voice_turn_valid_returns_pair():
    c = make_carrot()
    transcript, reply = run_voice_turn(c,
                                       MockSpeechRecognizer(["hello"]),
                                       MockSpeechSynthesizer())
    assert transcript == "hello"
    assert isinstance(reply, str) and reply


def test_run_voice_turn_speaks_reply():
    c = make_carrot()
    synth = MockSpeechSynthesizer()
    _, reply = run_voice_turn(c, MockSpeechRecognizer(["hello"]), synth)
    assert synth.spoken == [reply]


def test_run_voice_turn_speak_failure_swallowed():
    class LoudSynth:
        def speak(self, text):
            raise RuntimeError("speaker unplugged")

    c = make_carrot()
    transcript, reply = run_voice_turn(c, MockSpeechRecognizer(["hello"]),
                                       LoudSynth())
    assert transcript == "hello"
    assert isinstance(reply, str) and reply


def test_run_voice_turn_saves_nothing_itself(tmp_path):
    from carrot.state_store import StateStore

    c = make_carrot()
    path = tmp_path / "state.json"
    c.store = StateStore(str(path))
    run_voice_turn(c, MockSpeechRecognizer(["hello"]),
                   MockSpeechSynthesizer())
    assert not path.exists()


def test_run_voice_turn_deterministic():
    first = make_carrot()
    second = make_carrot()
    one = run_voice_turn(first, MockSpeechRecognizer(["hello"]),
                         MockSpeechSynthesizer())
    two = run_voice_turn(second, MockSpeechRecognizer(["hello"]),
                         MockSpeechSynthesizer())
    assert one == two
