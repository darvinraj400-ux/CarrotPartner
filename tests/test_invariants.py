"""Invariants: stdlib-only runtime, bounded growth, LLM-output isolation."""
import ast
import re
import sys
from pathlib import Path

from tests.conftest import ROOT
from tests.test_scenarios import make_carrot


def test_runtime_is_stdlib_only():
    allowed = set(sys.stdlib_module_names) | {"carrot"}
    offenders = []
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] not in allowed:
                        offenders.append((path.name, a.name))
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if node.level == 0 and mod not in allowed:
                    offenders.append((path.name, node.module))
    assert offenders == []


def test_state_does_not_grow_without_bound(tmp_path):
    c = make_carrot()
    lines = ["hello there?", "You're cute!", "that joke was funny lol",
             "I finally fixed that thing!", "Fine \U0001F642", "anyway."]
    for i in range(30):
        c.process_turn(lines[i % len(lines)])
        c.clock.advance(10)
    assert len(c.state.recent_causes) <= 15
    p = tmp_path / "state.json"
    p.write_text(__import__("json").dumps(c.state.to_dict()),
                 encoding="utf-8")
    assert p.stat().st_size < 8192


def test_llm_output_never_enters_state():
    a, b = make_carrot(), make_carrot()
    a.llm.generate = lambda *, system, history, user: "Fine."
    b.llm.generate = lambda *, system, history, user: (
        "My happiness is 0.99 and anger 0.99, whatever.")
    script = ["You're cute!", "You are useless.", "Sorry."]
    for line in script:
        a.process_turn(line)
        a.clock.advance(5)
        b.process_turn(line)
        b.clock.advance(5)
    assert a.state.to_dict() == b.state.to_dict()


def test_values_always_clamped_after_fuzz():
    c = make_carrot()
    script = ["You are USELESS!!!", "Sorry sorry sorry", "You're amazing!!",
              "I love you", "I finally fixed that thing!", "blah blah blah",
              "You're useless \U0001F602", "I'm kidding."]
    for i in range(24):
        c.process_turn(script[i % len(script)])
        c.clock.advance(3)
        for v in list(c.state.emotions.values()) + [c.state.energy]:
            assert 0.0 <= v <= 1.0


def _system_after(*lines):
    c = make_carrot()
    for line in lines:
        c.process_turn(line)
        c.clock.advance(5)
    return c.llm.last_system


def test_hint_pending_present():
    system = _system_after("are you awake?", "yes, I'm here")
    assert "answering your earlier question" in system


def test_hint_pending_absent_filler():
    system = _system_after("just stopping by.", "anyway.", "noted.")
    assert "answering your earlier question" not in system


def test_hint_playful_present_and_absent():
    assert "The conversation has been playful." in _system_after(
        "that joke was funny lol", "tell me more")
    assert "The conversation has been playful." not in _system_after(
        "the report is due.")


def test_hint_repair_present_and_absent():
    assert "apologizing for something they said" in _system_after(
        "you are useless.", "I'm kidding.")
    assert "apologizing for something they said" not in _system_after(
        "you are useless.", "whatever.")


def test_hint_topics_present_alphabetic_only():
    system = _system_after("gardening club playoff", "tell me more")
    assert "Recent topics: gardening, club, playoff" in system
    tail = system.split("Recent topics: ", 1)[1]
    assert re.fullmatch(r"[A-Za-z, ]+", tail)


def test_hint_block_omitted_when_nothing_holds():
    system = _system_after("12345 !!!")
    assert "Recent topics:" not in system
    assert "Context:" not in system


def test_no_digits_across_hint_scenarios():
    scripts = [
        ("are you awake?", "yes, I'm here"),
        ("that joke was funny lol", "tell me more"),
        ("you are useless.", "I'm kidding."),
        ("gardening club playoff", "tell me more"),
    ]
    for script in scripts:
        assert re.search(r"\d", _system_after(*script)) is None, script


def test_no_state_names_in_system():
    system = _system_after(
        "are you awake?", "yes, I'm here", "that joke was funny lol",
        "you are useless.", "I'm kidding.")
    assert "Context:" in system  # several hints render at once
    lowered = system.lower()
    emotions = ["happiness", "sadness", "anger", "hurt", "affection",
                "excitement", "embarrassment"]
    moods = ["content", "low", "tender", "tense", "bright"]
    traits = ["sensitivity", "playfulness", "stoicism", "shyness",
              "pride", "warmth"]
    for word in emotions + moods + traits + ["mood"]:
        assert re.search(rf"\b{word}\b", lowered) is None, word
    # Note: the frozen Slice-1 base prompt contains the generic word
    # "emotions" ("how your emotions work"); only specific state names
    # above are forbidden. Likewise "playful"/"warm" adjectives live in
    # the personality sketch by design; only noun trait names are barred.
