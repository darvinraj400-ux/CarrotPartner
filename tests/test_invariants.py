"""Invariants: stdlib-only runtime, bounded growth, LLM-output isolation."""
import ast
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
