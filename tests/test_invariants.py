"""Invariants: stdlib-only runtime, bounded growth, LLM-output isolation."""
import ast
import re
import sys
from pathlib import Path

from tests.conftest import ROOT
from tests.test_scenarios import make_carrot
from carrot.drives import DriveState, DriveStore
from carrot.environment import MockEnvironmentProbe, RawObservation


def _seeded_drives(tmp_path, social, novelty, name="d.json"):
    store = DriveStore(tmp_path / name)
    store.save(DriveState(social=social, novelty=novelty))
    return store


def _drive_carrot(tmp_path, social, novelty, name="d.json"):
    return make_carrot(
        drive_store=_seeded_drives(tmp_path, social, novelty, name))


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


def test_no_autonomy_primitives():
    """Pins the Slice 5/9 boundary before Slice 5 code exists: no
    schedulers, threads, processes, timers, or sleeps anywhere in
    runtime code. Any future drift fails loudly here."""
    banned = {"threading", "asyncio", "sched", "multiprocessing", "Timer"}
    offenders = []
    sleepers = []
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in banned:
                        offenders.append((path.name, a.name))
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if node.level == 0 and mod in banned:
                    offenders.append((path.name, node.module))
                for a in node.names:
                    if a.name in banned:
                        offenders.append((path.name, a.name))
            elif isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute)
                        and func.attr == "sleep"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "time"):
                    sleepers.append(path.name)
    assert offenders == []
    assert sleepers == []


# Slice 6 boundary: perception is read-only. `ctypes` is permitted only in
# `carrot/environment_win32.py` (the Windows probe, read-only use only);
# no other module may import it or call WinDLL/CDLL/windll.
# The accessor ban above is a *permission* check (who may reach the DLLs).
# The write-API ban below is an *action* check: no module, including the
# sanctioned one, may call a known write primitive. Together they make the
# test name honest. Full write-protection is Slice 8's job; this is a
# static tripwire, not a comprehensive policy.
_WRITE_APIS = frozenset({
    "SetForegroundWindow", "SendInput", "keybd_event", "mouse_event",
    "CreateFileW", "WriteFile", "ShellExecuteW", "ShellExecuteExW",
    "CreateProcessW", "TerminateProcess", "RegSetValueExW",
    "SystemParametersInfoW",
})


def _find_write_calls(tree):
    """Attribute names from the write-API ban list, whatever receiver."""
    return [node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in _WRITE_APIS]


def test_no_os_write_primitives():
    importers = []
    callers = []
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "ctypes":
                        importers.append(path.name)
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if node.level == 0 and mod == "ctypes":
                    importers.append(path.name)
                for a in node.names:
                    if a.name == "ctypes":
                        importers.append(path.name)
            elif isinstance(node, ast.Attribute):
                if node.attr in ("windll", "CDLL", "WinDLL"):
                    value = node.value
                    if isinstance(value, ast.Name) and value.id == "ctypes":
                        callers.append(path.name)
    assert all(name == "environment_win32.py" for name in callers), \
        callers
    writes = []
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        writes.extend(
            (path.name, hit) for hit in _find_write_calls(tree))
    assert writes == []


def test_write_api_ban_has_teeth():
    bad = ast.parse("import ctypes\nuser32.SetForegroundWindow(0)\n")
    assert _find_write_calls(bad) == ["SetForegroundWindow"]
    good = ast.parse("hwnd = user32.GetForegroundWindow()\n")
    assert _find_write_calls(good) == []


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


def _skeleton(system):
    # R1: quoted memory spans may contain anything; the invariant
    # applies to the unquoted skeleton only.
    return re.sub(r'"[^"]*"', "", system)


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
    assert "walking back what they said" in _system_after(
        "you are useless.", "I'm kidding.")
    assert "walking back what they said" not in _system_after(
        "you are useless.", "whatever.")


def test_repair_hint_is_backpedal_not_apology():
    system = _system_after("you are useless.", "I'm kidding.")
    assert "walking back" in system
    assert "apologizing" not in system


def test_plain_apology_gets_no_repair_hint():
    system = _system_after("you are useless.", "I'm sorry.")
    assert "walking back" not in system
    assert "apologizing" not in system
    # Topics still render; only the repair sentence is absent.
    assert "Recent topics: useless" in system


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
    # R1 scope change: digits may appear ONLY inside quoted memory
    # spans, so the skeleton (quotes stripped) must be digit-free.
    scripts = [
        ("are you awake?", "yes, I'm here"),
        ("that joke was funny lol", "tell me more"),
        ("you are useless.", "I'm kidding."),
        ("gardening club playoff", "tell me more"),
    ]
    for script in scripts:
        assert re.search(r"\d", _skeleton(
            _system_after(*script))) is None, script


def test_no_state_names_in_system():
    system = _system_after(
        "are you awake?", "yes, I'm here", "that joke was funny lol",
        "you are useless.", "I'm kidding.")
    assert "Context:" in system  # several hints render at once
    lowered = _skeleton(system).lower()  # R1: skeleton only
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


def _skeleton(system):
    return re.sub(r'"[^"]*"', "", system)


def test_drive_sentence_present_social(tmp_path):
    c = _drive_carrot(tmp_path, 0.9, 0.0)
    c.process_turn("hello there")
    assert c.last_drive_pull == "social"
    assert "You've been wanting to talk to them." in c.llm.last_system


def test_drive_sentence_present_novelty(tmp_path):
    c = _drive_carrot(tmp_path, 0.0, 0.9)
    c.process_turn("hello there")
    assert c.last_drive_pull == "novelty"
    assert "You've been wanting something new to happen." \
        in c.llm.last_system


def test_drive_sentence_absent_no_pull(tmp_path):
    c = _drive_carrot(tmp_path, 0.0, 0.0)
    c.process_turn("12345 !!!")
    assert c.last_drive_pull is None
    assert "You've been wanting to talk to them." not in c.llm.last_system
    assert "You've been wanting something new to happen." \
        not in c.llm.last_system


def test_drive_sentence_appended_last(tmp_path):
    c = _drive_carrot(tmp_path, 0.9, 0.0)
    c.process_turn("lol that was funny")
    c.clock.advance(5)
    c.process_turn("are you awake?")
    c.clock.advance(5)
    c.process_turn("yes, I'm here")
    system = c.llm.last_system
    assert "Recent topics:" in system
    assert system.index("You've been wanting to talk to them.") > \
        system.index("Recent topics:")


def test_drive_only_block_no_double_space(tmp_path):
    c = _drive_carrot(tmp_path, 0.9, 0.0)
    c.process_turn("12345 !!!")
    system = c.llm.last_system
    assert system.endswith("Context: You've been wanting to talk to them.")
    assert "  " not in system


def test_drive_sentence_skeleton_clean(tmp_path):
    c = _drive_carrot(tmp_path, 0.9, 0.9)
    c.process_turn("hello there")
    skeleton = _skeleton(c.llm.last_system)
    assert re.search(r"\d", skeleton) is None
    lowered = skeleton.lower()
    for word in ["happiness", "sadness", "anger", "hurt", "affection",
                 "excitement", "embarrassment", "content", "low",
                 "tender", "tense", "bright", "sensitivity", "playfulness",
                 "stoicism", "shyness", "pride", "warmth", "mood"]:
        assert re.search(rf"\b{word}\b", lowered) is None, word


def test_drive_sentence_adversarial_max(tmp_path):
    c = _drive_carrot(tmp_path, 1.0, 1.0)
    c.process_turn("hello there")
    # Presence satisfaction (-0.01 social) breaks the tie toward novelty;
    # either way the sentence is a static string carrying no data.
    assert c.last_drive_pull == "novelty"
    assert "You've been wanting something new to happen." \
        in c.llm.last_system
    assert re.search(r"\d", _skeleton(c.llm.last_system)) is None


def test_drive_prompt_deterministic(tmp_path):
    first = _drive_carrot(tmp_path, 0.9, 0.2, "a.json")
    second = _drive_carrot(tmp_path, 0.9, 0.2, "b.json")
    first.process_turn("hello there")
    second.process_turn("hello there")
    assert first.llm.last_system == second.llm.last_system


def _env_carrot(tmp_path, observations, name="e.json", drive=(0.0, 0.0)):
    c = make_carrot(drive_store=_seeded_drives(tmp_path, *drive, name=name))
    c.environment_probe = MockEnvironmentProbe(observations)
    return c


def test_extreme_sketch_passes_leakage():
    from carrot.personality import Personality

    from tests.test_scenarios import make_carrot
    extreme = Personality(sensitivity=0.9, playfulness=0.9, stoicism=0.1,
                          shyness=0.1, pride=0.9, warmth=0.9, patience=0.1)
    c = make_carrot(personality=extreme)
    c.process_turn("hello there")
    system = c.llm.last_system
    assert "takes words to heart" in system
    assert re.search(r"\d", re.sub(r'"[^"]*"', "", system)) is None
    lowered = re.sub(r'"[^"]*"', "", system).lower()
    for word in ["happiness", "sadness", "anger", "hurt", "affection",
                 "excitement", "embarrassment", "content", "low",
                 "tender", "tense", "bright", "sensitivity", "playfulness",
                 "stoicism", "shyness", "pride", "warmth", "patience",
                 "mood"]:
        assert re.search(rf"\b{word}\b", lowered) is None, word


def _env_skeleton(system):
    return re.sub(r'"[^"]*"', "", system)


def test_env_app_sentence_present(tmp_path):
    c = _env_carrot(tmp_path, [RawObservation("cmd.exe", False)])
    c.process_turn("hello there")
    assert "They're in the terminal." in c.llm.last_system


def test_env_idle_sentence_present(tmp_path):
    c = _env_carrot(tmp_path, [RawObservation(None, True)])
    c.process_turn("hello there")
    assert "They've been away for a bit." in c.llm.last_system


def test_env_both_sentences_ordered(tmp_path):
    c = _env_carrot(tmp_path, [RawObservation("cmd.exe", True)])
    c.process_turn("hello there")
    system = c.llm.last_system
    assert system.index("They're in the terminal.") < \
        system.index("They've been away for a bit.")


def test_env_neither_present(tmp_path):
    fresh = make_carrot()
    fresh.process_turn("12345 !!!")
    assert "They're in the" not in fresh.llm.last_system
    assert "been away for a bit" not in fresh.llm.last_system


def test_env_order_last_after_drive(tmp_path):
    c = _env_carrot(tmp_path, [RawObservation("cmd.exe", False)],
                    drive=(0.9, 0.0))
    c.process_turn("lol that was funny")
    c.clock.advance(5)
    c.process_turn("are you awake?")
    c.clock.advance(5)
    c.process_turn("yes, I'm here")
    system = c.llm.last_system
    assert "Recent topics:" in system
    assert "You've been wanting to talk to them." in system
    assert system.index("They're in the terminal.") > \
        system.index("You've been wanting to talk to them.")
    assert system.index("They're in the terminal.") > \
        system.index("Recent topics:")


def test_env_adversarial_raw_renders_nothing(tmp_path):
    c = _env_carrot(tmp_path,
                    [RawObservation("Report Q3 anger_management.exe", False)])
    c.process_turn("hello there")
    assert c.last_environment.app_noun is None
    system = c.llm.last_system
    skeleton = _env_skeleton(system)
    assert re.search(r"\d", skeleton) is None
    assert re.search(r"\banger\b", skeleton.lower()) is None


def test_env_known_good_skeleton_clean(tmp_path):
    c = _env_carrot(tmp_path, [RawObservation("cmd.exe", False)])
    c.process_turn("hello there")
    skeleton = _env_skeleton(c.llm.last_system)
    assert re.search(r"\d", skeleton) is None
    assert re.search(r"\d\.\d", skeleton) is None


def test_env_static_strings_all_nouns(tmp_path):
    import json

    config = json.loads((ROOT / "config" / "environment.json").read_text(
        encoding="utf-8"))
    by_noun = {}
    for process, noun in config["allowed_apps"].items():
        by_noun.setdefault(noun, process)
    assert len(by_noun) >= 5
    for noun, process in sorted(by_noun.items()):
        c = _env_carrot(tmp_path, [RawObservation(process, False)],
                        name=f"{noun}.json")
        c.process_turn("12345 !!!")
        system = c.llm.last_system
        assert f"They're in the {noun}." in system
        assert process not in system
        assert re.search(r"\d", _env_skeleton(system)) is None


def test_env_no_raw_process_name(tmp_path):
    c = _env_carrot(tmp_path, [RawObservation("code.exe", False)])
    c.process_turn("hello there")
    system = c.llm.last_system
    assert "They're in the editor." in system
    assert "code.exe" not in system
    assert "CODE.EXE" not in system


def test_env_deterministic(tmp_path):
    first = _env_carrot(tmp_path, [RawObservation("cmd.exe", True)],
                        "a.json")
    second = _env_carrot(tmp_path, [RawObservation("cmd.exe", True)],
                         "b.json")
    first.process_turn("hello there")
    second.process_turn("hello there")
    assert first.llm.last_system == second.llm.last_system


def test_env_block_omitted_when_empty(tmp_path):
    c = make_carrot()
    c.process_turn("12345 !!!")
    assert "Context:" not in c.llm.last_system


def test_env_block_present_when_only_env(tmp_path):
    c = _env_carrot(tmp_path, [RawObservation("cmd.exe", False)])
    c.process_turn("12345 !!!")
    system = c.llm.last_system
    assert system.endswith("Context: They're in the terminal.")
