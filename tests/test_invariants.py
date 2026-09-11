"""Invariants: stdlib-only runtime, bounded growth, LLM-output isolation."""
import ast
import json
import re
import sys
from pathlib import Path

import pytest

from tests.conftest import ROOT
from tests.test_scenarios import make_carrot
from carrot.carrot import (_FUSION_SENTENCE, _RELATIONSHIP_HIGH_SENTENCE,
                             _TOOLS_BLOCK, _context_hint, _episodes_block)
from carrot.behavior import Behavior
from carrot.context import ConversationContext
from carrot.drives import DriveState, DriveStore
from carrot.environment import (EMPTY_SNAPSHOT, EnvironmentSnapshot,
                                MockEnvironmentProbe, RawObservation)
from carrot.episodes import Episode
from carrot.memory import Memory
from carrot.relationship import RelationshipState, RelationshipStore
from carrot.personality import Personality
from carrot.tools import ToolRequest, ToolsConfig, decide


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
# `environment_win32` itself may be imported only in `cli.py`, lazily
# inside an `if ... win32 ...` branch (checked below); every other
# importer — guarded or not — fails.
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


def _is_env_win32_import(node) -> bool:
    if isinstance(node, ast.Import):
        return any(name.name.split(".")[-1] == "environment_win32"
                   for name in node.names)
    if isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        if node.level > 0:
            return mod == "environment_win32"
        return mod.split(".")[-1] == "environment_win32"
    return False


def _env_win32_imports(tree):
    """Split environment_win32 imports into (guarded, unguarded) linenos.

    Guarded means nested inside an `if` whose condition mentions win32.
    """
    guarded, unguarded = [], []

    def walk(node, inside_win32_branch):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                try:
                    cond = ast.unparse(child.test)
                except Exception:
                    cond = ""
                walk(child, inside_win32_branch or "win32" in cond)
            else:
                if _is_env_win32_import(child):
                    (guarded if inside_win32_branch else unguarded).append(
                        child.lineno)
                walk(child, inside_win32_branch)

    walk(tree, False)
    return guarded, unguarded


# Slice 9 boundary: exactly one unbounded loop is permitted, the CLI REPL.
# Away-mode actuation is bounded (max_n iterations, then exit). Any
# second while-True is a scheduler in disguise.
def test_no_unbounded_loops():
    offenders = []
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack = [(tree, None)]
        while stack:
            node, func = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = node.name
            if isinstance(node, ast.While):
                test = node.test
                is_true = isinstance(test, ast.Constant) and test.value in (
                    True, 1)
                if is_true and not (path.name == "cli.py"
                                    and func == "main"):
                    offenders.append((path.name, func, node.lineno))
            for child in ast.iter_child_nodes(node):
                stack.append((child, func))
    assert offenders == []


def test_initiative_module_is_pure():
    path = ROOT / "carrot" / "initiative.py"
    if not path.exists():
        pytest.skip("initiative.py lands in Phase 1")
    offenders = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == "carrot" and (
                        len(parts) < 2
                        or parts[1] not in ("drives", "environment")):
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".")
            if node.level == 0:
                if parts[0] == "carrot" and (
                        len(parts) < 2
                        or parts[1] not in ("drives", "environment")):
                    offenders.append(mod)
            elif mod not in ("drives", "environment"):
                offenders.append("." + mod if mod else ".")
    assert offenders == []


def test_generate_callers_restricted():
    offenders = []
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack = [(tree, None)]
        while stack:
            node, func = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = node.name
            if isinstance(node, ast.Call):
                called = node.func
                if isinstance(called, ast.Attribute) \
                        and called.attr == "generate":
                    if path.name not in ("carrot.py", "cli.py"):
                        offenders.append((path.name, func))
                    elif path.name == "cli.py" and func != "_run_away_mode":
                        offenders.append((path.name, func))
            for child in ast.iter_child_nodes(node):
                stack.append((child, func))
    assert offenders == []


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


# Slice 7 boundary: shelling out is permitted only inside carrot/tools.py
# (the sanctioned tool-execution module). Everywhere else is banned.
# webbrowser is intentionally not on this list — it is the sanctioned
# URL-launch mechanism and handles process spawning internally.
def test_no_subprocess_outside_tools():
    offenders = []
    for path in (ROOT / "carrot").glob("*.py"):
        if path.name == "tools.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "subprocess":
                        offenders.append((path.name, a.name))
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if node.level == 0 and mod == "subprocess":
                    offenders.append((path.name, node.module))
                if node.level == 0 and mod == "os":
                    for a in node.names:
                        if a.name in ("system", "popen", "startfile") \
                                or a.name.startswith(("exec", "spawn")):
                            offenders.append(
                                (path.name, "os." + a.name))
            elif isinstance(node, ast.Attribute):
                value = node.value
                if isinstance(value, ast.Name) and value.id == "os":
                    if node.attr in ("system", "popen", "startfile") \
                            or node.attr.startswith(("exec", "spawn")):
                        offenders.append(
                            (path.name, "os." + node.attr))
    assert offenders == []
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        guarded, unguarded = _env_win32_imports(tree)
        assert unguarded == [], (path.name, unguarded)
        if guarded:
            assert path.name == "cli.py", (path.name, guarded)


def test_write_api_ban_has_teeth():
    bad = ast.parse("import ctypes\nuser32.SetForegroundWindow(0)\n")
    assert _find_write_calls(bad) == ["SetForegroundWindow"]
    good = ast.parse("hwnd = user32.GetForegroundWindow()\n")
    assert _find_write_calls(good) == []


# Slice 8 boundary: blocking input is permitted only in the CLI REPL.
# Any handler or orchestrator that calls input() would freeze a turn
# and break deterministic testing. Prompting is injectable-only.
def test_no_input_outside_cli_main():
    offenders = []
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack = [(tree, None)]
        while stack:
            node, func = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = node.name
            if isinstance(node, ast.Call):
                called = node.func
                if isinstance(called, ast.Name) and called.id == "input":
                    if not (path.name == "cli.py" and func == "main"):
                        offenders.append((path.name, func))
            for child in ast.iter_child_nodes(node):
                stack.append((child, func))
    assert offenders == []


# Slice 8 consent lives in state/permissions.json, never state.json.
# Pinned before the store exists so a future schema drift is caught
# immediately.
def test_consent_not_in_state_json():
    state = make_carrot().state.to_dict()
    assert "permissions" not in state
    assert "grants" not in state
    assert "consent" not in state


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
    for word in _STATE_NAMES:
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
    for word in _STATE_NAMES:
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
    for word in _STATE_NAMES:
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


def test_tools_block_pinned():
    """The Tools: block is a stable capability advertisement.
    Its exact wording is pinned so any change is a deliberate diff.
    Corrected in Phase 5b: the original described a fenced format but
    omitted the fence characters the parser requires."""
    assert _TOOLS_BLOCK == 'Tools you may use. read_file: read a file inside an allowed folder. Args: {"path": "<absolute path>"}. open_url: open a link inside an allowed domain. Args: {"url": "<http or https url>"}. To use a tool, reply with exactly this block, including the triple backticks: ```tool:NAME {"arg": "value"}```'


def test_tools_block_contains_fence():
    """The Tools: block must contain literal triple backticks so the
    advertised format matches parse_tool_request's grammar. Without
    this, an LLM following the block literally would emit unfenced
    requests that parse as plain text."""
    from carrot.carrot import _TOOLS_BLOCK
    assert _TOOLS_BLOCK.count("```") >= 2
    assert "```tool:NAME" in _TOOLS_BLOCK
    assert _TOOLS_BLOCK.endswith("```")


def test_tools_block_present_every_turn():
    c = make_carrot()
    for text in ("hello there", "how are you?", "tell me more"):
        c.process_turn(text)
        c.clock.advance(5)
        assert "Tools you may use." in c.llm.last_system


def test_tools_block_before_memories():
    c = make_carrot()
    c.process_turn("my name is Alice")
    c.clock.advance(5)
    c.process_turn("what was my name")
    system = c.llm.last_system
    assert "Memories:" in system
    assert system.index("Tools you may use.") < system.index("Memories:")


def test_tools_block_before_context():
    c = make_carrot()
    c.process_turn("that joke was funny lol")
    c.clock.advance(5)
    c.process_turn("tell me more")
    system = c.llm.last_system
    assert "Context:" in system
    assert system.index("Tools you may use.") < system.index("Context:")


def test_tools_block_skeleton_clean():
    skeleton = re.sub(r'"[^"]*"', "", _TOOLS_BLOCK)
    assert re.search(r"\d", skeleton) is None
    lowered = skeleton.lower()
    for word in _STATE_NAMES:
        assert re.search(rf"\b{word}\b", lowered) is None, word


def test_tools_block_tool_names_digit_free():
    assert "read_file" in _TOOLS_BLOCK
    assert "open_url" in _TOOLS_BLOCK
    assert re.search(r"\d", _TOOLS_BLOCK) is None


_BANNED_PERMISSION_WORDS = (
    "granted", "grant", "denied", "deny",
    "consent", "policy", "revoke", "permission", "permissions",
)

_STATE_NAMES = (
    "happiness", "sadness", "anger", "hurt", "affection",
    "excitement", "embarrassment", "content", "low",
    "tender", "tense", "bright", "sensitivity", "playfulness",
    "stoicism", "shyness", "pride", "warmth", "patience",
    "mood",
)


def _assert_no_permission_words(system):
    lowered = _skeleton(system).lower()
    for word in _BANNED_PERMISSION_WORDS:
        assert re.search(rf"\b{word}\b", lowered) is None, word


def _alt_personality():
    return Personality.load(
        str(ROOT / "tests" / "fixtures" / "personality-alt.json"))


def test_grant_vocabulary_absent_from_system_prompt(tmp_path):
    """Slice 8 invariant: the permission layer is LLM-invisible. The LLM
    learns of a denial only via the tool-result string in the second-call
    history — never via the system prompt. If a future change adds a
    "Permissions:" block or a policy sentence to the prompt, this test
    fails and forces a deliberate planner review."""
    for personality in (None, _alt_personality()):
        c = make_carrot(personality=personality)
        c.process_turn("hello there")
        _assert_no_permission_words(c.llm.last_system)

        c = make_carrot(personality=personality)
        c.process_turn("my name is Alice")
        c.clock.advance(5)
        c.process_turn("what was my name")
        _assert_no_permission_words(c.llm.last_system)

        store = DriveStore(tmp_path / "d.json")
        store.save(DriveState(social=0.9, novelty=0.0))
        c = make_carrot(personality=personality, drive_store=store)
        c.process_turn("hello there")
        _assert_no_permission_words(c.llm.last_system)

        c = make_carrot(personality=personality)
        c.environment_probe = MockEnvironmentProbe(
            [RawObservation("cmd.exe", False)])
        c.process_turn("hello there")
        _assert_no_permission_words(c.llm.last_system)

        c = make_carrot(personality=personality)
        c.process_turn("are you awake?")
        c.clock.advance(5)
        c.process_turn("yes, I'm here")
        _assert_no_permission_words(c.llm.last_system)

        c = make_carrot(personality=personality)
        c.process_turn("you are useless.")
        c.clock.advance(5)
        c.process_turn("I'm kidding.")
        _assert_no_permission_words(c.llm.last_system)

        store = DriveStore(tmp_path / "d2.json")
        store.save(DriveState(social=0.9, novelty=0.0))
        c = make_carrot(personality=personality, drive_store=store)
        c.environment_probe = MockEnvironmentProbe(
            [RawObservation("cmd.exe", False)])
        c.process_turn("my name is Alice")
        c.clock.advance(5)
        c.process_turn("are you awake?")
        c.clock.advance(5)
        c.process_turn("yes, I'm here")
        c.clock.advance(5)
        c.process_turn("you are useless.")
        c.clock.advance(5)
        c.process_turn("I'm kidding.")
        c.clock.advance(5)
        c.process_turn("what was my name")
        _assert_no_permission_words(c.llm.last_system)


_ALLOWED_REASONS = {
    None,
    "unknown tool",
    "disabled",
    "not permitted",
    "requires persistent grant",
    "missing path",
    "path must be absolute",
    "path not allowed",
    "missing url",
    "unsupported scheme",
    "invalid url",
    "domain not allowed",
}


def _reason_cfg(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    return ToolsConfig(
        enabled={"read_file": True, "open_url": True},
        read_file_dirs=(str(notes),),
        open_url_domains=("example.com",),
        max_file_bytes=4096, max_iterations=2,
        levels={"read_file": 1, "open_url": 2}), notes


def test_decide_reason_strings_are_static(tmp_path):
    """Slice 8 invariant: reason strings are static. No domain, path, tool
    name, or count is interpolated. Interpolation would be an LLM leak
    vector (the reason reaches the second-call history text). If a future
    change adds interpolation, this test fails."""
    cfg, notes = _reason_cfg(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir(exist_ok=True)
    cases = [
        # (request, session, persistent, cfg override or None)
        (ToolRequest("rm_rf", {}), {}, {}, None),
        (ToolRequest("read_file", {"path": str(notes / "a.txt")}),
         {"read_file": True}, {}, {"enabled": {"read_file": False}}),
        (ToolRequest("read_file", {"path": str(notes / "a.txt")}),
         {}, {}, None),
        (ToolRequest("read_file", {"path": str(notes / "a.txt")}),
         {"read_file": False}, {"read_file": True}, None),
        (ToolRequest("read_file", {"path": str(notes / "a.txt")}),
         {"read_file": True}, {}, {"levels": {"read_file": 3}}),
        (ToolRequest("read_file", {"path": str(notes / "a.txt")}),
         {}, {"read_file": True}, {"levels": {"read_file": 3}}),
        (ToolRequest("read_file", {"path": str(notes / "a.txt")}),
         {"read_file": True}, {}, None),
        (ToolRequest("open_url", {"url": "https://example.com/"}),
         {"open_url": True}, {}, None),
        (ToolRequest("read_file", {}), {"read_file": True}, {}, None),
        (ToolRequest("read_file", {"path": "notes/a.txt"}),
         {"read_file": True}, {}, None),
        (ToolRequest("read_file", {"path": str(outside / "x.txt")}),
         {"read_file": True}, {}, None),
        (ToolRequest("open_url", {}), {"open_url": True}, {}, None),
        (ToolRequest("open_url", {"url": ""}), {"open_url": True}, {},
         None),
        (ToolRequest("open_url", {"url": "file:///etc/passwd"}),
         {"open_url": True}, {}, None),
        (ToolRequest("open_url", {"url": "http:///x"}),
         {"open_url": True}, {}, None),
        (ToolRequest("open_url", {"url": "https://evil.example/"}),
         {"open_url": True}, {}, None),
    ]
    for req, session, persistent, override in cases:
        use_cfg = cfg
        if override is not None:
            use_cfg = ToolsConfig(
                enabled=override.get("enabled", dict(cfg.enabled)),
                read_file_dirs=cfg.read_file_dirs,
                open_url_domains=cfg.open_url_domains,
                max_file_bytes=cfg.max_file_bytes,
                max_iterations=cfg.max_iterations,
                levels=override.get("levels", dict(cfg.levels)))
        assert decide(req, use_cfg, session,
                      persistent).reason in _ALLOWED_REASONS, req


def test_permissions_module_does_not_import_llm():
    """Complements test_permissions_module_boundary (Phase 1), which bans
    a broad project-module list: this pins the LLM-invisibility trio
    (llm, carrot, cli) explicitly, since cli was absent from that set."""
    banned = {"llm", "carrot", "cli"}
    offenders = []
    tree = ast.parse((ROOT / "carrot" / "permissions.py").read_text(
        encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in banned or (parts[0] == "carrot"
                                          and len(parts) > 1):
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".")
            if node.level == 0 and (parts[0] in banned or (
                    parts[0] == "carrot" and len(parts) > 1)):
                offenders.append(mod)
            elif node.level > 0:
                if parts and parts[0] in banned:
                    offenders.append(mod)
                else:
                    for a in node.names:
                        if a.name in banned:
                            offenders.append(a.name)
    assert offenders == []


def test_initiative_reason_lines_are_clean():
    """
    Slice 9 invariant: the initiative reason lines are LLM-facing. They
    must contain no digits, no permission vocabulary, no state names.
    A future edit that interpolates a value (e.g. 'away for N hours')
    or leaks permission vocabulary would be caught here.
    """
    from carrot.cli import _INITIATIVE_REASON_LINES
    banned = _BANNED_PERMISSION_WORDS  # existing tuple in this module
    for reason, line in _INITIATIVE_REASON_LINES.items():
        assert re.search(r"\d", line) is None, f"digit in {reason!r}: {line!r}"
        lower = line.lower()
        for word in banned:
            assert not re.search(rf"\b{re.escape(word)}\b", lower), \
                f"banned word {word!r} in {reason!r}: {line!r}"
        for word in _STATE_NAMES:
            assert not re.search(rf"\b{word}\b", lower), \
                f"state name {word!r} in {reason!r}: {line!r}"


def test_initiative_system_prompt_is_clean():
    """
    Slice 9 invariant: the full away-mode system prompt is LLM-facing.
    No digits, no permission vocabulary, no state names — extending the
    LLM-invisibility invariant (Slice 8) to the initiative path.
    """
    from carrot.cli import _initiative_system_prompt

    c = make_carrot()
    for reason in ("long-absence", "high-social-drive", "session-open"):
        system = _initiative_system_prompt(c, reason)
        assert re.search(r"\d", system) is None, reason
        lowered = system.lower()
        for word in _BANNED_PERMISSION_WORDS:
            assert re.search(rf"\b{word}\b", lowered) is None, \
                (reason, word)
        for word in _STATE_NAMES:
            assert re.search(rf"\b{word}\b", lowered) is None, \
                (reason, word)


# Slice 10 pre-emptive boundary: no new persistent stores. Fusion is
# recomputed each turn from existing primitives; persisting a fusion
# signal would create a second source of truth that rots.
_ALLOWED_JSON_BASENAMES = frozenset({
    "state.json", "memories.json", "drives.json", "permissions.json",
    "initiative.json", "appraisal.json", "behavior.json", "decay.json",
    "environment.json", "personality.json", "tools.json",
    # Slice 11 pre-entry: state/relationship.json + config/relationship.json.
    # The relationship slice is the only remaining slice expected to add a
    # new persistent store; any future addition needs a planner-approved diff.
    "relationship.json",
    "social.json",   # Slice 15: config/social.json (committed, handles only)
    "episodes.json",   # state/episodes.json (per-companion) + config/episodes.json (global)
})


def _json_basename_literals(tree):
    """Every string literal ending in .json, reduced to its basename,
    excluding the bare ".json" fragments of quarantine f-strings."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.endswith(".json"):
            base = node.value.replace("\\", "/").rsplit("/", 1)[-1]
            if base != ".json":
                found.append(base)
    return found


def test_no_new_state_files():
    offenders = []
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _json_basename_literals(tree):
            if name not in _ALLOWED_JSON_BASENAMES:
                offenders.append((path.name, name))
    assert offenders == []


# --- Slice 11 Phase 0: boundary pins ---


def test_pick_signature_unchanged():
    # Slice 11 boundary: relationship is a soft input via memory boost only.
    # If a future slice threads it through pick(), it must be a deliberate
    # planner-visible diff to this test — not a silent signature extension.
    tree = ast.parse((ROOT / "carrot" / "behavior.py").read_text(
        encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == "pick":
            params = [a.arg for a in node.args.args]
            assert node.args.vararg is None
            assert node.args.kwarg is None
            found.append(params)
    assert found == [["self", "state", "personality", "context",
                      "user_text"]]


def test_appraisal_keys_unchanged():
    # Slice 11 boundary: appraisal deltas are restricted to emotions + energy.
    # A relationship dimension (trust, closeness) must never enter the
    # appraisal table — relationship is orthogonal to affect.
    data = json.loads((ROOT / "config" / "appraisal.json").read_text(
        encoding="utf-8"))
    allowed = {"happiness", "sadness", "anger", "hurt", "affection",
               "excitement", "embarrassment", "energy"}
    keys = set()
    for deltas in data.values():
        keys.update(deltas)
    assert keys <= allowed, keys - allowed


def test_no_relationship_in_state_json():
    # Slice 11 boundary: relationship state lives in state/relationship.json.
    # state.json (EmotionState) schema is frozen — no new dimensions there.
    from carrot.emotions import EmotionState

    d = EmotionState().to_dict()
    assert "trust" not in d
    assert "closeness" not in d
    assert "relationship" not in d


def test_relationship_module_is_pure():
    # Relationship pure functions take plain floats and event-type strings,
    # not DriveState or EnvironmentSnapshot types. If a future phase needs
    # project types, that's a deliberate diff to this allowance set.
    path = ROOT / "carrot" / "relationship.py"
    if not path.exists():
        pytest.skip("relationship.py lands in Phase 1")
    offenders = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == "carrot" or a.name == "carrot":
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".")
            if node.level == 0 and parts[0] == "carrot":
                offenders.append(mod)
            elif node.level > 0:
                offenders.append("." + mod if mod else ".")
    assert offenders == []


# --- Slice 10 Phase 3: fusion sentence tests ---


def _fusion_carrot(memories):
    c = make_carrot()
    c.environment_probe = MockEnvironmentProbe(
        [RawObservation("chrome.exe", False)])
    for text, created in memories:
        c.memory_store.add(Memory(text=text, kind="fact", importance=0.5,
                                  created_ts=created,
                                  last_recalled_ts=1_000_000.0))
    return c


def test_fusion_sentence_static_and_clean():
    assert _FUSION_SENTENCE == \
        "Something here connects to what you've told me before."
    assert re.search(r"\d", _FUSION_SENTENCE) is None
    assert '"' not in _FUSION_SENTENCE
    lowered = _FUSION_SENTENCE.lower()
    for word in _BANNED_PERMISSION_WORDS:
        assert re.search(rf"\b{word}\b", lowered) is None, word
    for word in _STATE_NAMES:
        assert re.search(rf"\b{word}\b", lowered) is None, word


def test_fusion_last_among_all_hint_parts():
    ctx = ConversationContext(playful_frame=True, recent_harshness=True,
                              recent_topics=("gardening",))
    env = EnvironmentSnapshot(app_noun="browser", idle=True)
    hint = _context_hint(ctx, "I'm kidding.", drive_pull="social",
                         environment=env, boost_effect=True)
    idx_playful = hint.index("The conversation has been playful.")
    idx_repair = hint.index("walking back what they said")
    idx_topics = hint.index("Recent topics: gardening")
    idx_drive = hint.index("You've been wanting to talk to them.")
    idx_app = hint.index("They're in the browser.")
    idx_idle = hint.index("They've been away for a bit.")
    idx_fusion = hint.index(_FUSION_SENTENCE)
    assert idx_playful < idx_repair < idx_topics < idx_drive < idx_app \
        < idx_idle < idx_fusion


def test_fusion_only_part_renders_block():
    hint = _context_hint(ConversationContext(), "12345 !!!",
                         boost_effect=True)
    assert hint == " Context: " + _FUSION_SENTENCE
    # Omission preserved: no boost, no block (Phase 2 byte-identity).
    assert _context_hint(ConversationContext(), "12345 !!!",
                         boost_effect=False) == ""


def test_fusion_skeleton_clean_with_boost():
    c = _fusion_carrot([("i love pizza", 100.0),
                        ("my browser tabs are many", 200.0)])
    c.process_turn("hello there")
    assert c.last_boost_effect is True
    assert _FUSION_SENTENCE in c.llm.last_system
    skeleton = _skeleton(c.llm.last_system)
    assert re.search(r"\d", skeleton) is None
    lowered = skeleton.lower()
    for word in _BANNED_PERMISSION_WORDS:
        assert re.search(rf"\b{word}\b", lowered) is None, word
    for word in _STATE_NAMES:
        assert re.search(rf"\b{word}\b", lowered) is None, word


def test_fusion_skeleton_clean_with_adversarial_memories():
    # Digits ("May 3") and a state name ("anger") ride inside quoted
    # memory spans; the skeleton strips them, and the fusion sentence
    # itself carries no data.
    c = _fusion_carrot([("my birthday is May 3", 100.0),
                        ("my anger issues", 200.0),
                        ("my browser tabs are many", 300.0)])
    c.process_turn("hello there")
    assert c.last_boost_effect is True
    assert _FUSION_SENTENCE in c.llm.last_system
    assert '"my birthday is May 3"' in c.llm.last_system  # recalled
    skeleton = _skeleton(c.llm.last_system)
    assert re.search(r"\d", skeleton) is None
    lowered = skeleton.lower()
    for word in _BANNED_PERMISSION_WORDS:
        assert re.search(rf"\b{word}\b", lowered) is None, word
    for word in _STATE_NAMES:
        assert re.search(rf"\b{word}\b", lowered) is None, word


# --- Slice 11 Phase 5: relationship sentence tests ---


def _high_band_carrot(tmp_path, closeness, memories=()):
    store = RelationshipStore(tmp_path / "rel.json")
    store.save(RelationshipState(trust=0.5, closeness=closeness,
                                 updated_ts={}))
    c = make_carrot(relationship_store=store)
    c.environment_probe = MockEnvironmentProbe(
        [RawObservation("chrome.exe", False)])
    for text, created in memories:
        c.memory_store.add(Memory(text=text, kind="fact", importance=0.5,
                                  created_ts=created,
                                  last_recalled_ts=1_000_000.0))
    return c


def test_relationship_sentence_static_and_clean():
    assert _RELATIONSHIP_HIGH_SENTENCE == \
        "There's an easy familiarity here."
    assert re.search(r"\d", _RELATIONSHIP_HIGH_SENTENCE) is None
    assert '"' not in _RELATIONSHIP_HIGH_SENTENCE
    lowered = _RELATIONSHIP_HIGH_SENTENCE.lower()
    for word in _BANNED_PERMISSION_WORDS:
        assert re.search(rf"\b{word}\b", lowered) is None, word
    for word in _STATE_NAMES:
        assert re.search(rf"\b{word}\b", lowered) is None, word


def test_relationship_last_after_fusion():
    ctx = ConversationContext(playful_frame=True, recent_harshness=True,
                              recent_topics=("gardening",))
    env = EnvironmentSnapshot(app_noun="browser", idle=True)
    hint = _context_hint(ctx, "I'm kidding.", drive_pull="social",
                         environment=env, boost_effect=True,
                         relationship_band="high")
    idx_fusion = hint.index(_FUSION_SENTENCE)
    idx_rel = hint.index(_RELATIONSHIP_HIGH_SENTENCE)
    assert idx_fusion < idx_rel
    assert hint.endswith(_RELATIONSHIP_HIGH_SENTENCE)


def test_relationship_only_part_renders_block():
    hint = _context_hint(ConversationContext(), "12345 !!!",
                         relationship_band="high")
    assert hint == " Context: " + _RELATIONSHIP_HIGH_SENTENCE
    # Omission preserved: any other band value renders nothing.
    assert _context_hint(ConversationContext(), "12345 !!!",
                         relationship_band=None) == ""
    assert _context_hint(ConversationContext(), "12345 !!!",
                         relationship_band="low") == ""


def test_relationship_skeleton_clean_with_high_band(tmp_path):
    c = _high_band_carrot(tmp_path, 0.69,
                          [("i love pizza", 100.0),
                           ("my browser tabs are many", 200.0)])
    c.process_turn("i love you")
    assert c.last_relationship_band == "high"
    assert _RELATIONSHIP_HIGH_SENTENCE in c.llm.last_system
    skeleton = _skeleton(c.llm.last_system)
    assert re.search(r"\d", skeleton) is None
    lowered = skeleton.lower()
    for word in _BANNED_PERMISSION_WORDS:
        assert re.search(rf"\b{word}\b", lowered) is None, word
    for word in _STATE_NAMES:
        assert re.search(rf"\b{word}\b", lowered) is None, word


def test_relationship_skeleton_clean_adversarial(tmp_path):
    # Digits ("May 3") and a state name ("anger") ride inside quoted
    # memory spans while boost AND the high band both fire; the skeleton
    # strips user content and both static sentences carry no data.
    # Closeness is pre-written at exactly 0.70 so the neutral turn keeps
    # the band high without disturbing recall scores.
    c = _high_band_carrot(tmp_path, 0.70,
                          [("my birthday is May 3", 100.0),
                           ("my anger issues", 200.0),
                           ("my browser tabs are many", 300.0)])
    c.process_turn("hello there")
    assert c.last_boost_effect is True
    assert c.last_relationship_band == "high"
    assert _FUSION_SENTENCE in c.llm.last_system
    assert _RELATIONSHIP_HIGH_SENTENCE in c.llm.last_system
    assert '"my birthday is May 3"' in c.llm.last_system  # recalled
    skeleton = _skeleton(c.llm.last_system)
    assert re.search(r"\d", skeleton) is None
    lowered = skeleton.lower()
    for word in _BANNED_PERMISSION_WORDS:
        assert re.search(rf"\b{word}\b", lowered) is None, word
    for word in _STATE_NAMES:
        assert re.search(rf"\b{word}\b", lowered) is None, word


# --- Slice 12 Phase 0: avatar boundary pins ---


def _enclosing_funcs(tree):
    """Map every node to its enclosing function name (None at top)."""
    found = {}

    def walk(node, func):
        found[id(node)] = func
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, child.name)
            else:
                walk(child, func)

    walk(tree, None)
    return found


def test_mainloop_only_in_avatar_entry():
    # Slice 12 boundary: exactly one GUI pump, in avatar.launch_avatar.
    # The existing while-True matcher cannot see mainloop(); this pin is the
    # explicit sanction. Any second mainloop — in this file or elsewhere —
    # fails loudly.
    sites = set()
    for mod in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(mod.read_text(encoding="utf-8"))
        owners = _enclosing_funcs(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) \
                    and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "mainloop":
                sites.add((mod.name, owners[id(node)]))
    if not sites:
        pytest.skip("no mainloop call yet "
                    "(activates when avatar.launch_avatar lands)")
    assert sites == {("avatar.py", "launch_avatar")}


def test_no_tkinter_after():
    # Slice 12 boundary: no scheduled callbacks. after()/after_idle() are
    # schedulers in disguise — a foot-in-the-door for autonomous loops.
    # Refresh is user-triggered only. If a future slice wants polling, it
    # must retire this pin with a planner-visible diff.
    offenders = []
    for mod in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(mod.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) \
                    and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in ("after", "after_idle"):
                offenders.append((mod.name, node.lineno))
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "tkinter" \
                            and mod.name != "avatar.py":
                        offenders.append((mod.name, a.name))
            elif isinstance(node, ast.ImportFrom):
                mod_name = (node.module or "").split(".")[0]
                if node.level == 0 and mod_name == "tkinter" \
                        and mod.name != "avatar.py":
                    offenders.append((mod.name, node.module))
    assert offenders == []


def test_avatar_module_boundaries():
    # Slice 12 boundary: the avatar is a read-only expression layer.
    # It may read state via state_store/memory (read-only stores) and may
    # use stdlib + tkinter. It must never touch the orchestrator, the CLI
    # dispatcher, or the LLM. These are the invariants that keep Slice 12
    # a presentation layer rather than a second brain.
    path = ROOT / "carrot" / "avatar.py"
    if not path.exists():
        pytest.skip("avatar.py lands in Phase 2")
    banned_mods = {"carrot", "cli", "llm"}
    banned_calls = {"process_turn", "generate"}
    offenders = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in banned_mods or (parts[0] == "carrot"
                                               and len(parts) > 1):
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            parts = mod_name.split(".")
            if node.level == 0 and (parts[0] in banned_mods or (
                    parts[0] == "carrot" and len(parts) > 1)):
                offenders.append(mod_name)
            elif node.level > 0:
                if parts and parts[0] in banned_mods:
                    offenders.append(mod_name)
                else:
                    for a in node.names:
                        if a.name in banned_mods:
                            offenders.append(a.name)
        elif isinstance(node, ast.Call) \
                and isinstance(node.func, ast.Attribute) \
                and node.func.attr in banned_calls:
            offenders.append(node.func.attr)
    assert offenders == []
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"carrot"}
    assert mods <= allowed, mods


def test_avatar_degrade_message_pinned():
    """Slice 12: the degrade message is contract, not accident."""
    avatar = pytest.importorskip("carrot.avatar")
    msg = getattr(avatar, "AVATAR_UNAVAILABLE_MSG", None)
    if msg is None:
        pytest.skip("constant lands in Phase 2")
    assert msg == "(avatar unavailable: no display)"


# --- Slice 13 Phase 0: voice boundary pins ---


def test_voice_module_boundaries():
    # Slice 13 boundary: voice.py holds the channel protocols and mocks
    # only. The orchestrator/CLI/LLM are all off limits. No subprocess —
    # Option A never shells out; if a future slice sanctions a real TTS
    # backend, that lands in a separate voice_*.py with its own
    # planner-visible pin.
    path = ROOT / "carrot" / "voice.py"
    if not path.exists():
        pytest.skip("voice.py lands in Phase 1")
    banned_mods = {"carrot", "cli", "llm"}
    banned_calls = {"process_turn", "generate"}
    offenders = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in banned_mods or (parts[0] == "carrot"
                                               and len(parts) > 1):
                    offenders.append(a.name)
                if parts[0] == "subprocess":
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            parts = mod_name.split(".")
            if node.level == 0 and (parts[0] in banned_mods or (
                    parts[0] == "carrot" and len(parts) > 1)):
                offenders.append(mod_name)
            elif node.level > 0:
                if parts and parts[0] in banned_mods:
                    offenders.append(mod_name)
                else:
                    for a in node.names:
                        if a.name in banned_mods:
                            offenders.append(a.name)
            if node.level == 0 and parts[0] == "subprocess":
                offenders.append(mod_name)
        elif isinstance(node, ast.Call) \
                and isinstance(node.func, ast.Attribute) \
                and node.func.attr in banned_calls:
            offenders.append(node.func.attr)
    assert offenders == []
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"carrot"}
    assert mods <= allowed, mods


def test_voice_empty_msg_pinned():
    """Slice 13: the empty-utterance message is contract, not accident."""
    voice = pytest.importorskip("carrot.voice")
    msg = getattr(voice, "VOICE_EMPTY_MSG", None)
    if msg is None:
        pytest.skip("constant lands in Phase 1")
    assert msg == "(heard nothing)"


# --- Slice 14 Phase 0: channel boundary pins ---


def test_channel_module_boundaries():
    # Slice 14 boundary: channel.py holds the Channel protocol and
    # file-based adapter only. It must never call the orchestrator, the
    # CLI dispatcher, or the LLM. File I/O is the only side effect; no
    # subprocess, no ctypes. A future live-server module (Option C,
    # deferred) would need its own planner-visible pin.
    path = ROOT / "carrot" / "channel.py"
    if not path.exists():
        pytest.skip("channel.py lands in Phase 1")
    banned_mods = {"carrot", "cli", "llm"}
    banned_calls = {"process_turn", "generate"}
    banned_native = {"subprocess", "ctypes"}
    offenders = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in banned_mods or (parts[0] == "carrot"
                                               and len(parts) > 1):
                    offenders.append(a.name)
                if parts[0] in banned_native:
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            parts = mod_name.split(".")
            if node.level == 0 and (parts[0] in banned_mods or (
                    parts[0] == "carrot" and len(parts) > 1)):
                offenders.append(mod_name)
            elif node.level > 0:
                if parts and parts[0] in banned_mods:
                    offenders.append(mod_name)
                else:
                    for a in node.names:
                        if a.name in banned_mods:
                            offenders.append(a.name)
            if node.level == 0 and parts[0] in banned_native:
                offenders.append(mod_name)
        elif isinstance(node, ast.Call) \
                and isinstance(node.func, ast.Attribute) \
                and node.func.attr in banned_calls:
            offenders.append(node.func.attr)
    assert offenders == []
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"carrot"}
    assert mods <= allowed, mods


def test_channel_empty_msg_pinned():
    """Slice 14: the empty-inbox message is contract, not accident."""
    channel_mod = pytest.importorskip("carrot.channel")
    msg = getattr(channel_mod, "CHANNEL_EMPTY_MSG", None)
    if msg is None:
        pytest.skip("constant lands in Phase 1")
    assert msg == "(no messages)"


# --- Slice 15 Phase 0: social boundary pins ---


def test_social_module_boundaries():
    # Slice 15 boundary: social.py holds the Channel-reusing adapter and
    # config only. No orchestrator, CLI, or LLM contact. No subprocess,
    # ctypes, threads, or schedulers — a manual-mode file bridge needs
    # none.
    path = ROOT / "carrot" / "social.py"
    if not path.exists():
        pytest.skip("social.py lands in Phase 1")
    banned_mods = {"carrot", "cli", "llm"}
    banned_calls = {"process_turn", "generate"}
    banned_native = {"subprocess", "ctypes", "threading", "asyncio",
                     "sched", "multiprocessing", "Timer"}
    offenders = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in banned_mods or (parts[0] == "carrot"
                                               and len(parts) > 1):
                    offenders.append(a.name)
                if parts[0] in banned_native:
                    offenders.append(a.name)
                for a2 in node.names:
                    if a2.name in banned_native:
                        offenders.append(a2.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            parts = mod_name.split(".")
            if node.level == 0 and (parts[0] in banned_mods or (
                    parts[0] == "carrot" and len(parts) > 1)):
                offenders.append(mod_name)
            elif node.level > 0:
                if parts and parts[0] in banned_mods:
                    offenders.append(mod_name)
                else:
                    for a in node.names:
                        if a.name in banned_mods:
                            offenders.append(a.name)
            if node.level == 0 and parts[0] in banned_native:
                offenders.append(mod_name)
            for a in node.names:
                if a.name in banned_native:
                    offenders.append(a.name)
        elif isinstance(node, ast.Call) \
                and isinstance(node.func, ast.Attribute) \
                and node.func.attr in banned_calls:
            offenders.append(node.func.attr)
    assert offenders == []
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"carrot"}
    assert mods <= allowed, mods


def test_social_config_no_secrets():
    # Slice 15 boundary: config/ is committed (not gitignored). No
    # secret-shaped key may ever appear there. Future credentials live in
    # state/social/<platform>/credentials.json (gitignored via state/).
    # This pin preempts the leak class before its first victim.
    path = ROOT / "config" / "social.json"
    if not path.exists():
        pytest.skip("config/social.json lands in Phase 1")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    banned = {"token", "secret", "password", "api_key", "apikey",
              "auth", "credential", "oauth"}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in banned:
                    return key
                hit = walk(value)
                if hit is not None:
                    return hit
        elif isinstance(node, list):
            for item in node:
                hit = walk(item)
                if hit is not None:
                    return hit
        return None

    assert walk(data) is None


# --- Slice 16 Phase 0: companion pre-pins ---


def test_companion_module_boundaries():
    # Slice 16 boundary: companion.py holds the Companion value type,
    # path construction, name validation, and the factory that returns a
    # live Carrot instance. The factory composes existing Carrot + store
    # + config objects; it does not introduce a new orchestration
    # surface.
    path = ROOT / "carrot" / "companion.py"
    if not path.exists():
        pytest.skip("companion.py lands in Phase 1")
    banned_mods = {"carrot", "cli", "llm"}
    banned_calls = {"process_turn", "generate"}
    offenders = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in banned_mods or (parts[0] == "carrot"
                                               and len(parts) > 1):
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            parts = mod_name.split(".")
            if node.level == 0 and (parts[0] in banned_mods or (
                    parts[0] == "carrot" and len(parts) > 1)):
                offenders.append(mod_name)
            elif node.level > 0:
                if parts and parts[0] in banned_mods:
                    offenders.append(mod_name)
                else:
                    for a in node.names:
                        if a.name in banned_mods:
                            offenders.append(a.name)
        elif isinstance(node, ast.Call) \
                and isinstance(node.func, ast.Attribute) \
                and node.func.attr in banned_calls:
            offenders.append(node.func.attr)
    assert offenders == []
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"carrot"}
    assert mods <= allowed, mods


def test_companion_names_validated():
    # Slice 16 boundary: companion names are filesystem path segments.
    # Validation lives in companion.validate_name; any relaxation of the
    # rule must retire this pin with a planner-visible diff.
    companion = pytest.importorskip("carrot.companion")
    validate = getattr(companion, "validate_name", None)
    if validate is None:
        pytest.skip("validate_name lands in Phase 1")
    for good in ("carrot", "ada", "ada-2", "x" * 32, "a-b-c-1"):
        assert validate(good) == good
    for bad in ("", "x" * 33, "Ada", "ada_x", "../x", "/abs",
                "ada/sub", " ada", "ada "):
        with pytest.raises(ValueError):
            validate(bad)


def test_default_companion_uses_flat_paths():
    # Slice 16 boundary: the default companion ("carrot") keeps the flat
    # layout forever (zero migration); named companions namespace under
    # state/<name>/. Permissions stay global by design (user-level).
    companion = pytest.importorskip("carrot.companion")
    paths_for = getattr(companion, "companion_paths", None)
    if paths_for is None:
        pytest.skip("companion_paths lands in Phase 1")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="carrot-pin-") as tmp:
        from pathlib import Path as _Path

        root = _Path(tmp)
        flat = paths_for(root, "carrot")
        assert flat["state_json"] == root / "state" / "state.json"
        named = paths_for(root, "ada")
        assert named["state_json"] == \
            root / "state" / "ada" / "state.json"
        assert named["permissions_json"] == \
            root / "state" / "permissions.json"


# --- Episodic Continuity Phase 0: raw-turn persistence pin ---


def test_no_verbatim_turn_text_in_new_stores():
    # Research tier boundary: a persistent episode record must never
    # carry verbatim conversation text. Episodes store derived features
    # only (topic tokens, timestamps, turn count, mood label). If a
    # future Episode class gains a text-shaped field, this pin fails
    # loudly with the offending name.
    path = ROOT / "carrot" / "episodes.py"
    if not path.exists():
        pytest.skip("episodes.py lands in Phase 1")
    forbidden = {
        "text", "user_text", "reply", "message", "content",
        "transcript", "history", "turns", "conversation",
        "conversation_text", "user_message", "assistant_message",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Episode":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                        stmt.target, ast.Name):
                    names = [stmt.target.id]
                elif isinstance(stmt, ast.Assign):
                    names = [t.id for t in stmt.targets
                             if isinstance(t, ast.Name)]
                else:
                    continue
                for name in names:
                    if name in forbidden:
                        offenders.append(name)
    assert not offenders, \
        f"Episode declares verbatim-turn field(s): {sorted(set(offenders))}"


# --- Episodic Continuity Phase 5: prompt block tests ---


def _ep_block_episode(**over):
    fields = dict(first_seen_ts=1000.0, last_seen_ts=2000.0, turn_count=2,
                  topic_tokens=("work", "deadline"), mood_label="content",
                  last_behavior="normal")
    fields.update(over)
    return Episode(**fields)


def _ep_prompt_carrot():
    c = make_carrot()
    return c


def _ep_system(c, episodes=(), recalled=(), topics=("garden",)):
    ctx = ConversationContext(recent_topics=tuple(topics))
    return c._system_prompt(Behavior("normal", "low"), ctx, "tell me more",
                            recalled, None, EMPTY_SNAPSHOT, False, None,
                            episodes)


def test_episodes_block_empty():
    assert _episodes_block(()) == ""


def test_episodes_block_single():
    assert _episodes_block((_ep_block_episode(),)) == \
        " Previously: work, deadline"


def test_episodes_block_filtered():
    episode = _ep_block_episode(topic_tokens=("work", "anger", "deadline"))
    assert _episodes_block((episode,)) == " Previously: work, deadline"


def test_episodes_block_all_filtered():
    episode = _ep_block_episode(topic_tokens=("anger", "mood"))
    assert _episodes_block((episode,)) == ""


def test_episodes_block_first_only():
    first = _ep_block_episode(topic_tokens=("work",))
    second = _ep_block_episode(topic_tokens=("chess",))
    assert _episodes_block((first, second)) == " Previously: work"


def test_episodes_block_deterministic():
    episode = _ep_block_episode()
    assert _episodes_block((episode,)) == _episodes_block((episode,))


def test_episodes_block_in_prompt():
    c = _ep_prompt_carrot()
    system = _ep_system(c, episodes=(_ep_block_episode(),))
    assert " Previously: work, deadline" in system


def test_episodes_absent_byte_identical():
    c = _ep_prompt_carrot()
    assert _ep_system(c, episodes=()) == _ep_system(c)


def test_episodes_block_position():
    c = _ep_prompt_carrot()
    recalled = [Memory(text="my dog is funny", kind="fact",
                       importance=0.5, created_ts=1000.0,
                       last_recalled_ts=1000.0)]
    system = _ep_system(c, episodes=(_ep_block_episode(),),
                       recalled=recalled)
    assert system.index("Memories:") < system.index("Previously:")
    assert system.index("Previously:") < system.index("Context:")


def _ep_skeleton_clean(system):
    skeleton = re.sub(r'"[^"]*"', "", system)
    assert re.search(r"\d", skeleton) is None
    lowered = skeleton.lower()
    for word in _BANNED_PERMISSION_WORDS:
        assert re.search(rf"\b{word}\b", lowered) is None, word
    for word in _STATE_NAMES:
        assert re.search(rf"\b{word}\b", lowered) is None, word


def test_episodes_skeleton_clean():
    # Tokens of "My anger about the project deadline 12345 is real!":
    # ("anger", "project", "deadline", "real") — "anger" is filtered
    # at render; digits never reach tokens; the rest is clean.
    from carrot.context import tokenize

    tokens = tokenize("My anger about the project deadline 12345 is real!")
    assert tokens == ("anger", "project", "deadline", "real")
    c = _ep_prompt_carrot()
    episode = _ep_block_episode(topic_tokens=tokens)
    _ep_skeleton_clean(_ep_system(c, episodes=(episode,)))


def test_episodes_content_dropped():
    c = _ep_prompt_carrot()
    episode = _ep_block_episode(topic_tokens=("game", "content"))
    system = _ep_system(c, episodes=(episode,))
    assert " Previously: game" in system
    _ep_skeleton_clean(system)


def test_episodes_secret_renders():
    c = _ep_prompt_carrot()
    episode = _ep_block_episode(topic_tokens=("secret", "code"))
    system = _ep_system(c, episodes=(episode,))
    assert " Previously: secret, code" in system
    _ep_skeleton_clean(system)


def test_process_turn_no_episodes_no_block():
    c = make_carrot()
    c.process_turn("hello there")
    assert "Previously:" not in c.llm.last_system


def test_process_turn_renders_prior_episode():
    c = make_carrot()
    c.process_turn("talking about gardening")
    c.process_turn("talking about soil")
    c.clock.advance(3600.0)
    c.process_turn("talking about gardening")
    assert " Previously: talking, gardening, soil" in c.llm.last_system
