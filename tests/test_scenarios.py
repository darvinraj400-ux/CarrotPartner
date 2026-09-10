"""End-to-end behavioral scenarios 1-10 through the orchestrator."""
import json
import re
import tempfile
from pathlib import Path

import pytest

from carrot.appraisal import Appraisal
from carrot.behavior import BehaviorSelector
from carrot.carrot import Carrot
from carrot.classifier import DeterministicClassifier
from carrot.conversation import ConversationBuffer
from carrot.emotions import EmotionState
from carrot.llm import MockLLM
from carrot.memory import MemoryStore
from carrot.permissions import PermissionStore
from carrot.drives import DriveStore, DrivesConfig
from carrot.environment import EnvironmentConfig
from carrot.personality import Personality
from tests.conftest import ROOT
from tests.test_classifier import StubSemanticClassifier

LEAK = re.compile(r"\d\.\d|happiness level|sadness level|internal state|"
                  r"my .* (is|at) 0|emotion (value|vector)")


class FakeTime:
    def __init__(self, start=1_000_000.0):
        self.t = float(start)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make_carrot(personality=None, start=None, memory_store=None,
                drive_store=None, permission_store=None):
    clock = start or FakeTime()
    llm = MockLLM()
    if memory_store is None:
        # Isolated per-instance stores: the abstract defaults would
        # point at the real state dir. Kept alive via the instance.
        tmpdir = tempfile.TemporaryDirectory(prefix="carrot-test-mem-")
    else:
        tmpdir = None
    if drive_store is None:
        memdir = tempfile.TemporaryDirectory(prefix="carrot-test-drv-")
        drive_store = DriveStore(Path(memdir.name) / "drives.json",
                                 time_fn=clock)
    else:
        memdir = None
    if permission_store is None:
        permdir = tempfile.TemporaryDirectory(prefix="carrot-test-perm-")
        permission_store = PermissionStore(Path(permdir.name)
                                           / "permissions.json",
                                           time_fn=clock)
    else:
        permdir = None
    carrot = Carrot(
        state=EmotionState(),
        personality=personality or Personality.load(
            str(ROOT / "config" / "personality.json")),
        classifier=DeterministicClassifier(),
        appraisal=Appraisal.load(str(ROOT / "config" / "appraisal.json")),
        selector=BehaviorSelector.load(str(ROOT / "config" / "behavior.json")),
        llm=llm,
        decay_cfg=json.loads(
            (ROOT / "config" / "decay.json").read_text(encoding="utf-8")),
        buffer=ConversationBuffer(),
        memory_store=(memory_store if memory_store is not None
                      else MemoryStore(
                          Path(tmpdir.name) / "memories.json",
                          time_fn=clock)),
        drive_store=drive_store,
        drives_config=DrivesConfig.load(
            str(ROOT / "config" / "drives.json")),
        permission_store=permission_store,
        environment_config=EnvironmentConfig.load(
            str(ROOT / "config" / "environment.json")),
        time_fn=clock,
    )
    llm._behavior_getter = lambda: carrot.last_behavior
    carrot.clock = clock
    carrot._tmp_memdir = tmpdir
    carrot._tmp_drvdir = memdir
    carrot._tmp_permdir = permdir
    return carrot


def replies(*cs):
    out = []
    for c in cs:
        out.extend(t.text for t in c.buffer.recent(100)
                   if t.role == "carrot")
    return out


def test_1_compliment_warms():
    c = make_carrot()
    before = c.state.emotions
    c.process_turn("You're cute, thanks!")
    after = c.state.emotions
    assert after["happiness"] > before["happiness"]
    assert after["affection"] > before["affection"]
    assert c.last_behavior.name in ("normal", "playful", "affectionate",
                                   "excited", "proud", "curious")


def test_2_insult_hurts_and_shows():
    c = make_carrot()
    for _ in range(4):
        c.process_turn("You are USELESS!!!")
        c.clock.advance(5)
    assert c.state.emotions["hurt"] > 0.4
    assert c.state.emotions["anger"] > 0.4
    assert c.last_behavior.name in ("sad", "sulking", "annoyed", "frustrated")
    assert len(c.buffer.recent()[0].text) < 60  # short, clipped replies


def test_3_apology_repairs():
    c = make_carrot()
    for _ in range(3):
        c.process_turn("You are USELESS!!!")
        c.clock.advance(5)
    hurt, anger = c.state.emotions["hurt"], c.state.emotions["anger"]
    c.process_turn("Sorry, that was out of line.")
    assert c.state.emotions["hurt"] < hurt
    assert c.state.emotions["anger"] < anger
    assert c.last_event.type == "apology"


def test_4_joke_lifts():
    c = make_carrot()
    before = c.state.emotions
    c.process_turn("that joke was funny lol \U0001F602\U0001F602")
    assert c.state.emotions["happiness"] >= before["happiness"]
    assert c.state.emotions["excitement"] > before["excitement"]


def test_5_emoji_preserved_to_llm():
    c = make_carrot()
    text = "NO WAY \U0001F62D\U0001F62D\U0001F62D"
    c.process_turn(text)
    assert c.llm.last_user == text
    c.process_turn("Fine \U0001F642")
    assert c.llm.last_user == "Fine \U0001F642"


def test_6_return_after_absence():
    c = make_carrot()
    c.process_turn("hey")
    c.clock.advance(7 * 3600)
    c.process_turn("hey, I'm back")
    events = [x["event"] for x in c.state.recent_causes]
    assert "return" in events
    assert c.state.emotions["happiness"] > 0.55
    assert c.state.emotions["excitement"] > 0.2


def test_7_upset_persists_then_recovers():
    c = make_carrot()
    for _ in range(4):
        c.process_turn("You are USELESS!!!")
        c.clock.advance(5)
    anger_now = c.state.emotions["anger"]
    assert anger_now > 0.4
    # Seconds later: still upset, no instant reset.
    c.clock.advance(30)
    c.process_turn("anyway.")
    assert c.state.emotions["anger"] > anger_now * 0.8
    # Hours later: decayed back toward baseline.
    c.clock.advance(3 * 3600)
    c.process_turn("hey.")
    assert c.state.emotions["anger"] < anger_now * 0.5
    assert c.last_behavior.name in ("normal", "quiet", "curious")


def test_8_never_announces_machinery():
    c = make_carrot()
    for line in ["You're cute!", "You are useless.", "Sorry.",
                 "that joke was funny lol", "I finally fixed that thing!",
                 "Fine \U0001F642"]:
        c.process_turn(line)
        c.clock.advance(5)
    for reply in replies(c):
        assert not LEAK.search(reply), reply
    # R1: float rule applies to the unquoted skeleton (no memories render
    # in this script, so the skeleton equals the prompt here).
    skeleton = re.sub(r'"[^"]*"', "", c.llm.last_system)
    assert not re.search(r"\d\.\d", skeleton)
    assert "0." not in skeleton


def test_9_same_state_diverges_by_personality():
    stoic = Personality(id="s", sensitivity=0.9, playfulness=0.3,
                        stoicism=0.9, shyness=0.5, pride=0.5, warmth=0.4)
    openc = Personality(id="o", sensitivity=0.2, playfulness=0.7,
                        stoicism=0.1, shyness=0.2, pride=0.4, warmth=0.7)
    a, b = make_carrot(stoic), make_carrot(openc)
    for _ in range(4):
        a.process_turn("You are USELESS!!!")
        a.clock.advance(5)
        b.process_turn("You are USELESS!!!")
        b.clock.advance(5)
    assert a.last_behavior.name == "sulking"
    assert b.last_behavior.name == "sad"


def test_10_contextual_tease_then_repair():
    c = make_carrot()
    c.process_turn("that joke was funny lol")
    c.process_turn("haha nice")
    baseline_hurt = c.state.emotions["hurt"]
    c.process_turn("You're useless \U0001F602")
    assert c.last_event.type == "joke"
    assert c.state.emotions["hurt"] <= baseline_hurt + 1e-9
    c.process_turn("I'm kidding.")
    assert c.last_event.type == "repair"
    assert c.state.emotions["hurt"] <= baseline_hurt + 1e-9
    assert c.last_behavior.name in ("normal", "playful", "affectionate",
                                   "curious")
    # Counter-case: same words, cold context, stay an insult.
    d = make_carrot()
    d.process_turn("You're useless \U0001F602")
    assert d.last_event.type == "insult"


def test_stub_semantic_classifier_pluggable_through_orchestrator():
    from carrot.classifier import InteractionEvent

    c = make_carrot()
    stub = StubSemanticClassifier(InteractionEvent("compliment", 0.7,
                                                   "stub verdict"))
    c.classifier = stub  # no orchestrator edits needed
    reply = c.process_turn("utterly ordinary sentence")
    assert isinstance(reply, str) and reply
    assert c.last_event.type == "compliment"
    assert c.state.emotions["happiness"] > 0.5  # appraisal still applied
    text, ctx = stub.seen[0]
    assert text == "utterly ordinary sentence"
    assert hasattr(ctx, "recent_harshness")


def test_pending_filler_excluded_end_to_end():
    c = make_carrot()
    c.process_turn("just stopping by.")
    assert c.buffer.recent()[-1].text == "hey."
    c.process_turn("anyway.")
    assert c.buffer.recent()[-1].text == "yeah?"
    # "yeah?" is filler, not a question: answering it is not curiosity.
    c.process_turn("noted.")
    assert c.last_behavior.name == "normal"
