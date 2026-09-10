"""Phase-1 tests: Turn type, buffer shape, derive_context flags/topics."""
import ast
import dataclasses
import sys

import pytest

from carrot.context import derive_context
from carrot.conversation import ConversationBuffer, Turn
from tests.conftest import ROOT


def _buf(*pairs):
    buf = ConversationBuffer()
    for role, text in pairs:
        buf.add(role, text)
    return buf


# ---- Turn + buffer ----

def test_turn_is_frozen():
    t = Turn(role="user", text="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.text = "changed"


def test_buffer_rejects_bad_role():
    with pytest.raises(ValueError):
        ConversationBuffer().add("system", "x")


def test_recent_returns_oldest_to_newest_tuple():
    buf = _buf(("user", "a"), ("carrot", "b"), ("user", "c"))
    assert buf.recent() == (Turn("user", "a"), Turn("carrot", "b"),
                            Turn("user", "c"))
    assert isinstance(buf.recent(), tuple)
    assert buf.recent(2) == (Turn("carrot", "b"), Turn("user", "c"))
    assert ConversationBuffer().recent() == ()


def test_buffer_bound_still_10():
    buf = ConversationBuffer()
    for i in range(12):
        buf.add("user", f"m{i}")
    assert len(buf) == 10
    assert buf.recent()[0] == Turn("user", "m2")


def test_history_for_llm_shape_unchanged():
    buf = _buf(("user", "hi 😂"), ("carrot", "hey."))
    assert buf.history_for_llm() == [
        {"role": "user", "content": "hi 😂"},
        {"role": "carrot", "content": "hey."},
    ]


# ---- derive_context: pending_question ----

def test_empty_input_gives_blank_context():
    ctx = derive_context(())
    assert ctx.turns == ()
    assert ctx.pending_question is False
    assert ctx.playful_frame is False
    assert ctx.recent_harshness is False
    assert ctx.recent_topics == ()


def test_pending_question_positive():
    ctx = derive_context(_buf(("user", "hey"), ("carrot", "are you there?"))
                         .recent())
    assert ctx.pending_question is True


@pytest.mark.parametrize("filler", ["yeah", "ok", "okay", "right", "huh",
                                    "hm", "hmm", "mm", "sure", "yes", "no"])
def test_pending_question_filler_negative(filler):
    ctx = derive_context(_buf(("user", "hey"),
                              ("carrot", f"{filler}?")).recent())
    assert ctx.pending_question is False


def test_pending_question_requires_carrot_last():
    ctx = derive_context(_buf(("carrot", "are you there?"),
                              ("user", "i am")).recent())
    assert ctx.pending_question is False


def test_pending_question_bare_mark():
    ctx = derive_context(_buf(("carrot", "?")).recent())
    assert ctx.pending_question is False


# ---- derive_context: playful_frame ----

def test_playful_frame_newest_cue_wins():
    assert derive_context(_buf(("user", "that joke was funny lol"),
                               ("user", "you are useless")).recent()
                          ).playful_frame is False
    assert derive_context(_buf(("user", "you are useless"),
                               ("user", "haha nice")).recent()
                          ).playful_frame is True


def test_carrot_laughter_does_not_open_frame():
    ctx = derive_context(_buf(("user", "what is 2+2?"),
                              ("carrot", "haha nice \U0001F602")).recent())
    assert ctx.playful_frame is False


def test_playful_frame_no_cues():
    assert derive_context(_buf(("user", "what is 2+2?")).recent()
                          ).playful_frame is False


# ---- derive_context: recent_harshness ----

def test_repair_widened_window_harsh_five_back():
    ctx = derive_context(_buf(
        ("user", "you are useless"),
        ("user", "alpha"), ("user", "beta"), ("user", "gamma"),
        ("user", "I'm kidding")).recent())
    assert ctx.recent_harshness is True


def test_repair_harsh_six_back_out_of_window():
    ctx = derive_context(_buf(
        ("user", "you are useless"),
        ("user", "alpha"), ("user", "beta"), ("user", "gamma"),
        ("user", "delta"),
        ("user", "I'm kidding")).recent())
    assert ctx.recent_harshness is False


def test_recent_harshness_is_precondition_not_verdict():
    # Hedge alone is not harshness; harsh alone IS (the hedge half is
    # checked against the live message by the classifier, not here).
    assert derive_context(_buf(("user", "I'm kidding")).recent()
                          ).recent_harshness is False
    assert derive_context(_buf(("user", "you are useless")).recent()
                          ).recent_harshness is True


# ---- derive_context: recent_topics ----

def test_topics_digit_stripped_and_basic():
    ctx = derive_context(_buf(("user", "playing game2 after work")).recent())
    assert "game" in ctx.recent_topics
    assert "work" in ctx.recent_topics
    assert all(tok.isalpha() for tok in ctx.recent_topics)


def test_topics_cap_dedup_stopwords():
    ctx = derive_context(_buf(
        ("user", "the game and the game needs patch notes for launch day one"),
        ("user", "game server update")).recent())
    assert len(ctx.recent_topics) <= 5
    assert len(set(ctx.recent_topics)) == len(ctx.recent_topics)
    assert "the" not in ctx.recent_topics
    assert "and" not in ctx.recent_topics
    assert ctx.recent_topics[0] == "game"


def test_topics_user_only_and_empty():
    ctx = derive_context(_buf(("carrot", "i love chess club")).recent())
    assert ctx.recent_topics == ()
    assert derive_context(_buf(("user", "the and to")).recent()
                          ).recent_topics == ()


# ---- invariants ----

def test_context_has_no_floats_and_is_pure():
    turns = _buf(("user", "game night lol?"), ("carrot", "yeah?")).recent()
    first, second = derive_context(turns), derive_context(turns)
    assert first == second
    assert list(turns) == list(_buf(("user", "game night lol?"),
                                   ("carrot", "yeah?")).recent())
    for value in (first.pending_question, first.playful_frame,
                  first.recent_harshness):
        assert isinstance(value, bool)
    assert all(isinstance(t, str) for t in first.recent_topics)


def test_context_module_has_no_forbidden_imports():
    tree = ast.parse((ROOT / "carrot" / "context.py").read_text(
        encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"carrot"}
    assert mods <= allowed, mods
    src = (ROOT / "carrot" / "context.py").read_text(encoding="utf-8")
    assert "from .classifier import" not in src
    assert "from .carrot import" not in src
