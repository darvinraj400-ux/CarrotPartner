"""Phase-5 tests: /ctx handler, last_context retention, help text."""
import re

from carrot.cli import HELP_TEXT, NO_CONTEXT_YET, format_ctx, handle_ctx
from carrot.context import derive_context
from tests.test_scenarios import make_carrot


def test_ctx_before_any_turn():
    c = make_carrot()
    assert c.last_context is None
    assert handle_ctx(c) == NO_CONTEXT_YET


def test_last_context_matches_derive():
    c = make_carrot()
    c.process_turn("talking about gardening")
    # First turn was interpreted against an empty pre-turn context.
    assert c.last_context == derive_context(())
    assert "turns: 0" in handle_ctx(c)
    c.process_turn("my guitar needs tuning")
    prior = tuple(c.buffer.recent()[:-2])
    assert c.last_context == derive_context(prior)
    out = handle_ctx(c)
    assert "pending_question: False" in out
    assert "playful_frame: False" in out
    assert "recent_harshness: False" in out
    assert "recent_topics: [talking, about, gardening]" in out
    assert "turns: 2" in out


def test_ctx_shows_flags_and_recent_turns():
    c = make_carrot()
    c.process_turn("that joke was funny lol")
    assert "playful_frame: False" in handle_ctx(c)  # pre-turn was empty
    c.process_turn("more jokes please")
    out = handle_ctx(c)
    assert "playful_frame: True" in out
    assert "turns: 2" in out
    assert "user: that joke was funny lol" in out
    assert re.search(r"\d\.\d", out) is None
    assert "happiness" not in out


def test_ctx_does_not_mutate():
    c = make_carrot()
    c.process_turn("hello there?")
    before_state = c.state.to_dict()
    before_buffer = list(c.buffer.recent())
    handle_ctx(c)
    format_ctx(c.last_context)
    assert c.state.to_dict() == before_state
    assert list(c.buffer.recent()) == before_buffer


def test_help_lists_ctx():
    assert "/ctx" in HELP_TEXT
    assert "/dbg" in HELP_TEXT
    assert "/quit" in HELP_TEXT
