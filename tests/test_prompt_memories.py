"""Slice 3 Phase-4 tests: Memories: prompt block integration."""
import re

from carrot.memory import Memory, MemoryStore
from tests.test_scenarios import make_carrot


def _skeleton(system):
    return re.sub(r'"[^"]*"', "", system)


def _seed(carrot, *texts, importance=0.5, ts=1000.0):
    for i, text in enumerate(texts):
        assert carrot.memory_store.add(
            Memory(text=text, kind="fact", importance=importance,
                   created_ts=ts + i, last_recalled_ts=ts + i))


def test_memories_block_present():
    c = make_carrot()
    c.process_turn("my dog's name is Rex")
    c.clock.advance(5)
    c.process_turn("what is my dog called")
    assert "Memories: You told me: " in c.llm.last_system


def test_memories_block_absent():
    c = make_carrot()
    c.process_turn("12345 !!!")
    assert "Memories:" not in c.llm.last_system
    assert "Recent topics:" not in c.llm.last_system


def test_multiple_memories_one_block():
    c = make_carrot()
    _seed(c, "my dog is funny", "my cat is grumpy", "my bird is loud")
    c.process_turn("dog cat bird")
    system = c.llm.last_system
    assert system.count("Memories:") == 1
    for text in ("my dog is funny", "my cat is grumpy", "my bird is loud"):
        assert f'You told me: "{text}".' in system


def test_digit_inside_quotes_skeleton_clean():
    c = make_carrot()
    _seed(c, "my birthday is may 3")
    c.process_turn("when is my birthday")
    system = c.llm.last_system
    assert '"my birthday is may 3"' in system
    assert re.search(r"\d", _skeleton(system)) is None


def test_float_inside_quotes_skeleton_clean():
    c = make_carrot()
    _seed(c, "the price is 0.95 dollars")
    c.process_turn("what was the price")
    system = c.llm.last_system
    assert '"the price is 0.95 dollars"' in system
    assert re.search(r"\d", _skeleton(system)) is None


def test_state_name_inside_quotes_skeleton_clean():
    c = make_carrot()
    _seed(c, "my anger problems are bad")
    c.process_turn("tell me about anger problems")
    system = c.llm.last_system
    assert '"my anger problems are bad"' in system
    assert re.search(r"\banger\b", _skeleton(system)) is None


def test_mood_label_inside_quotes_skeleton_clean():
    c = make_carrot()
    _seed(c, "my mood is low today")
    c.process_turn("how is my mood today")
    system = c.llm.last_system
    assert '"my mood is low today"' in system
    skeleton = _skeleton(system)
    assert re.search(r"\blow\b", skeleton) is None
    assert re.search(r"\bmood\b", skeleton) is None


def test_long_memory_truncated_with_ellipsis():
    c = make_carrot()
    _seed(c, "z" * 140)
    c.process_turn("hello there")
    system = c.llm.last_system
    assert "…" in system
    assert "z" * 140 not in system
    assert ("z" * 120 + "…") in system


def test_embedded_quotes_escaped():
    c = make_carrot()
    _seed(c, 'she said "hi" loudly')
    c.process_turn("what did she say")
    assert 'she said \\"hi\\" loudly' in c.llm.last_system


def test_memories_before_context():
    c = make_carrot()
    c.process_turn("my dog's name is Rex")
    c.clock.advance(5)
    c.process_turn("that joke was funny lol")
    c.clock.advance(5)
    c.process_turn("tell me about rex lol")
    system = c.llm.last_system
    assert "Memories:" in system
    assert "Context:" in system
    assert system.index("Memories:") < system.index("Context:")


def test_end_to_end_extraction_to_recall():
    c = make_carrot()
    c.process_turn("my name is Alice")
    c.clock.advance(5)
    c.process_turn("nice weather today")
    c.clock.advance(5)
    c.process_turn("what was my name")
    assert '"my name is alice"' in c.llm.last_system


def test_recent_causes_never_rendered():
    c = make_carrot()
    c.process_turn("You are useless.")
    c.clock.advance(5)
    c.process_turn("You're cute!")
    system = c.llm.last_system
    assert "insult" not in system
    assert "compliment" not in system


def test_memory_survives_instance_restart(tmp_path):
    store_path = tmp_path / "mem.json"
    first = make_carrot(memory_store=MemoryStore(store_path))
    first.process_turn("my name is Alice")
    assert (tmp_path / "mem.json").exists()
    second = make_carrot(memory_store=MemoryStore(store_path))
    second.process_turn("what is my name")
    assert '"my name is alice"' in second.llm.last_system
