"""Slice 3 Phase-3 tests: recall ranking, importance decay, refresh."""
from carrot.memory import (
    IMPORTANCE_FLOOR,
    IMPORTANCE_HALF_LIFE_S,
    Memory,
    MemoryStore,
    apply_importance_decay,
    decay_all,
    overlap,
    recall,
)


def _mem(text, importance=0.5, created=1000.0, recalled=1000.0,
         kind="fact"):
    return Memory(text=text, kind=kind, importance=importance,
                  created_ts=created, last_recalled_ts=recalled)


def test_recall_empty():
    assert recall((), "anything about dogs", ()) == ()


def test_recall_k_zero_and_negative():
    mems = (_mem("my dog is funny"),)
    assert recall(mems, "dog", (), k=0) == ()
    assert recall(mems, "dog", (), k=-2) == ()


def test_recall_turn_match_ranks_first():
    mems = (_mem("my dog's name is Rex"), _mem("I love ramen"))
    got = recall(mems, "what is your dog's name?", ())
    assert [m.text for m in got] == ["my dog's name is Rex",
                                     "I love ramen"]


def test_turn_overlap_beats_topic_overlap():
    turn_hit = _mem("my dog is funny", importance=0.5)
    topic_hit = _mem("I love ramen", importance=0.5)
    got = recall((turn_hit, topic_hit), "dog training tips",
                 ("ramen", "sushi"))
    assert got[0].text == "my dog is funny"


def test_importance_breaks_overlap_tie():
    high = _mem("my dog is funny", importance=0.8)
    low = _mem("that dog is funny", importance=0.3)
    got = recall((low, high), "funny dog")
    assert [m.text for m in got] == ["my dog is funny",
                                     "that dog is funny"]


def test_created_ts_breaks_full_tie():
    old = _mem("my dog is funny", importance=0.5, created=100.0)
    new = _mem("my dog is funny", importance=0.5, created=200.0)
    assert recall((new, old), "dog")[0].created_ts == 100.0


def test_text_breaks_remaining_tie():
    b_memory = _mem("my dog is funny", importance=0.5, created=100.0)
    a_memory = _mem("my cat is funny", importance=0.5, created=100.0)
    got = recall((b_memory, a_memory), "funny")
    assert [m.text for m in got] == ["my cat is funny", "my dog is funny"]


def test_recall_caps_at_k():
    mems = tuple(_mem(f"fact about dogs number {i}", created=float(i))
                 for i in range(5))
    assert len(recall(mems, "dogs", (), k=3)) == 3
    assert len(recall(mems, "dogs", (), k=99)) == 5


def test_stopwords_do_not_score():
    assert overlap("the and of", "the and of") == 0
    mems = (_mem("my dog is funny"), _mem("I love ramen"))
    assert recall(mems, "the dog")[0].text == "my dog is funny"


def test_recall_deterministic():
    mems = (_mem("my dog is funny", 0.7, 100.0),
            _mem("I love ramen", 0.4, 200.0))
    assert recall(mems, "dog ramen", ("dog",)) == \
        recall(mems, "dog ramen", ("dog",))


def test_decay_zero_elapsed_unchanged():
    m = _mem("my dog is funny", importance=0.8)
    assert apply_importance_decay(m, 0).importance == 0.8
    assert apply_importance_decay(m, -5).importance == 0.8


def test_decay_one_half_life_halves():
    m = _mem("my dog is funny", importance=0.8)
    got = apply_importance_decay(m, IMPORTANCE_HALF_LIFE_S)
    assert got.importance == 0.8 * 0.5


def test_decay_two_half_lives_quarters():
    m = _mem("my dog is funny", importance=0.8)
    got = apply_importance_decay(m, 2 * IMPORTANCE_HALF_LIFE_S)
    assert got.importance == 0.8 * 0.25


def test_decay_clamps_at_one():
    m = _mem("my dog is funny", importance=1.0)
    assert apply_importance_decay(m, 0).importance == 1.0


def test_decay_returns_new_memory():
    m = _mem("my dog is funny", importance=0.8)
    before = m.to_dict()
    apply_importance_decay(m, IMPORTANCE_HALF_LIFE_S)
    assert m.to_dict() == before


def test_decay_all_negative_elapsed_unchanged():
    m = _mem("my dog is funny", importance=0.8, recalled=2000.0)
    assert decay_all((m,), 1000.0) == (m,)


def test_decay_all_drops_below_floor():
    m = _mem("stale fact", importance=0.04, recalled=1000.0)
    assert IMPORTANCE_FLOOR == 0.05
    assert decay_all((m,), 1000.0 + IMPORTANCE_HALF_LIFE_S) == ()


def test_decay_all_keeps_above_floor():
    m = _mem("fresh fact", importance=0.06, recalled=1000.0)
    assert decay_all((m,), 1000.0) == (m,)


def test_recall_and_refresh_persists_decay(tmp_path):
    now = 2000000000.0
    store = MemoryStore(tmp_path / "memories.json")
    store.save([_mem("my dog is funny", importance=0.8,
                     recalled=now - IMPORTANCE_HALF_LIFE_S)])
    store.recall_and_refresh("tell me about cats", (), now)
    assert store.load()[0].importance == 0.8 * 0.5


def test_recall_and_refresh_updates_recalled_ts(tmp_path):
    now = 2000000000.0
    store = MemoryStore(tmp_path / "memories.json")
    store.save([_mem("my dog is funny", importance=0.8,
                     recalled=now - 1000.0)])
    hits = store.recall_and_refresh("dog", (), now)
    assert [m.last_recalled_ts for m in hits] == [now]
    assert [m.last_recalled_ts for m in store.load()] == [now]


def test_recall_and_refresh_idempotent_same_now(tmp_path):
    now = 2000000000.0
    store = MemoryStore(tmp_path / "memories.json")
    store.save([_mem("my dog is funny", importance=0.8,
                     recalled=now - 1000.0),
                _mem("I love ramen", importance=0.9,
                     recalled=now - 1000.0)])
    first = store.recall_and_refresh("dog", (), now)
    assert len(first) == 2  # both survive: guards against vacuous pass
    after_first = [m.to_dict() for m in store.load()]
    second = store.recall_and_refresh("dog", (), now)
    assert [m.to_dict() for m in second] == [m.to_dict() for m in first]
    assert [m.to_dict() for m in store.load()] == after_first
