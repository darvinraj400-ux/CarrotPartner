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
    hits, _ = store.recall_and_refresh("dog", (), now)
    assert [m.last_recalled_ts for m in hits] == [now]
    assert [m.last_recalled_ts for m in store.load()] == [now]


def test_recall_and_refresh_idempotent_same_now(tmp_path):
    now = 2000000000.0
    store = MemoryStore(tmp_path / "memories.json")
    store.save([_mem("my dog is funny", importance=0.8,
                     recalled=now - 1000.0),
                _mem("I love ramen", importance=0.9,
                     recalled=now - 1000.0)])
    first, _ = store.recall_and_refresh("dog", (), now)
    assert len(first) == 2  # both survive: guards against vacuous pass
    after_first = [m.to_dict() for m in store.load()]
    second, _ = store.recall_and_refresh("dog", (), now)
    assert [m.to_dict() for m in second] == [m.to_dict() for m in first]
    assert [m.to_dict() for m in store.load()] == after_first


def test_recall_golden_baseline_default_args():
    # Golden baseline: pins the default-args ranking so the Phase 1
    # boost_text diff is reviewable. Scores: A = 2*2 + 1*1 + 0.2 = 5.2,
    # C = 2*1 + 1*0 + 0.1 = 2.1, B = 0 + 0 + 0.9 = 0.9. No ties.
    mems = [
        _mem("my dog rex rules the park", importance=0.2, created=100.0),
        _mem("i love pizza", importance=0.9, created=200.0),
        _mem("rex likes bones", importance=0.1, created=300.0),
    ]
    got = recall(mems, "my dog rex is funny", ("park",))
    assert [m.text for m in got] == [
        "my dog rex rules the park",
        "rex likes bones",
        "i love pizza",
    ]


# --- Slice 10 Phase 1: boost_text tests ---
# The Phase 0 golden test above passes unmodified with default
# boost_text="" (verified by the full suite run after this append).


def test_boost_matches_one_memory():
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    baseline = recall(mems, "hello there", ())
    boosted = recall(mems, "hello there", (), boost_text="games")
    assert [m.text for m in baseline] == ["i love pizza", "i love games"]
    assert [m.text for m in boosted] == ["i love games", "i love pizza"]


def test_boost_matches_multiple_memories():
    mems = [_mem("i love games", created=300.0),
            _mem("games are fun", created=200.0),
            _mem("i love pizza", created=100.0)]
    baseline = recall(mems, "hello there", ())
    boosted = recall(mems, "hello there", (), boost_text="games")
    assert [m.text for m in baseline] == \
        ["i love pizza", "games are fun", "i love games"]
    # Both boosted by +1; relative order preserved via created_ts tiebreak.
    assert [m.text for m in boosted] == \
        ["games are fun", "i love games", "i love pizza"]


def test_boost_is_additive_not_multiplier():
    # A: turn 1 (2.0) + imp 0.5 = 2.5; B: turn 1 + topics 1 + imp 0.2 = 3.2.
    # A +1.0 boost (3.5) flips the order; a x1.2 multiplier (3.0) would not.
    # (Importance clamps to [0, 1], so B's margin is built with topics.)
    mems = [_mem("my dog rex", importance=0.5, created=100.0),
            _mem("my dog park plaza", importance=0.2, created=200.0)]
    baseline = recall(mems, "dog", ("park",))
    boosted = recall(mems, "dog", ("park",), boost_text="rex")
    assert [m.text for m in baseline] == \
        ["my dog park plaza", "my dog rex"]
    assert [m.text for m in boosted] == \
        ["my dog rex", "my dog park plaza"]


def test_boost_counts_distinct_tokens_not_repetitions():
    # overlap() uses set(tokenize()) intersection: "games games games"
    # overlaps "games" by exactly 1, same as plain "games".
    assert overlap("games games games", "games") == 1
    mems = [_mem("games games games", created=100.0),
            _mem("games", created=200.0)]
    boosted = recall(mems, "hello there", (), boost_text="games")
    # Equal 1.5 scores; created_ts tiebreak puts the triple first.
    assert [m.text for m in boosted] == ["games games games", "games"]


def test_empty_boost_matches_default():
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    assert recall(mems, "hello", (), boost_text="") == \
        recall(mems, "hello", ())


def test_boost_stopwords_only_no_effect():
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    assert recall(mems, "hello", (), boost_text="the and of") == \
        recall(mems, "hello", ())


def test_boost_digits_only_no_effect():
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    assert recall(mems, "hello", (), boost_text="123 456") == \
        recall(mems, "hello", ())


def test_boost_no_overlap_no_change():
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    assert recall(mems, "hello", (), boost_text="zebra") == \
        recall(mems, "hello", ())


def test_boost_preserves_tiebreak():
    mems = [_mem("dog rex", created=200.0),
            _mem("dog rex", created=100.0)]
    boosted = recall(mems, "hello there", (), boost_text="rex")
    assert [m.text for m in boosted] == ["dog rex", "dog rex"]
    assert [m.created_ts for m in boosted] == [100.0, 200.0]


def test_turn_overlap_beats_boost():
    # A: turn 1 (2.0) + imp 0.5 = 2.5; B: boost 1 (1.0) + imp 0.5 = 1.5.
    mems = [_mem("my dog rex", created=200.0),
            _mem("my cat tom", created=100.0)]
    got = recall(mems, "dog", (), boost_text="tom")
    assert [m.text for m in got] == ["my dog rex", "my cat tom"]


def test_topics_and_boost_equal_weight():
    # A: topics 1 (1.0) + imp 0.5 = 1.5; B: boost 1 (1.0) + imp 0.5 = 1.5.
    # Tie at equal weight; created_ts tiebreak puts B first.
    mems = [_mem("my dog park", created=200.0),
            _mem("my cat tom", created=100.0)]
    got = recall(mems, "hello there", ("park",), boost_text="tom")
    assert [m.text for m in got] == ["my cat tom", "my dog park"]


def test_combined_turn_topics_boost_scores_four():
    # A: turn 1 + topics 1 + boost 1 + imp 0.0 = 4.0.
    # B: turn 1 + topics 1 + boost 0 + imp 1.0 = 4.0.
    # Tie is only consistent with boost contributing exactly +1.0.
    mems = [_mem("my dog rex park", importance=0.0, created=100.0),
            _mem("my dog park", importance=1.0, created=200.0)]
    got = recall(mems, "dog", ("park",), boost_text="rex")
    assert [m.text for m in got] == ["my dog rex park", "my dog park"]


def test_recall_and_refresh_forwards_boost(tmp_path):
    path = tmp_path / "memories.json"
    store = MemoryStore(path)
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    store.save(mems)
    got, _ = store.recall_and_refresh("hello there", (), 1000.0,
                                      boost_text="games")
    expected = recall(mems, "hello there", (), boost_text="games")
    assert [m.text for m in got] == [m.text for m in expected]


def test_recall_and_refresh_default_preserves_behavior(tmp_path):
    path = tmp_path / "memories.json"
    store = MemoryStore(path)
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    store.save(mems)
    assert [m.text for m in
            store.recall_and_refresh("hello there", (), 1000.0)[0]] == \
        [m.text for m in
         store.recall_and_refresh("hello there", (), 1000.0,
                                  boost_text="")[0]]


def test_boost_text_is_keyword_only():
    mems = [_mem("i love games")]
    try:
        recall(mems, "text", (), 3, "games")
    except TypeError:
        pass
    else:
        raise AssertionError("positional boost_text did not raise")


# --- Slice 10 Phase 2: recall_and_refresh pair tests ---


def test_recall_and_refresh_returns_pair(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.save([_mem("i love games", created=100.0)])
    pair = store.recall_and_refresh("hello there", (), 1000.0)
    assert isinstance(pair, tuple) and len(pair) == 2
    hits, changed = pair
    assert isinstance(hits, tuple)
    assert all(isinstance(m, Memory) for m in hits)
    assert isinstance(changed, bool)


def test_recall_and_refresh_empty_boost_never_changed(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.save([_mem("my browser tabs", created=100.0),
                _mem("i love pizza", created=200.0)])
    _, changed = store.recall_and_refresh("hello there", (), 1000.0,
                                          boost_text="")
    assert changed is False


def test_recall_and_refresh_boost_reorders_topk(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.save([_mem("i love pizza", created=100.0),
                _mem("my browser tabs", created=200.0)])
    hits, changed = store.recall_and_refresh("hello there", (), 1000.0,
                                             boost_text="browser")
    assert changed is True
    assert [m.text for m in hits] == ["my browser tabs", "i love pizza"]


def test_recall_and_refresh_boost_no_match_unchanged(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.save([_mem("i love pizza", created=100.0),
                _mem("my browser tabs", created=200.0)])
    hits, changed = store.recall_and_refresh("hello there", (), 1000.0,
                                             boost_text="zebra")
    assert changed is False
    assert [m.text for m in hits] == ["i love pizza", "my browser tabs"]


def test_recall_and_refresh_refreshes_only_boosted(tmp_path):
    now = 2000000000.0
    store = MemoryStore(tmp_path / "memories.json")
    store.save([_mem("i love pizza", created=100.0,
                      recalled=now - 1000.0),
                _mem("my browser tabs", created=200.0,
                      recalled=now - 1000.0),
                _mem("i love ramen", created=300.0,
                      recalled=now - 1000.0)])
    hits, changed = store.recall_and_refresh("hello there", (), now, k=1,
                                             boost_text="browser")
    assert changed is True
    assert [m.text for m in hits] == ["my browser tabs"]
    assert [m.last_recalled_ts for m in hits] == [now]
    by_text = {m.text: m for m in store.load()}
    assert by_text["my browser tabs"].last_recalled_ts == now
    assert by_text["i love pizza"].last_recalled_ts == now - 1000.0
    assert by_text["i love ramen"].last_recalled_ts == now - 1000.0


# --- Slice 11 Phase 4: boost_weight tests ---


def test_boost_weight_default_is_one():
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    assert recall(mems, "hello", (), boost_text="games") == \
        recall(mems, "hello", (), boost_text="games", boost_weight=1.0)


def test_boost_weight_is_keyword_only():
    mems = [_mem("i love games")]
    try:
        recall(mems, "text", (), 3, "", 1.0)
    except TypeError:
        pass
    else:
        raise AssertionError("positional boost_weight did not raise")


def test_boost_weight_scales_rank():
    # A matches boost (imp 0.0); B imp 1.0 created earlier.
    # w=1.0: tie 1.0/1.0 -> B first; w=2.0: A (2.0) outranks B.
    mems = [_mem("i love games", importance=0.0, created=200.0),
            _mem("i love pizza", importance=1.0, created=100.0)]
    light = recall(mems, "hello", (), boost_text="games",
                   boost_weight=1.0)
    heavy = recall(mems, "hello", (), boost_text="games",
                   boost_weight=2.0)
    assert [m.text for m in light] == ["i love pizza", "i love games"]
    assert [m.text for m in heavy] == ["i love games", "i love pizza"]


def test_boost_weight_zero_matches_empty_boost():
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    assert recall(mems, "hello", (), boost_text="games",
                  boost_weight=0.0) == \
        recall(mems, "hello", (), boost_text="")


def test_boost_weight_ignores_non_matching():
    mems = [_mem("i love pizza", created=100.0),
            _mem("i love ramen", created=200.0)]
    assert recall(mems, "hello", (), boost_text="games",
                  boost_weight=0.5) == \
        recall(mems, "hello", (), boost_text="games", boost_weight=2.0)


def test_boost_weight_negative_is_linear():
    # No clamp inside recall: contribution is weight * overlap, even
    # negative. The orchestrator never sends a negative weight
    # (0.5 + closeness >= 0.5); this pins the well-defined shape.
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    got = recall(mems, "hello", (), boost_text="games",
                 boost_weight=-1.0)
    assert [m.text for m in got] == ["i love pizza", "i love games"]


def test_recall_and_refresh_forwards_boost_weight(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    store.save(mems)
    got, _ = store.recall_and_refresh("hello", (), 1000.0,
                                      boost_text="games",
                                      boost_weight=2.0)
    expected = recall(mems, "hello", (), boost_text="games",
                      boost_weight=2.0)
    assert [m.text for m in got] == [m.text for m in expected]


def test_recall_and_refresh_default_weight_preserved(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    mems = [_mem("i love games", created=200.0),
            _mem("i love pizza", created=100.0)]
    store.save(mems)
    assert [m.text for m in
            store.recall_and_refresh("hello", (), 1000.0,
                                     boost_text="games")[0]] == \
        [m.text for m in
         store.recall_and_refresh("hello", (), 1000.0,
                                  boost_text="games",
                                  boost_weight=1.0)[0]]
