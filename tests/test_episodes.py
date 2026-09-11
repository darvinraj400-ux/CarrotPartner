"""Episodic Continuity Phase-1 tests: episode data layer (pure, headless)."""
import ast
import json
import sys

import pytest

from carrot.episodes import (
    JOIN_WINDOW_S,
    MAX_EPISODES,
    MAX_FILE_BYTES,
    MAX_TOPIC_TOKENS,
    Episode,
    EpisodeStore,
    EpisodesConfig,
    _merge_tokens,
    _overlap_count,
    derive,
    recall,
)
from tests.test_scenarios import make_carrot
from tests.conftest import ROOT


def _ep(**over):
    fields = dict(first_seen_ts=1000.0, last_seen_ts=2000.0, turn_count=3,
                  topic_tokens=("garden", "rain"), mood_label="content",
                  last_behavior="normal")
    fields.update(over)
    return Episode(**fields)


def _cfg():
    return EpisodesConfig.load(str(ROOT / "config" / "episodes.json"))


# ---- module boundary ----


def test_episodes_module_boundaries():
    banned_mods = {"carrot", "cli", "llm"}
    banned_calls = {"process_turn", "generate"}
    banned_native = {"subprocess", "ctypes", "threading", "asyncio",
                     "sched", "multiprocessing", "Timer"}
    offenders = []
    tree = ast.parse((ROOT / "carrot" / "episodes.py").read_text(
        encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in banned_mods or (parts[0] == "carrot"
                                               and len(parts) > 1):
                    offenders.append(a.name)
                if parts[0] in banned_native:
                    offenders.append(a.name)
                if a.name in banned_native:
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


# ---- Episode ----


def test_episode_construction_all_fields():
    e = _ep()
    assert e.first_seen_ts == 1000.0
    assert e.last_seen_ts == 2000.0
    assert e.turn_count == 3
    assert e.topic_tokens == ("garden", "rain")
    assert e.mood_label == "content"
    assert e.last_behavior == "normal"


def test_episode_construction_default_behavior():
    assert _ep(last_behavior=None).last_behavior is None


def test_episode_frozen():
    with pytest.raises(Exception):
        _ep().turn_count = 9


def test_episode_negative_ts_rejected():
    with pytest.raises(ValueError):
        _ep(first_seen_ts=-1.0)
    with pytest.raises(ValueError):
        _ep(last_seen_ts=-1.0)


def test_episode_last_before_first_rejected():
    with pytest.raises(ValueError):
        _ep(first_seen_ts=3000.0, last_seen_ts=2000.0)


def test_episode_bad_turn_count():
    with pytest.raises(ValueError):
        _ep(turn_count=0)
    with pytest.raises(ValueError):
        _ep(turn_count="one")
    with pytest.raises(ValueError):
        _ep(turn_count=True)


def test_episode_too_many_tokens():
    with pytest.raises(ValueError):
        _ep(topic_tokens=tuple(f"t{i}" for i in range(MAX_TOPIC_TOKENS + 1)))


def test_episode_bad_tokens():
    with pytest.raises(ValueError):
        _ep(topic_tokens=("ok", 42))
    with pytest.raises(ValueError):
        _ep(topic_tokens=("ok", ""))


def test_episode_bad_mood():
    with pytest.raises(ValueError):
        _ep(mood_label="manic")


def test_episode_empty_behavior_rejected():
    with pytest.raises(ValueError):
        _ep(last_behavior="")


def test_episode_bool_ts_rejected():
    with pytest.raises(ValueError):
        _ep(first_seen_ts=True)


def test_episode_round_trip():
    e = _ep()
    assert Episode.from_dict(e.to_dict()) == e


def test_episode_from_dict_non_dict():
    with pytest.raises(ValueError):
        Episode.from_dict(["nope"])


def test_episode_from_dict_skips_unknown_fields():
    e = Episode.from_dict(dict(_ep().to_dict(), future_field="x"))
    assert e == _ep()


def test_episode_deterministic():
    assert _ep() == _ep()


# ---- EpisodesConfig ----


def test_config_loads_defaults():
    cfg = _cfg()
    assert cfg.join_window_s == 1800.0
    assert cfg.max_episodes == 50
    assert cfg.max_topic_tokens == 10
    assert cfg.retrieval_min_overlap == 2
    assert cfg.recall_k == 1
    assert JOIN_WINDOW_S == 1800.0


def test_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        EpisodesConfig.load(str(tmp_path / "nope.json"))


def test_config_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        EpisodesConfig.load(str(path))


def test_config_missing_field_keeps_default(tmp_path):
    path = tmp_path / "part.json"
    path.write_text(json.dumps({"max_episodes": 10}), encoding="utf-8")
    cfg = EpisodesConfig.load(str(path))
    assert cfg.max_episodes == 10
    assert cfg.join_window_s == 1800.0


def test_config_bad_join_window(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"join_window_s": -1}), encoding="utf-8")
    with pytest.raises(ValueError):
        EpisodesConfig.load(str(path))


def test_config_bad_max_episodes(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"max_episodes": 0}), encoding="utf-8")
    with pytest.raises(ValueError):
        EpisodesConfig.load(str(path))


def test_config_bad_recall_k(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"recall_k": 0}), encoding="utf-8")
    with pytest.raises(ValueError):
        EpisodesConfig.load(str(path))


def test_config_frozen():
    with pytest.raises(Exception):
        _cfg().max_episodes = 10


# ---- EpisodeStore ----


def _store(tmp_path, name="episodes.json"):
    return EpisodeStore(tmp_path / name)


def test_store_missing_file(tmp_path):
    assert _store(tmp_path).load() == ()
    assert not (tmp_path / "episodes.json").exists()


def test_store_round_trip(tmp_path):
    store = _store(tmp_path)
    first = _ep()
    second = _ep(first_seen_ts=3000.0, last_seen_ts=4000.0,
                 turn_count=1, topic_tokens=("soup",), mood_label="bright",
                 last_behavior=None)
    store.save((first, second))
    assert store.load() == (first, second)


def test_store_corrupt_json(tmp_path):
    path = tmp_path / "episodes.json"
    path.write_text("{not json", encoding="utf-8")
    assert _store(tmp_path).load() == ()
    assert not path.exists()
    assert len(list(tmp_path.glob("quarantine-episodes-*.json"))) == 1


def test_store_unparseable_version(tmp_path):
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"version": "x", "episodes": []}),
                    encoding="utf-8")
    assert _store(tmp_path).load() == ()
    assert len(list(tmp_path.glob("quarantine-episodes-*.json"))) == 1


def test_store_unknown_version(tmp_path):
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"version": 2, "episodes": []}),
                    encoding="utf-8")
    assert _store(tmp_path).load() == ()
    assert list(tmp_path.glob("quarantine-episodes-*.json")) == []


def test_store_non_object_top_level(tmp_path):
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert _store(tmp_path).load() == ()
    assert len(list(tmp_path.glob("quarantine-episodes-*.json"))) == 1


def test_store_per_entry_skip(tmp_path):
    path = tmp_path / "episodes.json"
    good = _ep().to_dict()
    path.write_text(json.dumps({"version": 1,
                                "episodes": [good, {"nope": True}]}),
                    encoding="utf-8")
    assert _store(tmp_path).load() == (_ep(),)


def test_store_load_time_trim(tmp_path):
    path = tmp_path / "episodes.json"
    entries = []
    for i in range(60):
        entries.append(_ep(first_seen_ts=float(i),
                            last_seen_ts=float(1000 + i),
                            turn_count=1,
                            topic_tokens=(f"t{i % 10}",)).to_dict())
    path.write_text(json.dumps({"version": 1, "episodes": entries}),
                    encoding="utf-8")
    got = _store(tmp_path).load()
    assert len(got) == MAX_EPISODES == 50
    assert {e.last_seen_ts for e in got} == \
        {float(1000 + i) for i in range(10, 60)}
    assert [e.last_seen_ts for e in got] == sorted(
        e.last_seen_ts for e in got)


def test_store_cap_on_save(tmp_path):
    with pytest.raises(ValueError):
        _store(tmp_path).save(tuple(_ep() for _ in range(51)))


def test_store_type_check_on_save(tmp_path):
    with pytest.raises(ValueError):
        _store(tmp_path).save([_ep(), "not-an-episode"])


def test_store_size_cap(tmp_path, monkeypatch):
    import carrot.episodes as ep

    assert ep.MAX_FILE_BYTES == 16384
    monkeypatch.setattr(ep, "MAX_FILE_BYTES", 10)
    with pytest.raises(ValueError):
        _store(tmp_path).save((_ep(),))
    assert not (tmp_path / "episodes.json").exists()


def test_store_byte_cap_value_pinned():
    import carrot.episodes as ep

    # Deliberate: 50 full records serialize to ~8.5 KB, so the byte cap
    # must stay above that for the count cap to remain binding. Any
    # change here reopens the Phase 1 sizing question.
    assert ep.MAX_FILE_BYTES == 16384


def test_store_full_count_round_trip(tmp_path):
    store = _store(tmp_path)
    moods = ("content", "bright", "low", "tense", "tender")
    originals = tuple(
        _ep(first_seen_ts=float(1000 + i * 100),
            last_seen_ts=float(1100 + i * 100),
            turn_count=(i % 7) + 1,
            topic_tokens=(f"topic{i}a", f"topic{i}b"),
            mood_label=moods[i % len(moods)],
            last_behavior="normal" if i % 2 == 0 else None)
        for i in range(MAX_EPISODES))
    assert len(originals) == 50
    store.save(originals)
    assert (tmp_path / "episodes.json").is_file()
    assert store.load() == originals


def test_store_add_under_cap(tmp_path):
    store = _store(tmp_path)
    assert store.add(_ep()) is True
    assert store.load() == (_ep(),)


def test_store_add_at_cap(tmp_path):
    store = _store(tmp_path)
    store.save(tuple(_ep() for _ in range(MAX_EPISODES)))
    mtime = (tmp_path / "episodes.json").stat().st_mtime_ns
    assert store.add(_ep()) is False
    assert (tmp_path / "episodes.json").stat().st_mtime_ns == mtime


def test_store_remove_matching(tmp_path):
    store = _store(tmp_path)
    store.save((_ep(topic_tokens=("garden", "rain")),
                _ep(topic_tokens=("garden", "soup")),
                _ep(topic_tokens=("chess",))))
    assert store.remove_matching("garden") == 2
    assert store.load() == (_ep(topic_tokens=("chess",)),)


def test_store_remove_no_match(tmp_path):
    store = _store(tmp_path)
    store.save((_ep(),))
    mtime = (tmp_path / "episodes.json").stat().st_mtime_ns
    assert store.remove_matching("zebra") == 0
    assert (tmp_path / "episodes.json").stat().st_mtime_ns == mtime


def test_store_remove_empty_string(tmp_path):
    store = _store(tmp_path)
    store.save((_ep(),))
    assert store.remove_matching("") == 0
    assert store.remove_matching("   ") == 0


def test_store_remove_case_insensitive(tmp_path):
    store = _store(tmp_path)
    store.save((_ep(topic_tokens=("Garden",)),))
    assert store.remove_matching("garden") == 1
    assert store.load() == ()


def test_store_no_tmp_remnants(tmp_path):
    _store(tmp_path).save((_ep(),))
    assert list(tmp_path.glob("*.tmp")) == []


def test_store_deterministic_time_fn(tmp_path):
    calls = []
    store = EpisodeStore(tmp_path / "episodes.json",
                         time_fn=lambda: calls.append(1) or 777.0)
    (tmp_path / "episodes.json").write_text("{bad", encoding="utf-8")
    assert store.load() == ()
    assert calls == [1]
    assert (tmp_path / "quarantine-episodes-777.json").is_file()


def test_store_deterministic(tmp_path):
    first = EpisodeStore(tmp_path / "a.json")
    second = EpisodeStore(tmp_path / "b.json")
    first.save((_ep(),))
    second.save((_ep(),))
    assert first.load() == second.load()


# --- Episodic Continuity Phase 2: runtime content pin ---


def test_episode_store_never_contains_turn_text(tmp_path):
    # The current test suite has no structural defense against an episode
    # store quietly becoming a transcript. This pin asserts that the raw
    # user text of a turn never appears in the persisted episode file,
    # even when the turn's tokens are stored. Combined with the AST field-
    # name pin from Phase 0, this closes the DP1-for-episodes gap.
    from carrot.context import tokenize

    sentinel = "Hey, this is a SECRET-CODE-99 phrase!"
    turn_tokens = tokenize(sentinel)
    # Multi-word with digits, punctuation, mixed case, and stopwords:
    # tokenization strips everything but the content tokens.
    assert len(turn_tokens) >= 2
    assert turn_tokens != (sentinel,)
    store = EpisodeStore(tmp_path / "episodes.json")
    (result,) = derive((), turn_tokens, 5000.0, "content",
                       config=EpisodesConfig(),
                       last_behavior="normal")
    assert store.add(result) is True
    raw = (tmp_path / "episodes.json").read_bytes().decode("utf-8")
    # The tokens made it to disk (the pin is not vacuous)...
    for token in turn_tokens:
        assert token in raw
    # ...but the verbatim turn text did not.
    assert sentinel not in raw


# --- Episodic Continuity Phase 2: derivation tests ---


def _cfg(**over):
    return EpisodesConfig(**over)


def _old_ep(**over):
    fields = dict(first_seen_ts=1000.0, last_seen_ts=2000.0, turn_count=2,
                  topic_tokens=("garden", "rain"), mood_label="content",
                  last_behavior="normal")
    fields.update(over)
    return Episode(**fields)


def test_derive_empty_creates():
    (got,) = derive((), ("garden", "soil"), 5000.0, "bright",
                    config=_cfg(), last_behavior="curious")
    assert got.turn_count == 1
    assert got.first_seen_ts == got.last_seen_ts == 5000.0
    assert got.topic_tokens == ("garden", "soil")
    assert got.mood_label == "bright"
    assert got.last_behavior == "curious"


def test_derive_empty_no_behavior():
    (got,) = derive((), ("garden",), 5000.0, "content", config=_cfg())
    assert got.last_behavior is None


def test_derive_empty_tokens_empty():
    (got,) = derive((), (), 5000.0, "content", config=_cfg())
    assert got.topic_tokens == ()
    assert got.turn_count == 1


def test_derive_create_appends(tmp_path):
    first = _old_ep()
    got = derive((first,), ("chess",), 9000.0, "content", config=_cfg())
    assert len(got) == 2
    assert got[0] == first
    assert got[1].topic_tokens == ("chess",)
    assert got[1].turn_count == 1


def test_derive_join_updates_in_place():
    first = _old_ep()
    got = derive((first,), ("rain", "soil"), 2500.0, "bright",
                 config=_cfg(), last_behavior="curious")
    assert len(got) == 1
    updated = got[0]
    assert updated.turn_count == 3
    assert updated.first_seen_ts == 1000.0
    assert updated.last_seen_ts == 2500.0
    assert updated.mood_label == "bright"
    assert updated.last_behavior == "curious"
    assert updated.topic_tokens == ("garden", "rain", "soil")


def test_derive_join_preserves_behavior_when_none():
    first = _old_ep()
    (got,) = derive((first,), ("rain",), 2500.0, "content", config=_cfg())
    assert got.last_behavior == "normal"


def test_derive_join_merges_fifo_cap():
    cfg = _cfg(max_topic_tokens=3)
    first = _old_ep(topic_tokens=("a", "b", "c"))
    (got,) = derive((first,), ("c", "d", "e"), 2500.0, "content", config=cfg)
    assert got.topic_tokens == ("c", "d", "e")


def test_derive_earlier_untouched():
    first = _old_ep(first_seen_ts=100.0, last_seen_ts=200.0,
                    topic_tokens=("chess",))
    second = _old_ep()
    got = derive((first, second), ("rain",), 2500.0, "content",
                 config=_cfg())
    assert got[0] == first
    assert got[1].turn_count == 3


def test_derive_exact_window_joins():
    first = _old_ep()
    got = derive((first,), ("rain",), 2000.0 + 1800.0, "content",
                 config=_cfg())
    assert len(got) == 1
    assert got[0].turn_count == 3


def test_derive_past_window_creates():
    first = _old_ep()
    got = derive((first,), ("rain",), 2000.0 + 1800.0 + 0.001, "content",
                 config=_cfg())
    assert len(got) == 2
    assert got[0] == first


def test_derive_zero_elapsed_joins():
    first = _old_ep()
    (got,) = derive((first,), ("garden",), 2000.0, "content",
                    config=_cfg())
    assert got.turn_count == 3


def test_derive_pivot_creates():
    first = _old_ep()
    got = derive((first,), ("chess", "tournament"), 2500.0, "content",
                 config=_cfg())
    assert len(got) == 2
    assert got[0] == first
    assert got[1].topic_tokens == ("chess", "tournament")


def test_derive_cross_day_creates():
    first = _old_ep()
    got = derive((first,), ("rain",), 2000.0 + 86400.0, "content",
                 config=_cfg())
    assert len(got) == 2
    assert got[0] == first


def test_derive_latest_by_last_seen():
    older = _old_ep(first_seen_ts=100.0, last_seen_ts=3000.0,
                    topic_tokens=("chess",))
    newer = _old_ep(first_seen_ts=3500.0, last_seen_ts=4000.0,
                    topic_tokens=("garden",))
    got = derive((older, newer), ("garden",), 4500.0, "content",
                 config=_cfg())
    assert got[0] == older
    assert got[1].turn_count == 3


def test_merge_basic_order():
    assert _merge_tokens(("a", "b"), ("c", "d"), 10) == ("a", "b", "c", "d")


def test_merge_dedup_first():
    assert _merge_tokens(("a", "b"), ("b", "c"), 10) == ("a", "b", "c")


def test_merge_cap_fifo():
    assert _merge_tokens(("a", "b", "c"), ("d", "e"), 3) == ("c", "d", "e")


def test_merge_empty_incoming():
    assert _merge_tokens(("a", "b"), (), 10) == ("a", "b")


def test_merge_errors():
    with pytest.raises(ValueError):
        _merge_tokens(("a", 42), ("b",), 10)
    with pytest.raises(ValueError):
        _merge_tokens(("a",), ("b",), 0)
    with pytest.raises(ValueError):
        _merge_tokens(("a",), ("b",), -2)


def test_derive_no_mutation():
    first = _old_ep()
    before = (first, first.to_dict())
    got = derive((first,), ("rain",), 2500.0, "content", config=_cfg())
    assert first == before[0]
    assert first.to_dict() == before[1]
    assert got[0] is not first


def test_derive_deterministic():
    args = ((_old_ep(),), ("rain",), 2500.0, "content")
    first = derive(*args, config=_cfg(), last_behavior="curious")
    second = derive(*args, config=_cfg(), last_behavior="curious")
    assert first == second


def test_derive_validation():
    good = dict(config=_cfg())
    with pytest.raises(ValueError):
        derive((_old_ep(),), ("rain",), "now", "content", **good)
    with pytest.raises(ValueError):
        derive((_old_ep(),), ("rain",), -1.0, "content", **good)
    with pytest.raises(ValueError):
        derive((_old_ep(),), ("rain",), 2500.0, "manic", **good)
    with pytest.raises(ValueError):
        derive((_old_ep(),), ("rain",), 2500.0, "content", **good,
               last_behavior="")
    with pytest.raises(ValueError):
        derive((_old_ep(),), ("rain", 42), 2500.0, "content", **good)
    with pytest.raises(ValueError):
        derive((_old_ep(),), ("rain",), 2500.0, "content",
               config="not-a-config")


# --- Episodic Continuity Phase 3: retrieval tests ---


def _rep(**over):
    fields = dict(first_seen_ts=1000.0, last_seen_ts=2000.0, turn_count=2,
                  topic_tokens=("garden", "rain", "soil", "seeds"),
                  mood_label="content", last_behavior="normal")
    fields.update(over)
    return Episode(**fields)


def test_overlap_count_distinct():
    assert _overlap_count(("a", "b"), ("b", "c")) == 1
    assert _overlap_count(("a", "a"), ("a",)) == 1
    assert _overlap_count((), ("a",)) == 0
    assert _overlap_count(("a",), ()) == 0


def test_overlap_count_rejects_non_string():
    with pytest.raises(ValueError):
        _overlap_count(("a", 42), ("a",))


def test_recall_perfect_overlap():
    (got,) = recall((_rep(),), ("garden", "rain", "soil", "seeds"), (),
                    config=_cfg())
    assert got == _rep()


def test_recall_exact_threshold():
    # 2 turn matches -> 2*2 = 4, exactly the default threshold.
    (got,) = recall((_rep(),), ("garden", "rain"), (),
                    config=_cfg())
    assert got == _rep()


def test_recall_below_threshold():
    # 1 turn + 1 topic = 2 + 1 = 3 < 4.
    assert recall((_rep(),), ("garden",), ("soil",),
                  config=_cfg()) == ()
    # A single turn-token match alone (score 2) is below threshold.
    assert recall((_rep(),), ("garden",), (), config=_cfg()) == ()


def test_recall_topic_only():
    (got,) = recall((_rep(),), (),
                    ("garden", "rain", "soil", "seeds"),
                    config=_cfg())
    assert got == _rep()


def test_recall_combined_turn_topic():
    # 1 turn (2) + 2 topics (2) = 4, exactly at threshold.
    (got,) = recall((_rep(),), ("garden",), ("rain", "soil"),
                    config=_cfg())
    assert got == _rep()


def test_recall_no_overlap():
    assert recall((_rep(),), ("chess",), ("tournament",),
                  config=_cfg()) == ()


def test_recall_higher_score_first():
    low = _rep(topic_tokens=("garden", "rain", "soil", "seeds"))
    high = _rep(topic_tokens=("garden", "rain", "soil", "seeds",
                              "sprout"))
    got = recall((low, high),
                 ("garden", "rain", "soil", "seeds", "sprout"), (),
                 config=EpisodesConfig(recall_k=2))
    assert [e.topic_tokens for e in got] == [high.topic_tokens,
                                             low.topic_tokens]


def test_recall_newer_breaks_score_tie():
    older = _rep(last_seen_ts=2000.0)
    newer = _rep(last_seen_ts=3000.0)
    got = recall((older, newer), ("garden", "rain"), (),
                 config=EpisodesConfig(recall_k=2))
    assert got == (newer, older)


def test_recall_older_first_breaks_full_tie():
    early = _rep(first_seen_ts=100.0, last_seen_ts=2000.0)
    late = _rep(first_seen_ts=500.0, last_seen_ts=2000.0)
    got = recall((early, late), ("garden", "rain"), (),
                 config=EpisodesConfig(recall_k=2))
    assert got == (early, late)


def test_recall_default_cap_one():
    first = _rep(topic_tokens=("garden", "rain"))
    second = _rep(topic_tokens=("soil", "seeds"))
    got = recall((first, second), ("garden", "rain", "soil", "seeds"),
                 (), config=_cfg())
    assert len(got) == 1


def test_recall_cap_two():
    first = _rep(topic_tokens=("garden", "rain"))
    second = _rep(topic_tokens=("soil", "seeds"))
    got = recall((first, second), ("garden", "rain", "soil", "seeds"),
                 (), config=EpisodesConfig(recall_k=2))
    assert got == (first, second)


def test_recall_empty_episodes():
    assert recall((), ("garden",), (), config=_cfg()) == ()


def test_recall_empty_turn_and_topics():
    assert recall((_rep(),), (), (), config=_cfg()) == ()


def test_recall_all_below_threshold():
    first = _rep(topic_tokens=("garden",))
    second = _rep(topic_tokens=("soil",))
    assert recall((first, second), ("chess",), (), config=_cfg()) == ()


def test_recall_no_mutation():
    first = _rep()
    before = (first, first.to_dict())
    got = recall((first,), ("garden", "rain"), (), config=_cfg())
    assert first == before[0]
    assert first.to_dict() == before[1]
    assert got[0] is first


def test_recall_deterministic():
    args = ((_rep(),), ("garden", "rain"), ("soil",))
    assert recall(*args, config=_cfg()) == recall(*args, config=_cfg())


def test_recall_validation():
    good = dict(config=_cfg())
    with pytest.raises(ValueError):
        recall((_rep(),), ("garden",), (), config="not-a-config")
    with pytest.raises(ValueError):
        recall((_rep(),), ("garden", 42), (), **good)
    with pytest.raises(ValueError):
        recall((_rep(), "not-an-episode"), ("garden", "rain"), (), **good)


# --- Episodic Continuity Phase 4: orchestrator integration tests ---


def test_process_turn_creates_episode_first_turn():
    c = make_carrot()
    c.process_turn("talking about gardening")
    (episode,) = c.episode_store.load()
    assert episode.turn_count == 1
    assert episode.topic_tokens == ("talking", "gardening")
    assert episode.mood_label == c.state.mood


def test_process_turn_joins_within_window():
    c = make_carrot()
    c.process_turn("talking about gardening")
    c.process_turn("talking about soil")
    (episode,) = c.episode_store.load()
    assert episode.turn_count == 2
    assert "soil" in episode.topic_tokens


def test_process_turn_outside_window_creates_second():
    c = make_carrot()
    c.process_turn("talking about gardening")
    c.clock.advance(3600.0)
    c.process_turn("talking about gardening")
    episodes = c.episode_store.load()
    assert len(episodes) == 2
    assert all(e.turn_count == 1 for e in episodes)


def test_process_turn_pivot_creates_second():
    c = make_carrot()
    c.process_turn("talking about gardening")
    c.process_turn("playing chess daily")
    episodes = c.episode_store.load()
    assert len(episodes) == 2
    assert episodes[0].topic_tokens == ("talking", "gardening")


def test_episode_store_on_disk_matches():
    from carrot.episodes import EpisodeStore as _ES

    c = make_carrot()
    c.process_turn("talking about gardening")
    c.process_turn("talking about soil")
    assert _ES(c.episode_store.path).load() == c.episode_store.load()


def test_last_episodes_populated_on_match():
    # A matching second turn joins the ongoing thread, which is then
    # excluded from retrieval — last_episodes stays empty while the
    # thread is live. (Renamed semantics: "populated" now means the
    # store grew, asserted via turn_count below in the store test.)
    c = make_carrot()
    c.process_turn("talking about gardening")
    assert c.last_episodes == ()
    c.process_turn("talking about soil")
    assert c.last_episodes == ()
    (episode,) = c.episode_store.load()
    assert episode.turn_count == 2


def test_last_episodes_empty_without_match():
    # No *prior* episode matches the pivot turn, and the pivot's own
    # new thread is not retrieved back — last_episodes stays empty.
    c = make_carrot()
    c.process_turn("talking about gardening")
    c.process_turn("playing chess daily")
    assert c.last_episodes == ()
    assert len(c.episode_store.load()) == 2


def test_last_episodes_excludes_ongoing_thread():
    # Retrieval surfaces prior context, not the current turn's thread:
    # a turn that joins the latest episode must not retrieve it back.
    c = make_carrot()
    c.process_turn("talking about gardening")
    assert c.last_episodes == ()
    c.process_turn("talking about soil")
    assert c.last_episodes == ()


def test_last_episodes_retrieves_prior_after_new_thread():
    # A turn outside the join window starts a new thread; the prior
    # episode is now genuinely "previously" and is retrieved.
    c = make_carrot()
    c.process_turn("talking about gardening")
    c.process_turn("talking about soil")
    c.clock.advance(3600.0)
    c.process_turn("talking about gardening")
    assert len(c.last_episodes) == 1
    assert c.last_episodes[0].turn_count == 2
    assert set(c.last_episodes[0].topic_tokens) == \
        {"talking", "gardening", "soil"}


def test_last_episodes_empty_on_first_turn():
    # Pin against the original self-match bug: a fresh Carrot's first
    # turn has no prior episodes, so nothing is retrieved even for
    # contentful input.
    c = make_carrot()
    c.process_turn("talking about gardening")
    assert c.last_episodes == ()


def test_process_turn_prompt_stable_with_episodes_retained():
    # Preview pin for Phase 5: step 5e retains episodes without
    # consuming them, so identical turn sequences must still produce
    # byte-identical prompts even once last_episodes is non-empty.
    first = make_carrot()
    second = make_carrot()
    for text in ("talking about gardening", "talking about soil"):
        first.process_turn(text)
        second.process_turn(text)
    for carrot in (first, second):
        carrot.clock.advance(3600.0)
        carrot.process_turn("talking about gardening")
    assert len(first.last_episodes) == 1
    assert first.llm.last_system == second.llm.last_system
    assert "Previously:" not in first.llm.last_system


def test_process_turn_prompt_ignores_nonempty_last_episodes():
    c = make_carrot()
    c.process_turn("talking about gardening")
    c.process_turn("talking about soil")
    c.clock.advance(3600.0)
    c.process_turn("talking about gardening")
    assert len(c.last_episodes) == 1
    assert "Previously:" not in c.llm.last_system


def test_process_turn_deterministic_episodes():
    first = make_carrot()
    second = make_carrot()
    for text in ("talking about gardening", "talking about soil",
                 "playing chess daily"):
        first.process_turn(text)
        second.process_turn(text)
    assert first.episode_store.load() == second.episode_store.load()
    assert first.last_episodes == second.last_episodes
