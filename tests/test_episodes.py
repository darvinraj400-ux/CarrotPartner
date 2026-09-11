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
)
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
