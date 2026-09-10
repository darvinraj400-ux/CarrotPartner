"""Slice 3 Phase-1 tests: memory data layer (persistence only)."""
import ast
import json
import sys

import pytest

from carrot.memory import (
    MAX_FILE_BYTES,
    MAX_MEMORIES,
    MAX_MEMORY_TEXT_CHARS,
    MEMORY_SCHEMA_VERSION,
    Memory,
    MemoryStore,
)
from tests.conftest import ROOT


def _mem(text="my dog's name is Rex", kind="fact", importance=0.5, ts=1000.0):
    return Memory(text=text, kind=kind, importance=importance,
                  created_ts=ts, last_recalled_ts=ts)


def _store(tmp_path, name="memories.json"):
    return MemoryStore(tmp_path / name, time_fn=lambda: 1700000000.0)


def test_round_trip_field_by_field(tmp_path):
    store = _store(tmp_path)
    originals = [_mem("my dog's name is Rex", "fact", 0.7, 100.0),
                 _mem("I love ramen", "preference", 0.4, 200.0),
                 _mem("my sister lives in Oslo", "fact", 0.9, 300.0)]
    store.save(originals)
    loaded = store.load()
    assert len(loaded) == 3
    for got, want in zip(loaded, originals):
        assert got.to_dict() == want.to_dict()


def test_missing_file_loads_empty(tmp_path):
    assert _store(tmp_path).load() == ()


def test_corrupt_json_quarantined(tmp_path):
    p = tmp_path / "memories.json"
    p.write_text("{not json", encoding="utf-8")
    assert _store(tmp_path).load() == ()
    assert not p.exists()
    assert len(list(tmp_path.glob("quarantine-memories-*.json"))) == 1


def test_unparseable_version_quarantined(tmp_path):
    p = tmp_path / "memories.json"
    p.write_text('{"version": "x", "memories": []}', encoding="utf-8")
    assert _store(tmp_path).load() == ()
    assert len(list(tmp_path.glob("quarantine-memories-*.json"))) == 1


def test_unknown_version_no_quarantine(tmp_path):
    p = tmp_path / "memories.json"
    p.write_text('{"version": 2, "memories": []}', encoding="utf-8")
    assert _store(tmp_path).load() == ()
    assert list(tmp_path.glob("quarantine-memories-*.json")) == []
    assert p.exists()


def test_missing_top_level_keys_quarantined(tmp_path):
    p = tmp_path / "memories.json"
    p.write_text('{"version": 1}', encoding="utf-8")
    assert _store(tmp_path).load() == ()
    assert len(list(tmp_path.glob("quarantine-memories-*.json"))) == 1


def test_invalid_entry_dropped_without_quarantine(tmp_path):
    store = _store(tmp_path)
    store.save([_mem("kept fact")])
    data = json.loads((tmp_path / "memories.json").read_text(
        encoding="utf-8"))
    data["memories"].append({"text": "bad kind", "kind": "nonsense",
                             "importance": 0.5, "created_ts": 1.0,
                             "last_recalled_ts": 1.0})
    (tmp_path / "memories.json").write_text(json.dumps(data),
                                            encoding="utf-8")
    loaded = store.load()
    assert [m.text for m in loaded] == ["kept fact"]
    assert list(tmp_path.glob("quarantine-memories-*.json")) == []


def test_cap_on_add(tmp_path):
    store = _store(tmp_path)
    for i in range(MAX_MEMORIES):
        assert store.add(_mem(f"fact number {i}", ts=float(i))) is True
    assert store.add(_mem("one too many")) is False
    assert len(store.load()) == MAX_MEMORIES


def test_cap_on_save_raises(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.save([_mem(f"fact {i}", ts=float(i))
                    for i in range(MAX_MEMORIES + 1)])


def test_remove_matching(tmp_path):
    store = _store(tmp_path)
    store.save([_mem("my dog's name is Rex"),
                _mem("my Dog likes snow"),
                _mem("I love ramen")])
    assert store.remove_matching("DOG") == 2
    assert [m.text for m in store.load()] == ["I love ramen"]


def test_remove_matching_no_match(tmp_path):
    store = _store(tmp_path)
    store.save([_mem("I love ramen")])
    assert store.remove_matching("sushi") == 0
    assert [m.text for m in store.load()] == ["I love ramen"]


def test_text_length_validation():
    with pytest.raises(ValueError):
        _mem("x" * (MAX_MEMORY_TEXT_CHARS + 1))
    _mem("x" * MAX_MEMORY_TEXT_CHARS)  # boundary allowed


def test_memory_field_validation():
    with pytest.raises(ValueError):
        _mem(kind="nonsense")
    with pytest.raises(ValueError):
        Memory(text="ok", kind="fact", importance="a lot",
               created_ts=1.0, last_recalled_ts=1.0)
    with pytest.raises(ValueError):
        Memory(text="ok", kind="fact", importance=0.5,
               created_ts="yesterday", last_recalled_ts=1.0)
    m = Memory(text="ok", kind="fact", importance=1.5,
               created_ts=1.0, last_recalled_ts=1.0)
    assert m.importance == 1.0  # clamped, not rejected


def test_memory_does_not_import_emotions():
    tree = ast.parse((ROOT / "carrot" / "memory.py").read_text(
        encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                mods.add((node.module or "").split(".")[0])
            elif node.module:
                mods.add(node.module.split(".")[0])
    assert "emotions" not in mods, mods
    assert "context" in mods, mods  # tokenizer reuse (Phase 3)
    src = (ROOT / "carrot" / "memory.py").read_text(encoding="utf-8")
    assert "from .emotions import" not in src
    assert "from carrot.emotions import" not in src


def test_atomic_write_leaves_no_tmp(tmp_path):
    store = _store(tmp_path)
    store.save([_mem("first"), _mem("second")])
    names = [p.name for p in tmp_path.iterdir()]
    assert "memories.json" in names
    assert not [n for n in names if n.endswith(".tmp")]
    assert len(store.load()) == 2


def test_deterministic_with_fixed_time_fn(tmp_path):
    first = MemoryStore(tmp_path / "a.json", time_fn=lambda: 42.0)
    second = MemoryStore(tmp_path / "b.json", time_fn=lambda: 42.0)
    mems = [_mem("my dog's name is Rex", "fact", 0.7, 100.0),
            _mem("I love ramen", "preference", 0.4, 200.0)]
    first.save(mems)
    second.save(mems)
    assert (tmp_path / "a.json").read_bytes() == \
        (tmp_path / "b.json").read_bytes()


def test_full_store_stays_within_file_bound(tmp_path):
    store = _store(tmp_path)
    for i in range(MAX_MEMORIES):
        store.add(_mem(f"remembered fact number {i} about Oslo",
                       "fact", 0.5, ts=float(i)))
    size = (tmp_path / "memories.json").stat().st_size
    assert size < MAX_FILE_BYTES, size
    assert MEMORY_SCHEMA_VERSION == 1
