"""Slice 8 Phase-1 tests: permissions data layer (storage only)."""
import ast
import dataclasses
import json

import pytest

from carrot.permissions import (
    MAX_PERMISSIONS_BYTES,
    PERMISSIONS_SCHEMA_VERSION,
    PermissionState,
    PermissionStore,
)
from tests.conftest import ROOT


def _store(tmp_path, name="permissions.json"):
    return PermissionStore(tmp_path / name, time_fn=lambda: 1700000000.0)


def test_defaults_zero():
    assert PermissionState(grants={}) == \
        PermissionState(grants={}, policy_rev=0)


def test_round_trip(tmp_path):
    store = _store(tmp_path)
    assert store.set_grant("read_file", True) == \
        PermissionState(grants={"read_file": True}, policy_rev=1)
    assert store.load() == PermissionState(grants={"read_file": True},
                                           policy_rev=1)


def test_missing_file_loads_fresh(tmp_path):
    loaded = _store(tmp_path).load()
    assert loaded.grants == {}
    assert loaded.policy_rev == 0


def test_corrupt_json_quarantined(tmp_path):
    p = tmp_path / "permissions.json"
    p.write_text("{not json", encoding="utf-8")
    loaded = _store(tmp_path).load()
    assert loaded.grants == {}
    assert loaded.policy_rev == 0
    assert not p.exists()
    assert len(list(tmp_path.glob("quarantine-permissions-*.json"))) == 1


def test_unparseable_version_quarantined(tmp_path):
    p = tmp_path / "permissions.json"
    p.write_text('{"version": "x", "grants": {}, "policy_rev": 0}',
                 encoding="utf-8")
    loaded = _store(tmp_path).load()
    assert (loaded.grants, loaded.policy_rev) == ({}, 0)
    assert len(list(tmp_path.glob("quarantine-permissions-*.json"))) == 1


def test_unknown_version_no_quarantine(tmp_path):
    p = tmp_path / "permissions.json"
    p.write_text('{"version": 2, "grants": {"read_file": true},'
                 ' "policy_rev": 9}', encoding="utf-8")
    loaded = _store(tmp_path).load()
    assert (loaded.grants, loaded.policy_rev) == ({}, 0)
    assert list(tmp_path.glob("quarantine-permissions-*.json")) == []
    assert p.exists()


def test_wrong_top_level_shape_quarantined(tmp_path):
    p = tmp_path / "permissions.json"
    p.write_text('{"version": 1, "grants": "not a dict"}', encoding="utf-8")
    loaded = _store(tmp_path).load()
    assert (loaded.grants, loaded.policy_rev) == ({}, 0)
    assert len(list(tmp_path.glob("quarantine-permissions-*.json"))) == 1


def test_per_entry_fail_soft(tmp_path):
    p = tmp_path / "permissions.json"
    p.write_text('{"version": 1, "grants": {"read_file": true,'
                 ' "bad": "yes"}, "policy_rev": 0}', encoding="utf-8")
    loaded = _store(tmp_path).load()
    assert dict(loaded.grants) == {"read_file": True}
    assert list(tmp_path.glob("quarantine-permissions-*.json")) == []


def test_invalid_policy_rev_defaults(tmp_path):
    for rev in ("x", -1):
        p = tmp_path / "permissions.json"
        p.write_text(json.dumps({"version": 1, "grants": {},
                                 "policy_rev": rev}),
                     encoding="utf-8")
        assert _store(tmp_path).load().policy_rev == 0


def test_file_size_cap(tmp_path):
    store = _store(tmp_path)
    huge = {f"tool{i}": True for i in range(2000)}
    with pytest.raises(ValueError):
        store.save(PermissionState(grants=huge, policy_rev=0))
    assert not (tmp_path / "permissions.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_set_grant_increments_rev(tmp_path):
    store = _store(tmp_path)
    assert store.set_grant("read_file", True).policy_rev == 1
    assert store.set_grant("open_url", True).policy_rev == 2
    assert dict(store.load().grants) == {"read_file": True,
                                         "open_url": True}


def test_set_grant_idempotent_value_bumps_rev(tmp_path):
    store = _store(tmp_path)
    store.set_grant("read_file", True)
    second = store.set_grant("read_file", True)
    assert second.grants["read_file"] is True
    assert second.policy_rev == 2


def test_clear_grant(tmp_path):
    store = _store(tmp_path)
    store.set_grant("read_file", True)
    cleared = store.clear_grant("read_file")
    assert dict(cleared.grants) == {}
    assert cleared.policy_rev == 2
    assert dict(store.load().grants) == {}


def test_clear_all(tmp_path):
    store = _store(tmp_path)
    store.set_grant("read_file", True)
    cleared = store.clear_all()
    assert dict(cleared.grants) == {}
    assert cleared.policy_rev == 2


def test_state_frozen():
    state = PermissionState(grants={}, policy_rev=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.policy_rev = 9


def test_grants_mapping_immutable():
    state = PermissionState(grants={"read_file": True}, policy_rev=0)
    with pytest.raises(TypeError):
        state.grants["read_file"] = False


def test_permissions_module_boundary():
    banned = {"emotions", "drives", "memory", "context", "environment",
              "carrot", "classifier", "appraisal", "behavior",
              "personality", "llm", "tools"}
    offenders = []
    tree = ast.parse((ROOT / "carrot" / "permissions.py").read_text(
        encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in banned:
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if node.level == 0 and mod in banned:
                offenders.append(node.module)
            elif node.level > 0 and mod in banned:
                offenders.append(mod)
    assert offenders == []
