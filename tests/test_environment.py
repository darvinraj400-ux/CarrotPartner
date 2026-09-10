"""Slice 6 Phase-1 tests: environment data layer (types only)."""
import ast
import dataclasses
import json

import pytest

from carrot.environment import (
    EMPTY_RAW,
    EMPTY_SNAPSHOT,
    EnvironmentConfig,
    EnvironmentProbe,
    EnvironmentSnapshot,
    MockEnvironmentProbe,
    RawObservation,
    map_observation,
)
from tests.conftest import ROOT


def _cfg(**overrides):
    base = {
        "allowed_apps": {"code.exe": "editor", "cmd.exe": "terminal"},
        "idle_threshold_s": 300.0,
    }
    base.update(overrides)
    return base


def _write_config(tmp_path, data, name="environment.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _loaded(tmp_path):
    return EnvironmentConfig.load(_write_config(tmp_path, _cfg()))


# ---- value types ----

def test_raw_observation_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        RawObservation("code.exe", False).process_name = "x"


def test_snapshot_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        EnvironmentSnapshot("editor", False).app_noun = "x"


def test_empty_constants():
    assert EMPTY_RAW.process_name is None
    assert EMPTY_RAW.idle is False
    assert EMPTY_SNAPSHOT.app_noun is None
    assert EMPTY_SNAPSHOT.idle is False


# ---- config loader ----

def test_config_loads_valid():
    cfg = EnvironmentConfig.load(str(ROOT / "config" / "environment.json"))
    assert cfg.allowed_apps["windowsterminal.exe"] == "terminal"
    assert cfg.idle_threshold_s == 300.0
    assert len(cfg.allowed_apps) == 14


def test_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EnvironmentConfig.load(str(tmp_path / "nope.json"))


def test_config_malformed_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        EnvironmentConfig.load(str(path))


def test_config_missing_allowed_apps(tmp_path):
    data = _cfg()
    del data["allowed_apps"]
    assert EnvironmentConfig.load(
        _write_config(tmp_path, data)).allowed_apps == {}


def test_config_missing_idle_threshold(tmp_path):
    data = _cfg()
    del data["idle_threshold_s"]
    assert EnvironmentConfig.load(
        _write_config(tmp_path, data)).idle_threshold_s == 300.0


def test_config_invalid_idle_threshold(tmp_path):
    with pytest.raises(ValueError):
        EnvironmentConfig.load(_write_config(
            tmp_path, _cfg(idle_threshold_s="x")))
    with pytest.raises(ValueError):
        EnvironmentConfig.load(_write_config(
            tmp_path, _cfg(idle_threshold_s=-1.0)))


def test_config_frozen(tmp_path):
    cfg = _loaded(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.idle_threshold_s = 1.0


# ---- map_observation ----

def test_map_hit(tmp_path):
    cfg = _loaded(tmp_path)
    assert map_observation(RawObservation("code.exe", False), cfg) == \
        EnvironmentSnapshot("editor", False)


def test_map_miss(tmp_path):
    cfg = _loaded(tmp_path)
    assert map_observation(RawObservation("some_random_thing.exe", False),
                           cfg) == EnvironmentSnapshot(None, False)


def test_map_none_process_name(tmp_path):
    cfg = _loaded(tmp_path)
    assert map_observation(RawObservation(None, False), cfg) == \
        EnvironmentSnapshot(None, False)


def test_map_idle_passthrough(tmp_path):
    cfg = _loaded(tmp_path)
    assert map_observation(RawObservation(None, True), cfg) == \
        EnvironmentSnapshot(None, True)


def test_map_adversarial_digits(tmp_path):
    cfg = _loaded(tmp_path)
    assert map_observation(RawObservation("report_Q3_final.exe", False),
                           cfg).app_noun is None


def test_map_adversarial_banned_noun(tmp_path):
    cfg = _loaded(tmp_path)
    assert map_observation(RawObservation("anger_manager.exe", False),
                           cfg).app_noun is None


def test_map_adversarial_long_string(tmp_path):
    cfg = _loaded(tmp_path)
    assert map_observation(RawObservation("x" * 10000, False),
                           cfg).app_noun is None


def test_map_case_insensitive(tmp_path):
    # Loader lowercases keys; mapping lowercases input: both directions
    # are covered, so mixed-case process names still match.
    cfg = _loaded(tmp_path)
    assert map_observation(RawObservation("CODE.EXE", False), cfg) == \
        EnvironmentSnapshot("editor", False)


# ---- mock probe ----

def test_mock_empty_sequence():
    probe = MockEnvironmentProbe()
    assert probe.observe() is EMPTY_RAW
    assert probe.observe() is EMPTY_RAW


def test_mock_scripted_sequence_cycles_last():
    a = RawObservation("code.exe", False)
    b = RawObservation(None, True)
    c = RawObservation("cmd.exe", False)
    probe = MockEnvironmentProbe([a, b, c])
    assert probe.observe() is a
    assert probe.observe() is b
    assert probe.observe() is c
    assert probe.observe() is c


def test_mock_deterministic():
    seq = [RawObservation("code.exe", False), RawObservation(None, True)]
    first = MockEnvironmentProbe(seq)
    second = MockEnvironmentProbe(seq)
    assert [first.observe() for _ in range(4)] == \
        [second.observe() for _ in range(4)]


def test_probe_protocol_satisfied():
    import inspect

    assert callable(getattr(MockEnvironmentProbe, "observe", None))
    assert list(inspect.signature(
        MockEnvironmentProbe.observe).parameters) == ["self"]
    assert EnvironmentProbe is not None


# ---- boundaries ----

def test_environment_module_boundary():
    banned = {"emotions", "drives", "memory", "context", "carrot", "ctypes"}
    offenders = []
    tree = ast.parse((ROOT / "carrot" / "environment.py").read_text(
        encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in banned or parts[0] == "carrot":
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".")
            if node.level == 0 and parts[0] == "carrot" and len(parts) > 1 \
                    and parts[1] in banned:
                offenders.append(mod)
            elif node.level > 0 and parts and parts[0] in banned:
                offenders.append(mod)
            elif node.level > 0:
                for a in node.names:
                    if a.name in banned:
                        offenders.append(a.name)
            if "ctypes" in (mod,):
                offenders.append(mod)
    assert offenders == []


def test_allowed_apps_immutable(tmp_path):
    cfg = _loaded(tmp_path)
    with pytest.raises(TypeError):
        cfg.allowed_apps["evil.exe"] = "evil"
    assert cfg.allowed_apps == {"code.exe": "editor", "cmd.exe": "terminal"}
