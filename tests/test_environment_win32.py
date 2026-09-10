"""Slice 6 Phase-5 tests: real Windows probe (mocked everywhere except
the explicitly Windows-gated section at the bottom)."""
import inspect
import sys

import pytest

import carrot.environment_win32 as win32mod
from carrot.environment import (
    EMPTY_RAW,
    EnvironmentProbe,
    RawObservation,
)
from carrot.environment_win32 import WindowsEnvironmentProbe


class _TouchRecorder:
    def __init__(self):
        self.touched = False

    def __getattr__(self, name):
        self.touched = True
        raise AssertionError("ctypes.windll must not be touched")


def test_module_imports_cleanly():
    assert win32mod.WindowsEnvironmentProbe is WindowsEnvironmentProbe


def test_non_windows_short_circuit(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    recorder = _TouchRecorder()
    monkeypatch.setattr(win32mod.ctypes, "windll", recorder,
                        raising=False)
    probe = WindowsEnvironmentProbe()
    assert probe.observe() is EMPTY_RAW
    assert recorder.touched is False


def test_probe_protocol_conformance():
    assert callable(getattr(WindowsEnvironmentProbe, "observe", None))
    assert list(inspect.signature(
        WindowsEnvironmentProbe.observe).parameters) == ["self"]
    assert EnvironmentProbe is not None


def test_never_raises_on_helper_failure(monkeypatch):
    probe = WindowsEnvironmentProbe()

    def boom():
        raise RuntimeError("no OS here")

    def boom_idle():
        raise RuntimeError("no OS here")

    monkeypatch.setattr(probe, "_query_process_name", boom)
    assert probe.observe() == EMPTY_RAW
    monkeypatch.setattr(probe, "_query_process_name",
                        lambda: "code.exe")
    monkeypatch.setattr(probe, "_query_idle", boom_idle)
    assert probe.observe() == EMPTY_RAW


def test_idle_threshold_stored():
    assert WindowsEnvironmentProbe().idle_threshold_s == 300.0
    assert WindowsEnvironmentProbe(idle_threshold_s=600.0) \
        .idle_threshold_s == 600.0


def test_observe_returns_raw_observation(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    result = WindowsEnvironmentProbe().observe()
    assert isinstance(result, RawObservation)


@pytest.mark.skipif(sys.platform != "win32", reason="windows-only")
def test_real_probe_returns_observation():
    result = WindowsEnvironmentProbe().observe()
    assert isinstance(result, RawObservation)
    assert result.process_name is None or (
        isinstance(result.process_name, str)
        and result.process_name
        and result.process_name == result.process_name.lower())


@pytest.mark.skipif(sys.platform != "win32", reason="windows-only")
def test_real_probe_never_raises():
    WindowsEnvironmentProbe().observe()


@pytest.mark.skipif(sys.platform != "win32", reason="windows-only")
def test_real_idle_detection_best_effort():
    result = WindowsEnvironmentProbe(idle_threshold_s=0.0).observe()
    assert isinstance(result.idle, bool)
