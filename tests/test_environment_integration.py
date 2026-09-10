"""Slice 6 Phase-2 tests: environment wiring in process_turn.

The snapshot is retained only in Phase 2: behavior, classifier, and
prompt must be byte-identical with or without an observation.
"""
from carrot.environment import (
    EMPTY_SNAPSHOT,
    EnvironmentSnapshot,
    MockEnvironmentProbe,
    RawObservation,
)
from tests.test_scenarios import make_carrot


class CountingProbe:
    """Local stub: MockEnvironmentProbe keeps no call count, so this
    4-line counter covers the once-per-turn assertion instead."""

    def __init__(self, observation):
        self._observation = observation
        self.calls = 0

    def observe(self):
        self.calls += 1
        return self._observation


class FailingProbe:
    def observe(self):
        raise RuntimeError("no OS here")


def test_probe_called_once_per_turn():
    c = make_carrot()
    c.environment_probe = CountingProbe(RawObservation("code.exe", False))
    c.process_turn("hello there")
    assert c.environment_probe.calls == 1
    c.process_turn("hello again")
    assert c.environment_probe.calls == 2


def test_last_environment_set_after_turn():
    c = make_carrot()
    c.environment_probe = CountingProbe(RawObservation("code.exe", False))
    c.process_turn("hello there")
    assert c.last_environment == EnvironmentSnapshot("editor", False)


def test_last_environment_initialised_empty():
    assert make_carrot().last_environment == EMPTY_SNAPSHOT


def test_snapshot_does_not_leak_to_pick():
    full = make_carrot()
    full.environment_probe = CountingProbe(
        RawObservation("code.exe", False))
    blank = make_carrot()
    for text in ("hello there", "that joke was funny lol"):
        full.process_turn(text)
        blank.process_turn(text)
        assert full.last_behavior.name == blank.last_behavior.name


def test_snapshot_does_not_leak_to_classify():
    full = make_carrot()
    full.environment_probe = CountingProbe(
        RawObservation("code.exe", False))
    blank = make_carrot()
    full.process_turn("You're cute!")
    blank.process_turn("You're cute!")
    assert full.last_event.type == blank.last_event.type == "compliment"


def test_snapshot_does_not_leak_to_prompt():
    # Re-seeded in Slice 6 Phase 3 (Slice 5 Phase-4b pattern): an EMPTY
    # snapshot still renders nothing (Phase 2 behavior preserved), while
    # an allowlisted snapshot appends exactly the locked sentence(s).
    full = make_carrot()
    full.environment_probe = CountingProbe(
        RawObservation("code.exe", False))
    blank = make_carrot()
    full.process_turn("talking about gardening")
    blank.process_turn("talking about gardening")
    assert "Context:" not in blank.llm.last_system
    assert full.llm.last_system == blank.llm.last_system + \
        " Context: They're in the editor."


def test_default_probe_is_mock_not_real():
    probe = make_carrot().environment_probe
    assert isinstance(probe, MockEnvironmentProbe)
    assert type(probe).__name__ == "MockEnvironmentProbe"


def test_probe_failure_degrades_to_empty():
    c = make_carrot()
    c.environment_probe = FailingProbe()
    reply = c.process_turn("hello there")
    assert isinstance(reply, str) and reply
    assert c.last_environment == EMPTY_SNAPSHOT


def test_environment_wiring_deterministic():
    def _run():
        carrot = make_carrot()
        carrot.environment_probe = CountingProbe(
            RawObservation("cmd.exe", True))
        carrot.process_turn("hello there")
        return carrot
    first, second = _run(), _run()
    assert first.last_environment == second.last_environment
    assert first.state.to_dict() == second.state.to_dict()


def test_state_schema_has_no_environment_key():
    c = make_carrot()
    c.environment_probe = CountingProbe(
        RawObservation("code.exe", False))
    c.process_turn("my name is Alice")
    assert "environment" not in c.state.to_dict()
