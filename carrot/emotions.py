"""Emotional state.

EmotionState is the SOLE owner of emotion mutation, validation and
clamping. Appraisal and decay only compute *intents* (delta dicts);
the orchestrator feeds those intents back through ``apply_deltas`` /
``apply_decay``. There is no second mutation path.
"""
from __future__ import annotations

EMOTIONS = (
    "happiness",
    "sadness",
    "anger",
    "hurt",
    "affection",
    "excitement",
    "embarrassment",
)

STATE_VERSION = 1
RECENT_CAUSES_CAP = 15

# Resting values for a fresh companion. Mirrored in config/decay.json
# "baselines" (a test enforces they stay in sync); decay drifts state
# back toward these, so a new Carrot starts at rest, not flat-zero.
RESTING = {
    "happiness": 0.5,
    "sadness": 0.05,
    "anger": 0.05,
    "hurt": 0.05,
    # affection = emotional feeling toward the user, decays to baseline.
    # Distinct from a future "social" drive (accumulating need for interaction).
    "affection": 0.5,
    "excitement": 0.2,
    "embarrassment": 0.0,
}
RESTING_ENERGY = 0.7
# energy = current capacity. Not a "rest drive": Slice 5 deliberately does
# not model a rest need, because energy already represents that dimension.


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class EmotionState:
    """7 emotions plus separate general state.

    ``energy`` is deliberately NOT an emotion: it models
    capacity/readiness and lives next to ``mood`` / ``last_seen``.
    """

    def __init__(self, emotions=None, energy: float = RESTING_ENERGY,
                 mood: str = "content", last_seen: float = 0.0,
                 recent_causes=None,
                 personality_id: str = "carrot-default-v1",
                 version: int = STATE_VERSION,
                 session_count: int = 0, first_seen_ts: float = 0.0):
        self._emotions = dict(RESTING)
        if emotions:
            for key, value in emotions.items():
                if key not in EMOTIONS:
                    raise ValueError(f"unknown emotion: {key!r}")
                self._emotions[key] = clamp(value)
        self._energy = clamp(energy)
        self._mood = str(mood)
        self._last_seen = float(last_seen or 0.0)
        self._recent_causes: list = []
        for cause in recent_causes or []:
            self._push_cause(cause)
        self.personality_id = str(personality_id)
        self.version = int(version)
        if (isinstance(session_count, bool)
                or not isinstance(session_count, int)
                or session_count < 0):
            session_count = 0
        self.session_count = session_count
        try:
            first_seen_ts = float(first_seen_ts)
        except (TypeError, ValueError):
            first_seen_ts = 0.0
        self.first_seen_ts = first_seen_ts

    @property
    def emotions(self) -> dict:
        return dict(self._emotions)

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def mood(self) -> str:
        return self._mood

    @property
    def last_seen(self) -> float:
        return self._last_seen

    @property
    def recent_causes(self) -> list:
        return list(self._recent_causes)

    # ---- sole mutators (mutation path #1 and #2) ----

    def apply_deltas(self, deltas, cause="") -> None:
        """Apply an appraisal intent. Mutation path #1 (shared)."""
        for key, delta in (deltas or {}).items():
            try:
                amount = float(delta)
            except (TypeError, ValueError):
                raise ValueError(f"non-numeric delta for {key!r}")
            if key == "energy":
                self._energy = clamp(self._energy + amount)
            elif key in EMOTIONS:
                self._emotions[key] = clamp(self._emotions[key] + amount)
            else:
                raise ValueError(f"unknown state key: {key!r}")
        if cause:
            self._push_cause(cause)

    def apply_decay(self, elapsed_seconds: float, decay_cfg: dict) -> None:
        """Drift emotions/energy toward baselines. Mutation path #2.

        The drift itself is applied via ``apply_deltas`` so validation
        and clamping stay in exactly one place.
        """
        elapsed = max(0.0, float(elapsed_seconds or 0.0))
        if elapsed <= 0:
            return
        baselines = (decay_cfg or {}).get("baselines", {})
        half_lives = (decay_cfg or {}).get("half_lives", {})
        deltas = {}
        for key in list(EMOTIONS) + ["energy"]:
            current = self._energy if key == "energy" else self._emotions[key]
            baseline = clamp(baselines.get(key, current))
            try:
                half_life = float(half_lives.get(key, 3600.0))
            except (TypeError, ValueError):
                raise ValueError(f"bad half-life for {key!r}")
            if half_life <= 0:
                continue
            target = baseline + (current - baseline) * (0.5 ** (elapsed / half_life))
            deltas[key] = target - current
        self.apply_deltas(deltas)

    # ---- non-emotion setters (never touch emotion values) ----

    def set_mood(self, mood: str) -> None:
        self._mood = str(mood)

    def set_last_seen(self, ts: float) -> None:
        self._last_seen = float(ts)

    # ---- helpers ----

    def _push_cause(self, cause) -> None:
        if isinstance(cause, str):
            entry = {"event": cause}
        elif isinstance(cause, dict):
            entry = {"event": str(cause.get("event", "?"))}
            if "reason" in cause:
                entry["reason"] = str(cause["reason"])[:280]
            if "ts" in cause:
                entry["ts"] = cause["ts"]
        else:
            entry = {"event": str(cause)}
        self._recent_causes.append(entry)
        overflow = len(self._recent_causes) - RECENT_CAUSES_CAP
        if overflow > 0:
            del self._recent_causes[:overflow]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "personality_id": self.personality_id,
            "emotions": dict(self._emotions),
            "energy": self._energy,
            "mood": self._mood,
            "last_seen": self._last_seen,
            "recent_causes": list(self._recent_causes),
            "session_count": self.session_count,
            "first_seen_ts": self.first_seen_ts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionState":
        data = dict(data or {})
        try:
            version = int(data.get("version", STATE_VERSION))
        except (TypeError, ValueError):
            raise ValueError(f"bad version: {data.get('version')!r}")
        if version != STATE_VERSION:
            # Unknown schema: do not silently accept future versions.
            return cls()
        return cls(
            emotions=data.get("emotions", {}),
            energy=data.get("energy", 0.7),
            mood=data.get("mood", "content"),
            last_seen=data.get("last_seen", 0.0),
            recent_causes=data.get("recent_causes", []),
            personality_id=data.get("personality_id", "carrot-default-v1"),
            version=data.get("version", STATE_VERSION),
            session_count=data.get("session_count", 0),
            first_seen_ts=data.get("first_seen_ts", 0.0),
        )
