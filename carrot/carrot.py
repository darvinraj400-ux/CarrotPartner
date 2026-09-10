"""Orchestrator: the ONLY caller of EmotionState mutation methods.

One-way flow per turn:
  decay (elapsed) -> return-event? -> classify -> appraise -> apply ->
  mood -> behavior -> LLM wording.
LLM output is returned as text and NEVER parsed back into state.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import mood as mood_mod
from .behavior import Behavior
from .classifier import InteractionEvent, _INSULT_CUES, _REPAIR_CUES
from .context import ConversationContext, derive_context
from .drives import (
    DrivesConfig,
    DriveStore,
    apply_satisfaction,
    drive_pull,
    update_rise,
)
from .environment import (
    EMPTY_RAW,
    EMPTY_SNAPSHOT,
    EnvironmentConfig,
    EnvironmentSnapshot,
    MockEnvironmentProbe,
    map_observation,
)
from .memory import DeterministicExtractor, Memory, MemoryStore

_BAND_WORDS = {"low": "a little", "med": "clearly", "high": "strongly"}
_MEMORY_RENDER_CHARS = 120

# Drive-pull sentences: static strings only, never parameterized by
# drive values (no numbers may enter the prompt through this path).
_DRIVE_PULL_SENTENCES = {
    "social": "You've been wanting to talk to them.",
    "novelty": "You've been wanting something new to happen.",
}

# Mirrors cli.py: StateStore lives at <project>/state/state.json, so the
# default memory file lives next to it. Tests and callers pass an
# explicit store instead.
_DEFAULT_STATE_DIR = Path(__file__).resolve().parents[1] / "state"
# Same convention for config: <project>/config/drives.json. Absolute so
# behavior never depends on the process working directory.
_DEFAULT_DRIVES_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "drives.json")
_DEFAULT_ENVIRONMENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "environment.json")


def _has_cue(lowered_text: str, cues) -> bool:
    return any(cue in lowered_text for cue in cues)


def _escape_memory(text: str) -> str:
    clean = str(text or "").replace("\r", " ").replace("\n", " ")
    if len(clean) > _MEMORY_RENDER_CHARS:
        clean = clean[:_MEMORY_RENDER_CHARS] + "…"
    return clean.replace('"', '\\"')


def _memories_block(recalled) -> str:
    if not recalled:
        return ""
    parts = ['You told me: "' + _escape_memory(m.text) + '".'
             for m in recalled]
    return " Memories: " + " ".join(parts)


def _context_hint(context: ConversationContext, user_text: str,
                  drive_pull=None, environment=None) -> str:
    lowered = str(user_text or "").strip().lower()
    answered = (bool(lowered) and not lowered.endswith("?")
                and not _has_cue(lowered, _REPAIR_CUES)
                and not _has_cue(lowered, _INSULT_CUES))
    parts = []
    if context.pending_question and answered:
        parts.append("The user is answering your earlier question.")
    if context.playful_frame:
        parts.append("The conversation has been playful.")
    if context.recent_harshness and _has_cue(lowered, _REPAIR_CUES):
        parts.append("The user seems to be walking back what "
                     "they said.")
    if context.recent_topics:
        parts.append("Recent topics: " + ", ".join(context.recent_topics))
    sentence = _DRIVE_PULL_SENTENCES.get(drive_pull)
    if sentence is not None:
        parts.append(sentence)
    if environment is not None:
        if environment.app_noun is not None:
            parts.append("They're in the " + environment.app_noun + ".")
        if environment.idle:
            parts.append("They've been away for a bit.")
    if not parts:
        return ""
    return " Context: " + " ".join(parts)


class Carrot:
    def __init__(self, *, state, personality, classifier, appraisal,
                 selector, llm, decay_cfg, buffer=None,
                 absence_threshold_s: float = 21600.0,
                 memory_store=None, extractor=None,
                 drive_store=None, drives_config=None,
                 environment_probe=None, environment_config=None,
                 time_fn=None):
        self.state = state
        self.personality = personality
        self.classifier = classifier
        self.appraisal = appraisal
        self.selector = selector
        self.llm = llm
        self.decay_cfg = dict(decay_cfg or {})
        self.absence_threshold_s = float(absence_threshold_s)
        self._now = time_fn or time.time
        self.memory_store = (memory_store if memory_store is not None
                             else MemoryStore(_DEFAULT_STATE_DIR
                                              / "memories.json"))
        self.extractor = (extractor if extractor is not None
                          else DeterministicExtractor())
        if drive_store is None:
            drive_store = DriveStore(_DEFAULT_STATE_DIR / "drives.json")
        self.drive_store = drive_store
        if drives_config is None:
            # Config is required: a missing file is a hard error here.
            drives_config = DrivesConfig.load(_DEFAULT_DRIVES_CONFIG_PATH)
        self.drives_config = drives_config
        self.last_drive_pull: str | None = None
        self.environment_probe = (environment_probe
                                  if environment_probe is not None
                                  else MockEnvironmentProbe())
        if environment_config is None:
            # Config is required: a missing file is a hard error here.
            environment_config = EnvironmentConfig.load(
                _DEFAULT_ENVIRONMENT_CONFIG_PATH)
        self.environment_config = environment_config
        self.last_environment: EnvironmentSnapshot = EMPTY_SNAPSHOT
        self.last_behavior: Behavior = Behavior("normal", "low")
        self.last_event: InteractionEvent | None = None
        self.last_context = None
        if buffer is None:
            from .conversation import ConversationBuffer
            buffer = ConversationBuffer()
        self.buffer = buffer

    def _system_prompt(self, behavior: Behavior,
                         context: ConversationContext,
                         user_text: str,
                         recalled=(), drive_pull=None,
                         environment=EMPTY_SNAPSHOT) -> str:
        degree = _BAND_WORDS.get(behavior.band, "a little")
        return (
            f"{self.personality.sketch()} "
            f"Right now you are feeling {degree} {behavior.name}. "
            "Stay in character. Never mention numbers, levels, meters, "
            "internal state, or how your emotions work. Keep replies short "
            "and natural, like a person texting. Mirror the user's language "
            "and emoji naturally; do not force emoji."
            + _memories_block(recalled)
            + _context_hint(context, user_text, drive_pull, environment)
        )

    @staticmethod
    def _initial_importance(cand, user_text: str) -> float:
        # `cand` is currently unused; retained so future kind-weighting
        # can key off the candidate without a signature change.
        lowered = str(user_text or "").lower()
        if ("remember" in lowered or "don't forget" in lowered
                or "note that" in lowered):
            return 0.8
        return 0.5

    def _store_candidates(self, user_text: str) -> None:
        seen = {m.text.strip().lower() for m in self.memory_store.all()}
        now = self._now()
        for cand in self.extractor.extract(user_text):
            key = cand.text.strip().lower()
            if key in seen:
                continue
            if self.memory_store.add(Memory(
                    text=cand.text, kind=cand.kind,
                    importance=self._initial_importance(cand, user_text),
                    created_ts=now, last_recalled_ts=now)):
                seen.add(key)

    def process_turn(self, text: str, now: float | None = None) -> str:
        now = float(self._now() if now is None else now)
        elapsed = now - self.state.last_seen if self.state.last_seen else 0.0

        # 1. Background drift toward baselines (applied THROUGH the state).
        self.state.apply_decay(elapsed, self.decay_cfg)

        # 1b. Rise drives on the same elapsed; store is source of truth.
        drives = self.drive_store.load()
        drives = update_rise(drives, elapsed,
                             self.drives_config.rise_per_hour)

        # 2. Absence -> return is a background event, appraised like any other.
        if elapsed >= self.absence_threshold_s:
            hours = elapsed / 3600.0
            strength = min(1.0, 0.4 + (elapsed / self.absence_threshold_s) * 0.3)
            back = InteractionEvent(
                "return", round(strength, 3),
                f"user returned after {hours:.1f}h away")
            intent = self.appraisal.appraise(back, self.personality)
            self.state.apply_deltas(
                intent.deltas,
                {"event": "return", "reason": intent.reason, "ts": now})

        # 3. Interpret the live message with pre-turn context flags.
        ctx = derive_context(self.buffer.recent())
        self.last_context = ctx
        # 3a. Observe the environment. Retained only; the classifier,
        # behavior, and prompt do not consume it yet (Phase 3+). A
        # failing probe must never break a turn, so fall back to empty.
        try:
            raw = self.environment_probe.observe()
        except Exception:
            raw = EMPTY_RAW
        self.last_environment = map_observation(raw,
                                                self.environment_config)
        event = self.classifier.classify(text, ctx)
        self.last_event = event

        # 4. Appraise and apply -- the single state-write path for events.
        intent = self.appraisal.appraise(event, self.personality)
        self.state.apply_deltas(
            intent.deltas,
            {"event": event.type, "reason": intent.reason, "ts": now})

        # 4b. Satisfy drives from the classified event; persist rise and
        # satisfaction together in one atomic write.
        drives = apply_satisfaction(
            drives,
            event.type,
            self.drives_config.presence_satisfaction,
            self.drives_config.event_satisfaction,
        )
        self.drive_store.save(drives)

        # 5. Derived mood + behavior (read-only over state).
        self.state.set_mood(mood_mod.derive(self.state.emotions,
                                            self.state.mood))
        behavior = self.selector.pick(self.state, self.personality,
                                      ctx, user_text=str(text))
        self.last_behavior = behavior

        # 5b. Recall long-term memories (decay persisted, hits refreshed).
        recalled = self.memory_store.recall_and_refresh(
            current_user_text=str(text),
            recent_topics=ctx.recent_topics,
            now=self._now())
        # 5c. Extract candidate facts from the live turn only.
        self._store_candidates(str(text))

        # 5d. Per-turn drive pull, computed once for Phase 4/5 consumers.
        self.last_drive_pull = drive_pull(drives, self.drives_config)

        # 6. LLM renders wording only. Its output never re-enters state.
        reply = self.llm.generate(
            system=self._system_prompt(behavior, ctx, str(text), recalled,
                                       drive_pull=self.last_drive_pull,
                                       environment=self.last_environment),
            history=self.buffer.history_for_llm(),
            user=str(text))

        self.buffer.add("user", str(text))
        self.buffer.add("carrot", reply)
        self.state.set_last_seen(now)
        return reply
