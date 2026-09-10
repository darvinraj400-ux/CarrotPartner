"""Orchestrator: the ONLY caller of EmotionState mutation methods.

One-way flow per turn:
  decay (elapsed) -> return-event? -> classify -> appraise -> apply ->
  mood -> behavior -> LLM wording.
LLM output is returned as text and NEVER parsed back into state.
"""
from __future__ import annotations

import time

from . import mood as mood_mod
from .behavior import Behavior
from .classifier import InteractionEvent
from .context import derive_context

_BAND_WORDS = {"low": "a little", "med": "clearly", "high": "strongly"}


class Carrot:
    def __init__(self, *, state, personality, classifier, appraisal,
                 selector, llm, decay_cfg, buffer=None,
                 absence_threshold_s: float = 21600.0,
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
        self.last_behavior: Behavior = Behavior("normal", "low")
        self.last_event: InteractionEvent | None = None
        if buffer is None:
            from .conversation import ConversationBuffer
            buffer = ConversationBuffer()
        self.buffer = buffer

    def _system_prompt(self, behavior: Behavior) -> str:
        degree = _BAND_WORDS.get(behavior.band, "a little")
        return (
            f"{self.personality.sketch()} "
            f"Right now you are feeling {degree} {behavior.name}. "
            "Stay in character. Never mention numbers, levels, meters, "
            "internal state, or how your emotions work. Keep replies short "
            "and natural, like a person texting. Mirror the user's language "
            "and emoji naturally; do not force emoji."
        )

    def process_turn(self, text: str, now: float | None = None) -> str:
        now = float(self._now() if now is None else now)
        elapsed = now - self.state.last_seen if self.state.last_seen else 0.0

        # 1. Background drift toward baselines (applied THROUGH the state).
        self.state.apply_decay(elapsed, self.decay_cfg)

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
        event = self.classifier.classify(text, ctx)
        self.last_event = event

        # 4. Appraise and apply -- the single state-write path for events.
        intent = self.appraisal.appraise(event, self.personality)
        self.state.apply_deltas(
            intent.deltas,
            {"event": event.type, "reason": intent.reason, "ts": now})

        # 5. Derived mood + behavior (read-only over state).
        self.state.set_mood(mood_mod.derive(self.state.emotions,
                                            self.state.mood))
        behavior = self.selector.pick(self.state, self.personality,
                                      self.buffer.recent(),
                                      user_text=str(text))
        self.last_behavior = behavior

        # 6. LLM renders wording only. Its output never re-enters state.
        reply = self.llm.generate(
            system=self._system_prompt(behavior),
            history=self.buffer.history_for_llm(),
            user=str(text))

        self.buffer.add("user", str(text))
        self.buffer.add("carrot", reply)
        self.state.set_last_seen(now)
        return reply
