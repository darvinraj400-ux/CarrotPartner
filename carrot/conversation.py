"""Short-term conversation buffer. Text stored verbatim (emoji intact)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Turn:
    """One conversational turn. Immutable value type, no timestamps."""

    role: Literal["user", "carrot"]
    text: str


class ConversationBuffer:
    # Episodic Continuity (research tier) must never persist turn text.
    # Episodes store derived features only: topic tokens, timestamps,
    # turn count, mood label. Enforced by
    # test_no_verbatim_turn_text_in_new_stores (field-name) and
    # test_episode_store_never_contains_turn_text (runtime, Phase 2+).
    def __init__(self, capacity: int = 10):
        self._turns: deque = deque(maxlen=max(1, int(capacity)))

    def add(self, role: str, text: str) -> None:
        if role not in ("user", "carrot"):
            raise ValueError(f"bad role: {role!r}")
        self._turns.append(Turn(role=role, text=str(text)))

    def recent(self, n: int = 10) -> tuple:
        return tuple(list(self._turns)[-max(1, n):])

    def history_for_llm(self, n: int = 10) -> list:
        return [{"role": t.role, "content": t.text}
                for t in self.recent(n)]

    def __len__(self) -> int:
        return len(self._turns)
