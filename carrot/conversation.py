"""Short-term conversation buffer. Text stored verbatim (emoji intact)."""
from __future__ import annotations

from collections import deque


class ConversationBuffer:
    def __init__(self, capacity: int = 10):
        self._turns: deque = deque(maxlen=max(1, int(capacity)))

    def add(self, role: str, text: str) -> None:
        if role not in ("user", "carrot"):
            raise ValueError(f"bad role: {role!r}")
        self._turns.append({"role": role, "text": str(text)})

    def recent(self, n: int = 10) -> list:
        return list(self._turns)[-max(1, n):]

    def history_for_llm(self, n: int = 10) -> list:
        return [{"role": t["role"], "content": t["text"]}
                for t in self.recent(n)]

    def __len__(self) -> int:
        return len(self._turns)
