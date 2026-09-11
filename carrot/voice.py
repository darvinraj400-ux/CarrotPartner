"""Voice channel protocols and mocks.

Contract:
- Voice is a channel: text in (via a recognizer) and text out (via a
  synthesizer). The core is unaware of channels; transcription enters
  process_turn as ordinary user text, exactly like typed input.
- This module holds protocols and mocks only. The CLI owns the bounded
  voice turn (cli.run_voice_turn). No real STT/TTS ships in Slice 13
  (Option A per planner ruling).
- No imports of carrot.carrot, carrot.cli, carrot.llm. No subprocess.
  No tkinter. No threads. Stdlib only.
"""
from __future__ import annotations

from typing import Protocol, Sequence

VOICE_EMPTY_MSG = "(heard nothing)"


class SpeechRecognizer(Protocol):
    def listen(self) -> str:
        ...


class SpeechSynthesizer(Protocol):
    def speak(self, text: str) -> None:
        ...


class MockSpeechRecognizer:
    """Scripted recognizer. Pops transcripts in order; empty when exhausted."""

    def __init__(self, transcripts: Sequence[str] = ()) -> None:
        self._queue = list(transcripts)
        self.listen_count = 0

    def listen(self) -> str:
        self.listen_count += 1
        if self._queue:
            item = self._queue.pop(0)
            return "" if item is None else str(item)
        return ""


class MockSpeechSynthesizer:
    """Records spoken text in order."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(str(text or ""))
