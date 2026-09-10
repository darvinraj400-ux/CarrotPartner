"""Atomic JSON persistence for Slice 1. No database."""
from __future__ import annotations

import json
import os
import tempfile
import time

from .emotions import EmotionState


class StateStore:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)

    def _quarantine(self) -> None:
        """Move a corrupt file aside; never raise (fresh state follows)."""
        try:
            parent = os.path.dirname(self.path) or "."
            dest = os.path.join(parent,
                                f"quarantine-{int(time.time())}.json")
            os.replace(self.path, dest)
        except OSError:
            pass

    def load(self) -> EmotionState:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return EmotionState()
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            self._quarantine()
            return EmotionState()
        try:
            return EmotionState.from_dict(data)
        except (ValueError, AttributeError, TypeError):
            self._quarantine()
            return EmotionState()

    def save(self, state: EmotionState) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state.to_dict(), fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
