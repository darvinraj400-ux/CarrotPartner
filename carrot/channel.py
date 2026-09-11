"""Channel architecture for Carrot — the seam every transport plugs into.

Contract:
- A channel is a bounded pair: one message in (receive), one reply out
  (send). Transport details (files, sockets, HTTP) live behind the pair;
  the orchestrator never sees them.
- All channels share the same Carrot state on disk. Cross-channel
  continuity = shared persisted state; the in-memory conversation buffer
  stays per-process (DP1=NO).
- This module holds protocols, mocks, and the file-based adapter only.
  The CLI owns the drain (cli.drain_inbox). No process_turn, no generate.

DP1 clarification:
- DP1=NO forbids persisting the conversation buffer (the last 10 turns
  of in-session context). It does not forbid the transport layer from
  spooling reply text to disk for delivery. The outbox is a transport
  artifact, consumed and capped, not a conversation memory. Curated
  facts still go through the memory store; raw turns still live only
  in the buffer.
"""
from __future__ import annotations

import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Protocol, Sequence

CHANNEL_EMPTY_MSG = "(no messages)"
MAX_DRAIN = 100
MAX_BODY_BYTES = 4096
MAX_OUTBOX_FILES = 100

_FILENAME_RE = re.compile(r"^\d{1,6}-[0-9a-f]{1,32}\.msg$")


class Channel(Protocol):
    def receive(self) -> str:
        ...

    def send(self, text: str) -> None:
        ...


class MockChannel:
    """Scripted channel. Pops incoming in order; records outgoing."""

    def __init__(self, incoming: Sequence[str] = ()) -> None:
        self._queue: list[str] = list(incoming)
        self.sent: list[str] = []
        self.receive_count: int = 0

    def receive(self) -> str:
        self.receive_count += 1
        if self._queue:
            return str(self._queue.pop(0))
        return ""

    def send(self, text: str) -> None:
        self.sent.append(str(text or ""))


class FileChannel:
    """File-based channel. Directories under base_dir:

      inbox/        incoming .msg files (oldest-consumed first)
      inbox/done/   messages successfully read
      inbox/bad/    malformed filenames or oversized/invalid bodies
      outbox/       reply files written by send()

    receive(): find oldest valid inbox/*.msg (lexicographic by filename),
               read up to MAX_BODY_BYTES, move to done/, return content.
               Malformed filename or oversized body -> move to bad/,
               continue. Returns "" if nothing waiting.
    send(text): write <seq>-<hex>.msg to outbox/ atomically, prune oldest
                beyond MAX_OUTBOX_FILES.
    """

    def __init__(self, base_dir: Path | str) -> None:
        self.base = Path(base_dir)
        self.inbox = self.base / "inbox"
        self.done = self.inbox / "done"
        self.bad = self.inbox / "bad"
        self.outbox = self.base / "outbox"
        for directory in (self.inbox, self.done, self.bad, self.outbox):
            directory.mkdir(parents=True, exist_ok=True)

    def _move_to(self, src: Path, dest_dir: Path) -> None:
        try:
            os.replace(src, dest_dir / src.name)
        except FileExistsError:
            stem, suffix = src.stem, src.suffix
            dest_dir.mkdir(parents=True, exist_ok=True)
            os.replace(src, dest_dir / f"{stem}-{secrets.token_hex(4)}"
                       f"{suffix}")
        except OSError:
            pass

    def receive(self) -> str:
        candidates = sorted(p for p in self.inbox.glob("*.msg")
                            if p.is_file())
        for path in candidates:
            if not _FILENAME_RE.match(path.name):
                self._move_to(path, self.bad)
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                self._move_to(path, self.bad)
                continue
            if len(raw) > MAX_BODY_BYTES:
                self._move_to(path, self.bad)
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                self._move_to(path, self.bad)
                continue
            self._move_to(path, self.done)
            return text
        return ""

    def _next_outbox_seq(self) -> int:
        best = 0
        for path in self.outbox.glob("*.msg"):
            head = path.name.split("-", 1)[0]
            try:
                number = int(head)
            except (TypeError, ValueError):
                continue
            if number > best:
                best = number
        return best + 1

    def _atomic_write_text(self, path: Path, text: str) -> None:
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.write("\n")
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _prune_outbox(self) -> None:
        files = sorted(p for p in self.outbox.glob("*.msg")
                       if p.is_file())
        overflow = len(files) - MAX_OUTBOX_FILES
        for path in files[:max(0, overflow)]:
            try:
                path.unlink()
            except OSError:
                pass

    def send(self, text: str) -> None:
        body = str(text or "")
        raw = body.encode("utf-8")
        if len(raw) > MAX_BODY_BYTES:
            # Truncate to the byte cap, keeping only complete characters.
            body = raw[:MAX_BODY_BYTES].decode("utf-8", errors="ignore")
        seq = self._next_outbox_seq()
        name = f"{seq:06d}-{secrets.token_hex(4)}.msg"
        self._atomic_write_text(self.outbox / name, body)
        self._prune_outbox()
