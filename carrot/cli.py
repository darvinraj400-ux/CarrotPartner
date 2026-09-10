"""CLI chat loop. Slice 1 interface: terminal only."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .drives import drive_pull
from .memory import MAX_MEMORY_TEXT_CHARS, Memory

ROOT = Path(__file__).resolve().parents[1]

HELP_TEXT = ("  /dbg inner state (dev only) | /ctx context (dev only) | "
             "/quit exit\n"
             "  /memories list memories | /mem memory details (dev only) | "
             "/remember <text> save memory | /forget <substring> drop matches | "
             "/drives show drives | /env show environment")
NO_CONTEXT_YET = "(no context yet)"
NO_MEMORIES_YET = "(no memories yet)"


def format_ctx(ctx) -> str:
    """Render a ConversationContext as text. Reads only, no floats."""
    lines = [
        f"  pending_question: {ctx.pending_question}",
        f"  playful_frame: {ctx.playful_frame}",
        f"  recent_harshness: {ctx.recent_harshness}",
        f"  recent_topics: [{', '.join(ctx.recent_topics)}]",
        f"  turns: {len(ctx.turns)}",
    ]
    for turn in list(ctx.turns)[-3:]:
        lines.append(f"    {turn.role}: {turn.text}")
    return "\n".join(lines)


def handle_ctx(carrot) -> str:
    """Return /ctx output without touching state, buffer, or disk."""
    if carrot.last_context is None:
        return NO_CONTEXT_YET
    return format_ctx(carrot.last_context)


def format_memories(store) -> str:
    """User-facing memory list. Reads only: no floats, no timestamps."""
    mems = store.all()
    if not mems:
        return NO_MEMORIES_YET
    return "\n".join(f"[{m.kind}] {m.text}" for m in mems)


def format_mem_debug(store) -> str:
    """Debug memory view. Floats permitted (dev surface, like /dbg)."""
    mems = store.all()
    if not mems:
        return NO_MEMORIES_YET
    return "\n".join(
        f"[{m.kind}] {m.text} (importance={m.importance}, "
        f"created={m.created_ts}, recalled={m.last_recalled_ts})"
        for m in mems)


def handle_remember(carrot, arg: str) -> str:
    text = str(arg or "").strip()
    if not text:
        return "usage: /remember <text>"
    if len(text) > MAX_MEMORY_TEXT_CHARS:
        return "too long"
    existing = {m.text.strip().lower() for m in carrot.memory_store.all()}
    if text.lower() in existing:
        return "already remembered"
    now = carrot._now()
    ok = carrot.memory_store.add(Memory(
        text=text, kind="fact", importance=0.8,
        created_ts=now, last_recalled_ts=now))
    return "remembered" if ok else "memory full"


def handle_forget(carrot, arg: str) -> str:
    substring = str(arg or "").strip()
    if not substring:
        return "usage: /forget <substring>"
    removed = carrot.memory_store.remove_matching(substring)
    if removed == 0:
        return "no matches"
    return f"forgot {removed}"


def _drive_band(level: float, threshold: float) -> str:
    if level >= threshold:
        return "high"
    if level >= 0.3:
        return "medium"
    return "low"


def format_drives(drive_store, drives_config) -> str:
    """User-facing drives view. Reads only: labels, no floats."""
    state = drive_store.load()
    threshold = drives_config.pull_threshold
    pull = drive_pull(state, drives_config)
    return (f"social: {_drive_band(state.social, threshold)} "
            f"novelty: {_drive_band(state.novelty, threshold)}\n"
            f"pull: {pull if pull is not None else 'none'}")


def format_env(carrot) -> str:
    """User-facing environment view. Reads only: labels, no floats."""
    snapshot = carrot.last_environment
    if snapshot is None:
        return "app: unknown\nidle: False\nobserved: no"
    app = snapshot.app_noun if snapshot.app_noun is not None else "unknown"
    observed = snapshot.app_noun is not None or snapshot.idle
    return (f"app: {app}\n"
            f"idle: {snapshot.idle}\n"
            f"observed: {'yes' if observed else 'no'}")


def format_dbg(carrot) -> str:
    """Debug inner-state view. Floats permitted (dev surface)."""
    s = carrot.state
    lines = [f"  mood={s.mood} energy={s.energy:.2f} "
             f"behavior={carrot.last_behavior.name}/"
             f"{carrot.last_behavior.band}"]
    for k, v in sorted(s.emotions.items()):
        lines.append(f"    {k}={v:.2f}")
    drives = carrot.drive_store.load()
    lines.append(f"  social={drives.social:.2f} novelty={drives.novelty:.2f}")
    env = carrot.last_environment
    app = env.app_noun if env is not None and env.app_noun is not None \
        else None
    idle = env.idle if env is not None else False
    lines.append(f"  environment: app={app} idle={idle}")
    return "\n".join(lines)

def _load_json(name: str) -> dict:
    with open(ROOT / "config" / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_carrot(llm=None, time_fn=None):
    from .appraisal import Appraisal
    from .behavior import BehaviorSelector
    from .carrot import Carrot
    from .classifier import DeterministicClassifier
    from .conversation import ConversationBuffer
    from .emotions import EmotionState
    from .llm import MockLLM, make_provider
    from .personality import Personality
    from .state_store import StateStore

    store = StateStore(str(ROOT / "state" / "state.json"))
    state = store.load()
    # Session accounting: count the launch and stamp first contact.
    # last_seen is deliberately untouched so decay/return logic is
    # unaffected by launching.
    state.session_count += 1
    if state.first_seen_ts == 0.0:
        state.first_seen_ts = (time_fn or time.time)()
    store.save(state)
    personality = Personality.load(str(ROOT / "config" / "personality.json"))
    carrot = Carrot(
        state=state,
        personality=personality,
        classifier=DeterministicClassifier(),
        appraisal=Appraisal(_load_json("appraisal.json")),
        selector=BehaviorSelector(_load_json("behavior.json")),
        llm=llm or make_provider(),
        decay_cfg=_load_json("decay.json"),
        buffer=ConversationBuffer(),
    )
    if isinstance(carrot.llm, MockLLM) and carrot.llm._behavior_getter is None:
        carrot.llm._behavior_getter = lambda: carrot.last_behavior
    carrot.store = store
    return carrot


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    carrot = build_carrot()
    print("Carrot (slice 1) -- type /quit to exit, /dbg for inner state.")
    while True:
        try:
            line = input("you: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip() in ("/quit", "/exit"):
            break
        if line.strip() == "/dbg":
            print(format_dbg(carrot))
            continue
        if line.strip() == "/help":
            print(HELP_TEXT)
            continue
        if line.strip() == "/ctx":
            print(handle_ctx(carrot))
            continue
        if line.strip() == "/memories":
            print(format_memories(carrot.memory_store))
            continue
        if line.strip() == "/mem":
            print(format_mem_debug(carrot.memory_store))
            continue
        if line.strip() == "/drives":
            print(format_drives(carrot.drive_store, carrot.drives_config))
            continue
        if line.strip() == "/env":
            print(format_env(carrot))
            continue
        if (line.strip() == "/remember"
                or line.strip().startswith("/remember ")):
            print(handle_remember(carrot, line.strip()[len("/remember"):]))
            continue
        if (line.strip() == "/forget"
                or line.strip().startswith("/forget ")):
            print(handle_forget(carrot, line.strip()[len("/forget"):]))
            continue
        reply = carrot.process_turn(line)
        carrot.store.save(carrot.state)
        print(f"carrot: {reply}")


if __name__ == "__main__":
    main()
