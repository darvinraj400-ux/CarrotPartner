"""CLI chat loop. Slice 1 interface: terminal only."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .drives import drive_pull, update_rise
from .environment import MockEnvironmentProbe
from .initiative import ACTIVE_REASONS, should_initiate
from .memory import MAX_MEMORY_TEXT_CHARS, Memory
from .tools import TOOLS

ROOT = Path(__file__).resolve().parents[1]

HELP_TEXT = ("  /dbg inner state (dev only) | /ctx context (dev only) | "
             "/quit exit\n"
             "  /memories list memories | /mem memory details (dev only) | "
             "/remember <text> save memory | /forget <substring> drop matches | "
             "/drives show drives | /env show environment\n"
             "  /tools list tools | /calls recent tool calls | "
             "/allow <tool> allow this session | /deny <tool> deny this session\n"
             "  /permissions view consents | /allow <tool> [always] grant | "
             "/revoke <tool> clear consent | /revoke-all clear all\n"
             "  /initiative show initiative decision")
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


def format_tools(carrot) -> str:
    """User-facing tool list with config + session state. Reads only."""
    lines = []
    for name in TOOLS:
        enabled = "enabled" if carrot.tools_config.enabled.get(name) \
            else "disabled"
        grant = carrot.session_grants.get(name)
        session = "granted" if grant is True else \
            "denied" if grant is False else "none"
        lines.append(f"{name}: {enabled} (session: {session})")
    return "\n".join(lines)


def format_calls(carrot) -> str:
    """Recent tool-call log. Reads only: previews, no timestamps."""
    if not carrot.last_tool_calls:
        return "(no tool calls yet)"
    lines = []
    for call in carrot.last_tool_calls:
        allowed = "yes" if call.allowed else "no"
        ok = "yes" if call.ok else "no"
        lines.append(f"{call.tool} allowed={allowed} ok={ok} "
                     f"args={call.args_preview}")
    return "\n".join(lines)


def handle_allow(carrot, arg: str) -> str:
    parts = str(arg or "").strip().split()
    if not parts or len(parts) > 2 or (len(parts) == 2
                                       and parts[1] != "always"):
        return "usage: /allow <tool> [always]"
    tool = parts[0]
    if tool not in TOOLS:
        return f"unknown tool: {tool}"
    if len(parts) == 2:
        current = carrot.permission_state.grants.get(tool)
        if current is True and carrot.session_grants.get(tool) is True:
            return "already granted (always)"
        carrot.permission_state = carrot.permission_store.set_grant(
            tool, True)
        carrot.session_grants[tool] = True
        return f"granted: {tool} (always)"
    if carrot.session_grants.get(tool) is True:
        return "already granted (session)"
    carrot.session_grants[tool] = True
    return f"granted: {tool} (session)"


def handle_deny(carrot, arg: str) -> str:
    tool = str(arg or "").strip()
    if not tool:
        return "usage: /deny <tool>"
    if tool not in TOOLS:
        return f"unknown tool: {tool}"
    carrot.session_grants[tool] = False
    return f"denied: {tool}"


def handle_revoke(carrot, arg: str) -> str:
    tool = str(arg or "").strip()
    if not tool:
        return "usage: /revoke <tool>"
    if tool not in TOOLS:
        return f"unknown tool: {tool}"
    carrot.permission_state = carrot.permission_store.clear_grant(tool)
    carrot.session_grants[tool] = False
    return f"revoked: {tool}"


def handle_revoke_all(carrot) -> str:
    carrot.permission_state = carrot.permission_store.clear_all()
    carrot.session_grants.clear()
    return "revoked all"


def format_permissions(carrot) -> str:
    """Merged consent view: config level/enabled + session + persistent."""
    lines = []
    for name in TOOLS:
        level = carrot.tools_config.levels.get(name, 4)
        enabled = "yes" if carrot.tools_config.enabled.get(name) \
            else "no"
        session = carrot.session_grants.get(name)
        session_label = "granted" if session is True else \
            "denied" if session is False else "none"
        persistent = carrot.permission_state.grants.get(name)
        persistent_label = "granted" if persistent is True else \
            "denied" if persistent is False else "none"
        lines.append(f"{name}: level={level} enabled={enabled} "
                     f"session={session_label} "
                     f"persistent={persistent_label}")
    return "\n".join(lines)


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


def format_initiative(carrot) -> str:
    """Read-only initiative view: current decision + per-reason cooldowns."""
    now = carrot._now()
    last_seen = carrot.state.last_seen
    elapsed = (now - last_seen) if last_seen else 0.0
    drives = update_rise(carrot.drive_store.load(), elapsed,
                         carrot.drives_config.rise_per_hour)
    state = carrot.initiative_store.load()
    decision = should_initiate(
        drives, elapsed, carrot.last_environment,
        state.per_reason_last_fire, now, carrot.initiative_config)
    lines = [f"decision: {decision.reason if decision is not None else 'none'}"]
    for reason in ACTIVE_REASONS:
        last = state.per_reason_last_fire.get(reason, 0.0)
        cooldown = carrot.initiative_config.cooldowns.get(reason, 0.0)
        remaining = max(0.0, cooldown - (now - last))
        if remaining <= 0:
            lines.append(f"{reason}: ready")
        else:
            lines.append(f"{reason}: cooldown({int(remaining)})")
    return "\n".join(lines)


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


def build_carrot(llm=None, time_fn=None, environment_probe=None):
    from .appraisal import Appraisal
    from .behavior import BehaviorSelector
    from .carrot import Carrot
    from .classifier import DeterministicClassifier
    from .conversation import ConversationBuffer
    from .emotions import EmotionState
    from .environment import EnvironmentConfig
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
    environment_config = EnvironmentConfig.load(
        str(ROOT / "config" / "environment.json"))
    if environment_probe is None:
        if sys.platform == "win32":
            from .environment_win32 import WindowsEnvironmentProbe

            environment_probe = WindowsEnvironmentProbe(
                idle_threshold_s=environment_config.idle_threshold_s)
        else:
            environment_probe = MockEnvironmentProbe()
    carrot = Carrot(
        state=state,
        personality=personality,
        classifier=DeterministicClassifier(),
        appraisal=Appraisal(_load_json("appraisal.json")),
        selector=BehaviorSelector(_load_json("behavior.json")),
        llm=llm or make_provider(),
        decay_cfg=_load_json("decay.json"),
        buffer=ConversationBuffer(),
        environment_probe=environment_probe,
        environment_config=environment_config,
    )
    if isinstance(carrot.llm, MockLLM) and carrot.llm._behavior_getter is None:
        carrot.llm._behavior_getter = lambda: carrot.last_behavior
    carrot.store = store
    return carrot


_INITIATIVE_REASON_LINES = {
    "long-absence": "The user has been away for a while.",
    "high-social-drive": "You've been wanting to talk to them.",
    "session-open": "A fresh session just started.",
    "user-returned": "The user just came back.",
}


def _initiative_system_prompt(carrot, reason: str) -> str:
    sketch = carrot.personality.sketch()
    reason_line = _INITIATIVE_REASON_LINES[reason]
    return (
        sketch + " " + reason_line + " "
        "Open with one short, natural message. No numbers. No questions "
        "requiring urgent answers. If nothing meaningful to say, a brief "
        "greeting is fine."
    )


def _parse_away_args(argv) -> tuple[bool, int]:
    args = list(argv or [])
    if "--away" not in args:
        return False, 3
    rest = args[args.index("--away") + 1:]
    if not rest:
        return True, 3
    try:
        number = int(rest[0])
    except (TypeError, ValueError):
        return True, 3
    return True, max(1, min(10, number))


def _run_away_mode(carrot, max_n: int) -> None:
    """Bounded initiative actuation. Never calls process_turn, never writes
    to the conversation buffer. Each iteration loads drives read-only,
    decides, renders one message via llm.generate, prints it, and records
    the fire. Terminates after at most max_n iterations or on the first
    None decision."""
    for _ in range(max_n):
        now = carrot._now()
        last_seen = carrot.state.last_seen
        elapsed = (now - last_seen) if last_seen else 0.0
        raw_drives = carrot.drive_store.load()
        drives = update_rise(raw_drives, elapsed,
                             carrot.drives_config.rise_per_hour)
        state = carrot.initiative_store.load()
        decision = should_initiate(
            drives, elapsed, carrot.last_environment,
            state.per_reason_last_fire, now, carrot.initiative_config,
        )
        if decision is None:
            break
        system = _initiative_system_prompt(carrot, decision.reason)
        message = carrot.llm.generate(
            system=system, history=[], user="Open with a short greeting.",
        )
        print(message)
        carrot.initiative_store.record_fire(decision.reason, now)


def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = list(sys.argv[1:] if argv is None else argv)
    away, max_n = _parse_away_args(args)
    carrot = build_carrot()
    if away:
        _run_away_mode(carrot, max_n)
        return
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
        if line.strip() == "/initiative":
            print(format_initiative(carrot))
            continue
        if line.strip() == "/tools":
            print(format_tools(carrot))
            continue
        if line.strip() == "/calls":
            print(format_calls(carrot))
            continue
        if line.strip() == "/permissions":
            print(format_permissions(carrot))
            continue
        if line.strip() == "/revoke-all":
            print(handle_revoke_all(carrot))
            continue
        if line.strip() == "/revoke":
            print(handle_revoke(carrot, ""))
            continue
        if line.strip().startswith("/revoke "):
            print(handle_revoke(carrot, line.strip()[len("/revoke"):]))
            continue
        if (line.strip() == "/allow"
                or line.strip().startswith("/allow ")):
            print(handle_allow(carrot, line.strip()[len("/allow"):]))
            continue
        if (line.strip() == "/deny"
                or line.strip().startswith("/deny ")):
            print(handle_deny(carrot, line.strip()[len("/deny"):]))
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
