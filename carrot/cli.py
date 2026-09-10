"""CLI chat loop. Slice 1 interface: terminal only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_json(name: str) -> dict:
    with open(ROOT / "config" / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_carrot(llm=None):
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
            s = carrot.state
            print(f"  mood={s.mood} energy={s.energy:.2f} "
                  f"behavior={carrot.last_behavior.name}/"
                  f"{carrot.last_behavior.band}")
            for k, v in sorted(s.emotions.items()):
                print(f"    {k}={v:.2f}")
            continue
        if line.strip() == "/help":
            print("  /dbg inner state (dev only) | /quit exit")
            continue
        reply = carrot.process_turn(line)
        carrot.store.save(carrot.state)
        print(f"carrot: {reply}")


if __name__ == "__main__":
    main()
