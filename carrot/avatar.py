"""Avatar expression mapping — pure, no tkinter, no state, no I/O.

Contract:
- The avatar is a read-only expression layer over existing state.
- This module contains only the mapping from (mood, behavior, band,
  closeness band) to a visual Expression. Phase 2 adds the window.
- No imports of carrot.carrot, carrot.cli, carrot.llm. No process_turn.
  No generate. No after()/after_idle(). No file writes.
- Slice 12 is visual presence only; voice (Slice 13) attaches via
  expression_for, not via the window.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .mood import MOODS
from .relationship import RelationshipStore
from .state_store import StateStore

VISUAL_STATES = (
    "content", "happy", "tender", "sad", "tense", "shy", "curious",
)
INTENSITY_BANDS = ("low", "medium", "high")

AVATAR_UNAVAILABLE_MSG = "(avatar unavailable: no display)"


@dataclass(frozen=True)
class Expression:
    name: str            # one of VISUAL_STATES
    intensity_band: str  # one of INTENSITY_BANDS
    palette_key: str     # equals name for now; a stable hook for Phase 2
    presence_cue: str | None  # None or "warm"

    def __post_init__(self) -> None:
        if self.name not in VISUAL_STATES:
            raise ValueError(f"bad visual state: {self.name!r}")
        if self.intensity_band not in INTENSITY_BANDS:
            raise ValueError(
                f"bad intensity band: {self.intensity_band!r}")
        if self.presence_cue is not None and self.presence_cue != "warm":
            raise ValueError(
                f"bad presence cue: {self.presence_cue!r}")
        if not isinstance(self.palette_key, str) or not self.palette_key:
            raise ValueError("palette_key must be a non-empty string")


def expression_for(
    mood: str,
    behavior_name: str,
    band: str,
    closeness_band: str | None = None,
) -> Expression:
    """Map (mood, behavior_name, band, closeness_band) to an Expression.

    Behavior-name overrides (behavior wins over mood):
      - behavior_name in ("shy", "embarrassed") and mood != "low" -> "shy"
      - behavior_name == "curious" and mood != "bright"           -> "curious"

    Mood-driven (no override applies):
      - mood == "bright"  -> "happy"
      - mood == "low"     -> "sad"
      - mood == "tense"   -> "tense"
      - mood == "tender"  -> "tender"

    Fallback:
      - otherwise -> "content"

    intensity_band is passed through (normalized against INTENSITY_BANDS;
    unknown band -> "medium").

    presence_cue is "warm" iff closeness_band == "high", else None.

    Pure: no I/O, no clock, no state reads, deterministic.
    """
    clean_mood = mood if isinstance(mood, str) and mood in MOODS \
        else "content"
    clean_behavior = behavior_name \
        if isinstance(behavior_name, str) else ""
    clean_band = band if band in INTENSITY_BANDS else "medium"
    if clean_behavior in ("shy", "embarrassed") and clean_mood != "low":
        name = "shy"
    elif clean_behavior == "curious" and clean_mood != "bright":
        name = "curious"
    elif clean_mood == "bright":
        name = "happy"
    elif clean_mood == "low":
        name = "sad"
    elif clean_mood == "tense":
        name = "tense"
    elif clean_mood == "tender":
        name = "tender"
    else:
        name = "content"
    cue = "warm" if closeness_band == "high" else None
    return Expression(name=name, intensity_band=clean_band,
                      palette_key=name, presence_cue=cue)


def _closeness_band(closeness: float) -> str | None:
    """Return 'high', 'medium', 'low', or None (unknown)."""
    try:
        c = float(closeness)
    except (TypeError, ValueError):
        return None
    if c >= 0.7:
        return "high"
    if c >= 0.4:
        return "medium"
    return "low"


def _draw_eyes(canvas, name: str) -> None:
    """Eye geometry per visual state. Runs only inside the tkinter branch."""
    if name == "happy":
        canvas.create_arc(62, 78, 88, 100, start=0, extent=180,
                          outline="#333", width=3, style="arc")
        canvas.create_arc(112, 78, 138, 100, start=0, extent=180,
                          outline="#333", width=3, style="arc")
    elif name == "tender":
        canvas.create_arc(60, 85, 90, 100, start=180, extent=180,
                          outline="#333", width=3, style="arc")
        canvas.create_arc(110, 85, 140, 100, start=180, extent=180,
                          outline="#333", width=3, style="arc")
    elif name == "sad":
        canvas.create_arc(62, 88, 88, 110, start=180, extent=180,
                          outline="#333", width=3, style="arc")
        canvas.create_arc(112, 88, 138, 110, start=180, extent=180,
                          outline="#333", width=3, style="arc")
    elif name == "tense":
        canvas.create_oval(58, 80, 92, 104, outline="#333", width=3)
        canvas.create_oval(108, 80, 142, 104, outline="#333", width=3)
    elif name == "shy":
        canvas.create_oval(66, 92, 86, 108, outline="#333", width=2,
                           fill="#333")
        canvas.create_oval(114, 92, 134, 108, outline="#333", width=2,
                           fill="#333")
    elif name == "curious":
        canvas.create_oval(62, 82, 88, 104, outline="#333", width=3)
        canvas.create_oval(116, 86, 132, 100, outline="#333", width=2,
                           fill="#333")
    else:  # content and any fallback
        canvas.create_oval(64, 82, 88, 104, outline="#333", width=3)
        canvas.create_oval(112, 82, 136, 104, outline="#333", width=3)


def _draw_mouth(canvas, name: str) -> None:
    """Mouth geometry per visual state. Runs only inside the tkinter branch."""
    if name == "happy":
        canvas.create_arc(70, 110, 130, 148, start=200, extent=140,
                          outline="#333", width=3, style="arc")
    elif name == "tender":
        canvas.create_arc(75, 115, 125, 142, start=200, extent=140,
                          outline="#333", width=2, style="arc")
    elif name == "sad":
        canvas.create_arc(70, 122, 130, 152, start=20, extent=140,
                          outline="#333", width=3, style="arc")
    elif name == "tense":
        canvas.create_line(75, 132, 125, 132, fill="#333", width=3)
    elif name == "shy":
        canvas.create_arc(82, 122, 118, 142, start=200, extent=140,
                          outline="#333", width=2, style="arc")
    elif name == "curious":
        canvas.create_oval(92, 122, 108, 136, outline="#333", width=2)
    else:  # content and any fallback
        canvas.create_line(75, 130, 125, 130, fill="#333", width=3)


def launch_avatar(
    state_path: Path | str,
    *,
    launcher: Callable[[Path | str], str] | None = None,
) -> str:
    """Launch a minimal avatar window that reads state.json read-only.

    Returns a status string:
      - AVATAR_UNAVAILABLE_MSG if tkinter cannot be imported
      - "avatar closed" on normal close
      - launcher's return value when a custom launcher is injected

    Read-only over state.json, relationship.json (via RelationshipStore).
    No writes. No process_turn. No generate. No after()/after_idle().
    No threads. Refresh is via the window's Refresh button.

    If launcher is provided, delegates to it (test injection path).
    """
    if launcher is not None:
        return launcher(state_path)

    try:
        import tkinter as tk
    except ImportError:
        return AVATAR_UNAVAILABLE_MSG

    # ---- Window construction (only runs when tkinter is available) ----
    state = StateStore(str(state_path)).load()
    closeness = 0.5
    try:
        rel_path = Path(state_path).parent / "relationship.json"
        closeness = RelationshipStore(rel_path).load().closeness
    except Exception:
        pass  # relationship file may not exist; treat as default

    expression = expression_for(
        state.mood,
        behavior_name="",       # not persisted; see Slice 12 known limitations
        band="medium",
        closeness_band=_closeness_band(closeness),
    )

    root = tk.Tk()
    root.title("Carrot")
    root.geometry("220x260")

    frame_bg = "#f7d9a0" if expression.presence_cue == "warm" else "#f0f0f0"
    root.configure(bg=frame_bg)

    canvas = tk.Canvas(root, width=200, height=200,
                       bg=frame_bg, highlightthickness=0)
    canvas.pack(pady=10)

    def draw(expr):
        canvas.delete("all")
        # Face circle
        canvas.create_oval(40, 40, 160, 160, outline="#333", width=3,
                           fill="#fdf4e3")
        # Eyes (parameterized by expression.name)
        _draw_eyes(canvas, expr.name)
        # Mouth (parameterized by expression.name)
        _draw_mouth(canvas, expr.name)

    def refresh():
        try:
            s = StateStore(str(state_path)).load()
            c = 0.5
            try:
                c = RelationshipStore(rel_path).load().closeness
            except Exception:
                pass
            e = expression_for(
                s.mood, behavior_name="", band="medium",
                closeness_band=_closeness_band(c),
            )
            root.configure(
                bg="#f7d9a0" if e.presence_cue == "warm" else "#f0f0f0")
            canvas.configure(bg=root["bg"])
            draw(e)
        except Exception:
            pass  # silent — read failures show last-known state

    draw(expression)

    tk.Button(root, text="Refresh", command=refresh).pack(pady=4)

    root.mainloop()  # ← the single sanctioned GUI pump (Phase 0 pin)
    return "avatar closed"


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[1]
    print(launch_avatar(_root / "state" / "state.json"))
