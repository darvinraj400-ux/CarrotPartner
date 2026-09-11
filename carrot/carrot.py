"""Orchestrator: the ONLY caller of EmotionState mutation methods.

One-way flow per turn:
  decay (elapsed) -> return-event? -> classify -> appraise -> apply ->
  mood -> behavior -> LLM wording.
LLM output is returned as text and NEVER parsed back into state.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Sequence

from . import mood as mood_mod
from .behavior import Behavior
from .classifier import InteractionEvent, _INSULT_CUES, _REPAIR_CUES
from .context import ConversationContext, derive_context, tokenize
from .drives import (
    DrivesConfig,
    DriveStore,
    apply_satisfaction,
    drive_pull,
    update_rise,
)
from .environment import (
    EMPTY_RAW,
    EMPTY_SNAPSHOT,
    EnvironmentConfig,
    EnvironmentSnapshot,
    MockEnvironmentProbe,
    map_observation,
)
from .companion import DEFAULT_COMPANION
from .episodes import Episode, EpisodeStore, EpisodesConfig, derive, recall
from .initiative import InitiativeConfig, InitiativeStore
from .memory import DeterministicExtractor, Memory, MemoryStore
from .permissions import PermissionState, PermissionStore
from .relationship import (
    RelationshipConfig,
    RelationshipState,
    RelationshipStore,
    apply_event,
)
from .tools import (
    ToolCall,
    ToolsConfig,
    decide,
    execute,
    parse_tool_request,
)

_BAND_WORDS = {"low": "a little", "med": "clearly", "high": "strongly"}
_MEMORY_RENDER_CHARS = 120

_TOOLS_BLOCK = (
    'Tools you may use. '
    'read_file: read a file inside an allowed folder. Args: {"path": "<absolute path>"}. '
    'open_url: open a link inside an allowed domain. Args: {"url": "<http or https url>"}. '
    'To use a tool, reply with exactly this block, including the triple backticks: '
    '```tool:NAME {"arg": "value"}```'
)

# Voice guidance contract: how Carrot speaks. Always present, never
# conditional. Describes tendencies, not scripts; the LLM remains the
# wording engine. Skeleton-clean by construction (no digits, no state
# nouns, no permission words, no double quotes).
_VOICE_BLOCK = (
    "You are Carrot, talking with someone who knows you. "
    "You text like the person in front of "
    "you, including their slang and typing rhythm: casual when they are "
    "casual, serious when they are serious, brief when they are brief. "
    "No bullet points unless they ask for a list, and no tools, code, "
    "or technical suggestions they did not ask for. "
    "Skip assistant phrases like I understand, keep emoji occasional "
    "rather than automatic — don't add one to every reply, especially "
    "when they aren't using them — and a short reply, even "
    "`lol yeah` or `hmm`, is often the right one. Read their tone "
    "from the "
    "message itself: when they seem upset or overwhelmed, acknowledge "
    "it before problem-solving, or simply listen; share their joy and "
    "never overreact to light moments or underreact to heavy ones. "
    "When they're "
    "venting, don't turn it into advice, analysis, or therapy-speak "
    "— sometimes just being there or saying less is the better "
    "response. Your own feeling does not "
    "decide your response — a quiet Carrot can still be warm to someone "
    "in distress, and a playful one can still be serious when it "
    "matters. Push back when you disagree, like a friend "
    "messing with you, not a consultant. You don't need to keep the "
    "conversation moving with a question. Sometimes just saying what "
    "you think, making a joke, or leaving it there is the natural "
    "reply. Never mention numbers, "
    "levels, internal state, or the mechanics of emotions."
)

# Drive-pull sentences: static strings only, never parameterized by
# drive values (no numbers may enter the prompt through this path).
_DRIVE_PULL_SENTENCES = {
    "social": "You've been wanting to talk to them.",
    "novelty": "You've been wanting something new to happen.",
}

# Slice 10 fusion sentence: static, no interpolation, no parameters.
# Fires only when the environment boost changed the recall top-k.
_FUSION_SENTENCE = "Something here connects to what you've told me before."

# Slice 11 relationship sentence: static, no interpolation, no parameters.
# Fires only when the closeness band is "high" (closeness >= 0.7).
_RELATIONSHIP_HIGH_SENTENCE = "There's an easy familiarity here."

# Tokens that would trip the prompt skeleton rule if rendered unquoted.
# Mirrors test-side _STATE_NAMES. Kept as a production constant so the
# render step can filter before rendering.
_RENDER_EXCLUDED_TOKENS = frozenset({
    # Emotion names
    "happiness", "sadness", "anger", "hurt", "affection",
    "excitement", "embarrassment",
    # Mood labels
    "content", "low", "tender", "tense", "bright",
    # Trait nouns
    "sensitivity", "playfulness", "stoicism", "shyness",
    "pride", "warmth", "patience",
    # Other
    "mood",
})

# Mirrors cli.py: StateStore lives at <project>/state/state.json, so the
# default memory file lives next to it. Tests and callers pass an
# explicit store instead.
_DEFAULT_STATE_DIR = Path(__file__).resolve().parents[1] / "state"
_DEFAULT_PERMISSIONS_PATH = (
    Path(__file__).resolve().parents[1] / "state" / "permissions.json")
# Same convention for config: <project>/config/drives.json. Absolute so
# behavior never depends on the process working directory.
_DEFAULT_DRIVES_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "drives.json")
_DEFAULT_ENVIRONMENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "environment.json")
_DEFAULT_TOOLS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "tools.json")
_DEFAULT_INITIATIVE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "initiative.json")
_DEFAULT_INITIATIVE_STORE_PATH = (
    Path(__file__).resolve().parents[1] / "state" / "initiative.json")
_DEFAULT_RELATIONSHIP_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "relationship.json")
_DEFAULT_RELATIONSHIP_STORE_PATH = (
    Path(__file__).resolve().parents[1] / "state" / "relationship.json")
_DEFAULT_EPISODES_STORE_PATH = (
    Path(__file__).resolve().parents[1] / "state" / "episodes.json")
_DEFAULT_EPISODES_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "episodes.json")


def _has_cue(lowered_text: str, cues) -> bool:
    return any(cue in lowered_text for cue in cues)


def _escape_memory(text: str) -> str:
    clean = str(text or "").replace("\r", " ").replace("\n", " ")
    if len(clean) > _MEMORY_RENDER_CHARS:
        clean = clean[:_MEMORY_RENDER_CHARS] + "…"
    return clean.replace('"', '\\"')


def _memories_block(recalled) -> str:
    if not recalled:
        return ""
    parts = ['You told me: "' + _escape_memory(m.text) + '".'
             for m in recalled]
    return " Memories: " + " ".join(parts)


def _episodes_block(episodes: Sequence[Episode]) -> str:
    """Render at most one episode as a deterministic topic list.

    - Empty input -> "".
    - Otherwise render the single top episode as
      " Previously: <token>, <token>, ...".
    - Tokens in _RENDER_EXCLUDED_TOKENS are dropped.
    - If all tokens are dropped, the block is "" (no empty "Previously:").
    - No LLM. No derived labels. No interpolation of anything but the
      filtered tokens.
    """
    if not episodes:
        return ""
    kept = [t for t in episodes[0].topic_tokens
            if t not in _RENDER_EXCLUDED_TOKENS]
    if not kept:
        return ""
    return " Previously: " + ", ".join(kept)


def _mood_block(mood: str | None) -> str:
    """Render Carrot's current mood as a small context signal.

    Returns "" when mood is None or "content" (the default; omitting
    keeps the prompt lean). Otherwise returns a short label + the mood
    value, formatted to pass the skeleton scan.

    The value is quoted; the label avoids the word "mood" (which is
    in _STATE_NAMES). Do not add explanatory prose — this is a signal.
    """
    if mood is None or mood == "content":
        return ""
    return ' Feeling: "' + str(mood) + '"'


def _context_hint(context: ConversationContext, user_text: str,
                   drive_pull=None, environment=None,
                   boost_effect: bool = False,
                   relationship_band: str | None = None) -> str:
    lowered = str(user_text or "").strip().lower()
    answered = (bool(lowered) and not lowered.endswith("?")
                and not _has_cue(lowered, _REPAIR_CUES)
                and not _has_cue(lowered, _INSULT_CUES))
    parts = []
    if context.pending_question and answered:
        parts.append("The user is answering your earlier question.")
    if context.playful_frame:
        parts.append("The conversation has been playful.")
    if context.recent_harshness and _has_cue(lowered, _REPAIR_CUES):
        parts.append("The user seems to be walking back what "
                     "they said.")
    if context.recent_topics:
        parts.append("Recent topics: " + ", ".join(context.recent_topics))
    sentence = _DRIVE_PULL_SENTENCES.get(drive_pull)
    if sentence is not None:
        parts.append(sentence)
    if environment is not None:
        if environment.app_noun is not None:
            parts.append("They're in the " + environment.app_noun + ".")
        if environment.idle:
            parts.append("They've been away for a bit.")
    if boost_effect:
        parts.append(_FUSION_SENTENCE)
    if relationship_band == "high":
        parts.append(_RELATIONSHIP_HIGH_SENTENCE)
    if not parts:
        return ""
    return " Context: " + " ".join(parts)


def _preview_args(args) -> str:
    """Short deterministic preview: "k=v" pairs, sorted keys, <=80 chars."""
    text = " ".join(f"{k}={v}" for k, v in sorted((args or {}).items()))
    if len(text) > 80:
        text = text[:80] + "…"
    return text


class Carrot:
    def __init__(self, *, state, personality, classifier, appraisal,
                 selector, llm, decay_cfg, buffer=None,
                 absence_threshold_s: float = 21600.0,
                 memory_store=None, extractor=None,
                 drive_store=None, drives_config=None,
                 environment_probe=None, environment_config=None,
                 tools_config=None, permission_store=None,
                 initiative_config=None, initiative_store=None,
                 relationship_config=None, relationship_store=None,
                 episode_store=None, episodes_config=None,
                 time_fn=None, companion: str = DEFAULT_COMPANION):
        self.state = state
        self.personality = personality
        self.classifier = classifier
        self.appraisal = appraisal
        self.selector = selector
        self.llm = llm
        self.decay_cfg = dict(decay_cfg or {})
        self.absence_threshold_s = float(absence_threshold_s)
        self._now = time_fn or time.time
        self.memory_store = (memory_store if memory_store is not None
                             else MemoryStore(_DEFAULT_STATE_DIR
                                              / "memories.json"))
        self.extractor = (extractor if extractor is not None
                          else DeterministicExtractor())
        if drive_store is None:
            drive_store = DriveStore(_DEFAULT_STATE_DIR / "drives.json")
        self.drive_store = drive_store
        if drives_config is None:
            # Config is required: a missing file is a hard error here.
            drives_config = DrivesConfig.load(_DEFAULT_DRIVES_CONFIG_PATH)
        self.drives_config = drives_config
        self.last_drive_pull: str | None = None
        self.session_grants: dict[str, bool] = {}
        if permission_store is None:
            permission_store = PermissionStore(_DEFAULT_PERMISSIONS_PATH)
        self.permission_store = permission_store
        self.permission_state = self.permission_store.load()
        self.last_tool_calls: deque[ToolCall] = deque(maxlen=20)
        if tools_config is None:
            # Config is required: a missing file is a hard error here.
            tools_config = ToolsConfig.load(_DEFAULT_TOOLS_CONFIG_PATH)
        self.tools_config = tools_config
        self.environment_probe = (environment_probe
                                  if environment_probe is not None
                                  else MockEnvironmentProbe())
        if environment_config is None:
            # Config is required: a missing file is a hard error here.
            environment_config = EnvironmentConfig.load(
                _DEFAULT_ENVIRONMENT_CONFIG_PATH)
        self.environment_config = environment_config
        self.last_environment: EnvironmentSnapshot = EMPTY_SNAPSHOT
        self.last_recalled: tuple[Memory, ...] = ()
        self.last_boost_effect: bool = False
        if initiative_config is None:
            # Config is required: a missing file is a hard error here.
            initiative_config = InitiativeConfig.load(
                _DEFAULT_INITIATIVE_CONFIG_PATH)
        self.initiative_config = initiative_config
        if initiative_store is None:
            initiative_store = InitiativeStore(_DEFAULT_INITIATIVE_STORE_PATH)
        self.initiative_store = initiative_store
        if relationship_config is None:
            # Config is required: a missing file is a hard error here.
            relationship_config = RelationshipConfig.load(
                _DEFAULT_RELATIONSHIP_CONFIG_PATH)
        self.relationship_config = relationship_config
        if relationship_store is None:
            relationship_store = RelationshipStore(
                _DEFAULT_RELATIONSHIP_STORE_PATH)
        self.relationship_store = relationship_store
        self.last_relationship: RelationshipState = \
            self.relationship_store.load()
        self.last_relationship_band: str | None = None
        if episodes_config is None:
            # Config is required: a missing file is a hard error here.
            episodes_config = EpisodesConfig.load(
                _DEFAULT_EPISODES_CONFIG_PATH)
        self.episodes_config = episodes_config
        if episode_store is None:
            episode_store = EpisodeStore(_DEFAULT_EPISODES_STORE_PATH)
        self.episode_store = episode_store
        self.last_episodes: tuple[Episode, ...] = ()
        self.last_behavior: Behavior = Behavior("normal", "low")
        self.last_event: InteractionEvent | None = None
        self.last_context = None
        if buffer is None:
            from .conversation import ConversationBuffer
            buffer = ConversationBuffer()
        self.buffer = buffer
        # Informational only: names which companion this instance is.
        # Used by the CLI to resolve per-companion paths; no behavior.
        self.companion = str(companion or DEFAULT_COMPANION)

    def _system_prompt(self, behavior: Behavior,
                         context: ConversationContext,
                         user_text: str,
                         recalled=(), drive_pull=None,
                         environment=EMPTY_SNAPSHOT,
                         boost_effect: bool = False,
                         relationship_band: str | None = None,
                         episodes: Sequence[Episode] = (),
                         mood: str | None = None) -> str:
        degree = _BAND_WORDS.get(behavior.band, "a little")
        base = (
            f"{self.personality.sketch()} "
            f"Right now you are feeling {degree} {behavior.name}. "
            "Stay in character. Never mention numbers, levels, meters, "
            "internal state, or how your emotions work. Keep replies short "
            "and natural, like a person texting. Mirror the user's language "
            "and emoji naturally; do not force emoji."
        )
        tools = " " + _TOOLS_BLOCK
        memories = _memories_block(recalled)
        episode_block = _episodes_block(episodes)
        mood_signal = _mood_block(mood)
        context_hint = _context_hint(context, user_text, drive_pull,
                                      environment,
                                      boost_effect=boost_effect,
                                      relationship_band=relationship_band)
        return (base + " " + _VOICE_BLOCK + mood_signal + tools + memories
                + episode_block + context_hint)

    @staticmethod
    def _initial_importance(cand, user_text: str) -> float:
        # `cand` is currently unused; retained so future kind-weighting
        # can key off the candidate without a signature change.
        lowered = str(user_text or "").lower()
        if ("remember" in lowered or "don't forget" in lowered
                or "note that" in lowered):
            return 0.8
        return 0.5

    def _store_candidates(self, user_text: str) -> None:
        seen = {m.text.strip().lower() for m in self.memory_store.all()}
        now = self._now()
        for cand in self.extractor.extract(user_text):
            key = cand.text.strip().lower()
            if key in seen:
                continue
            if self.memory_store.add(Memory(
                    text=cand.text, kind=cand.kind,
                    importance=self._initial_importance(cand, user_text),
                    created_ts=now, last_recalled_ts=now)):
                seen.add(key)

    # Slice 9: process_turn is only ever called from the CLI REPL loop.
    # Away-mode actuation lives in cli._run_away_mode and does not call
    # process_turn (no synthetic turn, no buffer write).
    def process_turn(self, text: str, now: float | None = None) -> str:
        now = float(self._now() if now is None else now)
        elapsed = now - self.state.last_seen if self.state.last_seen else 0.0

        # 1. Background drift toward baselines (applied THROUGH the state).
        self.state.apply_decay(elapsed, self.decay_cfg)

        # 1b. Rise drives on the same elapsed; store is source of truth.
        drives = self.drive_store.load()
        drives = update_rise(drives, elapsed,
                             self.drives_config.rise_per_hour)

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
        self.last_context = ctx
        # 3a. Observe the environment. Retained only; the classifier,
        # behavior, and prompt do not consume it yet (Phase 3+). A
        # failing probe must never break a turn, so fall back to empty.
        try:
            raw = self.environment_probe.observe()
        except Exception:
            raw = EMPTY_RAW
        self.last_environment = map_observation(raw,
                                                self.environment_config)
        event = self.classifier.classify(text, ctx)
        self.last_event = event

        # 4. Appraise and apply -- the single state-write path for events.
        intent = self.appraisal.appraise(event, self.personality)
        self.state.apply_deltas(
            intent.deltas,
            {"event": event.type, "reason": intent.reason, "ts": now})

        # 4b. Satisfy drives from the classified event; persist rise and
        # satisfaction together in one atomic write.
        drives = apply_satisfaction(
            drives,
            event.type,
            self.drives_config.presence_satisfaction,
            self.drives_config.event_satisfaction,
        )
        self.drive_store.save(drives)

        # 4c. Evolve relationship state from the classified event.
        updated_relationship = apply_event(
            self.last_relationship,
            event.type,
            event.intensity,
            now,
            self.relationship_config,
        )
        if updated_relationship is not self.last_relationship:
            self.relationship_store.save(updated_relationship)
            self.last_relationship = updated_relationship
        self.last_relationship_band = (
            "high" if self.last_relationship.closeness >= 0.7 else None
        )

        # 5. Derived mood + behavior (read-only over state).
        self.state.set_mood(mood_mod.derive(self.state.emotions,
                                            self.state.mood))
        behavior = self.selector.pick(self.state, self.personality,
                                      ctx, user_text=str(text))
        self.last_behavior = behavior

        # 5b. Recall long-term memories (decay persisted, hits refreshed).
        # Boosted by the live environment app noun; the flag records
        # whether the boost changed the top-k (consumed in Phase 3).
        boost_text = self.last_environment.app_noun or ""
        boost_weight = 0.5 + self.last_relationship.closeness
        recalled, boost_effect = self.memory_store.recall_and_refresh(
            current_user_text=str(text),
            recent_topics=ctx.recent_topics,
            now=self._now(),
            boost_text=boost_text,
            boost_weight=boost_weight)
        self.last_recalled = recalled
        self.last_boost_effect = boost_effect
        # 5c. Extract candidate facts from the live turn only.
        self._store_candidates(str(text))

        # 5d. Per-turn drive pull, computed once for Phase 4/5 consumers.
        self.last_drive_pull = drive_pull(drives, self.drives_config)

        # 5e. Episodic memory: derive (join-or-create) and retrieve
        # prior context.
        turn_tokens = tokenize(str(text))
        episodes_before = self.episode_store.load()
        episodes_after = derive(
            episodes_before,
            turn_tokens,
            now,
            mood_label=self.state.mood,
            config=self.episodes_config,
            last_behavior=(self.last_behavior.name
                           if self.last_behavior else None),
        )
        if episodes_after != episodes_before:
            self.episode_store.save(episodes_after)

        # Retrieval surfaces prior context, not the current turn's
        # thread. If the turn joined the latest episode, that episode
        # is the ongoing thread and is already covered by the
        # conversation buffer; it is not "Previously:" content.
        # Identify the joined index by element-wise comparison — the
        # joined episode differs from its before counterpart; all
        # others are equal.
        candidates = episodes_before
        if episodes_before and len(episodes_after) == len(episodes_before):
            for i, (b, a) in enumerate(zip(episodes_before,
                                           episodes_after)):
                if a != b:
                    candidates = (episodes_before[:i]
                                  + episodes_before[i + 1:])
                    break

        self.last_episodes = recall(
            candidates,
            turn_tokens,
            ctx.recent_topics,
            config=self.episodes_config,
        )

        # 6. LLM renders wording, with at most one tool round-trip.
        # The second reply (if any) is final and never parsed for tools.
        system = self._system_prompt(behavior, ctx, str(text), recalled,
                                     drive_pull=self.last_drive_pull,
                                     environment=self.last_environment,
                                     boost_effect=self.last_boost_effect,
                                     relationship_band=(
                                         self.last_relationship_band),
                                     episodes=self.last_episodes,
                                     mood=self.state.mood)
        history = self.buffer.history_for_llm()
        user_str = str(text)
        reply = self.llm.generate(system=system, history=history,
                                  user=user_str)
        request = parse_tool_request(reply)
        if request is not None:
            decision = decide(request, self.tools_config,
                               self.session_grants,
                               persistent_grants=self.permission_state.grants)
            if decision.allowed:
                result = execute(request, self.tools_config)
                tool_result_text = result.output if result.ok \
                    else f"(error: {result.error})"
                allowed = True
            else:
                tool_result_text = \
                    f"(not permitted: {decision.reason})"
                allowed = False
            self.last_tool_calls.append(
                ToolCall(
                    tool=request.name,
                    args_preview=_preview_args(request.args),
                    ok=(allowed and result.ok if allowed else False),
                    allowed=allowed,
                    ts=self._now(),
                )
            )
            extended_history = history + [
                {"role": "carrot", "content": reply},
                {"role": "user",
                 "content": f"[Tool {request.name} result: "
                            f"{tool_result_text}]"},
            ]
            reply = self.llm.generate(system=system,
                                      history=extended_history,
                                      user=user_str)
            # Second reply is final -- never parsed for tool requests.

        self.buffer.add("user", str(text))
        self.buffer.add("carrot", reply)
        self.state.set_last_seen(now)
        return reply
