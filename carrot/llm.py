"""LLM provider abstraction. Core never imports a provider SDK.

Providers: MockLLM (tests/dev, deterministic), OpenAICompatLLM
(cloud, env-configured), OllamaLLM (local, env-configured).
HTTP clients use stdlib urllib only. Switching providers changes no
core code -- only which object is handed to the orchestrator.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol

MOCK_REPLIES = {
    "normal": ["hey.", "yeah?", "got it.", "mm, okay."],
    "playful": ["WAIT, seriously? \U0001F602", "haha, nice \U0001F602",
                "okay that's funny \U0001F602"],
    "excited": ["NO WAY \U0001F62D", "that's awesome!!", "omg really?!"],
    "sad": ["oh. okay.", "\u2026alright.", "yeah. fine."],
    "annoyed": ["seriously?", "ugh, stop.", "what was that for?"],
    "sulking": ["Fine.", "Whatever.",
                "\u2026I'm still annoyed, by the way."],
    "quiet": ["\u2026", "mm.", "yeah."],
    "shy": ["oh\u2026 um, thanks \U0001F642", "hehe\u2026 stop \U0001F642"],
    "affectionate": ["that means a lot \U0001F642", "aww \U0001F642"],
    "embarrassed": ["okay, that was embarrassing \U0001F605",
                    "um. moving on \U0001F605"],
    "curious": ["wait, what?", "what happened?", "hmm?"],
    "proud": ["told you \U0001F60C", "we did that \U0001F642"],
    "frustrated": ["ugh. fine.", "this is getting annoying."],
}


class LLMProvider(Protocol):
    # Signature frozen. Slice 7 tool requests are parsed from the text
    # return value by a free function; generate() is not extended.
    def generate(self, *, system: str, history: list, user: str) -> str:
        ...


class MockLLM:
    """Deterministic behavior-aware stand-in. Records last call."""

    def __init__(self, behavior_getter=None):
        self._counter = 0
        self._behavior_getter = behavior_getter
        self.last_system = ""
        self.last_history: list = []
        self.last_user = ""

    def generate(self, *, system: str, history: list, user: str) -> str:
        self.last_system = system
        self.last_history = list(history)
        self.last_user = user
        behavior = "normal"
        if self._behavior_getter is not None:
            try:
                behavior = self._behavior_getter().name
            except Exception:
                behavior = "normal"
        options = MOCK_REPLIES.get(behavior, MOCK_REPLIES["normal"])
        reply = options[self._counter % len(options)]
        self._counter += 1
        return reply


def _post_json(url: str, payload: dict, headers: dict,
               timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {**headers, "User-Agent": "Carrot/1.0"}
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OpenAICompatLLM:
    """Cloud provider via an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self._model = model

    def generate(self, *, system: str, history: list, user: str) -> str:
        messages = [{"role": "system", "content": system}]
        for turn in history[-10:]:
            role = "assistant" if turn.get("role") == "carrot" else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user})
        try:
            data = _post_json(
                self._url,
                {"model": self._model, "messages": messages},
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {self._key}"})
        except Exception as exc:
            raise RuntimeError(f"cloud LLM request failed: {exc}") from exc
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("cloud LLM returned an unexpected shape") from exc


class OllamaLLM:
    """Local provider via Ollama's /api/chat endpoint."""

    def __init__(self, base_url: str, model: str):
        self._url = base_url.rstrip("/") + "/api/chat"
        self._model = model

    def generate(self, *, system: str, history: list, user: str) -> str:
        messages = [{"role": "system", "content": system}]
        for turn in history[-10:]:
            role = "assistant" if turn.get("role") == "carrot" else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user})
        try:
            data = _post_json(
                self._url,
                {"model": self._model, "messages": messages, "stream": False},
                {"Content-Type": "application/json"})
        except Exception as exc:
            raise RuntimeError(f"ollama request failed: {exc}") from exc
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("ollama returned an unexpected shape") from exc


def make_provider(name: str = "", env=None) -> LLMProvider:
    """Select a provider by name. Defaults to the deterministic mock."""
    env = env if env is not None else os.environ
    kind = (name or env.get("CARROT_LLM_PROVIDER", "mock")).lower()
    if kind == "openai":
        return OpenAICompatLLM(
            env.get("CARROT_LLM_BASE_URL", "https://api.openai.com/v1"),
            env.get("CARROT_LLM_API_KEY", ""),
            env.get("CARROT_LLM_MODEL", "gpt-4o-mini"))
    if kind == "ollama":
        return OllamaLLM(env.get("CARROT_OLLAMA_URL", "http://localhost:11434"),
                         env.get("CARROT_OLLAMA_MODEL", "llama3.1"))
    return MockLLM()
