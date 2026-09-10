"""Slice 7 Phase-4 tests: the step-6 tool loop (custom generate lambdas
only; MockLLM itself is never modified)."""
import ast
import json

from carrot.tools import ToolsConfig
from tests.conftest import ROOT
from tests.test_scenarios import make_carrot


def _tool_carrot(tmp_path, replies, max_bytes=4096):
    c = make_carrot()
    c.tools_config = ToolsConfig(
        enabled={"read_file": True, "open_url": True},
        read_file_dirs=(str(tmp_path),),
        open_url_domains=("example.com",),
        max_file_bytes=max_bytes, max_iterations=2)
    calls = {"n": 0, "histories": [], "users": []}

    def generate(*, system, history, user):
        calls["n"] += 1
        calls["histories"].append(list(history))
        calls["users"].append(user)
        return replies[min(calls["n"] - 1, len(replies) - 1)]

    c.llm.generate = generate
    c.calls = calls
    return c


def _request(path):
    return "```tool:read_file\n" + json.dumps({"path": path}) + "\n```"


def test_no_tool_request_single_call(tmp_path):
    c = _tool_carrot(tmp_path, ["just chatting."])
    c.process_turn("hello there")
    assert c.calls["n"] == 1
    assert [t.text for t in c.buffer.recent()] == \
        ["hello there", "just chatting."]
    assert len(c.last_tool_calls) == 0


def test_granted_tool_two_calls_final_text(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("file contents here", encoding="utf-8")
    c = _tool_carrot(tmp_path, [_request(str(target)),
                                "Here's what I found."])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    assert c.calls["n"] == 2
    assert [t.text for t in c.buffer.recent()] == \
        ["please read the file", "Here's what I found."]
    assert len(c.last_tool_calls) == 1
    entry = c.last_tool_calls[0]
    assert (entry.tool, entry.allowed, entry.ok) == \
        ("read_file", True, True)


def test_denied_tool_two_calls_denial_in_transcript(tmp_path):
    req = _request(str(tmp_path / "a.txt"))
    c = _tool_carrot(tmp_path, [req, "I can't do that."])
    c.process_turn("please read the file")
    assert c.calls["n"] == 2
    assert len(c.last_tool_calls) == 1
    assert c.last_tool_calls[0].allowed is False
    assert "(not permitted: not permitted)" in c.calls["histories"][1][-1][
        "content"]
    assert [t.text for t in c.buffer.recent()] == \
        ["please read the file", "I can't do that."]


def test_second_call_tool_request_ignored(tmp_path):
    req = _request(str(tmp_path / "a.txt"))
    c = _tool_carrot(tmp_path, [req, req])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    assert c.calls["n"] == 2
    assert [t.text for t in c.buffer.recent()] == \
        ["please read the file", req]
    assert len(c.last_tool_calls) == 1


def test_tool_failure_logged_ok_false(tmp_path):
    c = _tool_carrot(tmp_path, [_request(str(tmp_path / "missing.txt")),
                                "Sorry."])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    assert c.last_tool_calls[0].ok is False
    assert [t.text for t in c.buffer.recent()] == \
        ["please read the file", "Sorry."]


def test_second_call_history_shape(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("file contents here", encoding="utf-8")
    req = _request(str(target))
    c = _tool_carrot(tmp_path, [req, "done."])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    assert c.calls["histories"][1] == c.calls["histories"][0] + [
        {"role": "carrot", "content": req},
        {"role": "user",
         "content": "[Tool read_file result: file contents here]"},
    ]


def test_second_call_same_user_arg(tmp_path):
    c = _tool_carrot(tmp_path, [_request(str(tmp_path / "a.txt")), "done."])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    assert c.calls["users"] == ["please read the file"] * 2


def test_buffer_never_contains_tool_block(tmp_path):
    req = _request(str(tmp_path / "a.txt"))
    c = _tool_carrot(tmp_path, [req, "done."])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    recent = list(c.buffer.recent())
    assert [t.text for t in recent] == ["please read the file", "done."]
    assert all(t.role in ("user", "carrot") for t in recent)


def test_buffer_unchanged_for_plain_turns(tmp_path):
    c = _tool_carrot(tmp_path, ["just chatting."])
    c.process_turn("hello there")
    recent = list(c.buffer.recent())
    assert [(t.role, t.text) for t in recent] == \
        [("user", "hello there"), ("carrot", "just chatting.")]


def test_last_tool_calls_bounded_at_20(tmp_path):
    c = _tool_carrot(tmp_path, ["x"])
    c.session_grants["read_file"] = True
    state = {"t": 0}

    def generate(*, system, history, user):
        i = state["t"]
        state["t"] += 1
        if i % 2 == 0:
            # Short synthetic paths: logging happens whether the gate
            # allows or denies, and short paths survive preview intact.
            return _request(f"/f{i // 2}.txt")
        return "done."

    c.llm.generate = generate
    for _ in range(25):
        c.process_turn("read it")
    assert len(c.last_tool_calls) == 20
    previews = [e.args_preview for e in c.last_tool_calls]
    assert any("f24.txt" in p for p in previews)
    assert not any("f0.txt" in p for p in previews)


def test_last_tool_calls_persist_across_turns(tmp_path):
    req = _request(str(tmp_path / "a.txt"))
    c = _tool_carrot(tmp_path, [req, "done."] * 3)
    c.session_grants["read_file"] = True
    for _ in range(3):
        c.process_turn("read it")
    assert len(c.last_tool_calls) == 3


def test_last_tool_calls_not_persisted(tmp_path):
    assert len(make_carrot().last_tool_calls) == 0


def test_args_preview_truncated(tmp_path):
    long_path = "x" * 200
    req = "```tool:read_file\n" + json.dumps({"path": long_path}) + "\n```"
    c = _tool_carrot(tmp_path, [req, "done."])
    c.process_turn("read it")
    preview = c.last_tool_calls[0].args_preview
    assert len(preview) <= 81
    assert preview.endswith("…")


def test_tool_result_never_enters_state(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("neutral file text", encoding="utf-8")
    granted = _tool_carrot(tmp_path, [_request(str(target)), "done."])
    granted.session_grants["read_file"] = True
    plain = _tool_carrot(tmp_path, ["done."])
    granted.process_turn("hello there")
    plain.process_turn("hello there")
    assert granted.state.to_dict() == plain.state.to_dict()


def test_denied_tool_never_enters_state(tmp_path):
    granted = _tool_carrot(tmp_path, ["done."])
    plain = _tool_carrot(tmp_path, ["done."])
    granted.process_turn("hello there")
    plain.process_turn("hello there")
    assert granted.state.to_dict() == plain.state.to_dict()


def test_tool_result_never_enters_memory(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("my name is Xena", encoding="utf-8")
    c = _tool_carrot(tmp_path, [_request(str(target)), "done."])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    assert all("Xena" not in m.text for m in c.memory_store.all())


def test_tool_result_never_enters_prompt(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("my name is Xena", encoding="utf-8")
    c = _tool_carrot(tmp_path, [_request(str(target)), "done."])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    c.process_turn("hello")
    assert "Xena" not in c.llm.last_system


def test_tool_loop_deterministic(tmp_path):
    first = _tool_carrot(tmp_path, [_request(str(tmp_path / "a.txt")),
                                    "done."])
    second = _tool_carrot(tmp_path, [_request(str(tmp_path / "a.txt")),
                                     "done."])
    for carrot in (first, second):
        carrot.session_grants["read_file"] = True
        carrot.process_turn("please read the file")
    assert list(first.last_tool_calls) == list(second.last_tool_calls)
    assert first.state.to_dict() == second.state.to_dict()


def test_execute_called_only_from_process_turn():
    callers = set()
    for path in (ROOT / "carrot").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "execute":
                    callers.add(path.name)
    assert callers == {"carrot.py"}


def test_session_grants_unaffected_by_tool_turn(tmp_path):
    c = _tool_carrot(tmp_path, [_request(str(tmp_path / "a.txt")), "done."])
    c.session_grants["read_file"] = True
    c.process_turn("please read the file")
    assert c.session_grants == {"read_file": True}
