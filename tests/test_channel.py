"""Slice 14 Phase-1 tests: channel protocols, mocks, file adapter."""
import re

from carrot.channel import (
    CHANNEL_EMPTY_MSG,
    MAX_BODY_BYTES,
    MAX_DRAIN,
    MAX_OUTBOX_FILES,
    FileChannel,
    MockChannel,
    _FILENAME_RE,
)
from carrot.cli import drain_inbox
from tests.test_scenarios import make_carrot


# ---- constants ----


def test_channel_empty_msg():
    assert CHANNEL_EMPTY_MSG == "(no messages)"


def test_channel_caps_positive_ints():
    assert isinstance(MAX_DRAIN, int) and MAX_DRAIN > 0
    assert isinstance(MAX_BODY_BYTES, int) and MAX_BODY_BYTES > 0
    assert isinstance(MAX_OUTBOX_FILES, int) and MAX_OUTBOX_FILES > 0


# ---- MockChannel ----


def test_mock_empty_queue():
    assert MockChannel().receive() == ""


def test_mock_scripted_sequence():
    chan = MockChannel(["a", "b"])
    assert chan.receive() == "a"
    assert chan.receive() == "b"
    assert chan.receive() == ""


def test_mock_receive_count():
    chan = MockChannel(["a"])
    chan.receive()
    chan.receive()
    chan.receive()
    assert chan.receive_count == 3


def test_mock_send_records():
    chan = MockChannel()
    chan.send("hi")
    assert chan.sent == ["hi"]


def test_mock_send_none_coerced():
    chan = MockChannel()
    chan.send(None)
    assert chan.sent == [""]


# ---- FileChannel: construction ----


def test_file_channel_creates_directories(tmp_path):
    chan = FileChannel(tmp_path / "ch")
    for sub in ("inbox", "inbox/done", "inbox/bad", "outbox"):
        assert (tmp_path / "ch" / sub).is_dir()
    assert isinstance(chan, FileChannel)


def test_file_channel_construction_idempotent(tmp_path):
    FileChannel(tmp_path / "ch")
    FileChannel(tmp_path / "ch")
    assert (tmp_path / "ch" / "inbox").is_dir()


# ---- FileChannel.receive: happy path ----


def test_receive_empty_inbox(tmp_path):
    assert FileChannel(tmp_path).receive() == ""


def _inbox_write(inbox, name, data):
    path = inbox / name
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def test_receive_single_message(tmp_path):
    chan = FileChannel(tmp_path)
    _inbox_write(chan.inbox, "000001-aabbccdd.msg", "hello")
    assert chan.receive() == "hello"
    assert (chan.done / "000001-aabbccdd.msg").is_file()
    assert not (chan.inbox / "000001-aabbccdd.msg").exists()


def test_receive_fifo_order(tmp_path):
    chan = FileChannel(tmp_path)
    for n, word in ((3, "three"), (1, "one"), (2, "two")):
        _inbox_write(chan.inbox, f"{n:06d}-aa.msg", word)
    assert chan.receive() == "one"
    assert chan.receive() == "two"
    assert chan.receive() == "three"
    assert chan.receive() == ""


def test_receive_utf8_preserved(tmp_path):
    chan = FileChannel(tmp_path)
    _inbox_write(chan.inbox, "000001-aa.msg", "héllo 😀")
    assert chan.receive() == "héllo 😀"


def test_receive_interleaved_done_bad(tmp_path):
    chan = FileChannel(tmp_path)
    _inbox_write(chan.inbox, "000001-aa.msg", "good")
    _inbox_write(chan.inbox, "ZZZZ-ff.msg", "bad-name")
    assert chan.receive() == "good"
    assert (chan.done / "000001-aa.msg").is_file()
    assert chan.receive() == ""
    assert (chan.bad / "ZZZZ-ff.msg").is_file()


# ---- FileChannel.receive: validation ----


def test_receive_ignores_non_msg(tmp_path):
    chan = FileChannel(tmp_path)
    _inbox_write(chan.inbox, "not-valid.txt", "x")
    _inbox_write(chan.inbox, "abc.txt", "y")
    assert chan.receive() == ""
    assert (chan.inbox / "not-valid.txt").is_file()
    assert (chan.inbox / "abc.txt").is_file()


def test_receive_bad_pattern_to_bad(tmp_path):
    chan = FileChannel(tmp_path)
    _inbox_write(chan.inbox, "AAAA-ffff.msg", "upper")
    _inbox_write(chan.inbox, "000002-aa.msg", "good")
    assert chan.receive() == "good"
    assert chan.receive() == ""
    assert (chan.bad / "AAAA-ffff.msg").is_file()


def test_receive_oversized_to_bad(tmp_path):
    chan = FileChannel(tmp_path)
    _inbox_write(chan.inbox, "000001-aabb.msg", "x" * 5000)
    assert chan.receive() == ""
    assert (chan.bad / "000001-aabb.msg").is_file()


def test_receive_non_utf8_to_bad(tmp_path):
    chan = FileChannel(tmp_path)
    _inbox_write(chan.inbox, "000001-aabb.msg", b"\xff\xfe")
    assert chan.receive() == ""
    assert (chan.bad / "000001-aabb.msg").is_file()


def test_malformed_does_not_stop_drain(tmp_path):
    chan = FileChannel(tmp_path)
    _inbox_write(chan.inbox, "000001-aa.msg", "x" * 5000)
    _inbox_write(chan.inbox, "000002-bb.msg", "good")
    _inbox_write(chan.inbox, "000003-cc.msg", b"\xff")
    assert chan.receive() == "good"
    assert (chan.bad / "000001-aa.msg").is_file()
    assert chan.receive() == ""
    assert (chan.bad / "000003-cc.msg").is_file()


# ---- FileChannel.send ----


def test_send_single_file(tmp_path):
    chan = FileChannel(tmp_path)
    chan.send("reply")
    files = list(chan.outbox.glob("*.msg"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8").rstrip("\n") == "reply"
    assert _FILENAME_RE.match(files[0].name)


def test_send_sequence_advances(tmp_path):
    chan = FileChannel(tmp_path)
    chan.send("one")
    chan.send("two")
    names = sorted(p.name for p in chan.outbox.glob("*.msg"))
    assert len(names) == 2
    assert names[0] < names[1]
    assert chan.receive() == ""  # outbox is not an inbox


def test_send_utf8_preserved(tmp_path):
    chan = FileChannel(tmp_path)
    chan.send("héllo 😀")
    files = list(chan.outbox.glob("*.msg"))
    assert files[0].read_text(encoding="utf-8").rstrip("\n") == "héllo 😀"


def test_send_none_coerced(tmp_path):
    chan = FileChannel(tmp_path)
    chan.send(None)
    files = list(chan.outbox.glob("*.msg"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8").rstrip("\n") == ""


def test_send_oversize_truncated(tmp_path):
    # Truncate (not raise): the transport must never lose a reply to a
    # size cap; MAX_BODY_BYTES bounds disk use instead.
    chan = FileChannel(tmp_path)
    chan.send("x" * 5000)
    files = list(chan.outbox.glob("*.msg"))
    assert len(files) == 1
    body = files[0].read_text(encoding="utf-8").rstrip("\n")
    assert len(body.encode("utf-8")) <= MAX_BODY_BYTES


def test_send_multibyte_truncation_stays_valid_utf8(tmp_path):
    chan = FileChannel(tmp_path)
    chan.send("é" * 5000)  # 2 bytes each; cap splits a character
    files = list(chan.outbox.glob("*.msg"))
    raw = files[0].read_text(encoding="utf-8").rstrip("\n").encode("utf-8")
    assert len(raw) <= MAX_BODY_BYTES
    assert raw.decode("utf-8")


def test_send_prune_at_cap(tmp_path):
    chan = FileChannel(tmp_path)
    for i in range(105):
        chan.send(f"msg-{i}")
    files = sorted(chan.outbox.glob("*.msg"))
    assert len(files) == MAX_OUTBOX_FILES == 100
    first_seq = int(files[0].name.split("-", 1)[0])
    last_seq = int(files[-1].name.split("-", 1)[0])
    assert (first_seq, last_seq) == (6, 105)


# ---- determinism ----


def test_file_channel_deterministic(tmp_path):
    first = FileChannel(tmp_path / "a")
    second = FileChannel(tmp_path / "b")
    for chan in (first, second):
        _inbox_write(chan.inbox, "000001-aa.msg", "hello")
        chan.send("reply")
    assert first.receive() == second.receive() == "hello"
    out_first = sorted(p.read_bytes() for p in first.outbox.glob("*.msg"))
    out_second = sorted(p.read_bytes() for p in second.outbox.glob("*.msg"))
    assert out_first == out_second
    assert re.search(r"\d", first.receive()) is None


# --- Slice 14 Phase 2: drain_inbox tests ---


def _store_carrot(tmp_path, name="state.json"):
    from carrot.state_store import StateStore

    c = make_carrot()
    c.store = StateStore(str(tmp_path / name))
    return c


def test_drain_empty_channel():
    c = make_carrot()
    assert drain_inbox(c, MockChannel([])) == 0
    assert len(c.buffer) == 0


def test_drain_single_message(tmp_path):
    c = _store_carrot(tmp_path)
    assert drain_inbox(c, MockChannel(["hello"])) == 1
    texts = [t.text for t in c.buffer.recent()]
    assert "hello" in texts
    assert len(texts) == 2  # user turn + carrot reply


def test_drain_reply_sent(tmp_path):
    c = _store_carrot(tmp_path)
    chan = MockChannel(["hello"])
    assert drain_inbox(c, chan) == 1
    assert len(chan.sent) == 1
    assert isinstance(chan.sent[0], str) and chan.sent[0]


def test_drain_multiple_in_order(tmp_path):
    c = _store_carrot(tmp_path)
    assert drain_inbox(c, MockChannel(["a", "b", "c"])) == 3
    roles = [t.role for t in c.buffer.recent(6)]
    assert roles == ["user", "carrot"] * 3


def test_drain_bounded_at_max(tmp_path):
    c = _store_carrot(tmp_path)
    chan = MockChannel(["x"] * 150)
    assert drain_inbox(c, chan, max_messages=5) == 5
    assert len(c.buffer.recent(100)) == 10


def test_drain_whitespace_stops():
    c = make_carrot()
    assert drain_inbox(c, MockChannel([" "])) == 0
    assert len(c.buffer) == 0


def test_drain_whitespace_after_valid_stops(tmp_path):
    c = _store_carrot(tmp_path)
    assert drain_inbox(c, MockChannel(["a", " ", "b"])) == 1
    texts = [t.text for t in c.buffer.recent()]
    assert "b" not in texts


def test_drain_send_failure_swallowed(tmp_path):
    class LoudChannel(MockChannel):
        def send(self, text):
            raise RuntimeError("outbox unwritable")

    c = _store_carrot(tmp_path)
    assert drain_inbox(c, LoudChannel(["hello"])) == 1
    assert len(c.buffer.recent()) == 2


def test_drain_saves_state_per_turn(tmp_path):
    path = tmp_path / "state.json"
    c = _store_carrot(tmp_path)
    assert not path.exists()
    assert drain_inbox(c, MockChannel(["hello"])) == 1
    assert path.exists()


def test_drain_deterministic(tmp_path):
    first = _store_carrot(tmp_path, "a.json")
    second = _store_carrot(tmp_path, "b.json")
    one = drain_inbox(first, MockChannel(["hello", "how are you"]))
    two = drain_inbox(second, MockChannel(["hello", "how are you"]))
    assert one == two == 2
    assert first.state.to_dict() == second.state.to_dict()
    assert [t.text for t in first.buffer.recent()] == \
        [t.text for t in second.buffer.recent()]
