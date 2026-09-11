"""Slice 15 Phase-1 tests: social adapter, config, status view."""
import pytest

from carrot.channel import FileChannel, MockChannel, _FILENAME_RE
from carrot.cli import drain_inbox
from carrot.drives import DriveStore
from carrot.initiative import InitiativeStore
from carrot.memory import MemoryStore
from carrot.permissions import PermissionStore
from carrot.relationship import RelationshipStore
from carrot.social import (
    PLATFORMS,
    MockSocialChannel,
    SocialChannel,
    SocialConfig,
    social_status,
)
from tests.conftest import ROOT
from tests.test_scenarios import make_carrot


def _cfg():
    return SocialConfig.load(str(ROOT / "config" / "social.json"))


# ---- constants ----


def test_platforms():
    assert PLATFORMS == ("instagram",)


# ---- SocialConfig.load ----


def test_config_loads_valid():
    cfg = _cfg()
    assert cfg.platforms["instagram"]["handle"] == ""
    assert cfg.platforms["instagram"]["enabled"] is True
    assert cfg.platforms["instagram"]["inbox_dir"] == \
        "state/social/instagram"


def test_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        SocialConfig.load(str(tmp_path / "nope.json"))


def test_config_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        SocialConfig.load(str(path))


def test_config_missing_platforms(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    assert dict(SocialConfig.load(str(path)).platforms) == {}


def test_config_non_dict_platforms(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"platforms": ["instagram"]}', encoding="utf-8")
    with pytest.raises(ValueError):
        SocialConfig.load(str(path))


def test_config_non_mapping_inner(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"platforms": {"instagram": "not-a-dict"}}',
                    encoding="utf-8")
    with pytest.raises(ValueError):
        SocialConfig.load(str(path))


def test_config_frozen():
    with pytest.raises(Exception):
        _cfg().platforms = {}


def test_config_inner_immutable():
    with pytest.raises(TypeError):
        _cfg().platforms["instagram"]["handle"] = "x"


# ---- SocialChannel ----


def test_social_channel_creates_dirs(tmp_path):
    chan = SocialChannel("instagram", tmp_path / "ig")
    for sub in ("inbox", "inbox/done", "inbox/bad", "outbox"):
        assert (tmp_path / "ig" / sub).is_dir()
    assert chan.platform == "instagram"


def test_social_channel_receive_delegation(tmp_path):
    chan = SocialChannel("instagram", tmp_path / "ig")
    (chan.inbox / "000001-aaaa.msg").write_text("hello", encoding="utf-8")
    assert chan.receive() == "hello"
    assert (chan.done / "000001-aaaa.msg").is_file()


def test_social_channel_send_delegation(tmp_path):
    chan = SocialChannel("instagram", tmp_path / "ig")
    chan.send("hello")
    files = list(chan.outbox.glob("*.msg"))
    assert len(files) == 1
    assert _FILENAME_RE.match(files[0].name)


def test_social_channel_is_file_channel(tmp_path):
    assert isinstance(SocialChannel("instagram", tmp_path / "ig"),
                      FileChannel)


# ---- MockSocialChannel ----


def test_mock_platform_default():
    assert MockSocialChannel().platform == "instagram"


def test_mock_receive_send_inherited():
    chan = MockSocialChannel("instagram", ["a"])
    assert chan.receive() == "a"
    chan.send("b")
    assert chan.sent == ["b"]


def test_mock_is_mock_channel():
    assert isinstance(MockSocialChannel(), MockChannel)


# ---- social_status ----


def test_status_empty_config(tmp_path):
    assert social_status(SocialConfig(platforms={}),
                         tmp_path) == "(no social platforms configured)"


def test_status_single_empty(tmp_path):
    cfg = SocialConfig(platforms={"instagram": {"enabled": True,
                                                "inbox_dir": "soc/ig"}})
    assert social_status(cfg, tmp_path) == \
        "instagram: enabled pending=0 sent=0"


def test_status_counts_files(tmp_path):
    root = tmp_path
    inbox = root / "soc" / "ig" / "inbox"
    outbox = root / "soc" / "ig" / "outbox"
    inbox.mkdir(parents=True)
    outbox.mkdir(parents=True)
    (inbox / "000001-aa.msg").write_text("a", encoding="utf-8")
    (inbox / "000002-bb.msg").write_text("b", encoding="utf-8")
    (outbox / "000001-cc.msg").write_text("c", encoding="utf-8")
    cfg = SocialConfig(platforms={"instagram": {"enabled": True,
                                                "inbox_dir": "soc/ig"}})
    assert social_status(cfg, root) == \
        "instagram: enabled pending=2 sent=1"


def test_status_disabled_platform(tmp_path):
    cfg = SocialConfig(platforms={"instagram": {"enabled": False,
                                                "inbox_dir": "soc/ig"}})
    assert social_status(cfg, tmp_path).startswith("instagram: disabled ")


def test_status_missing_inbox_dir_defaults(tmp_path):
    cfg = SocialConfig(platforms={"instagram": {"enabled": True}})
    assert social_status(cfg, tmp_path) == \
        "instagram: enabled pending=0 sent=0"


def test_status_absolute_inbox_dir(tmp_path):
    absdir = tmp_path / "elsewhere"
    (absdir / "inbox").mkdir(parents=True)
    (absdir / "inbox" / "000001-aa.msg").write_text("a", encoding="utf-8")
    cfg = SocialConfig(platforms={"instagram": {"enabled": True,
                                                "inbox_dir": str(absdir)}})
    assert social_status(cfg, tmp_path / "other-root") == \
        "instagram: enabled pending=1 sent=0"


def test_status_deterministic(tmp_path):
    cfg = SocialConfig(platforms={"instagram": {"enabled": True,
                                                "inbox_dir": "soc/ig"}})
    assert social_status(cfg, tmp_path) == social_status(cfg, tmp_path)


# --- Slice 15 Phase 2: no-separate-bot identity proof ---


def test_social_path_answers_identically_to_cli(tmp_path):
    # One core, two doors: shared store files + same clock origin, the
    # same input through process_turn (CLI path) and through drain_inbox
    # over a social channel must produce the identical reply.
    stores = dict(
        memory_store=MemoryStore(tmp_path / "m.json"),
        drive_store=DriveStore(tmp_path / "d.json"),
        permission_store=PermissionStore(tmp_path / "p.json"),
        initiative_store=InitiativeStore(tmp_path / "i.json"),
        relationship_store=RelationshipStore(tmp_path / "r.json"),
    )
    cli_carrot = make_carrot(**stores)
    soc_carrot = make_carrot(**stores)
    from carrot.state_store import StateStore

    soc_carrot.store = StateStore(str(tmp_path / "state.json"))
    cli_reply = cli_carrot.process_turn("hello there")
    chan = MockSocialChannel("instagram", ["hello there"])
    assert drain_inbox(soc_carrot, chan) == 1
    assert chan.sent == [cli_reply]
