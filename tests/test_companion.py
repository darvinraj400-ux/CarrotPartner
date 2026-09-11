"""Slice 16 Phase-1 tests: companion value type, names, paths."""
import json

import pytest

from carrot.cli import format_companions
from carrot.companion import (
    DEFAULT_COMPANION,
    Companion,
    companion_paths,
    list_companions,
    validate_name,
)
from carrot.paths import _paths
from tests.test_scenarios import make_companion


# ---- validate_name ----


def test_valid_names_returned_unchanged():
    for good in ("carrot", "ada", "ada-2", "a", "x" * 32, "a-b-c-1",
                 "0", "123"):
        assert validate_name(good) == good


def test_empty_name_rejected():
    with pytest.raises(ValueError):
        validate_name("")


def test_too_long_name_rejected():
    with pytest.raises(ValueError):
        validate_name("x" * 33)


def test_uppercase_rejected():
    with pytest.raises(ValueError):
        validate_name("Ada")


def test_underscore_rejected():
    with pytest.raises(ValueError):
        validate_name("ada_x")


def test_traversal_rejected():
    with pytest.raises(ValueError):
        validate_name("../x")


def test_absolute_rejected():
    with pytest.raises(ValueError):
        validate_name("/abs")


def test_slash_rejected():
    with pytest.raises(ValueError):
        validate_name("ada/sub")


def test_whitespace_rejected():
    for bad in (" ada", "ada ", " "):
        with pytest.raises(ValueError):
            validate_name(bad)


def test_non_string_rejected():
    for bad in (None, 123, b"carrot"):
        with pytest.raises(ValueError):
            validate_name(bad)


def test_returns_input():
    assert validate_name("carrot") == "carrot"
    assert DEFAULT_COMPANION == "carrot"


# ---- companion_paths ----


def test_default_identical_to_base(tmp_path):
    assert companion_paths(tmp_path, "carrot") == _paths(tmp_path)


def test_default_state_json_flat(tmp_path):
    assert companion_paths(tmp_path, "carrot")["state_json"] == \
        tmp_path / "state" / "state.json"


def test_non_default_state_json_namespaced(tmp_path):
    assert companion_paths(tmp_path, "ada")["state_json"] == \
        tmp_path / "state" / "ada" / "state.json"


def test_permissions_stays_global(tmp_path):
    assert companion_paths(tmp_path, "ada")["permissions_json"] == \
        tmp_path / "state" / "permissions.json"


def test_kind_configs_stay_global(tmp_path):
    named = companion_paths(tmp_path, "ada")
    base = companion_paths(tmp_path, "carrot")
    for key in ("appraisal_json", "behavior_json", "relationship_config_json"):
        assert named[key] == base[key]


def test_personality_routes_per_companion(tmp_path):
    assert companion_paths(tmp_path, "ada")["personality_json"] == \
        tmp_path / "config" / "companions" / "ada.json"
    assert companion_paths(tmp_path, "carrot")["personality_json"] == \
        tmp_path / "config" / "personality.json"


def test_relationship_routes_per_companion(tmp_path):
    assert companion_paths(tmp_path, "ada")["relationship_json"] == \
        tmp_path / "state" / "ada" / "relationship.json"


def test_initiative_routes_per_companion(tmp_path):
    assert companion_paths(tmp_path, "ada")["initiative_json"] == \
        tmp_path / "state" / "ada" / "initiative.json"


def test_social_dir_routes_per_companion(tmp_path):
    assert companion_paths(tmp_path, "ada")["social_dir"] == \
        tmp_path / "state" / "ada" / "social"


def test_channel_dir_routes_per_companion(tmp_path):
    assert companion_paths(tmp_path, "ada")["channel_dir"] == \
        tmp_path / "state" / "ada" / "channel"


def test_invalid_name_raises(tmp_path):
    with pytest.raises(ValueError):
        companion_paths(tmp_path, "Ada")


def test_paths_pure_no_writes(tmp_path):
    companion_paths(tmp_path, "ada")
    companion_paths(tmp_path, "carrot")
    assert list(tmp_path.iterdir()) == []


def test_paths_deterministic(tmp_path):
    assert companion_paths(tmp_path, "ada") == \
        companion_paths(tmp_path, "ada")


def test_paths_no_aliasing(tmp_path):
    assert companion_paths(tmp_path, "ada") is not \
        companion_paths(tmp_path, "carrot")


# ---- Companion dataclass ----


def test_companion_construction(tmp_path):
    c = Companion(name="carrot", personality_id="carrot-default-v1",
                  state_dir=tmp_path / "state",
                  config_dir=tmp_path / "config")
    assert c.name == "carrot"
    assert c.state_dir == tmp_path / "state"


def test_companion_frozen(tmp_path):
    c = Companion(name="carrot", personality_id="x",
                  state_dir=tmp_path / "state",
                  config_dir=tmp_path / "config")
    with pytest.raises(Exception):
        c.name = "ada"


def test_companion_invalid_name_raises(tmp_path):
    with pytest.raises(ValueError):
        Companion(name="Ada", personality_id="x",
                  state_dir=tmp_path / "state",
                  config_dir=tmp_path / "config")


def test_companion_equality(tmp_path):
    kwargs = dict(name="ada", personality_id="ada-v1",
                  state_dir=tmp_path / "state" / "ada",
                  config_dir=tmp_path / "config" / "companions")
    assert Companion(**kwargs) == Companion(**kwargs)


# ---- paths helper cleanup ----


def test_paths_requires_root():
    with pytest.raises(TypeError):
        _paths()


# --- Slice 16 Phase 2: list_companions + factory tests ---


def _stage_personality(root, name="personality.json", ident="carrot-default-v1"):
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / name).write_text(json.dumps({"id": ident}),
                                        encoding="utf-8")


def test_list_no_companions_dir(tmp_path):
    _stage_personality(tmp_path)
    found = list_companions(tmp_path)
    assert len(found) == 1
    assert found[0].name == "carrot"


def test_list_default_first(tmp_path):
    _stage_personality(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True)
    for name in ("zoe", "ada", "bob"):
        (companions / f"{name}.json").write_text(
            json.dumps({"id": f"{name}-v1"}), encoding="utf-8")
    found = list_companions(tmp_path)
    assert [c.name for c in found] == ["carrot", "ada", "bob", "zoe"]


def test_list_discovers_personality_id(tmp_path):
    _stage_personality(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True)
    (companions / "ada.json").write_text(json.dumps({"id": "ada-v1"}),
                                         encoding="utf-8")
    found = list_companions(tmp_path)
    assert [c.name for c in found] == ["carrot", "ada"]
    assert found[1].personality_id == "ada-v1"


def test_list_sorted_by_name(tmp_path):
    _stage_personality(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True)
    for name in ("zoe", "bob", "ada"):
        (companions / f"{name}.json").write_text(json.dumps({"id": name}),
                                                 encoding="utf-8")
    assert [c.name for c in list_companions(tmp_path)] == \
        ["carrot", "ada", "bob", "zoe"]


def test_list_skips_malformed(tmp_path):
    _stage_personality(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True)
    (companions / "broken.json").write_text("{not json", encoding="utf-8")
    (companions / "ada.json").write_text(json.dumps({"id": "ada-v1"}),
                                         encoding="utf-8")
    assert [c.name for c in list_companions(tmp_path)] == ["carrot", "ada"]


def test_list_skips_invalid_stem(tmp_path):
    _stage_personality(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True)
    (companions / "Ada.json").write_text(json.dumps({"id": "Ada"}),
                                         encoding="utf-8")
    assert [c.name for c in list_companions(tmp_path)] == ["carrot"]


def test_list_missing_id_falls_back_to_name(tmp_path):
    _stage_personality(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True)
    (companions / "ada.json").write_text(json.dumps({}), encoding="utf-8")
    found = list_companions(tmp_path)
    assert found[1].personality_id == "ada"


def test_list_deterministic(tmp_path):
    _stage_personality(tmp_path)
    assert list_companions(tmp_path) == list_companions(tmp_path)


def test_list_pure_no_writes(tmp_path):
    list_companions(tmp_path)
    assert list(tmp_path.iterdir()) == []


def _stage_build_root(tmp_path):
    import shutil

    from tests.conftest import ROOT

    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    for name in ("personality.json", "appraisal.json", "behavior.json",
                 "decay.json", "environment.json"):
        shutil.copy(ROOT / "config" / name, tmp_path / "config" / name)


def test_build_default_companion_flat(tmp_path, monkeypatch):
    from carrot.cli import build_carrot

    _stage_build_root(tmp_path)
    monkeypatch.setattr("carrot.cli.ROOT", tmp_path)
    c = build_carrot(companion="carrot")
    assert (tmp_path / "state" / "state.json").is_file()
    assert c.state.session_count == 1
    assert c.personality.id == "carrot-default-v1"


def test_build_named_companion_namespaced(tmp_path, monkeypatch):
    from carrot.cli import build_carrot

    _stage_build_root(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(exist_ok=True)
    (companions / "ada.json").write_text(json.dumps({"id": "ada-v1"}),
                                         encoding="utf-8")
    monkeypatch.setattr("carrot.cli.ROOT", tmp_path)
    c = build_carrot(companion="ada")
    assert (tmp_path / "state" / "ada" / "state.json").is_file()
    assert c.personality.id == "ada-v1"


def test_build_unknown_companion_raises(tmp_path, monkeypatch):
    from carrot.cli import build_carrot

    _stage_build_root(tmp_path)
    monkeypatch.setattr("carrot.cli.ROOT", tmp_path)
    with pytest.raises(ValueError):
        build_carrot(companion="Ada")


def test_make_companion_working_turn(tmp_path):
    c = make_companion("ada", tmp_path)
    assert isinstance(c.process_turn("hello"), str) \
        and c.process_turn("hello again")


def test_make_companion_isolated_from_real_state(tmp_path):
    from tests.conftest import ROOT

    real = [p for p in (ROOT / "state").iterdir() if p.is_file()]
    before = {p.name: p.stat().st_mtime_ns for p in real}
    c = make_companion("ada", tmp_path)
    c.process_turn("hello there")
    after = {p.name: p.stat().st_mtime_ns for p in (ROOT / "state").iterdir()
             if p.is_file()}
    assert after == before


def test_make_companion_namespaced_paths(tmp_path):
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True)
    (companions / "ada.json").write_text(json.dumps({"id": "ada-v1"}),
                                         encoding="utf-8")
    c = make_companion("ada", tmp_path)
    assert c.memory_store.path == tmp_path / "state" / "ada" / "memories.json"
    assert c.personality.id == "ada-v1"


def test_companions_do_not_share_state(tmp_path):
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True)
    (companions / "ada.json").write_text(json.dumps({"id": "ada-v1"}),
                                         encoding="utf-8")
    first = make_companion("carrot", tmp_path)
    second = make_companion("ada", tmp_path)
    first.process_turn("You're cute, thanks!")
    second.process_turn("You are USELESS!!!")
    from carrot.emotions import EmotionState
    from carrot.state_store import StateStore

    StateStore(str(tmp_path / "state" / "state.json")).save(first.state)
    StateStore(str(tmp_path / "state" / "ada" / "state.json")).save(
        second.state)
    a_state = EmotionState.from_dict(
        json.loads((tmp_path / "state" / "state.json").read_text(
            encoding="utf-8")))
    b_state = EmotionState.from_dict(
        json.loads((tmp_path / "state" / "ada" / "state.json").read_text(
            encoding="utf-8")))
    assert a_state.to_dict() != b_state.to_dict()
    assert a_state.emotions["happiness"] > b_state.emotions["happiness"]
    assert b_state.emotions["hurt"] > a_state.emotions["hurt"]


def test_companions_share_global_permissions(tmp_path):
    first = make_companion("carrot", tmp_path)
    second = make_companion("ada", tmp_path)
    expected = tmp_path / "state" / "permissions.json"
    assert first.permission_store.path == expected
    assert second.permission_store.path == expected


# --- Slice 16 Phase 3: format_companions tests ---


def _stage_social_root(tmp_path):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "personality.json").write_text(
        json.dumps({"id": "carrot-default-v1"}), encoding="utf-8")


def test_format_default_only(tmp_path):
    _stage_social_root(tmp_path)
    assert format_companions(tmp_path) == \
        "carrot: personality=carrot-default-v1 state=state/"


def test_format_named_companions(tmp_path):
    _stage_social_root(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True)
    (companions / "ada.json").write_text(json.dumps({"id": "ada-v1"}),
                                         encoding="utf-8")
    assert format_companions(tmp_path) == \
        ("carrot: personality=carrot-default-v1 state=state/\n"
         "ada: personality=ada-v1 state=state/ada/")


def test_format_pending_count_shown(tmp_path):
    _stage_social_root(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True)
    (companions / "ada.json").write_text(json.dumps({"id": "ada-v1"}),
                                         encoding="utf-8")
    inbox = tmp_path / "state" / "ada" / "social" / "instagram" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "000001-aa.msg").write_text("hi", encoding="utf-8")
    lines = format_companions(tmp_path).splitlines()
    assert lines[1] == \
        "ada: personality=ada-v1 state=state/ada/ pending=1"


def test_format_no_pending_when_zero(tmp_path):
    _stage_social_root(tmp_path)
    assert "pending=" not in format_companions(tmp_path)


def test_format_deterministic(tmp_path):
    _stage_social_root(tmp_path)
    assert format_companions(tmp_path) == format_companions(tmp_path)


def test_format_non_mutating(tmp_path):
    _stage_social_root(tmp_path)
    inbox = tmp_path / "state" / "ada" / "social" / "instagram" / "inbox"
    inbox.mkdir(parents=True)
    target = inbox / "000001-aa.msg"
    target.write_text("hi", encoding="utf-8")
    before = (target.stat().st_mtime_ns, target.read_bytes())
    format_companions(tmp_path)
    assert (target.stat().st_mtime_ns, target.read_bytes()) == before
