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
                 "decay.json", "environment.json", "episodes.json"):
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


# --- Slice 16 Phase 4: distinctness + isolation proofs ---


_ALT_PERSONALITY = {
    "id": "ada-v1", "sensitivity": 0.9, "playfulness": 0.1,
    "stoicism": 0.8, "shyness": 0.7, "pride": 0.3, "warmth": 0.3,
    "patience": 0.9,
}


def _stage_alt_personality(tmp_path):
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True, exist_ok=True)
    (companions / "ada.json").write_text(json.dumps(_ALT_PERSONALITY),
                                         encoding="utf-8")


def _tree_snapshot(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_mtime_ns,
                                             p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_two_companions_distinct_replies_from_same_input(tmp_path):
    # Thesis: companions are not interchangeable. Four insults push both
    # past the sulk gate; default stoicism reads it as sad, ada's high
    # stoicism as sulking — different behaviors, different replies.
    # (Single-turn replies coincide across personalities under MockLLM's
    # canned set — verified by probe — so the test drives accumulation.)
    _stage_alt_personality(tmp_path)
    carrot = make_companion("carrot", tmp_path)
    ada = make_companion("ada", tmp_path)
    assert carrot.personality.sketch() != ada.personality.sketch()
    for _ in range(4):
        carrot_reply = carrot.process_turn("You are USELESS!!!")
        ada_reply = ada.process_turn("You are USELESS!!!")
    assert carrot.last_behavior.name == "sad"
    assert ada.last_behavior.name == "sulking"
    assert carrot_reply != ada_reply


def test_same_companion_same_input_same_reply_from_fresh_state(tmp_path):
    # Guards against "different because nondeterministic": two fully
    # separate tmp roots (no shared files at all), same input.
    roots = [tmp_path / "r1", tmp_path / "r2"]
    for root in roots:
        root.mkdir()
    replies = []
    for root in roots:
        c = make_companion("carrot", root)
        replies.append(c.process_turn("hello there"))
    assert replies[0] == replies[1]


def test_turn_on_one_companion_does_not_touch_another(tmp_path):
    _stage_alt_personality(tmp_path)
    carrot = make_companion("carrot", tmp_path)
    ada = make_companion("ada", tmp_path)
    ada.process_turn("my name is Alice")  # materialize ada's files
    ada_root = tmp_path / "state" / "ada"
    assert any(ada_root.rglob("*.json"))  # non-vacuous: ada wrote files
    before = _tree_snapshot(ada_root)
    carrot.process_turn("hello there")
    assert _tree_snapshot(ada_root) == before
    assert any((tmp_path / "state").glob("memories.json"))


def test_turn_on_named_companion_does_not_touch_default(tmp_path):
    _stage_alt_personality(tmp_path)
    carrot = make_companion("carrot", tmp_path)
    ada = make_companion("ada", tmp_path)
    carrot.process_turn("my name is Alice")  # materialize default's files
    default_files = [p for p in (tmp_path / "state").iterdir()
                     if p.is_file()]
    assert default_files  # non-vacuous
    before = _tree_snapshot(tmp_path / "state")
    ada.process_turn("hello there")
    after = _tree_snapshot(tmp_path / "state")
    # Only ada's subtree may have changed; top-level default files fixed.
    for rel, stamp in after.items():
        if not rel.startswith("ada" + "/"):
            assert before[rel] == stamp, rel


def test_memory_extraction_is_per_companion(tmp_path):
    _stage_alt_personality(tmp_path)
    carrot = make_companion("carrot", tmp_path)
    ada = make_companion("ada", tmp_path)
    carrot.process_turn("my name is Alice")
    assert any("alice" in m.text
               for m in carrot.memory_store.all())
    assert not any("alice" in m.text for m in ada.memory_store.all())


def test_drive_rise_is_per_companion(tmp_path):
    _stage_alt_personality(tmp_path)
    carrot = make_companion("carrot", tmp_path, start=1_000_000.0)
    ada = make_companion("ada", tmp_path, start=1_000_000.0)
    carrot.process_turn("hello there")
    carrot.clock.advance(7200.0)
    carrot.process_turn("hello there")
    # Rise minus presence satisfaction leaves a positive level; ada
    # never turned, so its store is untouched at the default.
    assert carrot.drive_store.load().social > 0.0
    assert ada.drive_store.load().social == 0.0


def test_relationship_is_per_companion(tmp_path):
    _stage_alt_personality(tmp_path)
    carrot = make_companion("carrot", tmp_path)
    ada = make_companion("ada", tmp_path)
    carrot.process_turn("You're cute, thanks!")
    assert carrot.relationship_store.load().trust > 0.5
    assert ada.relationship_store.load().trust == 0.5


def test_global_stores_are_shared(tmp_path):
    _stage_alt_personality(tmp_path)
    carrot = make_companion("carrot", tmp_path)
    ada = make_companion("ada", tmp_path)
    expected = tmp_path / "state" / "permissions.json"
    assert carrot.permission_store.path == expected
    assert ada.permission_store.path == expected
    # Configs are frozen dataclasses with MappingProxyType fields, whose
    # identity is per-object — so compare field contents, not identity.
    assert dict(carrot.tools_config.levels) == \
        dict(ada.tools_config.levels)
    assert dict(carrot.tools_config.enabled) == \
        dict(ada.tools_config.enabled)
    assert dict(carrot.environment_config.allowed_apps) == \
        dict(ada.environment_config.allowed_apps)
    assert carrot.environment_config.idle_threshold_s == \
        ada.environment_config.idle_threshold_s


def test_social_inbox_is_per_companion(tmp_path):
    _stage_social_root(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True, exist_ok=True)
    (companions / "ada.json").write_text(json.dumps({"id": "ada-v1"}),
                                         encoding="utf-8")
    inbox = tmp_path / "state" / "ada" / "social" / "instagram" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "000001-aa.msg").write_text("hi", encoding="utf-8")
    lines = format_companions(tmp_path).splitlines()
    assert lines[1].endswith("pending=1")
    assert "pending=" not in lines[0]


def test_list_companions_returns_default_first(tmp_path):
    _stage_social_root(tmp_path)
    companions = tmp_path / "config" / "companions"
    companions.mkdir(parents=True, exist_ok=True)
    for name in ("zoe", "ada"):
        (companions / f"{name}.json").write_text(
            json.dumps({"id": f"{name}-v1"}), encoding="utf-8")
    found = list_companions(tmp_path)
    assert isinstance(found[0], Companion) and found[0].name == "carrot"
    assert [c.name for c in found] == ["carrot", "ada", "zoe"]


def test_companion_paths_disjoint_for_two_names(tmp_path):
    default = companion_paths(tmp_path, "carrot")
    named = companion_paths(tmp_path, "ada")
    per_companion = {"state_dir", "state_json", "memories_json",
                     "drives_json", "relationship_json", "initiative_json",
                     "episodes_json", "social_dir", "channel_dir",
                     "personality_json", "companion_config_dir"}
    for key in per_companion:
        # companion_config_dir exists only for named companions (the
        # default's config root is config_dir itself) — absence there
        # is itself a difference.
        if key in default:
            assert default[key] != named[key], key
        else:
            assert key in named, key
    for key in set(default) | set(named):
        if key not in per_companion:
            assert default[key] == named[key], key


def test_default_and_named_do_not_share_state_file(tmp_path):
    from carrot.state_store import StateStore

    carrot = make_companion("carrot", tmp_path)
    ada = make_companion("ada", tmp_path)
    carrot.store = StateStore(str(tmp_path / "sc.json"))
    ada.store = StateStore(str(tmp_path / "sa.json"))
    carrot.state.session_count = 5
    ada.state.session_count = 7
    carrot.store.save(carrot.state)
    ada.store.save(ada.state)
    assert carrot.store.path != ada.store.path
    assert StateStore(str(tmp_path / "sc.json")).load().session_count == 5
    assert StateStore(str(tmp_path / "sa.json")).load().session_count == 7
