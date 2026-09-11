"""Slice 16 Phase-1 tests: companion value type, names, paths."""
import pytest

from carrot.companion import (
    DEFAULT_COMPANION,
    Companion,
    companion_paths,
    validate_name,
)
from carrot.paths import _paths


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
