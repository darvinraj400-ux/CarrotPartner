"""Slice 12 Phase-1 tests: avatar expression mapping (pure, headless)."""
import pytest

from carrot.avatar import (
    INTENSITY_BANDS,
    VISUAL_STATES,
    Expression,
    expression_for,
)


# ---- Expression ----


def test_expression_construction():
    expr = Expression(name="content", intensity_band="medium",
                      palette_key="content", presence_cue=None)
    assert expr.name == "content"
    assert expr.intensity_band == "medium"
    assert expr.palette_key == "content"
    assert expr.presence_cue is None


def test_expression_frozen():
    expr = Expression(name="content", intensity_band="medium",
                      palette_key="content", presence_cue=None)
    with pytest.raises(Exception):
        expr.name = "happy"


def test_expression_bad_name():
    with pytest.raises(ValueError):
        Expression(name="furious", intensity_band="medium",
                   palette_key="furious", presence_cue=None)


def test_expression_bad_band():
    with pytest.raises(ValueError):
        Expression(name="content", intensity_band="xtreme",
                   palette_key="content", presence_cue=None)


def test_expression_presence_cue_values():
    ok = Expression(name="content", intensity_band="medium",
                    palette_key="content", presence_cue="warm")
    assert ok.presence_cue == "warm"
    with pytest.raises(ValueError):
        Expression(name="content", intensity_band="medium",
                   palette_key="content", presence_cue="hot")


def test_expression_empty_palette_key():
    with pytest.raises(ValueError):
        Expression(name="content", intensity_band="medium",
                   palette_key="", presence_cue=None)


# ---- expression_for: behavior overrides ----


def test_shy_behavior_non_low_mood():
    expr = expression_for("content", "shy", "medium")
    assert expr.name == "shy"


def test_embarrassed_behavior_non_low_mood():
    expr = expression_for("bright", "embarrassed", "medium")
    assert expr.name == "shy"


def test_shy_suppressed_by_low_mood():
    expr = expression_for("low", "shy", "medium")
    assert expr.name == "sad"


def test_curious_behavior_non_bright_mood():
    expr = expression_for("content", "curious", "medium")
    assert expr.name == "curious"


def test_curious_suppressed_by_bright_mood():
    expr = expression_for("bright", "curious", "medium")
    assert expr.name == "happy"


# ---- expression_for: mood-driven ----


def test_bright_mood_maps_happy():
    assert expression_for("bright", "normal", "medium").name == "happy"


def test_low_mood_maps_sad():
    assert expression_for("low", "normal", "medium").name == "sad"


def test_tense_mood_maps_tense():
    assert expression_for("tense", "normal", "medium").name == "tense"


def test_tender_mood_maps_tender():
    assert expression_for("tender", "normal", "medium").name == "tender"


def test_content_fallback():
    assert expression_for("content", "normal", "medium").name == "content"


# ---- expression_for: band passthrough ----


def test_band_passthrough_each():
    for band in ("low", "medium", "high"):
        assert expression_for("content", "normal",
                              band).intensity_band == band


def test_unknown_band_maps_medium():
    assert expression_for("content", "normal",
                          "xtreme").intensity_band == "medium"


def test_non_string_band_maps_medium():
    assert expression_for("content", "normal",
                          42).intensity_band == "medium"


# ---- expression_for: closeness cue ----


def test_no_closeness_band_no_cue():
    assert expression_for("content", "normal",
                          "medium").presence_cue is None


def test_low_closeness_no_cue():
    assert expression_for("content", "normal", "medium",
                          closeness_band="low").presence_cue is None


def test_medium_closeness_no_cue():
    assert expression_for("content", "normal", "medium",
                          closeness_band="medium").presence_cue is None


def test_high_closeness_warm_cue():
    assert expression_for("content", "normal", "medium",
                          closeness_band="high").presence_cue == "warm"


# ---- defensive normalization ----


def test_unknown_mood_falls_back():
    assert expression_for("manic", "normal", "medium").name == "content"


def test_non_string_behavior_no_crash():
    assert expression_for("bright", None, "medium").name == "happy"


def test_non_string_mood_no_crash():
    assert expression_for(123, "normal", "medium").name == "content"


# ---- purity / determinism ----


def test_expression_for_deterministic():
    first = expression_for("tender", "affectionate", "high",
                           closeness_band="high")
    second = expression_for("tender", "affectionate", "high",
                            closeness_band="high")
    assert first == second


def test_palette_key_equals_name_all_states():
    assert VISUAL_STATES == ("content", "happy", "tender", "sad",
                             "tense", "shy", "curious")
    assert INTENSITY_BANDS == ("low", "medium", "high")
    for state in VISUAL_STATES:
        expr = Expression(name=state, intensity_band="medium",
                          palette_key=state, presence_cue=None)
        assert expr.palette_key == expr.name
