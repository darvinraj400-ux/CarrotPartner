"""Centralized filesystem layout for the default companion (Slice 16).

Pure path construction: no I/O, no state reads, no project imports.
Every production site that builds a store or config path goes through
_paths(); Phase 1's companion module derives namespaced variants from
the same roots.
"""
from __future__ import annotations

from pathlib import Path


def _paths(root: Path) -> dict:
    """Map of layout names to absolute paths for the default companion.

    root is required: there is no fallback, so a caller can never
    silently diverge from a patched project root in tests.
    """
    r = Path(root)
    state = r / "state"
    config = r / "config"
    return {
        "root": r,
        "state_dir": state,
        "state_json": state / "state.json",
        "memories_json": state / "memories.json",
        "drives_json": state / "drives.json",
        "permissions_json": state / "permissions.json",
        "relationship_json": state / "relationship.json",
        "initiative_json": state / "initiative.json",
        "channel_dir": state / "channel",
        "social_dir": state / "social",
        "config_dir": config,
        "appraisal_json": config / "appraisal.json",
        "behavior_json": config / "behavior.json",
        "decay_json": config / "decay.json",
        "personality_json": config / "personality.json",
        "environment_json": config / "environment.json",
        "tools_json": config / "tools.json",
        "initiative_config_json": config / "initiative.json",
        "relationship_config_json": config / "relationship.json",
        "social_config_json": config / "social.json",
    }
