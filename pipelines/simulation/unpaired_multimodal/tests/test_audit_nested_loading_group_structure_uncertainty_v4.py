from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_nested_loading_group_structure_uncertainty_v4 as audit  # noqa: E402
import loading_group_structure_uncertainty_guard_v4 as guard  # noqa: E402


def test_threshold_reconstruction_matches_fixed_guard_boundary() -> None:
    frame = pd.DataFrame(
        [
            {
                "v429_retained": True,
                "v429_guard_route": guard.STATE_ROUTE,
                "complete_direct_collider_overlap": False,
                "minimum_member_centroid_cosine": 0.90,
            },
            {
                "v429_retained": True,
                "v429_guard_route": guard.STATE_ROUTE,
                "complete_direct_collider_overlap": False,
                "minimum_member_centroid_cosine": 0.899999,
            },
            {
                "v429_retained": True,
                "v429_guard_route": guard.AXIS_ROUTE,
                "complete_direct_collider_overlap": False,
                "minimum_member_centroid_cosine": 0.10,
            },
            {
                "v429_retained": True,
                "v429_guard_route": guard.AXIS_ROUTE,
                "complete_direct_collider_overlap": True,
                "minimum_member_centroid_cosine": 1.0,
            },
        ]
    )

    selected = audit._threshold_selection(
        frame, guard.MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
    )

    assert selected.tolist() == [True, False, True, False]
    assert guard.MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE in audit.THRESHOLD_GRID


def test_random_cached_panels_are_deterministic_and_block_balanced() -> None:
    worlds = pd.DataFrame(
        [
            {"block": block, "seed": block_index * 100 + offset}
            for block_index, block in enumerate(("a", "b", "c"), start=1)
            for offset in range(8)
        ]
    )

    first = audit._balanced_panels(
        worlds, panels=2, panel_size=9, selection_seed=123
    )
    second = audit._balanced_panels(
        worlds, panels=2, panel_size=9, selection_seed=123
    )

    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["panel_id", "block", "seed"]).any()
    for _, panel in first.groupby("panel_id"):
        assert panel["block"].value_counts().to_dict() == {"a": 3, "b": 3, "c": 3}


def test_audit_has_no_simulation_or_missing_world_generation_entrypoint() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")

    assert "simulate_world" not in source
    assert "generate_missing" not in source
    assert audit.SCHEMA.endswith(".v1")
