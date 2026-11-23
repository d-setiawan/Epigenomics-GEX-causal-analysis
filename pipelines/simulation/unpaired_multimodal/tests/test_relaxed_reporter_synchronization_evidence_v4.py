from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import relaxed_reporter_synchronization_evidence_v4 as evidence  # noqa: E402


def _pairwise() -> pd.DataFrame:
    orders = {"a": [0, 1, 2], "b": [1, 2, 0], "c": [2, 0, 1]}
    rows = []
    for left, right in combinations(sorted(orders), 2):
        matches = set(zip(orders[left], orders[right], strict=True))
        for left_component in range(3):
            for right_component in range(3):
                rows.append(
                    {
                        "left_modality": left,
                        "right_modality": right,
                        "left_component": left_component,
                        "right_component": right_component,
                        "scaled_ks_statistic": (
                            0.1
                            if (left_component, right_component) in matches
                            else 3.0
                        ),
                        "asymptotic_critical_value": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _assays() -> pd.DataFrame:
    components = {"a": 0, "b": 1, "c": 2}
    rows = []
    for modality, component in components.items():
        reporter = f"{modality}_s"
        rows.append(
            {
                "modality": modality,
                "external_state_reporter_members": reporter,
                "reporter_component_identified": True,
                "reporter_local_component": component,
                "reporter_bootstrap_component_assignment_supported": True,
                "reporter_feature_fold_components_json": json.dumps(
                    {reporter: {"0": component, "1": component}}
                ),
                "reporter_feature_fold_bootstrap_supported_json": json.dumps(
                    {"0": True, "1": True}
                ),
                # Reporter-only synchronization must not use this as a gate.
                "within_assay_pair_resolved_by_component": modality == "c",
            }
        )
    return pd.DataFrame(rows)


def test_relaxed_evidence_decouples_reporters_from_direct_structure() -> None:
    decisions = pd.DataFrame(
        [
            {
                "left_modality": "a",
                "right_modality": "b",
                "statistical_context_support_pattern": 110,
                "reporter_bootstrap_context_support_pattern": 111,
            }
        ]
    )
    context = _pairwise()

    pairs, contexts = evidence.build_relaxed_synchronization_evidence(
        _assays(),
        pd.DataFrame(columns=["candidate_assay_pairs_json"]),
        decisions,
        [context.copy(), context.copy(), context.copy()],
        ["full_fit", "feature_fold_0", "feature_fold_1"],
        [0, 1],
    )

    assert len(contexts) == 3
    assert contexts["eligible_modality_count"].eq(3).all()
    assert len(pairs) == 1
    pair = pairs.iloc[0]
    assert pair["global_reporter_coassignment_context_pattern"] == "111"
    assert pair["heldout_global_reporter_coassignment_count"] == 2
    assert pair["minimum_heldout_coassigned_pair_exclusion_relative_margin"] > 0
    assert not pair["both_assays_within_assay_pair_resolved_by_component"]
    assert pair["eligible_for_cross_block_empirical_null_calibration"]
    assert not pair["hypothesis_selected"]
    assert not pair["within_assay_direct_structure_claimed"]
    assert not pair["candidate_pairs_may_be_expanded_as_edges"]
