from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_nested_loading_group_competing_parent_observability_v4 as audit  # noqa: E402
import loading_group_competing_parent_observability_guard_v4 as guard  # noqa: E402
import loading_group_state_guard_v4 as v429  # noqa: E402


def test_fixed_guard_payload_is_self_hashed_and_contains_no_truth_path() -> None:
    artifact = {
        "schema_version": guard.SCHEMA,
        "fixed_rule": guard.FIXED_RULE,
        "artifact_role": "future_application_rule_binding",
        "application_requires_truth": False,
        "numeric_threshold_fitted": False,
        "exact_group_construction_competition_is_bound": True,
        "new_simulation_worlds_or_datasets_created": False,
    }
    artifact["artifact_payload_sha256"] = audit._canonical_sha256(artifact)

    assert audit._validate_self_hash(
        artifact,
        field="artifact_payload_sha256",
        label="test fixed guard",
    ) == artifact["artifact_payload_sha256"]
    assert artifact["application_requires_truth"] is False
    assert not any(str(key).endswith("truth_path") for key in artifact)

    artifact["fixed_rule"] = "tampered"
    with pytest.raises(ValueError, match="payload hash mismatch"):
        audit._validate_self_hash(
            artifact,
            field="artifact_payload_sha256",
            label="test fixed guard",
        )


def test_phase_a_refuses_overwrite_before_opening_any_input(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit.run_phase_a(
            out_dir=tmp_path,
            v430_dir=tmp_path / "missing_v430",
            v427_dir=tmp_path / "missing_v427",
            selection_seed=1,
            panels=1,
            panel_size=1,
        )


def test_phase_a_loader_rejects_a_source_hash_change(tmp_path: Path) -> None:
    manifest = {
        "schema_version": audit.SCHEMA,
        "algorithm_schema_version": guard.SCHEMA,
        "phase": "a",
        "scorer_or_truth_opened": False,
        "all_observable_inputs_loaded_before_truth": True,
        "new_simulation_worlds_or_datasets_created": False,
        "source_sha256": {"audit.py": "tampered"},
    }
    manifest["manifest_payload_sha256"] = audit._canonical_sha256(manifest)
    (tmp_path / "phase_a_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="sources changed"):
        audit._load_phase_a(tmp_path)


def test_independent_boolean_reconstruction_matches_guard_output() -> None:
    decisions = pd.DataFrame(
        [
            {
                "loading_group": "state_keep",
                "modality": "chrom_A",
                "members": "p;s",
                "retained": True,
                "guard_route": v429.STATE_ROUTE,
                "guard_status": "v430_state",
            },
            {
                "loading_group": "state_drop",
                "modality": "chrom_B",
                "members": "p2;s2",
                "retained": True,
                "guard_route": v429.STATE_ROUTE,
                "guard_status": "v430_state",
            },
            {
                "loading_group": "axis_keep",
                "modality": "chrom_C",
                "members": "x;y",
                "retained": True,
                "guard_route": v429.AXIS_ROUTE,
                "guard_status": "v430_axis",
            },
        ]
    )
    structure = pd.DataFrame(
        [
            {
                "pair_key": "p|||s",
                "opposite_parent_pair_key": "q|||s",
                "modality": "chrom_A",
                "parent_candidate": "p",
                "noncollider_sibling_candidate": "s",
                "collider_center": "c",
                "other_parent_candidate": "q",
                "selected": True,
                "selection_stage": "formal_baseline",
            },
            {
                "pair_key": "p2|||s2",
                "opposite_parent_pair_key": "q2|||s2",
                "modality": "chrom_B",
                "parent_candidate": "p2",
                "noncollider_sibling_candidate": "s2",
                "collider_center": "c2",
                "other_parent_candidate": "q2",
                "selected": True,
                "selection_stage": "formal_baseline",
            },
        ]
    )
    features = pd.DataFrame(
        [
            {
                "feature": feature,
                "modality": modality,
                "formal_test_eligible": True,
                "oof_marginal_state_fit": 0.1,
                "oof_partial_state_fit": partial,
                "within_assay_direct_candidate_degree": 1,
                "topology_consensus_state_specific_gate": partial > 0.0,
                "topology_consensus_residual_structure_support": True,
                "topology_consensus_fixed_rule_eligible": partial > 0.0,
            }
            for feature, modality, partial in (
                ("q", "chrom_A", 0.1),
                ("q2", "chrom_B", -0.1),
            )
        ]
    )

    result = guard.apply_loading_group_competing_parent_observability_guard(
        decisions, structure, features
    ).decisions
    reconstructed = result["v430_retained"].astype(bool) & (
        result["v430_guard_route"].astype(str).eq(guard.AXIS_ROUTE)
        | (
            result["v430_guard_route"].astype(str).eq(guard.STATE_ROUTE)
            & result["complete_two_parent_comparison_available"].astype(bool)
            & result["competing_parent_fixed_rule_eligible"].astype(bool)
        )
    )

    assert reconstructed.equals(result["retained"].astype(bool))
    assert result.set_index("loading_group")["retained"].to_dict() == {
        "state_keep": True,
        "state_drop": False,
        "axis_keep": True,
    }
