from __future__ import annotations

import json
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_derive_zero_edge_rank_fusion_v1 as audit  # noqa: E402
import crossfit_hierarchical_collider_v4 as runtime  # noqa: E402


EXPECTED_CONTRACT_SHA256 = (
    "130a4c7d5c2236d293850c95c60388981cdfc87e0672ed8b33c61843542d233a"
)
EXPECTED_RUNTIME_PAYLOAD_SHA256 = (
    "a8b42323b49dac5e1c40485c236169eb3156f2fe1bb0d4ab0817925c9037cd6b"
)


def test_retired_fit_reproduces_the_frozen_runtime_exactly(tmp_path: Path) -> None:
    manifest = audit.derive_manifest()

    assert manifest["allowed_training_blocks"] == [
        "TRAIN330",
        "TRAIN360",
        "TRAIN370",
        "TRAIN380",
    ]
    assert manifest["train400_inputs_used_for_fitting_or_selection"] is False
    assert manifest["dev_or_locked_inputs_opened"] is False
    assert manifest["runtime_source_closure_dependency"] is False
    assert manifest["input_binding"]["candidate_table"]["sha256"] == (
        audit.EXPECTED_CANDIDATE_SHA256
    )
    assert manifest["input_binding"]["phase_a_manifest"]["sha256"] == (
        audit.EXPECTED_PHASE_A_MANIFEST_SHA256
    )
    assert manifest["input_binding"]["phase_b_manifest"]["sha256"] == (
        audit.EXPECTED_PHASE_B_MANIFEST_SHA256
    )
    assert manifest["input_shape"] == {
        "candidate_rows": 34_560,
        "worlds": 96,
        "assays": 576,
        "exact_star_labels": 576,
        "directed_edge_rows": 17_280,
        "positive_directed_arms": 1_152,
    }
    assert manifest["directed_edge_design"]["matrix_sha256"] == (
        audit.EXPECTED_PAIR_MATRIX_SHA256
    )

    grid = manifest["block_held_out_grid"]
    assert [
        (
            row["C"],
            row["exact_assays"],
            row["all_six_exact_worlds"],
            row["full_retired_refit_nonzero_coefficients"],
        )
        for row in grid
    ] == [
        (0.003, 562, 84, 9),
        (0.01, 563, 84, 19),
        (0.03, 563, 84, 28),
        (0.1, 562, 83, 46),
        (0.3, 563, 84, 64),
        (1.0, 561, 84, 76),
    ]
    assert manifest["selection"]["selected_C"] == 0.003
    assert manifest["selection"]["maximum_block_oof_exact_assays"] == 563
    assert manifest["selection"]["minimum_eligible_exact_assays"] == 562
    assert manifest["selection"]["maximum_eligible_all_six_exact_worlds"] == 84

    selected = manifest["selected_model"]
    assert selected["feature_schema"] == [
        f"{context}__{feature}"
        for context, feature in runtime.ZERO_EDGE_RANK_FUSION_SCHEMA
    ]
    assert selected["imputation_medians"] == list(
        runtime.ZERO_EDGE_RANK_FUSION_MEDIANS
    )
    assert selected["raw_scale_weights"] == list(
        runtime.ZERO_EDGE_RANK_FUSION_WEIGHTS
    )
    assert selected["raw_scale_intercept"] == (
        runtime.ZERO_EDGE_RANK_FUSION_INTERCEPT
    )
    assert selected["coefficient_canonical_decimal_places"] == 9
    assert selected["runtime_payload"] == (
        runtime._zero_edge_rank_fusion_model_payload()
    )
    assert selected["runtime_payload_sha256"] == EXPECTED_RUNTIME_PAYLOAD_SHA256
    assert runtime._zero_edge_rank_fusion_model_sha256() == (
        EXPECTED_RUNTIME_PAYLOAD_SHA256
    )

    assert manifest["derivation_contract"]["sha256"] == EXPECTED_CONTRACT_SHA256
    assert runtime.ZERO_EDGE_RANK_FUSION_DERIVATION_CONTRACT_SHA256 == (
        EXPECTED_CONTRACT_SHA256
    )
    assert manifest["runtime_verification"]["all_exact"] is True
    assert all(
        value is True
        for key, value in manifest["runtime_verification"].items()
        if key.endswith("_exact")
    )
    assert set(manifest["software_versions"]) == {
        "numpy",
        "pandas",
        "scikit_learn",
    }

    audit.verify_manifest_payload(manifest)
    output = tmp_path / "development_only_derivation" / "manifest.json"
    audit.write_manifest(manifest, output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == manifest
    audit.verify_manifest_payload(written)

