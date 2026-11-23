from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_quantile_geometry_rescue_v4 as diagnostic  # noqa: E402
import quantile_geometry_synchronization_evidence_v4 as evidence  # noqa: E402


def test_signed_quantile_cost_is_sign_invariant() -> None:
    left = np.asarray([-2.0, -1.0, 0.0, 1.0, 3.0])
    reflected_right = -left[::-1]

    assert diagnostic._signed_quantile_cost(
        left, reflected_right, power=1
    ) == pytest.approx(0.0)
    assert diagnostic._signed_quantile_cost(
        left, reflected_right, power=2
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Unsupported Wasserstein power"):
        diagnostic._signed_quantile_cost(left, reflected_right, power=3)


def test_local_diagnostics_requires_unique_reciprocal_low_cost() -> None:
    matrix = np.asarray(
        [
            [0.1, 0.8, 0.9],
            [0.7, 0.2, 0.6],
            [0.8, 0.5, 0.3],
        ]
    )
    cost, margin, mutual = diagnostic._local_diagnostics(matrix, 0, 0)

    assert cost > 0.0
    assert margin > 0.0
    assert mutual

    tied = matrix.copy()
    tied[0, 1] = tied[0, 0]
    _, tied_margin, tied_mutual = diagnostic._local_diagnostics(tied, 0, 0)
    assert tied_margin == pytest.approx(0.0)
    assert not tied_mutual


def test_feature_payload_requires_one_relaxed_row_per_assay_pair(
    tmp_path: Path,
) -> None:
    relaxed_dir = tmp_path / "relaxed"
    relaxed_dir.mkdir()
    relaxed_path = relaxed_dir / "relaxed_reporter_synchronization_pairs.tsv"
    pd.DataFrame(
        [
            {
                "left_modality": "a",
                "right_modality": "b",
                "candidate_cross_assay_feature_pairs_json": '[["a_s","b_s"]]',
            }
        ]
    ).to_csv(relaxed_path, sep="\t", index=False)
    artifacts = diagnostic._WorldArtifacts(
        relaxed_dir=relaxed_dir,
        mdcr_dir=tmp_path,
        assignment_path=tmp_path / "unused.tsv",
        evidence_manifest={},
    )
    frame = pd.DataFrame(
        [{"left_modality": "a", "right_modality": "b"}]
    )

    enriched = evidence._add_feature_payload(frame, artifacts)
    assert enriched["candidate_cross_assay_feature_pairs_json"].tolist() == [
        '[["a_s","b_s"]]'
    ]

    missing = pd.DataFrame(
        [{"left_modality": "a", "right_modality": "c"}]
    )
    with pytest.raises(ValueError, match="lacks assay pairs"):
        evidence._add_feature_payload(missing, artifacts)


def test_frozen_evidence_schema_contains_no_truth_columns() -> None:
    forbidden_tokens = ("truth", "label", "matched")
    assert not any(
        token in column
        for column in evidence.EVIDENCE_COLUMNS
        for token in forbidden_tokens
    )
    assert evidence.IDENTITY_COLUMNS[-1] == (
        "candidate_cross_assay_feature_pairs_json"
    )

