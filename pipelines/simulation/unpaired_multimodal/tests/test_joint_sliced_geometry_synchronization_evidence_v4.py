from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import joint_sliced_geometry_synchronization_evidence_v4 as evidence  # noqa: E402


def test_evidence_schema_contains_both_joint_views_without_truth() -> None:
    assert evidence.SCHEMA.endswith(".v1")
    assert len(evidence.JOINT_COLUMNS) == 18
    assert any(column.startswith("joint_sliced_w2_") for column in evidence.JOINT_COLUMNS)
    assert any(column.startswith("copula_sliced_w2_") for column in evidence.JOINT_COLUMNS)
    assert not {"truth_pair", "role", "label"}.intersection(evidence.EVIDENCE_COLUMNS)


def test_evidence_retains_feature_pair_payload() -> None:
    assert "candidate_cross_assay_feature_pairs_json" in evidence.IDENTITY_COLUMNS
