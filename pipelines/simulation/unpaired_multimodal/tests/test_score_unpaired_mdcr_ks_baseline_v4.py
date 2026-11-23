from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_unpaired_mdcr_ks_baseline_v4 as scorer  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_component_alignment_recovers_permutation_and_sign() -> None:
    rng = np.random.default_rng(9)
    latent = rng.normal(size=(500, 3))
    scores = np.column_stack(
        [latent[:, 2], -latent[:, 0], latent[:, 1]]
    )

    result = scorer.align_components_to_latent_state(scores, latent)

    assert result["state_dimension"].tolist() == [2, 0, 1]
    assert result["absolute_correlation"].min() > 0.999


def test_assignment_scoring_distinguishes_pure_and_impure_sets() -> None:
    lookup = {
        ("a", 0): 0,
        ("a", 1): 1,
        ("b", 0): 1,
        ("b", 1): 0,
        ("c", 0): 0,
        ("c", 1): 1,
    }
    pure = scorer._score_assignment(
        assignment_json=json.dumps({"a": 0, "b": 1, "c": 0}),
        component_to_dimension=lookup,
    )
    impure = scorer._score_assignment(
        assignment_json=json.dumps({"a": 0, "b": 0, "c": 0}),
        component_to_dimension=lookup,
    )

    assert pure["biological_axis_pure"]
    assert pure["majority_assay_fraction"] == 1.0
    assert not impure["biological_axis_pure"]
    assert impure["majority_assay_fraction"] == 2.0 / 3.0


def test_manifest_verification_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "component_scores.tsv"
    output.write_text("x\n1\n", encoding="utf-8")
    manifest = {
        "schema_version": scorer.EXPECTED_EVIDENCE_SCHEMA,
        "output_file_sha256": {output.name: _sha256(output)},
        "truth_or_scorer_used": False,
        "new_worlds_or_datasets_created": False,
    }
    manifest["manifest_payload_sha256"] = scorer._canonical_sha256(manifest)
    (tmp_path / "frozen_evidence_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    scorer.verify_frozen_evidence(tmp_path)
    output.write_text("x\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash mismatch"):
        scorer.verify_frozen_evidence(tmp_path)


def test_empty_clique_is_an_explicit_abstention() -> None:
    result = scorer._score_assignment(
        assignment_json=json.dumps({"a": 0, "b": 1}),
        component_to_dimension={("a", 0): 0, ("b", 1): 0},
        modalities=[""],
    )

    assert result["n_scored_modalities"] == 0
    assert not result["biological_axis_pure"]
