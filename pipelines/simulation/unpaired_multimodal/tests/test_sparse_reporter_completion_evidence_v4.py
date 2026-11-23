from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sparse_reporter_completion_evidence_v4 as evidence  # noqa: E402


def _direct() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "block": "block_a",
                "seed": 7,
                "source_assay": "assay_a",
                "node1": "p",
                "endpoint1": "TAIL",
                "node2": "c",
                "endpoint2": "ARROW",
            },
            {
                "block": "block_a",
                "seed": 7,
                "source_assay": "assay_a",
                "node1": "q",
                "endpoint1": "TAIL",
                "node2": "c",
                "endpoint2": "ARROW",
            },
        ]
    )


def _association_rows(*, fold: int | None = None) -> pd.DataFrame:
    components = {
        "p": (0, 0.70, 0.40, 80),
        "q": (1, 0.65, 0.35, 75),
        "c": (2, 0.60, 0.30, 70),
        "s": (0, 0.55, 0.25, 50),
        "n1": (2, 0.20, 0.05, 40),
        "n2": (1, 0.15, 0.02, 30),
    }
    rows = []
    for feature, (component, strength, margin, detected) in components.items():
        row = {
            "modality": "assay_a",
            "feature": feature,
            "dominant_component": component,
            "dominant_absolute_partial_correlation": strength,
            "component_relative_margin": margin,
            "detected_cells": detected,
            "cells": 100,
            "component_0_partial_correlation": 1.0 if component == 0 else 0.1,
            "component_1_partial_correlation": 1.0 if component == 1 else 0.1,
            "component_2_partial_correlation": 1.0 if component == 2 else 0.1,
        }
        if fold is not None:
            row["feature_fold"] = fold
        rows.append(row)
    return pd.DataFrame(rows)


def test_motif_by_assay_extracts_two_parents_and_child() -> None:
    observed = evidence._motif_by_assay(_direct(), block="block_a", seed=7)

    assert observed == {"assay_a": (("p", "q"), "c")}


def test_motif_by_assay_rejects_non_parent_endpoint() -> None:
    direct = _direct()
    direct.loc[0, "endpoint1"] = "CIRCLE"

    with pytest.raises(ValueError, match="TAIL"):
        evidence._motif_by_assay(direct, block="block_a", seed=7)


def test_feature_candidates_are_only_the_three_nonmotif_features() -> None:
    associations = _association_rows()
    folds = pd.concat(
        [_association_rows(fold=0), _association_rows(fold=1)],
        ignore_index=True,
    )
    upstream = pd.DataFrame(
        [
            {
                "modality": "assay_a",
                "hypothesis_selected": False,
                "external_state_reporter_members": "",
            }
        ]
    )

    observed = evidence._feature_candidates(
        world_key="block_a__world_seed7",
        block="block_a",
        seed=7,
        associations=associations,
        fold_associations=folds,
        upstream=upstream,
        motifs=evidence._motif_by_assay(
            _direct(), block="block_a", seed=7
        ),
    )

    assert observed["feature"].tolist() == ["n1", "n2", "s"]
    assert not observed["assay_upstream_selected"].any()
    assert observed.loc[observed["feature"].eq("s"), "component_stable"].item()
    assert (
        observed.loc[
            observed["feature"].eq("s"), "parent_component_match_count"
        ].item()
        == 3
    )
    assert observed["hypothesis_type"].eq(
        "unresolved_reporter_candidate"
    ).all()
    assert not observed["hypothesis_selected"].any()

