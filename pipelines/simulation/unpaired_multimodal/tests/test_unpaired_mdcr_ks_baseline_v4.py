from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import unpaired_mdcr_ks_baseline_v4 as mdcr  # noqa: E402


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)


def _distinct_component_samples(seed: int, n: int = 5_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _standardize(
        np.column_stack(
            [
                rng.exponential(scale=1.0, size=n),
                rng.beta(2.0, 5.0, size=n),
                rng.chisquare(df=5.0, size=n),
            ]
        )
    )


def test_published_ks_rule_matches_distinct_shared_distributions() -> None:
    scores = {
        "assay_a": _distinct_component_samples(1),
        "assay_b": _distinct_component_samples(2)[:, [2, 0, 1]],
        "assay_c": _distinct_component_samples(3)[:, [1, 2, 0]],
    }

    pairwise, adjusted_alpha, critical = mdcr._pairwise_component_table(
        scores, alpha=0.05
    )
    shared = mdcr._consistent_shared_tuples(scores, pairwise)
    consensus, diagnostics = mdcr._best_global_consensus(scores, pairwise)

    assert adjusted_alpha < 0.05
    assert critical > 0.0
    assert len(shared) == 3
    assert len(consensus) == 3
    assert pairwise["matched"].astype(bool).sum() == 9
    assert shared["all_pairwise_matches"].astype(bool).all()
    assert consensus["satisfies_published_all_pair_rule"].astype(bool).all()
    assert diagnostics["best_global_pairwise_match_support"] == 1.0


def test_public_fit_api_excludes_focal_truth_and_correspondence() -> None:
    parameters = set(inspect.signature(mdcr.fit_unpaired_mdcr_backgrounds).parameters)

    assert parameters == {"background_matrices", "background_depths", "config"}
    assert not parameters.intersection(
        {"focal", "truth", "latent_state", "cell_ids", "pairing", "roles"}
    )


def test_observed_rank_audit_exposes_measurement_noise() -> None:
    rng = np.random.default_rng(8)
    normalized = rng.normal(size=(300, 12))

    assert mdcr._empirical_rank(normalized, gamma=0.2) > 3


def test_background_loader_refuses_scorer_path(tmp_path: Path) -> None:
    scorer = tmp_path / "scorer" / "backgrounds"
    scorer.mkdir(parents=True)

    with pytest.raises(ValueError, match="scorer/truth"):
        mdcr._safe_discovery_dir(scorer)


def test_freeze_writes_only_background_derived_outputs(tmp_path: Path) -> None:
    discovery = tmp_path / "world" / "discovery"
    discovery.mkdir(parents=True)
    rng = np.random.default_rng(19)
    for index, modality in enumerate(("assay_a", "assay_b")):
        latent = _distinct_component_samples(30 + index, n=500)
        loadings = rng.normal(size=(18, 3))
        rate = np.exp(1.0 + 0.25 * latent @ loadings.T)
        matrix = rng.poisson(rate).astype(float)
        empty = matrix.sum(axis=1) == 0
        matrix[empty, 0] = 1.0
        np.savez_compressed(
            discovery / f"background_matrix__{modality}.npz",
            matrix=matrix,
            background_depth=matrix.sum(axis=1),
            cell_ids=np.asarray(
                [f"{modality}_{row:04d}" for row in range(len(matrix))]
            ),
        )
    output = tmp_path / "published_baseline"

    manifest = mdcr.freeze_unpaired_mdcr_evidence(discovery, output)

    assert manifest["truth_or_scorer_used"] is False
    assert manifest["fixed_rank_three_adaptation"] is True
    assert manifest["published_method_assumptions_satisfied"] is False
    assert set(path.name for path in output.iterdir()) == {
        "component_scores.tsv",
        "fold_component_scores.tsv",
        "mixing_matrices.tsv",
        "pairwise_component_matches.tsv",
        "shared_component_tuples.tsv",
        "consensus_component_tuples.tsv",
        "fold_component_evidence.tsv",
        "fold_pairwise_component_matches.tsv",
        "fold_consensus_component_tuples.tsv",
        "provenance.json",
        "frozen_evidence_manifest.json",
    }
    fold_scores = np.genfromtxt(
        output / "fold_component_scores.tsv",
        delimiter="\t",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    assert len(fold_scores) == 2 * 2 * 500
