from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from channel_aware_boss_v4 import (  # noqa: E402
    CABossV4Config,
    _global_cross_state_matches,
    apply_evidence_thresholds,
    attach_cell_loading_evidence,
    build_pair_evidence,
    build_typed_graph,
    decompose_features,
    load_state_centroids,
)


def test_membership_reader_ignores_truth_columns(tmp_path: Path) -> None:
    embeddings = pd.DataFrame(
        {
            "modality": ["rna", "rna", "chrom_H3K27ac", "chrom_H3K27ac"],
            "cell": ["r1", "r2", "a1", "a2"],
            "integration__dim0": [-1.0, 1.0, -0.8, 0.9],
            "integration__dim1": [0.2, -0.2, 0.1, -0.1],
            "true_z0": [99.0, 99.0, 99.0, 99.0],
        }
    )
    membership = pd.DataFrame(
        {
            "sample_id": ["M1", "M2", "M1", "M2"],
            "modality": embeddings["modality"],
            "cell": embeddings["cell"],
            "true_cell_type": ["leak"] * 4,
            "true_z0": [99.0] * 4,
        }
    )
    embedding_path = tmp_path / "analysis_embeddings.tsv"
    membership_path = tmp_path / "analysis_membership.tsv"
    embeddings.to_csv(embedding_path, sep="\t", index=False)
    membership.to_csv(membership_path, sep="\t", index=False)

    centroids, provenance = load_state_centroids(
        embedding_path,
        membership_path,
        ["M1", "M2"],
    )

    assert len(centroids) == 2
    assert provenance["membership_columns_read"] == ["sample_id", "modality", "cell"]
    assert "true_z0" not in provenance["embedding_columns_read"]
    assert not any("true" in column for column in centroids.columns)


def test_explicit_state_decomposition_separates_shared_state_from_residual() -> None:
    rng = np.random.default_rng(12)
    n = 90
    state = rng.normal(size=n)
    observed = pd.DataFrame(
        {
            "sample_id": [f"M{i}" for i in range(n)],
            "chr1:100-199__H3K27ac": state + rng.normal(scale=0.18, size=n),
            "chr1:300-399__H3K4me3": -state + rng.normal(scale=0.18, size=n),
            "libsize_background__H3K27ac": np.exp(5.0 + 0.1 * state),
            "libsize_background__H3K4me3": np.exp(5.2 - 0.1 * state),
        }
    )
    basis = pd.DataFrame(
        {
            "state__chrom_H3K27ac__dim0": state + rng.normal(scale=0.05, size=n),
            "state__chrom_H3K4me3__dim0": -state + rng.normal(scale=0.05, size=n),
            "state__consensus__dim0": state,
            "state__assay_uncertainty": np.full(n, 0.05),
        }
    )
    cfg = CABossV4Config(
        residual_permutations=49,
        stability_bootstraps=50,
        min_state_score=0.10,
        residual_nonsignificance_p=0.05,
        seed=4,
    )

    decomposition = decompose_features(observed, basis, cfg)
    evidence = build_pair_evidence(decomposition, cfg).iloc[0]
    strict_cfg = CABossV4Config(
        residual_permutations=49,
        stability_bootstraps=50,
        min_abs_predicted_r=0.99,
        min_state_score=0.95,
        residual_nonsignificance_p=0.05,
        seed=4,
    )
    strict_evidence = build_pair_evidence(decomposition, strict_cfg).iloc[0]

    assert abs(evidence["total_r"]) > 0.8
    assert abs(evidence["predicted_r"]) > 0.8
    assert abs(evidence["residual_r"]) < 0.35
    assert evidence["state_eligible"]
    assert not evidence["direct_eligible"]
    assert strict_evidence["state_bootstrap_support"] == evidence["state_bootstrap_support"]
    assert strict_evidence["direct_bootstrap_support"] == evidence["direct_bootstrap_support"]


def _pair(
    left: str,
    right: str,
    *,
    same: bool,
    state: bool = False,
    direct: bool = False,
    score: float = 0.8,
    residual_q: float = 0.8,
) -> dict[str, object]:
    return {
        "node1": left,
        "node2": right,
        "channel1": left.rsplit("__", 1)[-1] if not left.startswith("expr__") else "RNA",
        "channel2": right.rsplit("__", 1)[-1] if not right.startswith("expr__") else "RNA",
        "same_assay": same,
        "total_r": 0.8,
        "predicted_r": 0.85,
        "residual_r": 0.05 if not direct else 0.5,
        "residual_empirical_p": residual_q,
        "residual_empirical_q": residual_q,
        "loading_cosine": 0.95,
        "state_score": score,
        "state_explained_covariance_fraction": 0.9,
        "state_bootstrap_support": 0.9,
        "direct_bootstrap_support": 0.9,
        "state_eligible": state,
        "direct_eligible": direct,
    }


def test_typed_graph_uses_mutual_state_matches_and_orients_residual_collider() -> None:
    parent_a = "chr1:100-199__H3K27ac"
    parent_b = "chr1:200-299__H3K27ac"
    child = "chr1:300-399__H3K27ac"
    shared = "chr1:400-499__H3K4me3"
    decoy = "chr1:500-599__H3K4me3"
    gene = "expr__SIMG_log1p"
    rows = [
        _pair(parent_a, child, same=True, direct=True, residual_q=0.01),
        _pair(parent_b, child, same=True, direct=True, residual_q=0.01),
        _pair(parent_a, parent_b, same=True, residual_q=0.8),
        _pair(parent_a, shared, same=False, state=True, score=0.9),
        _pair(parent_a, decoy, same=False, state=True, score=0.2),
        # A direct collider child can have a strong state loading, but its
        # redundant pair projection is suppressed in favor of the explicit
        # direct and state-loading layers.
        _pair(child, decoy, same=False, state=True, score=0.95),
        _pair(gene, shared, same=False, state=True, score=0.85),
        _pair(gene, decoy, same=False, state=True, score=0.1),
    ]
    evidence = pd.DataFrame(rows)
    graph = build_typed_graph(evidence, CABossV4Config(stability_bootstraps=0))

    direct = graph.loc[graph["edge_type"] == "within_assay_direct_candidate"]
    assert len(direct) == 2
    for row in direct.to_dict(orient="records"):
        child_endpoint = row["endpoint1"] if row["node1"] == child else row["endpoint2"]
        assert child_endpoint == "ARROW"
        assert row["orientation_source"] == "unshielded_residual_collider"

    shared_rows = graph.loc[graph["edge_type"] == "cross_assay_shared_state"]
    assert len(shared_rows) == 1
    assert {shared_rows.iloc[0]["node1"], shared_rows.iloc[0]["node2"]} == {
        parent_a,
        shared,
    }

    gene_rows = graph.loc[graph["edge_type"] == "prior_constrained_peak_gene"]
    assert len(gene_rows) == 1
    gene_row = gene_rows.iloc[0]
    gene_endpoint = gene_row["endpoint1"] if gene_row["node1"] == gene else gene_row["endpoint2"]
    assert gene_endpoint == "ARROW"
    assert gene_row["orientation_source"] == "gene_sink_prior"


def test_cell_loading_evidence_can_support_state_when_metacell_rows_cannot() -> None:
    left = "chr1:100-199__H3K27ac"
    right = "chr2:200-299__H3K4me3"
    metacell = pd.DataFrame(
        [
            {
                "node1": right,
                "node2": left,
                "channel1": "H3K4me3",
                "channel2": "H3K27ac",
                "same_assay": False,
                "left_state_r2": 0.0,
                "right_state_r2": 0.0,
                "total_r": 0.0,
                "predicted_r": 0.0,
                "loading_cosine": 0.0,
                "state_score": 0.0,
                "state_bootstrap_support": 0.0,
                "residual_empirical_p": 1.0,
                "residual_r": 0.0,
                "residual_to_pair_null95": 0.0,
                "residual_empirical_q": 1.0,
                "direct_bootstrap_support": 0.0,
            }
        ]
    )
    loading = pd.DataFrame(
        [
            {
                "node1": left,
                "node2": right,
                "loading_cosine": -0.95,
                "absolute_loading_cosine": 0.95,
                "sign_invariant_unit_loading_distance": 0.32,
                "minimum_loading_norm": 0.8,
                "node1_oof_marginal_state_fit": 0.30,
                "node2_oof_marginal_state_fit": 0.25,
                "node1_oof_partial_state_fit": 0.24,
                "node2_oof_partial_state_fit": 0.20,
                "node1_loading_stability": 0.9,
                "node2_loading_stability": 0.8,
                "loading_stability_support": 0.85,
                "pair_state_score": 0.20,
                "loading_empirical_null_p": 0.04,
                "loading_empirical_null_q": 0.10,
                "unit_distance_null_percentile": 0.04,
                "selected_sparse": True,
            }
        ]
    )

    combined = attach_cell_loading_evidence(metacell, loading)
    selected = apply_evidence_thresholds(combined, CABossV4Config())

    assert not bool(selected.iloc[0]["metacell_state_eligible"])
    assert bool(selected.iloc[0]["cell_loading_state_eligible"])
    assert bool(selected.iloc[0]["state_eligible"])
    # The cell evidence was canonicalized even though metacell node order was reversed.
    assert selected.iloc[0]["cell_left_oof_partial_state_fit"] == 0.20
    assert selected.iloc[0]["cell_right_oof_partial_state_fit"] == 0.24


def test_cross_loading_matches_use_one_global_neighbor_budget() -> None:
    evidence = pd.DataFrame(
        [
            {"node1": "a__A", "node2": "b__B", "same_assay": False, "ok": True, "score": 0.9},
            {"node1": "a__A", "node2": "c__C", "same_assay": False, "ok": True, "score": 0.8},
            {"node1": "d__A", "node2": "c__C", "same_assay": False, "ok": True, "score": 0.85},
        ]
    )

    selected = _global_cross_state_matches(
        evidence,
        1,
        eligibility_column="ok",
        score_column="score",
    )

    assert selected == {("a__A", "b__B"), ("c__C", "d__A")}


def test_hierarchical_direct_layer_requires_positive_residual_dependence() -> None:
    true_parent = "chr1:100-199__H3K27ac"
    child = "chr1:200-299__H3K27ac"
    compositional_decoy = "chr1:300-399__H3K27ac"
    evidence = pd.DataFrame(
        [
            _pair(true_parent, child, same=True),
            _pair(compositional_decoy, child, same=True),
        ]
    )
    hierarchy = pd.DataFrame(
        [
            {
                "node1": true_parent,
                "node2": child,
                "posterior_probability_positive": 0.91,
                "posterior_probability_negative": 0.01,
                "posterior_probability_residual_dependence": 0.92,
            },
            {
                "node1": compositional_decoy,
                "node2": child,
                "posterior_probability_positive": 0.01,
                "posterior_probability_negative": 0.94,
                "posterior_probability_residual_dependence": 0.95,
            },
        ]
    )

    graph = build_typed_graph(
        evidence,
        CABossV4Config(
            hierarchical_probability_threshold=0.88,
            hierarchical_positive_effect_only=True,
            stability_bootstraps=0,
        ),
        hierarchical_evidence=hierarchy,
    )

    selected = {
        tuple(sorted((str(row.node1), str(row.node2))))
        for row in graph.itertuples()
        if row.edge_type == "within_assay_direct_candidate"
    }
    assert selected == {tuple(sorted((true_parent, child)))}
