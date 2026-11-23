from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_ca_boss_v4 as benchmark  # noqa: E402
from benchmark_ca_boss_v4 import (  # noqa: E402
    COLLIDER_MOTIF_SEED_STRIDE,
    COLLIDER_MOTIF_VOTES,
    axis_eligible_modalities_from_prepared,
    candidate_grid,
    combine_attainable_metrics,
    fit_collider_motif_vote_evidence,
    load_scorer_artifacts_after_pretruth_freeze,
    replace_direct_layer_with_collider_motifs,
    resolve_postconsensus_candidate,
    write_collider_motif_vote_evidence,
)
from channel_aware_boss_v4 import CABossV4Config  # noqa: E402
from consensus_ca_boss_v4 import direct_collider_children  # noqa: E402


def _write_bound_world(root: Path, seed: int) -> Path:
    world = root / f"world_seed{seed}"
    discovery = world / "discovery"
    scorer = world / "scorer"
    discovery.mkdir(parents=True)
    scorer.mkdir()

    discovery_payload = discovery / "observed.tsv"
    discovery_payload.write_text(f"seed\n{seed}\n")
    discovery_manifest = {
        "schema_version": benchmark.DISCOVERY_MANIFEST_SCHEMA_VERSION,
        "modalities": list(benchmark.MODALITIES),
        "files": [discovery_payload.name],
        "file_sha256": {
            discovery_payload.name: benchmark._file_sha256(discovery_payload)
        },
    }
    discovery_manifest_path = discovery / "manifest.json"
    discovery_manifest_path.write_text(json.dumps(discovery_manifest, sort_keys=True))

    truth = scorer / "truth_typed_edges.tsv"
    parameters = scorer / "world_parameters.json"
    roles = scorer / "role_assignments.tsv"
    pd.DataFrame(
        [{"node1": "a", "node2": "b", "truth_type": "within_assay_direct"}]
    ).to_csv(truth, sep="\t", index=False)
    parameters.write_text("{}\n")
    pd.DataFrame([{"node": "a", "role": "P"}]).to_csv(
        roles, sep="\t", index=False
    )
    scorer_files = [truth.name, parameters.name, roles.name]
    scorer_manifest = {
        "schema_version": benchmark.SCORER_MANIFEST_SCHEMA_VERSION,
        "scorer_only": True,
        "must_not_be_read_by_discovery": True,
        "files": scorer_files,
        "file_sha256": {
            name: benchmark._file_sha256(scorer / name) for name in scorer_files
        },
    }
    scorer_manifest_path = scorer / "manifest.json"
    scorer_manifest_path.write_text(json.dumps(scorer_manifest, sort_keys=True))

    run_summary = {
        "schema_version": benchmark.RUN_SUMMARY_SCHEMA_VERSION,
        "seed": seed,
        "discovery_directory": "discovery",
        "scorer_directory": "scorer",
        "n_modalities": len(benchmark.MODALITIES),
        "information_domains_separated_by_directory": True,
        "raw_only_preparer_required_for_discovery": True,
        "domain_payloads_content_addressed": True,
        "discovery_manifest_sha256": benchmark._file_sha256(
            discovery_manifest_path
        ),
        "scorer_manifest_sha256": benchmark._file_sha256(scorer_manifest_path),
    }
    (world / "run_summary.json").write_text(json.dumps(run_summary, sort_keys=True))
    return world


def test_candidate_grid_changes_only_selection_thresholds() -> None:
    base = CABossV4Config(seed=9, residual_permutations=17, stability_bootstraps=19)
    candidates = candidate_grid(base)
    assert len(candidates) == 216
    assert len({name for name, _ in candidates}) == len(candidates)
    for _, config in candidates:
        assert config.seed == 9
        assert config.ridge_alpha == base.ridge_alpha
        assert config.residual_permutations == 17
        assert config.stability_bootstraps == 19


def test_benchmark_binds_complete_default_repair_configs() -> None:
    observed = asdict(benchmark.FROZEN_OBSERVED_COLLIDER_STRUCTURE_CONFIG)
    loading = asdict(benchmark.FROZEN_STRUCTURE_ANCHORED_GROUP_CONFIG)
    assert observed == asdict(benchmark.ObservedColliderStructureConfig())
    assert loading == asdict(benchmark.StructureAnchoredGroupConfig())
    assert observed["required_rescue_only_base_graph_view_count"] == 6
    assert observed["required_rescue_only_loading_view_count"] == 6
    assert observed["required_rescue_only_beta_vote_count"] == 6
    assert observed["require_unique_rescue_only_candidate"] is True
    assert loading["enable_abstention_only_empirical_null_rescue"] is True
    assert loading["rescue_empirical_null_p_threshold"] == 0.125
    assert loading["rescue_minimum_empirical_null_view_support"] == 1.0
    assert loading["rescue_minimum_parent_win_view_support"] == 1.0
    assert loading["rescue_required_view_count"] == 6
    assert loading["rescue_requires_unique_candidate"] is True
    assert loading["enable_layer_separated_tiered_rescue"] is True
    assert loading["tiered_rescue_empirical_null_p_threshold"] == 0.50
    assert loading["tiered_rescue_minimum_empirical_null_view_support"] == 1.0
    assert loading["tiered_rescue_minimum_parent_win_view_support"] == 1.0
    assert loading[
        "tiered_rescue_requires_sibling_assay_global_maxT_admission"
    ] is True
    assert loading[
        "tiered_rescue_minimum_sibling_conditional_wald_view_support"
    ] == 1.0
    assert loading["tiered_rescue_required_view_count"] == 6
    assert loading["tiered_rescue_requires_unique_candidate"] is True


def test_partial_soft_axis_eligibility_comes_only_from_safe_provenance(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "benchmark_output" / "prepared_discovery"
    prepared.mkdir(parents=True)
    (prepared / "alignment_provenance.json").write_text(
        json.dumps(
            {
                "alignment_selection": {
                    "selected_mode": "soft_partial",
                    "cross_assay_axis_eligible_modalities": [
                        "rna",
                        "chrom_H3K4me1",
                        "chrom_H3K4me2",
                    ],
                }
            }
        )
    )
    assert axis_eligible_modalities_from_prepared(prepared) == (
        "chrom_H3K4me1",
        "chrom_H3K4me2",
        "rna",
    )


def test_attainable_gate_uses_latent_axes_and_rejects_mismatched_claims() -> None:
    pair = {
        "direct_class_success": 1,
        "collider_class_success": 1,
        "shared_class_success": 0,
        "slow_class_success": 0,
        "null_class_success": 1,
        "density_class_success": 1,
        "precision_class_success": 1,
        "world_success": 0,
    }
    latent = {
        "combined_within_assay_ps_group_recall": 5 / 6,
        "combined_cross_assay_latent_axis_coverage": 1.0,
        "combined_slow_gene_axis_support": 1,
        "n_selected_axis_mismatch_claims": 0,
        "n_selected_wrong_layer_claims": 0,
    }

    passing = combine_attainable_metrics(pair, latent)
    assert passing["latent_state_class_success"] == 1
    assert passing["shared_class_success"] == 1
    assert passing["slow_class_success"] == 1
    assert passing["world_success"] == 1
    assert passing["exact_pair_shared_class_success"] == 0

    failing = combine_attainable_metrics(
        pair,
        {**latent, "n_selected_axis_mismatch_claims": 1},
    )
    assert failing["latent_state_class_success"] == 0
    assert failing["world_success"] == 0

    imprecise = combine_attainable_metrics(
        {**pair, "precision_class_success": 0},
        latent,
    )
    assert imprecise["world_success"] == 0


def test_beta_votes_use_only_discovery_inputs_and_deterministic_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = tmp_path / "world_seed30001" / "discovery"
    discovery.mkdir(parents=True)
    tables: list[Path] = []
    for assay_i in range(6):
        table = discovery / f"observed_focal__chrom_MARK{assay_i}.tsv"
        pd.DataFrame(
            {
                "cell": ["c1", "c2"],
                "observed_background_library_size": [10, 11],
                f"chr{assay_i + 1}:1-2__MARK{assay_i}": [0, 1],
                f"chr{assay_i + 1}:3-4__MARK{assay_i}": [1, 0],
            }
        ).to_csv(table, sep="\t", index=False)
        tables.append(table)
    prepared = tmp_path / "benchmark_output" / "prepared_discovery"
    prepared.mkdir(parents=True)
    embedding = prepared / "analysis_embeddings.tsv"
    pd.DataFrame(
        {
            "cell": ["c1", "c2"],
            "integration__dim0": [0.0, 1.0],
        }
    ).to_csv(embedding, sep="\t", index=False)

    calls: list[dict[str, object]] = []

    def fake_estimator(table, embedding_table, *, feature_columns, config):
        calls.append(
            {
                "table": Path(table),
                "embedding": Path(embedding_table),
                "features": tuple(feature_columns),
                "seed": config.random_state,
                "permutations": config.n_permutations,
            }
        )
        return SimpleNamespace(
            edge_evidence=pd.DataFrame(
                [
                    {
                        "node1": feature_columns[0],
                        "node2": feature_columns[1],
                        "selected": True,
                        "empirical_p_positive": 0.01,
                        "pooled_log_odds": 1.2,
                    }
                ]
            ),
            metadata={"truth_pairing_or_external_inputs": False},
        )

    monkeypatch.setattr(
        benchmark,
        "estimate_beta_binomial_from_observed_tables",
        fake_estimator,
    )
    combined, vote_frames, vote_metadata = fit_collider_motif_vote_evidence(
        tables,
        embedding,
    )

    assert len(calls) == COLLIDER_MOTIF_VOTES * 6
    assert len(combined) == COLLIDER_MOTIF_VOTES * 6
    assert set(vote_frames) == set(range(COLLIDER_MOTIF_VOTES))
    assert set(vote_metadata) == set(range(COLLIDER_MOTIF_VOTES))
    for vote in range(COLLIDER_MOTIF_VOTES):
        expected = {
            vote + COLLIDER_MOTIF_SEED_STRIDE * (assay_i + 1)
            for assay_i in range(6)
        }
        observed = {
            int(call["seed"])
            for call in calls[vote * 6 : (vote + 1) * 6]
        }
        assert observed == expected
    assert all("discovery" in Path(call["table"]).parts for call in calls)
    assert all(Path(call["embedding"]) == embedding for call in calls)
    assert all(int(call["permutations"]) == 399 for call in calls)
    assert all(
        all(str(feature).startswith("chr") for feature in call["features"])
        for call in calls
    )

    hashes = write_collider_motif_vote_evidence(
        tmp_path / "world_output",
        vote_frames,
        vote_metadata,
    )
    assert len(hashes) == 2 * COLLIDER_MOTIF_VOTES
    assert all("collider_motif_evidence/" in filename for filename in hashes)

    forbidden_embedding = tmp_path / "world_seed30001" / "scorer" / "truth_embedding.tsv"
    with pytest.raises(ValueError, match="discovery artifacts only"):
        fit_collider_motif_vote_evidence(tables, forbidden_embedding)


def test_direct_layer_is_replaced_and_demotions_remain_unresolved() -> None:
    preliminary = pd.DataFrame(
        [
            {
                "node1": "chr1:a__H3K27ac",
                "endpoint1": "CIRCLE",
                "node2": "chr1:c__H3K27ac",
                "endpoint2": "CIRCLE",
                "edge_type": "within_assay_direct_candidate",
                "selection_count": 6,
                "selection_frequency": 1.0,
                "n_nested_runs": 6,
            },
            {
                "node1": "chr1:p__H3K27ac",
                "endpoint1": "CIRCLE",
                "node2": "chr1:q__H3K27ac",
                "endpoint2": "CIRCLE",
                "edge_type": "within_assay_direct_candidate",
                "selection_count": 4,
                "selection_frequency": 4 / 6,
                "n_nested_runs": 6,
            },
            {
                "node1": "chr1:x__H3K27ac",
                "endpoint1": "CIRCLE",
                "node2": "chr1:y__H3K27ac",
                "endpoint2": "CIRCLE",
                "edge_type": "within_assay_shared_state",
                "selection_count": 5,
                "selection_frequency": 5 / 6,
                "n_nested_runs": 6,
            },
        ]
    )
    motif = SimpleNamespace(
        direct_candidates=pd.DataFrame(
            [
                {
                    "node1": "chr1:a__H3K27ac",
                    "endpoint1": "TAIL",
                    "node2": "chr1:c__H3K27ac",
                    "endpoint2": "ARROW",
                    "edge_type": "within_assay_direct_candidate",
                    "orientation_source": "unique_positive_residual_wedge",
                    "collider_center": "chr1:c__H3K27ac",
                    "component_id": "component0",
                    "wedge_score": 1.5,
                    "wedge_margin": 0.4,
                    "selected_vote_fraction": 1.0,
                    "mean_empirical_p_positive": 0.01,
                    "mean_pooled_log_odds": 1.7,
                },
                {
                    "node1": "chr1:b__H3K27ac",
                    "endpoint1": "TAIL",
                    "node2": "chr1:c__H3K27ac",
                    "endpoint2": "ARROW",
                    "edge_type": "within_assay_direct_candidate",
                    "orientation_source": "unique_positive_residual_wedge",
                    "collider_center": "chr1:c__H3K27ac",
                    "component_id": "component0",
                    "wedge_score": 1.5,
                    "wedge_margin": 0.4,
                    "selected_vote_fraction": 1.0,
                    "mean_empirical_p_positive": 0.01,
                    "mean_pooled_log_odds": 1.5,
                },
            ]
        ),
        demoted_or_unresolved=pd.DataFrame(
            [
                {
                    "node1": "chr1:p__H3K27ac",
                    "node2": "chr1:q__H3K27ac",
                    "component_id": "component1",
                    "reason": "component_not_identifiable",
                    "was_base_selected": True,
                    "was_rescue_selected": True,
                    "mean_empirical_p_positive": 0.03,
                    "mean_pooled_log_odds": 1.1,
                },
                {
                    "node1": "chr1:null1__H3K27ac",
                    "node2": "chr1:null2__H3K27ac",
                    "component_id": "component2",
                    "reason": "component_not_identifiable",
                    "was_base_selected": False,
                    "was_rescue_selected": True,
                    "mean_empirical_p_positive": 0.01,
                    "mean_pooled_log_odds": 1.3,
                }
            ]
        ),
    )

    resolved = replace_direct_layer_with_collider_motifs(preliminary, motif)
    direct = resolved.loc[
        resolved["edge_type"].eq("within_assay_direct_candidate")
    ]
    unresolved = resolved.loc[resolved["edge_type"].eq("unresolved")]
    shared = resolved.loc[resolved["edge_type"].eq("within_assay_shared_state")]
    assert len(direct) == 2
    assert len(unresolved) == 1
    assert len(shared) == 1
    assert set(direct_collider_children(resolved)) == {"chr1:c__H3K27ac"}
    assert unresolved.iloc[0]["endpoint1"] == "CIRCLE"
    assert unresolved.iloc[0]["endpoint2"] == "CIRCLE"
    assert unresolved.iloc[0]["collider_motif_status"] == "demoted_unresolved"
    assert not resolved["node1"].astype(str).str.contains("null").any()
    assert not resolved["node2"].astype(str).str.contains("null").any()


def test_motif_resolution_precedes_children_and_loading_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    preliminary = pd.DataFrame(
        [
            {
                "node1": "chr1:a__H3K27ac",
                "endpoint1": "CIRCLE",
                "node2": "chr1:c__H3K27ac",
                "endpoint2": "CIRCLE",
                "edge_type": "within_assay_direct_candidate",
            }
        ]
    )
    motif = SimpleNamespace(
        direct_candidates=pd.DataFrame(),
        demoted_or_unresolved=pd.DataFrame(),
        component_diagnostics=pd.DataFrame(),
        wedge_diagnostics=pd.DataFrame(),
        aggregated_evidence=pd.DataFrame(),
        metadata={},
    )
    refinement = SimpleNamespace(decisions=pd.DataFrame())

    def fake_selector(base, evidence, config):
        calls.append("motif")
        assert set(base["source_assay"]) == {"chrom_H3K27ac"}
        return motif

    def fake_replace(frame, result):
        calls.append("replace_direct")
        assert result is motif
        return frame

    def fake_refinement_selector(
        components,
        wedges,
        unresolved,
        evidence,
        base_views,
        residual_views,
        feature_views,
        config,
        *,
        loading_pair_views=None,
        state_membership=None,
    ):
        calls.append("refine")
        assert set(base_views) == {"view0"}
        assert set(residual_views) == {"view0"}
        assert set(feature_views) == {"view0"}
        assert loading_pair_views is None
        assert state_membership is None
        return refinement

    def fake_refinement_apply(result, observed):
        calls.append("apply_refinement")
        assert result is motif
        assert observed is refinement
        return motif

    def fake_children(frame):
        calls.append("derive_children")
        return ("chr1:c__H3K27ac",)

    projection = SimpleNamespace(
        sparse_pair_projection=pd.DataFrame(
            [
                {
                    "node1": "chr1:p__H3K27ac",
                    "node2": "chr1:s__H3K27ac",
                    "edge_type": "within_assay_shared_state",
                },
                {
                    "node1": "chr1:p__H3K27ac",
                    "node2": "chr2:s__H3K4me1",
                    "edge_type": "cross_assay_shared_state",
                },
            ]
        ),
        groups=pd.DataFrame(columns=["loading_group", "modality"]),
        group_membership=pd.DataFrame(columns=["loading_group", "feature"]),
    )
    axis_projection = SimpleNamespace(
        typed_axis_hyperedges=pd.DataFrame(),
        provenance={},
    )

    def fake_projection(
        views, config, *, excluded_features, forbidden_within_pairs
    ):
        calls.append("loading_groups")
        assert excluded_features == ("chr1:c__H3K27ac",)
        assert forbidden_within_pairs == ()
        assert config.min_within_absolute_cosine == pytest.approx(0.70)
        assert config.min_group_relative_margin == pytest.approx(0.10)
        return projection

    def fake_state_replacement(frame, sparse_projection):
        calls.append("replace_state")
        assert sparse_projection["edge_type"].tolist() == [
            "within_assay_shared_state"
        ]
        return frame

    def fake_axes(groups, membership, views, config):
        calls.append("latent_axes")
        assert groups.equals(projection.groups)
        assert membership.equals(projection.group_membership)
        return axis_projection

    monkeypatch.setattr(benchmark, "select_unique_collider_motifs", fake_selector)
    monkeypatch.setattr(
        benchmark,
        "select_observed_collider_refinements",
        fake_refinement_selector,
    )
    monkeypatch.setattr(
        benchmark,
        "apply_observed_collider_refinements",
        fake_refinement_apply,
    )
    monkeypatch.setattr(
        benchmark,
        "replace_direct_layer_with_collider_motifs",
        fake_replace,
    )
    monkeypatch.setattr(benchmark, "direct_collider_children", fake_children)
    monkeypatch.setattr(
        benchmark,
        "estimate_loading_group_projection",
        fake_projection,
    )
    monkeypatch.setattr(
        benchmark,
        "replace_pairwise_state_layer",
        fake_state_replacement,
    )
    monkeypatch.setattr(
        benchmark,
        "estimate_latent_axis_clustering",
        fake_axes,
    )

    (
        consensus,
        observed_motif,
        observed_projection,
        observed_axes,
        observed_refinement,
        observed_structure_anchor,
        observed_crossfit_collider,
    ) = resolve_postconsensus_candidate(
        preliminary,
        pd.DataFrame(),
        [],
        CABossV4Config(),
        {},
        base_edge_views={"view0": pd.DataFrame()},
        residual_views={"view0": pd.DataFrame()},
        feature_model_views={"view0": pd.DataFrame()},
    )
    assert consensus.equals(preliminary)
    assert observed_motif is motif
    assert observed_projection is projection
    assert observed_axes is axis_projection
    assert observed_refinement is refinement
    assert observed_structure_anchor is None
    assert observed_crossfit_collider is None
    assert calls == [
        "motif",
        "refine",
        "apply_refinement",
        "replace_direct",
        "derive_children",
        "loading_groups",
        "latent_axes",
        "replace_state",
    ]


def test_frozen_layered_loading_config_reaches_production_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preliminary = pd.DataFrame(
        [
            {
                "node1": "p",
                "endpoint1": "TAIL",
                "node2": "c",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
            }
        ]
    )
    motif = SimpleNamespace(
        direct_candidates=pd.DataFrame(),
        demoted_or_unresolved=pd.DataFrame(),
        component_diagnostics=pd.DataFrame(),
        wedge_diagnostics=pd.DataFrame(),
        aggregated_evidence=pd.DataFrame(),
        metadata={},
    )
    projection = SimpleNamespace(
        within_group_candidates=pd.DataFrame(),
        sparse_pair_projection=pd.DataFrame(),
        groups=pd.DataFrame(columns=["loading_group", "modality"]),
        group_membership=pd.DataFrame(columns=["loading_group", "feature"]),
    )
    axis_projection = SimpleNamespace(
        typed_axis_hyperedges=pd.DataFrame(), provenance={}
    )
    structure_anchor = SimpleNamespace(
        typed_groups=pd.DataFrame(columns=["members"])
    )
    observed_configs = []

    monkeypatch.setattr(
        benchmark, "select_unique_collider_motifs", lambda *args: motif
    )
    monkeypatch.setattr(
        benchmark,
        "replace_direct_layer_with_collider_motifs",
        lambda frame, result: frame,
    )
    monkeypatch.setattr(benchmark, "state_group_exclusions", lambda *args: ())
    monkeypatch.setattr(
        benchmark, "direct_collider_parent_pairs", lambda *args: ()
    )
    monkeypatch.setattr(
        benchmark,
        "estimate_loading_group_projection",
        lambda *args, **kwargs: projection,
    )

    def fake_structure_selector(
        consensus, candidates, empirical_null_views, state_membership, config
    ):
        observed_configs.append(config)
        return structure_anchor

    monkeypatch.setattr(
        benchmark, "estimate_structure_anchored_groups", fake_structure_selector
    )
    monkeypatch.setattr(
        benchmark,
        "estimate_latent_axis_clustering",
        lambda *args, **kwargs: axis_projection,
    )
    monkeypatch.setattr(
        benchmark, "replace_pairwise_state_layer", lambda frame, sparse: frame
    )
    loading_view = SimpleNamespace(
        view_id="view0",
        pair_evidence=pd.DataFrame(
            columns=[
                "pair_key",
                "absolute_loading_cosine",
                "loading_empirical_null_p",
            ]
        ),
    )

    resolved = resolve_postconsensus_candidate(
        preliminary,
        pd.DataFrame(),
        [loading_view],
        CABossV4Config(),
        {},
        state_membership=pd.DataFrame({"feature": ["s"]}),
    )

    assert resolved[-2] is structure_anchor
    assert resolved[-1] is None
    assert observed_configs == [benchmark.FROZEN_STRUCTURE_ANCHORED_GROUP_CONFIG]
    frozen = asdict(observed_configs[0])
    assert frozen["rescue_empirical_null_p_threshold"] == 0.125
    assert frozen["tiered_rescue_empirical_null_p_threshold"] == 0.50
    assert frozen["tiered_rescue_required_view_count"] == 6


def test_crossfit_collider_is_applied_before_direct_and_loading_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    preliminary = pd.DataFrame(
        [
            {
                "node1": "p__H3K27ac",
                "endpoint1": "TAIL",
                "node2": "c__H3K27ac",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
            }
        ]
    )
    initial = benchmark.ColliderMotifResult(
        direct_candidates=pd.DataFrame(
            {
                "source_assay": ["chrom_H3K27ac"],
                "node1": ["p__H3K27ac"],
                "node2": ["c__H3K27ac"],
            }
        ),
        demoted_or_unresolved=pd.DataFrame(),
        component_diagnostics=pd.DataFrame(),
        wedge_diagnostics=pd.DataFrame(),
        aggregated_evidence=pd.DataFrame(),
        metadata={"selector": "initial"},
    )
    revised_direct = pd.DataFrame(
        {
            "source_assay": ["chrom_H3K27ac"],
            "node1": ["q__H3K27ac"],
            "node2": ["c__H3K27ac"],
        }
    )
    crossfit_result = SimpleNamespace(
        candidates=pd.DataFrame(),
        decisions=pd.DataFrame(),
        direct_candidates=revised_direct,
        demoted_or_unresolved=pd.DataFrame(),
        provenance={
            "schema_version": "crossfit_hierarchical_collider.v4",
            "n_selected_replacements": 1,
        },
    )
    refinement = SimpleNamespace()
    projection = SimpleNamespace(
        within_group_candidates=pd.DataFrame(),
        sparse_pair_projection=pd.DataFrame(columns=["edge_type"]),
        groups=pd.DataFrame(columns=["loading_group", "modality"]),
        group_membership=pd.DataFrame(columns=["loading_group", "feature"]),
    )
    axes = SimpleNamespace(typed_axis_hyperedges=pd.DataFrame(), provenance={})
    anchor = SimpleNamespace(typed_groups=pd.DataFrame(columns=["members"]))
    loading_view = SimpleNamespace(
        view_id="view0", pair_evidence=pd.DataFrame()
    )
    state = pd.DataFrame({"feature": ["state_feature"]})
    world_evidence = object()

    monkeypatch.setattr(
        benchmark,
        "select_unique_collider_motifs",
        lambda *args: calls.append("motif") or initial,
    )
    monkeypatch.setattr(
        benchmark,
        "select_observed_collider_refinements",
        lambda *args, **kwargs: calls.append("refine") or refinement,
    )
    monkeypatch.setattr(
        benchmark,
        "apply_observed_collider_refinements",
        lambda *args: calls.append("apply_refinement") or initial,
    )

    def fake_crossfit(**kwargs: object) -> object:
        calls.append("crossfit")
        assert kwargs["world_evidence"] is world_evidence
        assert kwargs["state_membership"] is state
        assert kwargs["current_direct_candidates"] is initial.direct_candidates
        assert kwargs["config"] is benchmark.FROZEN_CROSSFIT_HIERARCHICAL_COLLIDER_CONFIG
        return crossfit_result

    monkeypatch.setattr(
        benchmark, "resolve_crossfit_hierarchical_colliders_v4", fake_crossfit
    )

    def fake_replace(frame: pd.DataFrame, motif: object) -> pd.DataFrame:
        calls.append("replace_direct")
        assert motif.direct_candidates.equals(revised_direct)
        return frame

    monkeypatch.setattr(
        benchmark, "replace_direct_layer_with_collider_motifs", fake_replace
    )
    monkeypatch.setattr(benchmark, "state_group_exclusions", lambda *args: ())
    monkeypatch.setattr(
        benchmark, "direct_collider_parent_pairs", lambda *args: ()
    )
    monkeypatch.setattr(
        benchmark,
        "estimate_loading_group_projection",
        lambda *args, **kwargs: calls.append("loading_groups") or projection,
    )
    monkeypatch.setattr(
        benchmark,
        "estimate_structure_anchored_groups",
        lambda *args, **kwargs: calls.append("structure_anchor") or anchor,
    )
    monkeypatch.setattr(
        benchmark,
        "estimate_latent_axis_clustering",
        lambda *args, **kwargs: calls.append("latent_axes") or axes,
    )
    monkeypatch.setattr(
        benchmark,
        "replace_pairwise_state_layer",
        lambda frame, sparse: calls.append("replace_state") or frame,
    )

    resolved = resolve_postconsensus_candidate(
        preliminary,
        pd.DataFrame(),
        [loading_view],
        CABossV4Config(),
        {},
        base_edge_views={"view0": pd.DataFrame()},
        residual_views={"view0": pd.DataFrame()},
        feature_model_views={"view0": pd.DataFrame()},
        state_membership=state,
        crossfit_world_evidence=world_evidence,
    )

    assert resolved[-1] is crossfit_result
    assert calls.index("crossfit") > calls.index("apply_refinement")
    assert calls.index("crossfit") < calls.index("replace_direct")
    assert calls.index("crossfit") < calls.index("loading_groups")


def test_state_group_exclusions_are_limited_to_selected_collider_children() -> None:
    resolved = pd.DataFrame(
        [
            {
                "node1": "p",
                "endpoint1": "TAIL",
                "node2": "c",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
            },
            {
                "node1": "q",
                "endpoint1": "TAIL",
                "node2": "c",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
            },
        ]
    )
    unresolved = pd.DataFrame(
        [
            {
                "component_id": "ambiguous",
                "node1": "a",
                "node2": "b",
                "reason": "component_not_identifiable",
                "was_base_selected": False,
            },
            {
                "component_id": "ambiguous",
                "node1": "a",
                "node2": "d",
                "reason": "component_not_identifiable",
                "was_base_selected": False,
            },
            {
                "component_id": "ambiguous",
                "node1": "b",
                "node2": "d",
                "reason": "component_not_identifiable",
                "was_base_selected": False,
            },
            {
                "component_id": "isolated_rescue",
                "node1": "null",
                "node2": "signal",
                "reason": "component_not_identifiable",
                "was_base_selected": False,
            },
            {
                "component_id": "base_demoted",
                "node1": "u",
                "node2": "v",
                "reason": "weak_unique_wedge",
                "was_base_selected": True,
            },
            {
                "component_id": "child_link",
                "node1": "c",
                "node2": "state_sibling",
                "reason": "weak_unique_wedge",
                "was_base_selected": True,
            },
        ]
    )
    motif = SimpleNamespace(demoted_or_unresolved=unresolved)

    observed = set(benchmark.state_group_exclusions(resolved, motif))

    assert observed == {"c"}
    assert "null" not in observed
    assert "signal" not in observed
    assert "state_sibling" not in observed


def test_direct_collider_parent_pair_is_an_observable_pair_constraint() -> None:
    direct = pd.DataFrame(
        [
            {
                "source_assay": "chrom_A",
                "node1": "parent1",
                "node2": "child",
                "collider_center": "child",
            },
            {
                "source_assay": "chrom_A",
                "node1": "child",
                "node2": "parent2",
                "collider_center": "child",
            },
            {
                "source_assay": "chrom_B",
                "node1": "other_parent1",
                "node2": "other_child",
                "collider_center": "other_child",
            },
            {
                "source_assay": "chrom_B",
                "node1": "other_parent2",
                "node2": "other_child",
                "collider_center": "other_child",
            },
            {
                "source_assay": "chrom_C",
                "node1": "unoriented_left",
                "endpoint1": "CIRCLE",
                "node2": "unoriented_right",
                "endpoint2": "CIRCLE",
                "collider_center": "",
            },
        ]
    )
    motif = SimpleNamespace(direct_candidates=direct)

    assert benchmark.direct_collider_parent_pairs(motif) == (
        ("other_parent1", "other_parent2"),
        ("parent1", "parent2"),
    )


def test_world_input_binding_rejects_stale_tampered_and_swapped_worlds(
    tmp_path: Path,
) -> None:
    first = _write_bound_world(tmp_path / "valid", 30001)
    binding = benchmark.validate_world_input_binding(
        first,
        verify_scorer_payload=True,
    )
    assert binding["seed"] == 30001
    assert binding["run_summary"]["schema_version"] == benchmark.RUN_SUMMARY_SCHEMA_VERSION
    assert binding["discovery_manifest"]["payloads_content_addressed_verified"]
    assert binding["scorer_manifest"]["payloads_content_addressed_verified"]
    payload = dict(binding)
    claimed = payload.pop("binding_payload_sha256")
    assert claimed == benchmark._canonical_sha256(payload)

    stale = _write_bound_world(tmp_path / "stale", 30002)
    summary_path = stale / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "multimodal_simulation_v4.run.v1"
    summary_path.write_text(json.dumps(summary, sort_keys=True))
    with pytest.raises(ValueError, match="stale or unsupported run-summary"):
        benchmark.validate_world_input_binding(stale, verify_scorer_payload=False)

    stale_discovery = _write_bound_world(tmp_path / "stale_discovery", 30007)
    discovery_manifest_path = stale_discovery / "discovery" / "manifest.json"
    discovery_manifest = json.loads(discovery_manifest_path.read_text())
    discovery_manifest["schema_version"] = "multimodal_simulation_v4.discovery.v1"
    discovery_manifest_path.write_text(json.dumps(discovery_manifest, sort_keys=True))
    stale_summary_path = stale_discovery / "run_summary.json"
    stale_summary = json.loads(stale_summary_path.read_text())
    stale_summary["discovery_manifest_sha256"] = benchmark._file_sha256(
        discovery_manifest_path
    )
    stale_summary_path.write_text(json.dumps(stale_summary, sort_keys=True))
    with pytest.raises(ValueError, match="stale or unsupported discovery"):
        benchmark.validate_world_input_binding(
            stale_discovery,
            verify_scorer_payload=False,
        )

    tampered = _write_bound_world(tmp_path / "tampered", 30003)
    (tampered / "discovery" / "observed.tsv").write_text("seed\n99999\n")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        benchmark.validate_world_input_binding(tampered, verify_scorer_payload=False)

    swap_root = tmp_path / "swapped"
    left = _write_bound_world(swap_root, 30004)
    right = _write_bound_world(swap_root, 30005)
    temporary = swap_root / "discovery_swap"
    (left / "discovery").rename(temporary)
    (right / "discovery").rename(left / "discovery")
    temporary.rename(right / "discovery")
    with pytest.raises(ValueError, match="not bound to its run summary"):
        benchmark.validate_world_input_binding(left, verify_scorer_payload=False)

    scorer_tampered = _write_bound_world(tmp_path / "scorer_tampered", 30006)
    (scorer_tampered / "scorer" / "world_parameters.json").write_text(
        '{"changed": true}\n'
    )
    # The pretruth phase deliberately leaves scorer bytes unopened.
    benchmark.validate_world_input_binding(
        scorer_tampered,
        verify_scorer_payload=False,
    )
    with pytest.raises(ValueError, match="payload hash mismatch"):
        benchmark.validate_world_input_binding(
            scorer_tampered,
            verify_scorer_payload=True,
        )


def test_scorer_access_requires_complete_pretruth_motif_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _write_bound_world(tmp_path, 30001)
    world_out = tmp_path / "output" / world.name
    world_out.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="before pretruth manifest"):
        load_scorer_artifacts_after_pretruth_freeze(world, world_out)

    incomplete = {
        "all_consensuses_built_before_truth_open": True,
        "consensus_sha256": {"candidate": "abc"},
    }
    (world_out / "pretruth_consensus_manifest.json").write_text(
        json.dumps(incomplete)
    )
    with pytest.raises(RuntimeError, match="complete pretruth freeze"):
        load_scorer_artifacts_after_pretruth_freeze(world, world_out)

    complete = {
        **incomplete,
        "all_collider_motif_outputs_built_before_truth_open": True,
        "collider_motif_discovery_inputs_only": True,
        "collider_motif_applied_after_preliminary_nested_consensus": True,
        "collider_motif_applied_before_direct_collider_children": True,
        "collider_motif_applied_before_loading_group_projection": True,
        "collider_motif_base_demotions_retained_as_unresolved": True,
        "observed_collider_rescue_only_completion_requires_state_loading_stability": True,
        "collider_motif_beta_vote_sha256": {"vote0": "def"},
        "collider_motif_result_sha256": {"candidate": "ghi"},
        "crossfit_hierarchical_collider_discovery_inputs_only": True,
        "crossfit_hierarchical_collider_truth_pairing_external_inputs": False,
        "crossfit_hierarchical_collider_paired_cells_used": False,
        "crossfit_hierarchical_collider_cross_assay_coordinate_registration_required": False,
        "latent_axis_hyperedges_built_before_truth_open": True,
        "latent_axis_hyperedges_attached_once_post_consensus": True,
        "ordinary_state_skeleton_contains_within_assay_groups_only": True,
        "pairwise_cross_group_matches_are_diagnostics_only": True,
        "loading_views_reuse_same_observed_cells": True,
        "loading_views_are_algorithmic_stability_repeats": True,
        "loading_views_are_independent_worlds": False,
        "loading_view_support_not_world_sample_size": True,
        "latent_axis_projection_sha256": {"candidate": "jkl"},
        "world_input_discovery_payloads_content_addressed": True,
        "world_input_pretruth_binding": benchmark.validate_world_input_binding(
            world,
            verify_scorer_payload=False,
        ),
    }
    (world_out / "pretruth_consensus_manifest.json").write_text(
        json.dumps(complete)
    )
    pre_scorer_order: list[str] = []
    original_crossfit_check = (
        benchmark._validate_crossfit_world_evidence_freeze
    )
    original_binding_check = benchmark.validate_world_input_binding

    def record_crossfit_check(*args: object, **kwargs: object) -> None:
        pre_scorer_order.append("crossfit_world_freeze")
        original_crossfit_check(*args, **kwargs)

    def record_binding_check(*args: object, **kwargs: object) -> object:
        pre_scorer_order.append("scorer_binding")
        return original_binding_check(*args, **kwargs)

    monkeypatch.setattr(
        benchmark,
        "_validate_crossfit_world_evidence_freeze",
        record_crossfit_check,
    )
    monkeypatch.setattr(
        benchmark, "validate_world_input_binding", record_binding_check
    )
    truth_path = world / "scorer" / "truth_typed_edges.tsv"
    original_read_csv = benchmark.pd.read_csv
    parse_started = False

    def mutate_truth_when_parse_starts(source: object, *args: object, **kwargs: object):
        nonlocal parse_started
        if not parse_started:
            parse_started = True
            truth_path.write_text("mutated_during_parse\n")
        assert not isinstance(source, (str, Path))
        return original_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(benchmark.pd, "read_csv", mutate_truth_when_parse_starts)
    truth, parameters, roles, recorded_binding = load_scorer_artifacts_after_pretruth_freeze(
        world,
        world_out,
        return_binding=True,
    )
    assert len(truth) == 1
    assert parameters == {}
    assert roles.loc[0, "role"] == "P"
    assert truth_path.read_text() == "mutated_during_parse\n"
    assert not (world_out / "world_input_binding.json").exists()
    assert recorded_binding["scorer_manifest"][
        "payloads_content_addressed_verified"
    ]
    assert pre_scorer_order[:2] == [
        "crossfit_world_freeze",
        "scorer_binding",
    ]


def _write_minimal_frozen_candidate(
    tmp_path: Path,
) -> tuple[benchmark.FrozenWorldPretruth, Path]:
    world = _write_bound_world(tmp_path / "inputs", 30001)
    world_out = tmp_path / "outputs" / world.name
    candidate = "frozen_candidate"
    candidate_out = world_out / candidate
    group_out = candidate_out / "loading_group_projection"
    axis_out = candidate_out / "latent_axis_projection"
    group_out.mkdir(parents=True)
    axis_out.mkdir()
    consensus = pd.DataFrame(
        [
            {
                "node1": "p__M",
                "endpoint1": "CIRCLE",
                "node2": "c__M",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
            }
        ]
    )
    group_hyperedges = pd.DataFrame(
        columns=["hyperedge_id", "edge_type", "group1", "group2"]
    )
    group_sparse = pd.DataFrame(columns=["node1", "node2", "edge_type"])
    axis_hyperedges = pd.DataFrame(
        columns=["axis_id", "edge_type", "members", "modalities"]
    )
    consensus_path = candidate_out / "consensus_edges.tsv"
    group_hyperedges_path = group_out / "typed_group_hyperedges.tsv"
    group_sparse_path = group_out / "sparse_pair_projection.tsv"
    axis_hyperedges_path = axis_out / "typed_axis_hyperedges.tsv"
    consensus.to_csv(consensus_path, sep="\t", index=False)
    group_hyperedges.to_csv(group_hyperedges_path, sep="\t", index=False)
    group_sparse.to_csv(group_sparse_path, sep="\t", index=False)
    axis_hyperedges.to_csv(axis_hyperedges_path, sep="\t", index=False)
    manifest = {
        "candidate_count": 1,
        "all_consensuses_built_before_truth_open": True,
        "consensus_sha256": {candidate: "logical-consensus-hash"},
        "consensus_file_sha256": {
            candidate: benchmark._file_sha256(consensus_path)
        },
        "loading_group_projection_sha256": {
            candidate: "logical-group-hash"
        },
        "loading_group_projection_file_sha256": {
            candidate: {
                path.name: benchmark._file_sha256(path)
                for path in sorted(group_out.iterdir())
            }
        },
        "latent_axis_projection_file_sha256": {
            candidate: {
                path.name: benchmark._file_sha256(path)
                for path in sorted(axis_out.iterdir())
            }
        },
    }
    manifest_path = world_out / "pretruth_consensus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    frozen = benchmark.FrozenWorldPretruth(
        world_dir=str(world.resolve()),
        world_name=world.name,
        world_out=str(world_out.resolve()),
        candidate_names=(candidate,),
        pretruth_manifest=str(manifest_path.resolve()),
        pretruth_manifest_sha256=benchmark._file_sha256(manifest_path),
    )
    return frozen, consensus_path


def test_frozen_scoring_inputs_are_content_addressed_before_truth(
    tmp_path: Path,
) -> None:
    frozen, consensus_path = _write_minimal_frozen_candidate(tmp_path)

    _, _, loaded = benchmark._load_frozen_scoring_inputs(frozen)
    assert list(loaded) == ["frozen_candidate"]
    assert len(loaded["frozen_candidate"][0]) == 1

    consensus_path.write_text(consensus_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="consensus changed"):
        benchmark._load_frozen_scoring_inputs(frozen)


def test_frozen_scoring_rejects_crossfit_world_and_candidate_tampering(
    tmp_path: Path,
) -> None:
    frozen, _ = _write_minimal_frozen_candidate(tmp_path)
    world_out = Path(frozen.world_out)
    candidate = frozen.candidate_names[0]
    world_crossfit = world_out / "crossfit_hierarchical_collider"
    candidate_crossfit = (
        world_out / candidate / "crossfit_hierarchical_collider"
    )
    world_crossfit.mkdir()
    candidate_crossfit.mkdir()
    world_result_hash = "a" * 64
    candidate_result_hash = "b" * 64
    world_files = {
        "world_generative_candidates.tsv": "source_assay\nchrom_A\n",
        "world_evidence_provenance.json": json.dumps(
            {
                "world_evidence_result_sha256": world_result_hash,
                "randomization_key": "30001",
            }
        ),
    }
    candidate_files = {
        "candidate_evidence.tsv": "source_assay\nchrom_A\n",
        "assay_decisions.tsv": "source_assay\nchrom_A\n",
        "direct_candidates.tsv": "source_assay\nchrom_A\n",
        "demoted_or_unresolved.tsv": "source_assay\nchrom_A\n",
        "provenance.json": json.dumps(
            {
                "result_sha256": candidate_result_hash,
                "world_evidence_result_sha256": world_result_hash,
            }
        ),
    }
    for filename, payload in world_files.items():
        (world_crossfit / filename).write_text(payload)
    for filename, payload in candidate_files.items():
        (candidate_crossfit / filename).write_text(payload)

    manifest_path = Path(frozen.pretruth_manifest)
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "crossfit_hierarchical_collider_applied": True,
            "crossfit_hierarchical_collider_randomization_key": 30001,
            "crossfit_hierarchical_collider_randomization_key_bound_to_world_input_seed": True,
            "world_input_pretruth_binding": {"seed": 30001},
            "crossfit_hierarchical_collider_world_result_sha256": (
                world_result_hash
            ),
            "crossfit_hierarchical_collider_world_file_sha256": {
                path.name: benchmark._file_sha256(path)
                for path in sorted(world_crossfit.iterdir())
            },
            "crossfit_hierarchical_collider_result_sha256": {
                candidate: candidate_result_hash
            },
            "crossfit_hierarchical_collider_file_sha256": {
                candidate: {
                    path.name: benchmark._file_sha256(path)
                    for path in sorted(candidate_crossfit.iterdir())
                }
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    frozen = benchmark.FrozenWorldPretruth(
        world_dir=frozen.world_dir,
        world_name=frozen.world_name,
        world_out=frozen.world_out,
        candidate_names=frozen.candidate_names,
        pretruth_manifest=frozen.pretruth_manifest,
        pretruth_manifest_sha256=benchmark._file_sha256(manifest_path),
    )

    benchmark._load_frozen_scoring_inputs(frozen)
    candidate_extra = candidate_crossfit / "unexpected_directory"
    candidate_extra.mkdir()
    with pytest.raises(RuntimeError, match="unsafe entry set"):
        benchmark._load_frozen_scoring_inputs(frozen)
    candidate_extra.rmdir()

    float_manifest = dict(manifest)
    float_manifest["crossfit_hierarchical_collider_randomization_key"] = 30001.5
    manifest_path.write_text(json.dumps(float_manifest, sort_keys=True))
    float_frozen = benchmark.FrozenWorldPretruth(
        world_dir=frozen.world_dir,
        world_name=frozen.world_name,
        world_out=frozen.world_out,
        candidate_names=frozen.candidate_names,
        pretruth_manifest=frozen.pretruth_manifest,
        pretruth_manifest_sha256=benchmark._file_sha256(manifest_path),
    )
    with pytest.raises(RuntimeError, match="randomization binding is malformed"):
        benchmark._load_frozen_scoring_inputs(float_frozen)
    leading_zero_manifest = dict(manifest)
    leading_zero_manifest[
        "crossfit_hierarchical_collider_randomization_key"
    ] = "030001"
    manifest_path.write_text(json.dumps(leading_zero_manifest, sort_keys=True))
    leading_zero_frozen = benchmark.FrozenWorldPretruth(
        world_dir=frozen.world_dir,
        world_name=frozen.world_name,
        world_out=frozen.world_out,
        candidate_names=frozen.candidate_names,
        pretruth_manifest=frozen.pretruth_manifest,
        pretruth_manifest_sha256=benchmark._file_sha256(manifest_path),
    )
    with pytest.raises(RuntimeError, match="randomization binding is malformed"):
        benchmark._load_frozen_scoring_inputs(leading_zero_frozen)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    candidate_table = candidate_crossfit / "candidate_evidence.tsv"
    original_candidate = candidate_table.read_text()
    candidate_table.write_text(original_candidate + "chrom_B\n")
    with pytest.raises(RuntimeError, match="crossfit collider evidence changed"):
        benchmark._load_frozen_scoring_inputs(frozen)
    candidate_table.write_text(original_candidate)

    world_extra = world_crossfit / "unexpected_directory"
    world_extra.mkdir()
    with pytest.raises(RuntimeError, match="unsafe entry set"):
        benchmark._load_frozen_scoring_inputs(frozen)
    world_extra.rmdir()

    world_table = world_crossfit / "world_generative_candidates.tsv"
    world_table.write_text(world_table.read_text() + "chrom_B\n")
    with pytest.raises(RuntimeError, match="crossfit collider world evidence changed"):
        benchmark._load_frozen_scoring_inputs(frozen)


def test_global_pretruth_manifest_binds_all_registered_attempts(
    tmp_path: Path,
) -> None:
    frozen, _ = _write_minimal_frozen_candidate(tmp_path)
    payload = {
        "schema_version": benchmark.GLOBAL_PRETRUTH_FREEZE_SCHEMA_VERSION,
        "all_registered_discovery_attempts_finished": True,
        "registered_world_names": [frozen.world_name, "world_seed30002"],
        "world_records": [
            {
                "world_name": frozen.world_name,
                "world_dir": frozen.world_dir,
                "status": "ok",
                "pretruth_manifest_sha256": frozen.pretruth_manifest_sha256,
            },
            {
                "world_name": "world_seed30002",
                "world_dir": "/not/scored/here",
                "status": "timeout",
            },
        ],
    }
    manifest = {
        **payload,
        "manifest_payload_sha256": benchmark._canonical_sha256(payload),
    }
    path = tmp_path / "global_pretruth_freeze_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True))

    observed = benchmark._validate_global_pretruth_freeze(frozen, path)
    assert observed["all_registered_discovery_attempts_finished"] is True

    manifest["world_records"][1]["status"] = "ok"
    path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(RuntimeError, match="self-hash mismatch"):
        benchmark._validate_global_pretruth_freeze(frozen, path)
def test_frozen_state_membership_loader_rejects_semantic_leakage(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "feature": ["peak_a"],
            "modality": ["chrom_A"],
            "selection_status": ["selected"],
            "feature_global_maxT_confirmation_gate": [True],
            "assay_global_maxT_admission_gate": [True],
            "conditional_wald_view_support": [1.0],
        }
    )
    safe = tmp_path / "subspace_feature_membership.tsv"
    frame.to_csv(safe, sep="\t", index=False)
    loaded, digest = benchmark.load_frozen_state_membership(safe)
    assert loaded["feature"].tolist() == ["peak_a"]
    assert digest == benchmark._file_sha256(safe)

    leaked = tmp_path / "leaked_membership.tsv"
    frame.assign(simulator_role="P").to_csv(leaked, sep="\t", index=False)
    with pytest.raises(ValueError, match="forbidden semantic columns"):
        benchmark.load_frozen_state_membership(leaked)
