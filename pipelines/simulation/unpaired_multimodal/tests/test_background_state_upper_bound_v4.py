from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import background_state_upper_bound_v4 as upper_bound  # noqa: E402
import multimodal_simulation_v4 as simulation  # noqa: E402


@pytest.fixture(scope="module")
def oracle_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write raw observed/scorer domains without invoking simulated alignment."""

    output = tmp_path_factory.mktemp("ca-boss-v4-upper-bound") / "world_seed30001"
    discovery = output / "discovery"
    scorer = output / "scorer"
    discovery.mkdir(parents=True)
    world = simulation.generate_world(
        simulation.SimulationV4Config(
            seed=30_001,
            base_cells_override=1_200,
            n_background_features=4,
            n_microstates=5,
            min_cells_per_microstate=4,
        )
    )
    for modality, frame in world.focal.items():
        frame.to_csv(
            discovery / f"observed_focal__{modality}.tsv",
            sep="\t",
            index=False,
        )
    simulation._write_scorer(world, scorer)
    return output


@pytest.fixture(scope="module")
def oracle_result(
    oracle_world: Path,
) -> upper_bound.WorldUpperBoundResult:
    return upper_bound.evaluate_world_upper_bound(oracle_world)


def test_oracle_ceiling_passes_observation_and_group_contract(
    oracle_world: Path,
    oracle_result: upper_bound.WorldUpperBoundResult,
) -> None:
    metrics = oracle_result.metrics
    assert metrics["world_success"] == 1
    assert metrics["all_positive_targets_observation_supported"] is True
    assert metrics["direct_adjacency_macro_recall"] == 1.0
    assert metrics["shared_state_adjacency_recall"] == 1.0
    assert metrics["slow_peak_gene_present"] == 1
    assert metrics["matched_null_false_positive_rate"] <= 0.10
    assert metrics["n_estimated_focal_edges"] == 22

    truth = pd.read_csv(
        oracle_world / "scorer" / "truth_typed_edges.tsv",
        sep="\t",
        keep_default_na=False,
    ).set_index("label")
    group_rows = oracle_result.diagnostics.loc[
        oracle_result.diagnostics["truth_type"].isin(
            {upper_bound.CROSS_SHARED, upper_bound.SLOW_GENE}
        )
    ]
    assert len(group_rows) == 4
    for row in group_rows.itertuples(index=False):
        registered = truth.loc[str(row.label)]
        assert str(row.candidate_node1) in str(registered["node_set1"]).split(";")
        assert str(row.candidate_node2) in str(registered["node_set2"]).split(";")
        assert bool(row.observation_supported)

    provenance = oracle_result.provenance
    assert provenance["information_domain"] == "scorer_only_validation_ceiling"
    assert provenance["discovery_eligible"] is False
    assert provenance["must_not_be_consumed_by_discovery"] is True
    assert provenance["truth_used_for_target_definition"] is True
    assert provenance["oracle_latent_state_used"] is True
    assert provenance["cross_assay_cell_rows_paired"] is False
    assert provenance["truth_graph_copied_without_observation_test"] is False


def test_ceiling_is_not_a_truth_graph_copier(
    oracle_world: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "world_seed30001"
    shutil.copytree(oracle_world, broken)
    truth = pd.read_csv(
        broken / "scorer" / "truth_typed_edges.tsv",
        sep="\t",
        keep_default_na=False,
    )
    first_mark = str(
        truth.loc[truth["truth_type"] == upper_bound.DIRECT, "mark"].iloc[0]
    )
    child = str(
        truth.loc[
            (truth["truth_type"] == upper_bound.DIRECT)
            & (truth["mark"] == first_mark),
            "child",
        ].iloc[0]
    )
    focal_path = broken / "discovery" / f"observed_focal__chrom_{first_mark}.tsv"
    focal = pd.read_csv(focal_path, sep="\t")
    focal[child] = 0
    focal.to_csv(focal_path, sep="\t", index=False)

    result = upper_bound.evaluate_world_upper_bound(broken)
    affected = result.diagnostics.loc[
        (result.diagnostics["truth_type"] == upper_bound.DIRECT)
        & (result.diagnostics["mark"] == first_mark)
    ]
    assert not affected["observation_supported"].any()
    assert result.metrics["direct_adjacency_macro_recall"] <= 5 / 6
    assert result.metrics["world_success"] == 0


def test_scorer_bundle_is_tainted_and_cannot_enter_discovery(
    oracle_world: Path,
    oracle_result: upper_bound.WorldUpperBoundResult,
    tmp_path: Path,
) -> None:
    output = tmp_path / "scorer_validation_output"
    manifest = upper_bound.write_world_upper_bound(
        oracle_result,
        output,
        world_dir=oracle_world,
    )
    persisted = json.loads((output / "oracle_upper_bound_manifest.json").read_text())
    assert persisted["manifest_payload_sha256"] == manifest["manifest_payload_sha256"]
    assert persisted["discovery_eligible"] is False
    assert persisted["must_not_be_consumed_by_discovery"] is True
    assert set(persisted["artifacts"]) == {
        "oracle_upper_bound_edges.tsv",
        "oracle_upper_bound_feature_state_fits.tsv",
        "oracle_upper_bound_metrics.json",
        "oracle_upper_bound_target_diagnostics.tsv",
    }

    with pytest.raises(upper_bound.UpperBoundError, match="may not enter discovery"):
        upper_bound.write_world_upper_bound(
            oracle_result,
            oracle_world / "discovery" / "oracle_leak",
            world_dir=oracle_world,
        )


def test_cell_key_mismatch_is_rejected(
    oracle_world: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "world_seed30001"
    shutil.copytree(oracle_world, broken)
    focal_path = broken / "discovery" / "observed_focal__rna.tsv"
    focal = pd.read_csv(focal_path, sep="\t")
    focal.loc[0, "cell"] = "not-an-oracle-cell"
    focal.to_csv(focal_path, sep="\t", index=False)
    with pytest.raises(upper_bound.UpperBoundError, match="cell sets differ"):
        upper_bound.evaluate_world_upper_bound(broken)


def test_aggregate_ceiling_applies_registered_world_class_null_density_gates(
    oracle_result: upper_bound.WorldUpperBoundResult,
) -> None:
    results = [
        replace(
            oracle_result,
            metrics=dict(oracle_result.metrics),
            world_seed=42_001 + index,
        )
        for index in range(40)
    ]
    aggregate = upper_bound.aggregate_upper_bound_results(
        results,
        required_world_successes=34,
        required_class_successes=34,
    )
    assert aggregate["pass"] is True
    assert aggregate["gates"]["world_success"] == {
        "successes": 40,
        "required": 34,
        "pass": True,
    }
    assert aggregate["gates"]["pooled_sentinel_null"]["events"] == 600
    assert aggregate["gates"]["pooled_sentinel_null"]["wilson_one_sided_upper_95"] < 0.10
    assert aggregate["gates"]["density"]["mean_focal_edges"] == 22.0
    assert aggregate["discovery_eligible"] is False

    failed = replace(results[0], metrics=dict(results[0].metrics))
    failed.metrics["n_estimated_focal_edges"] = 31
    dense_results = [failed, *results[1:]]
    dense = upper_bound.aggregate_upper_bound_results(
        dense_results,
        required_world_successes=34,
        required_class_successes=34,
    )
    assert dense["gates"]["density"]["all_worlds_at_most_30"] is False
    assert dense["pass"] is False

    registered_results = [
        replace(
            oracle_result,
            metrics=dict(oracle_result.metrics),
            provenance={
                **oracle_result.provenance,
                "world_design": simulation.design_for_seed(seed),
            },
            world_seed=seed,
        )
        for seed in range(42_001, 42_041)
    ]
    registered = upper_bound.aggregate_registered_upper_bound(
        registered_results,
        split="locked",
    )
    assert registered["pass"] is True
    assert registered["exact_seed_block"] is True
    design_gate = registered["gates"]["resolution_iv_design"]
    assert design_gate["evaluated"] is True
    assert len(design_gate["factor_levels"]) == 8
    assert len(design_gate["design_cells"]) == 8
    assert all(row["worlds"] == 20 for row in design_gate["factor_levels"].values())
    assert all(row["worlds"] == 5 for row in design_gate["design_cells"].values())

    with pytest.raises(upper_bound.UpperBoundError, match="exact registered seed block"):
        upper_bound.aggregate_registered_upper_bound(
            registered_results[:-1],
            split="locked",
        )


def test_bh_adjustment_is_monotone_in_rank() -> None:
    raw = [0.04, 0.001, 0.02, 0.9]
    adjusted = upper_bound._bh_adjust(raw)
    ordered = sorted(zip(raw, adjusted, strict=True))
    assert all(
        float(ordered[index][1]) <= float(ordered[index + 1][1])
        for index in range(len(ordered) - 1)
    )
    assert all(raw_value <= adjusted_value <= 1.0 for raw_value, adjusted_value in zip(raw, adjusted, strict=True))
