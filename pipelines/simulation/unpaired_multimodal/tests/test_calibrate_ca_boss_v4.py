from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_ca_boss_v4 as benchmark_v4  # noqa: E402
from calibrate_ca_boss_v4 import (  # noqa: E402
    CalibrationError,
    CalibrationSpec,
    calibrate_benchmark,
    canonical_sha256,
    cluster_equal_weighted_quantile,
    world_seed,
)


def _write_source_world(source: Path, world: int, n_nulls: int) -> dict[str, object]:
    source_world = source / f"world_seed{world}"
    discovery = source_world / "discovery"
    scorer = source_world / "scorer"
    discovery.mkdir(parents=True)
    scorer.mkdir()

    observed = discovery / "observed.tsv"
    observed.write_text(f"seed\n{world}\n")
    discovery_manifest = {
        "schema_version": benchmark_v4.DISCOVERY_MANIFEST_SCHEMA_VERSION,
        "modalities": list(benchmark_v4.MODALITIES),
        "files": [observed.name],
        "file_sha256": {
            observed.name: benchmark_v4._file_sha256(observed),
        },
    }
    discovery_manifest_path = discovery / "manifest.json"
    discovery_manifest_path.write_text(json.dumps(discovery_manifest, sort_keys=True))

    truth_path = scorer / "truth_typed_edges.tsv"
    _truth_rows(world, n_nulls).to_csv(truth_path, sep="\t", index=False)
    scorer_manifest = {
        "schema_version": benchmark_v4.SCORER_MANIFEST_SCHEMA_VERSION,
        "scorer_only": True,
        "must_not_be_read_by_discovery": True,
        "files": [truth_path.name],
        "file_sha256": {
            truth_path.name: benchmark_v4._file_sha256(truth_path),
        },
    }
    scorer_manifest_path = scorer / "manifest.json"
    scorer_manifest_path.write_text(json.dumps(scorer_manifest, sort_keys=True))

    run_summary = {
        "schema_version": benchmark_v4.RUN_SUMMARY_SCHEMA_VERSION,
        "seed": world,
        "discovery_directory": "discovery",
        "scorer_directory": "scorer",
        "n_modalities": len(benchmark_v4.MODALITIES),
        "information_domains_separated_by_directory": True,
        "raw_only_preparer_required_for_discovery": True,
        "domain_payloads_content_addressed": True,
        "discovery_manifest_sha256": benchmark_v4._file_sha256(
            discovery_manifest_path
        ),
        "scorer_manifest_sha256": benchmark_v4._file_sha256(scorer_manifest_path),
    }
    (source_world / "run_summary.json").write_text(
        json.dumps(run_summary, sort_keys=True)
    )
    return benchmark_v4.validate_world_input_binding(
        source_world,
        verify_scorer_payload=True,
    )


def _raw_rows(world: int, n_nulls: int) -> pd.DataFrame:
    rows = []
    for null_i in range(n_nulls):
        rows.append(
            {
                "node1": f"null_{world}_{null_i}",
                "node2": f"match_{world}_{null_i}",
                "residual_r": 0.02 + 0.001 * null_i,
                "residual_empirical_p": 0.40 + 0.01 * (null_i % 5),
                "residual_empirical_q": 0.70,
                "residual_null_abs_r_95": 0.15,
                "residual_null_abs_r_mean": 0.06,
                "state_score": 0.01 + 0.001 * null_i,
                "predicted_r": 0.10,
                "direct_bootstrap_support": 0.05,
                "state_bootstrap_support": 0.05,
            }
        )
    return pd.DataFrame(rows)


def _truth_rows(world: int, n_nulls: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "node1": f"null_{world}_{null_i}",
                "node2": f"match_{world}_{null_i}",
                "truth_type": "sentinel_null",
                "label": f"null_{world}_{null_i}",
            }
            for null_i in range(n_nulls)
        ]
    )


def _write_training_benchmark(
    root: Path,
    *,
    n_worlds: int = 8,
    nested_runs: int = 2,
    n_nulls: int = 14,
) -> tuple[Path, Path]:
    benchmark = root / "training_benchmark"
    source = root / "training_worlds"
    benchmark.mkdir()
    source.mkdir()
    candidates = {
        "controlled": {
            "min_state_score": 0.18,
            "direct_residual_q": 0.02,
            "seed": 11,
        },
        "dense": {
            "min_state_score": 0.10,
            "direct_residual_q": 0.10,
            "seed": 11,
        },
    }
    manifest_worlds = []
    world_bindings = []
    result_rows = []
    for world in range(30001, 30001 + n_worlds):
        world_name = f"world_seed{world}"
        binding = _write_source_world(source, world, n_nulls)
        world_bindings.append(binding)
        manifest_worlds.append(str((source / world_name).absolute()))
        for nested in range(nested_runs):
            nested_dir = benchmark / world_name / f"nested_{nested}"
            nested_dir.mkdir(parents=True)
            _raw_rows(world, n_nulls).to_csv(
                nested_dir / "raw_pair_evidence.tsv", sep="\t", index=False
            )
            (nested_dir / "discovery_summary.json").write_text(
                json.dumps(
                    {
                        "method": "ca_boss_v4_explicit_state",
                        "config": {
                            "residual_permutations": 199,
                            "null_strata": 6,
                        },
                        "preprocessing": {
                            "n_null_strata": 6,
                            "focal_features_excluded_from_state_estimation": True,
                        },
                    }
                )
            )
        for candidate in candidates:
            controlled = candidate == "controlled"
            row = {
                "world": world_name,
                "candidate": candidate,
                "direct_adjacency_macro_recall": 0.92 if controlled else 1.0,
                "collider_arrowhead_recall": 0.90 if controlled else 1.0,
                "shared_state_adjacency_recall": 0.93 if controlled else 1.0,
                "slow_peak_gene_present": 1,
                "matched_null_false_positive_rate": 0.0 if controlled else 0.5,
                "primary_skeleton_precision": 0.95 if controlled else 0.60,
                "primary_skeleton_f1": 0.93 if controlled else 0.75,
                "n_estimated_focal_edges": 18 if controlled else 31,
                "world_success": 1 if controlled else 0,
            }
            for null_i in range(n_nulls):
                row[f"present__null_{world}_{null_i}"] = (
                    0 if controlled else int(null_i % 2 == 0)
                )
            result_rows.append(row)
    pd.DataFrame(result_rows).to_csv(
        benchmark / "world_results.tsv", sep="\t", index=False
    )
    (benchmark / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": benchmark_v4.BENCHMARK_MANIFEST_SCHEMA_VERSION,
                "worlds": manifest_worlds,
                "world_input_binding_schema_version": (
                    benchmark_v4.WORLD_INPUT_BINDING_SCHEMA_VERSION
                ),
                "world_input_bindings": world_bindings,
                "world_input_bindings_canonical_sha256": canonical_sha256(
                    world_bindings
                ),
                "all_world_inputs_content_addressed_and_verified": True,
                "candidates": [
                    {"name": name, "config": config}
                    for name, config in candidates.items()
                ],
                "truth_opened_after_consensus_only": True,
            }
        )
    )
    return benchmark, root / "calibration"


def test_cluster_equal_quantile_gives_worlds_equal_total_weight() -> None:
    # World A has many low nested values; world B has one high value.  Equal-row
    # weighting would put the median at zero, while equal-world weighting does not.
    values = [0.0] * 100 + [10.0]
    worlds = ["A"] * 100 + ["B"]
    assert cluster_equal_weighted_quantile(values, worlds, 0.75) == 10.0


def test_world_seed_rejects_development_and_locked_inputs() -> None:
    assert world_seed("/tmp/training/world_seed30001") == 30001
    with pytest.raises(CalibrationError, match="Non-training"):
        world_seed("/tmp/development/world_seed31001")
    with pytest.raises(CalibrationError, match="Locked path"):
        world_seed("/tmp/locked/world_seed42001")


def test_calibration_controls_nulls_before_maximizing_recall(tmp_path: Path) -> None:
    benchmark, output = _write_training_benchmark(tmp_path)
    artifacts = calibrate_benchmark(
        benchmark,
        output,
        CalibrationSpec(
            expected_worlds=8,
            expected_nested_runs=2,
            bootstrap_replicates=500,
            random_seed=123,
        ),
    )

    assert artifacts.selected_candidate == "controlled"
    assert artifacts.frozen_config["direct_residual_q"] == 0.02
    ranking = artifacts.ranking.set_index("candidate")
    assert ranking.loc["controlled", "controls_target_null_fpr"] == 1
    assert ranking.loc["dense", "controls_target_null_fpr"] == 0
    assert ranking.loc["dense", "typed_motif_macro_recall"] > ranking.loc[
        "controlled", "typed_motif_macro_recall"
    ]

    frozen = json.loads(artifacts.config_path.read_text())
    audit = json.loads(artifacts.audit_path.read_text())
    assert audit["truth_boundary"]["raw_discovery_evidence_loaded_before_truth"]
    assert audit["truth_boundary"][
        "scorer_payload_hashes_verified_after_discovery_before_truth_parse"
    ]
    assert not audit["truth_boundary"]["locked_worlds_accessed"]
    bindings = audit["training_world_bindings"]
    assert bindings["required_for_frozen_candidate_use"]
    assert bindings["benchmark_bindings_verified_against_current_worlds"]
    assert bindings["world_count"] == 8
    assert bindings["bindings_canonical_sha256"] == canonical_sha256(
        bindings["bindings"]
    )
    for binding in bindings["bindings"]:
        assert len(binding["run_summary"]["sha256"]) == 64
        assert len(binding["discovery_manifest"]["sha256"]) == 64
        assert len(binding["scorer_manifest"]["sha256"]) == 64
    assert audit["selected"]["config_canonical_sha256"] == canonical_sha256(frozen)
    payload_hash = audit.pop("audit_payload_sha256")
    assert payload_hash == canonical_sha256(audit)
    assert len(artifacts.null_world_summary) == 8


def test_calibrator_rejects_stale_or_changed_content_bound_worlds(
    tmp_path: Path,
) -> None:
    benchmark, output = _write_training_benchmark(tmp_path)
    manifest = json.loads((benchmark / "benchmark_manifest.json").read_text())
    first_world = Path(manifest["worlds"][0])

    run_summary_path = first_world / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text())
    run_summary["schema_version"] = "multimodal_simulation_v4.run.v1"
    run_summary_path.write_text(json.dumps(run_summary, sort_keys=True))
    with pytest.raises(CalibrationError, match="stale or unsupported run-summary"):
        calibrate_benchmark(
            benchmark,
            output,
            CalibrationSpec(
                expected_worlds=8,
                expected_nested_runs=2,
                bootstrap_replicates=200,
            ),
        )

    # Restore the exact content-bound summary, then alter a discovery payload.
    run_summary["schema_version"] = benchmark_v4.RUN_SUMMARY_SCHEMA_VERSION
    run_summary_path.write_text(json.dumps(run_summary, sort_keys=True))
    (first_world / "discovery" / "observed.tsv").write_text("seed\nchanged\n")
    with pytest.raises(CalibrationError, match="payload hash mismatch"):
        calibrate_benchmark(
            benchmark,
            output,
            CalibrationSpec(
                expected_worlds=8,
                expected_nested_runs=2,
                bootstrap_replicates=200,
            ),
        )


def test_refuses_to_freeze_if_no_candidate_controls_nulls(tmp_path: Path) -> None:
    benchmark, output = _write_training_benchmark(tmp_path)
    results_path = benchmark / "world_results.tsv"
    results = pd.read_csv(results_path, sep="\t")
    controlled = results["candidate"].eq("controlled")
    sentinel_columns = [column for column in results if column.startswith("present__")]
    for column in sentinel_columns:
        results.loc[controlled & results[column].notna(), column] = 1
    results.loc[controlled, "matched_null_false_positive_rate"] = 1.0
    results.to_csv(results_path, sep="\t", index=False)

    with pytest.raises(CalibrationError, match="No candidate controls"):
        calibrate_benchmark(
            benchmark,
            output,
            CalibrationSpec(
                expected_worlds=8,
                expected_nested_runs=2,
                bootstrap_replicates=200,
            ),
        )
