from __future__ import annotations

from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import types
from typing import Any

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_ca_boss_v4 as validation  # noqa: E402
from channel_aware_boss_v4 import CABossV4Config  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _frozen_training_candidate(tmp_path: Path) -> tuple[Path, Path, str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "ca_boss_v4_frozen_config.json"
    config = asdict(
        CABossV4Config(
            seed=17,
            residual_permutations=19,
            stability_bootstraps=23,
        )
    )
    _write_json(config_path, config)
    candidate = "v4_frozen_confirmatory"
    bindings: list[dict[str, Any]] = []
    for seed in validation.TRAINING_SEEDS:
        digest = f"{seed:064x}"
        binding: dict[str, Any] = {
            "schema_version": validation.WORLD_INPUT_BINDING_SCHEMA_VERSION,
            "seed": seed,
            "world": f"/registered_training/world_seed{seed}",
            "run_summary": {
                "path": f"/registered_training/world_seed{seed}/run_summary.json",
                "schema_version": validation.RUN_SUMMARY_SCHEMA_VERSION,
                "sha256": digest,
            },
            "discovery_manifest": {
                "path": f"/registered_training/world_seed{seed}/discovery/manifest.json",
                "schema_version": validation.DISCOVERY_MANIFEST_SCHEMA_VERSION,
                "sha256": digest,
                "payloads_content_addressed_verified": True,
            },
            "scorer_manifest": {
                "path": f"/registered_training/world_seed{seed}/scorer/manifest.json",
                "schema_version": validation.SCORER_MANIFEST_SCHEMA_VERSION,
                "sha256": digest,
                "payloads_content_addressed_verified": True,
            },
            "directory_seed_matches_run_summary": True,
            "information_domains_separated": True,
            "scorer_payload_verification_performed": True,
        }
        binding["binding_payload_sha256"] = validation.canonical_sha256(binding)
        bindings.append(binding)
    audit: dict[str, Any] = {
        "schema_version": validation.CALIBRATION_AUDIT_SCHEMA_VERSION,
        "selection_split": "training_only",
        "registered_training_seed_block": [30_001, 30_024],
        "world_seeds": list(validation.TRAINING_SEEDS),
        "truth_boundary": {
            "raw_discovery_evidence_loaded_before_truth": True,
            "locked_worlds_accessed": False,
            "external_data_accessed": False,
        },
        "training_world_bindings": {
            "schema_version": validation.TRAINING_BINDING_AUDIT_SCHEMA_VERSION,
            "required_for_frozen_candidate_use": True,
            "benchmark_schema_version": validation.BENCHMARK_MANIFEST_SCHEMA_VERSION,
            "world_binding_schema_version": validation.WORLD_INPUT_BINDING_SCHEMA_VERSION,
            "benchmark_bindings_verified_against_current_worlds": True,
            "world_count": len(bindings),
            "world_seeds": list(validation.TRAINING_SEEDS),
            "bindings_canonical_sha256": validation.canonical_sha256(bindings),
            "bindings": bindings,
        },
        "selected": {
            "candidate": candidate,
            "metrics": {"controls_target_null_fpr": 1},
            "config_sha256": validation.file_sha256(config_path),
            "config_canonical_sha256": validation.canonical_sha256(config),
        },
    }
    audit["audit_payload_sha256"] = validation.canonical_sha256(audit)
    audit_path = tmp_path / "calibration_audit_manifest.json"
    _write_json(audit_path, audit)
    return audit_path, config_path, validation.file_sha256(audit_path), candidate


def _dummy_worlds(tmp_path: Path, seeds: tuple[int, ...]) -> list[Path]:
    worlds: list[Path] = []
    for seed in seeds:
        world = tmp_path / f"world_seed{seed}"
        (world / "discovery").mkdir(parents=True)
        scorer = world / "scorer"
        scorer.mkdir()
        design = validation.design_for_seed(seed)
        discovery_payload = world / "discovery" / "observed_payload.tsv"
        discovery_payload.write_text(f"seed\n{seed}\n")
        _write_json(
            world / "discovery" / "manifest.json",
            {
                "schema_version": "multimodal_simulation_v4.discovery.v2",
                "modalities": list(validation.MODALITIES),
                "files": [discovery_payload.name],
                "file_sha256": {
                    discovery_payload.name: validation.file_sha256(discovery_payload)
                },
            },
        )
        (scorer / "truth_typed_edges.tsv").write_text("dummy\n")
        (scorer / "role_assignments.tsv").write_text("dummy\n")
        (scorer / "generative_dag_nodes.tsv").write_text("dummy\n")
        (scorer / "generative_dag_edges.tsv").write_text("dummy\n")
        _write_json(
            scorer / "world_parameters.json",
            {
                "schema_version": "multimodal_simulation_v4.parameters.v2",
                "seed": seed,
                "design": design,
            },
        )
        _write_json(
            scorer / "manifest.json",
            {
                "schema_version": "multimodal_simulation_v4.scorer.v2",
                "files": [
                    "truth_typed_edges.tsv",
                    "role_assignments.tsv",
                    "generative_dag_nodes.tsv",
                    "generative_dag_edges.tsv",
                    "world_parameters.json",
                ],
                "file_sha256": {
                    name: validation.file_sha256(scorer / name)
                    for name in (
                        "truth_typed_edges.tsv",
                        "role_assignments.tsv",
                        "generative_dag_nodes.tsv",
                        "generative_dag_edges.tsv",
                        "world_parameters.json",
                    )
                },
            },
        )
        _write_json(
            world / "run_summary.json",
            {
                "schema_version": "multimodal_simulation_v4.run.v2",
                "seed": seed,
                "discovery_directory": "discovery",
                "scorer_directory": "scorer",
                "information_domains_separated_by_directory": True,
                "raw_only_preparer_required_for_discovery": True,
                "domain_payloads_content_addressed": True,
                "discovery_manifest_sha256": validation.file_sha256(
                    world / "discovery" / "manifest.json"
                ),
                "scorer_manifest_sha256": validation.file_sha256(
                    scorer / "manifest.json"
                ),
                "n_modalities": 7,
            },
        )
        worlds.append(world)
    return worlds


def _passing_score(world: Path, candidate: str, *, density: int = 10) -> dict[str, Any]:
    row: dict[str, Any] = {
        key: (0 if isinstance(value, int) else value)
        for key, value in validation.RESULT_DEFAULTS.items()
    }
    row.update(
        {
            "world": world.name,
            "candidate": candidate,
            "direct_adjacency_macro_recall": 1.0,
            "collider_arrowhead_recall": 1.0,
            "shared_state_adjacency_recall": 1.0,
            "slow_peak_gene_present": 1,
            "matched_null_false_positive_rate": 0.0,
            "primary_skeleton_precision": 1.0,
            "primary_skeleton_recall": 1.0,
            "primary_skeleton_f1": 1.0,
            "n_estimated_focal_edges": density,
            "n_unlabeled_focal_edges": 0,
            "density_class_success": 1,
            "world_success": 1,
            **{column: 1 for column in validation.CLASS_SUCCESS_COLUMNS},
        }
    )
    return row


def _passing_audit(_world: Path, _out: Path, _candidate: str) -> dict[str, Any]:
    return {
        "pass": True,
        "reasons": [],
        "null_false_adjacencies": 0,
        "null_adjacency_opportunities": 15,
    }


def _frozen_discovery_handle(
    world: Path,
    candidate: str,
    out_dir: Path,
) -> validation.FrozenWorldPretruth:
    world_out = out_dir / world.name
    world_out.mkdir(parents=True, exist_ok=True)
    manifest_path = world_out / "pretruth_consensus_manifest.json"
    _write_json(
        manifest_path,
        {
            "world": world.name,
            "candidate": candidate,
            "all_consensuses_built_before_truth_open": True,
        },
    )
    return validation.FrozenWorldPretruth(
        world_dir=str(world.resolve()),
        world_name=world.name,
        world_out=str(world_out.resolve()),
        candidate_names=(candidate,),
        pretruth_manifest=str(manifest_path.resolve()),
        pretruth_manifest_sha256=validation.file_sha256(manifest_path),
    )


def _passing_discovery_executor(
    world: Path,
    candidate: str,
    _config: CABossV4Config,
    out_dir: Path,
    _timeout: float,
) -> validation.DiscoveryExecution:
    return validation.DiscoveryExecution(
        status="ok",
        frozen=_frozen_discovery_handle(world, candidate, out_dir),
        elapsed_seconds=0.01,
    )


def _passing_scoring_executor(
    frozen: validation.FrozenWorldPretruth,
    global_freeze_manifest: Path,
    _timeout: float,
) -> validation.WorldExecution:
    global_manifest = json.loads(global_freeze_manifest.read_text())
    payload_hash = global_manifest.pop("manifest_payload_sha256")
    assert payload_hash == validation.canonical_sha256(global_manifest)
    assert global_manifest["all_registered_discovery_attempts_finished"] is True
    assert frozen.world_name in global_manifest["registered_world_names"]
    return validation.WorldExecution(
        status="ok",
        rows=[
            _passing_score(
                Path(frozen.world_dir),
                frozen.candidate_names[0],
            )
        ],
        elapsed_seconds=0.01,
    )


def _passing_upper_bound(
    worlds: dict[int, Path],
    split: str,
    out_dir: Path,
) -> dict[str, Any]:
    del out_dir
    return {
        "pass": True,
        "scorer_only": True,
        "discovery_eligible": False,
        "candidate_outputs_frozen_before_gate": True,
        "split": split,
        "world_seeds": list(worlds),
    }


def _passing_baselines(
    worlds: dict[int, Path],
    split: str,
    evaluation_out: Path,
    out_dir: Path,
    timeout_seconds: float,
    candidate_freeze: dict[str, Any],
) -> dict[str, Any]:
    del evaluation_out, out_dir, timeout_seconds
    assert candidate_freeze["all_registered_v4_execution_attempts_finished"]
    return {
        "pass": True,
        "diagnostic_only": True,
        "outcomes_enter_candidate_gate": False,
        "split": split,
        "world_seeds": list(worlds),
        "methods": {
            method: {
                "registered_worlds": len(worlds),
                "execution_status_counts": {"ok": len(worlds)},
                "pass": True,
            }
            for method in validation.BASELINE_METHODS
        },
    }


def _sleeping_benchmark_evaluator(
    _world: Path,
    _candidates: list[tuple[str, CABossV4Config]],
    *,
    out_dir: Path,
    save_candidate_names: set[str],
) -> list[dict[str, Any]]:
    del out_dir, save_candidate_names
    time.sleep(1.0)
    return []


def _sleeping_pretruth_discoverer(
    _world: Path,
    _candidates: list[tuple[str, CABossV4Config]],
    *,
    out_dir: Path,
    save_candidate_names: set[str],
) -> validation.FrozenWorldPretruth:
    del out_dir, save_candidate_names
    time.sleep(1.0)
    raise AssertionError("sleeping discovery should be terminated")


def _sleeping_frozen_scorer(
    _frozen: validation.FrozenWorldPretruth,
    *,
    global_freeze_manifest: Path,
) -> list[dict[str, Any]]:
    del global_freeze_manifest
    time.sleep(1.0)
    return []


def _sleeping_baseline_evaluator(
    _method: str,
    _inputs: validation.BaselineObservableInputs,
    _out_dir: Path,
) -> dict[str, Any]:
    time.sleep(1.0)
    return {}


def _aggregate_frame(seeds: tuple[int, ...], candidate: str = "frozen") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        world = Path(f"world_seed{seed}")
        execution = validation.WorldExecution(
            status="ok",
            rows=[_passing_score(world, candidate)],
            elapsed_seconds=0.01,
        )
        rows.append(
            validation._successful_result(
                seed,
                world,
                candidate,
                execution,
                {
                    "pass": True,
                    "null_false_adjacencies": 0,
                    "null_adjacency_opportunities": 15,
                },
            )
        )
    return pd.DataFrame(rows)


def _dummy_baseline_inputs(
    tmp_path: Path,
    world: Path,
) -> validation.BaselineObservableInputs:
    evaluation = tmp_path / "evaluation"
    prepared = evaluation / world.name / "prepared_discovery"
    prepared.mkdir(parents=True)
    embedding = prepared / "analysis_embeddings.tsv"
    embedding.write_text("cell\tmodality\tstate_0\n")
    for target in (8, 14, 24):
        for seed in (0, 1):
            nested = prepared / "nested" / f"cells{target}_seed{seed}"
            nested.mkdir(parents=True)
            (nested / "observed_metacell_matrix.tsv").write_text(
                "sample_id\texpr__GENE_log1p\nMC1\t0.0\n"
            )
    for index, modality in enumerate(validation.MODALITIES[1:]):
        (world / "discovery" / f"observed_focal__{modality}.tsv").write_text(
            f"cell\tpeak_{index}\ncell0\t0\n"
        )
    return validation.collect_baseline_observable_inputs(world, evaluation)


def test_frozen_candidate_requires_training_audit_and_both_hashes(tmp_path: Path) -> None:
    audit, config, audit_hash, candidate = _frozen_training_candidate(tmp_path)

    frozen = validation.load_frozen_candidate(config, audit, audit_hash)

    assert frozen.name == candidate
    assert frozen.config.seed == 17
    assert frozen.config_bytes_sha256 == validation.file_sha256(config)
    with pytest.raises(validation.ValidationError, match="file hash mismatch"):
        validation.load_frozen_candidate(config, audit, "0" * 64)

    original = json.loads(audit.read_text())
    rebound = json.loads(audit.read_text())
    rebound["training_world_bindings"]["bindings"][0]["run_summary"]["sha256"] = "f" * 64
    rebound["training_world_bindings"]["bindings_canonical_sha256"] = (
        validation.canonical_sha256(
            rebound["training_world_bindings"]["bindings"]
        )
    )
    rebound.pop("audit_payload_sha256")
    rebound["audit_payload_sha256"] = validation.canonical_sha256(rebound)
    _write_json(audit, rebound)
    with pytest.raises(validation.ValidationError, match="binding payload hash mismatch"):
        validation.load_frozen_candidate(
            config,
            audit,
            validation.file_sha256(audit),
        )

    _write_json(audit, original)
    tampered = json.loads(audit.read_text())
    tampered["selection_split"] = "development"
    _write_json(audit, tampered)
    with pytest.raises(validation.ValidationError, match="payload hash mismatch"):
        validation.load_frozen_candidate(
            config,
            audit,
            validation.file_sha256(audit),
        )


def test_world_contract_rejects_historical_mixed_and_incomplete_blocks(tmp_path: Path) -> None:
    historical = _dummy_worlds(tmp_path / "historical", (10_001,))
    with pytest.raises(validation.ValidationError, match="Historical locked"):
        validation.validate_world_set(historical, validation.SPLITS["development"])

    development = _dummy_worlds(tmp_path / "development", validation.DEVELOPMENT_SEEDS)
    with pytest.raises(validation.ValidationError, match="do not exactly match"):
        validation.validate_world_set(development[:-1], validation.SPLITS["development"])

    mixed = development[:-1] + _dummy_worlds(tmp_path / "mixed", (42_001,))
    with pytest.raises(validation.ValidationError, match="not in the registered development"):
        validation.validate_world_set(mixed, validation.SPLITS["development"])

    summary_path = development[0] / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["seed"] = validation.DEVELOPMENT_SEEDS[1]
    _write_json(summary_path, summary)
    with pytest.raises(validation.ValidationError, match="run-summary seed"):
        validation.validate_world_set(development, validation.SPLITS["development"])


def test_world_contract_rejects_discovery_domain_swapped_between_worlds(
    tmp_path: Path,
) -> None:
    first, second = _dummy_worlds(tmp_path, (31_001, 31_002))
    original = first / "discovery_original"
    (first / "discovery").rename(original)
    shutil.copytree(second / "discovery", first / "discovery")

    with pytest.raises(validation.ValidationError, match="discovery manifest is not bound"):
        validation._validate_safe_world_identity(first, 31_001)


def test_registered_locked_aggregate_gates_include_factor_and_design_cells() -> None:
    results = _aggregate_frame(validation.FRESH_LOCKED_SEEDS)
    gates = validation.aggregate_gates(results, validation.SPLITS["locked"])
    assert gates["pass"]
    assert gates["world_gate"]["successes"] == 40
    assert gates["pooled_null_gate"]["one_sided_wilson_upper_95"] < 0.10
    assert all(
        level["total"] == 20
        for levels in gates["factor_level_gate"]["levels"].values()
        for level in levels.values()
    )
    assert all(
        cell["total"] == 5 for cell in gates["design_cell_gate"]["cells"].values()
    )

    # Three failures in one five-world DOE cell leave overall success and each
    # factor level above their thresholds, but correctly fail the 3/5 cell gate.
    same_cell = [42_001, 42_009, 42_017]
    stressed = results.copy()
    stressed.loc[stressed["world_seed"].isin(same_cell), "validation_world_success"] = 0
    stressed_gates = validation.aggregate_gates(stressed, validation.SPLITS["locked"])
    assert stressed_gates["world_gate"]["pass"]
    assert stressed_gates["factor_level_gate"]["pass"]
    assert not stressed_gates["design_cell_gate"]["pass"]
    assert not stressed_gates["pass"]


def test_density_and_provenance_are_hard_aggregate_gates() -> None:
    results = _aggregate_frame(validation.FRESH_LOCKED_SEEDS)
    results["n_estimated_focal_edges"] = 23
    density = validation.aggregate_gates(results, validation.SPLITS["locked"])
    assert density["density_gate"]["mean_estimated_focal_edges"] == 23.0
    assert not density["density_gate"]["pass"]
    assert not density["pass"]

    results["n_estimated_focal_edges"] = 10
    results.loc[0, "provenance_violation"] = 1
    provenance = validation.aggregate_gates(results, validation.SPLITS["locked"])
    assert provenance["provenance_gate"]["violations"] == 1
    assert not provenance["provenance_gate"]["pass"]


def test_development_runner_uses_one_frozen_candidate_and_writes_manifest(
    tmp_path: Path,
) -> None:
    audit, config, audit_hash, candidate = _frozen_training_candidate(tmp_path / "freeze")
    worlds = _dummy_worlds(tmp_path / "worlds", validation.DEVELOPMENT_SEEDS)
    discovery_calls: list[tuple[int, str]] = []
    scoring_calls: list[int] = []
    candidate_outputs_completed = False

    def upper_bound(
        registered_worlds: dict[int, Path],
        split: str,
        out_dir: Path,
    ) -> dict[str, Any]:
        nonlocal candidate_outputs_completed
        assert split == "development"
        assert tuple(registered_worlds) == validation.DEVELOPMENT_SEEDS
        assert candidate_outputs_completed
        del out_dir
        return {"pass": True, "scorer_only": True, "discovery_eligible": False}

    def discovery_executor(
        world: Path,
        name: str,
        cfg: CABossV4Config,
        out: Path,
        _timeout: float,
    ) -> validation.DiscoveryExecution:
        discovery_calls.append((validation.extract_world_seed(world), name))
        assert cfg == CABossV4Config(**json.loads(config.read_text()))
        return validation.DiscoveryExecution(
            status="ok",
            frozen=_frozen_discovery_handle(world, name, out),
            elapsed_seconds=0.01,
        )

    def scoring_executor(
        frozen_world: validation.FrozenWorldPretruth,
        global_manifest: Path,
        _timeout: float,
    ) -> validation.WorldExecution:
        nonlocal candidate_outputs_completed
        assert len(discovery_calls) == len(validation.DEVELOPMENT_SEEDS)
        assert global_manifest.is_file()
        seed = validation.extract_world_seed(frozen_world.world_dir)
        scoring_calls.append(seed)
        candidate_outputs_completed = len(scoring_calls) == len(
            validation.DEVELOPMENT_SEEDS
        )
        return validation.WorldExecution(
            status="ok",
            rows=[
                _passing_score(
                    Path(frozen_world.world_dir),
                    frozen_world.candidate_names[0],
                )
            ],
            elapsed_seconds=0.01,
        )

    manifest = validation.run_validation(
        split="development",
        world_dirs=worlds,
        frozen_config=config,
        calibration_audit=audit,
        calibration_audit_sha256=audit_hash,
        out_dir=tmp_path / "validation",
        timeout_seconds=60.0,
        discovery_executor=discovery_executor,
        scoring_executor=scoring_executor,
        provenance_auditor=_passing_audit,
        baseline_runner=_passing_baselines,
        upper_bound_runner=upper_bound,
    )

    assert discovery_calls == [
        (seed, candidate) for seed in validation.DEVELOPMENT_SEEDS
    ]
    assert scoring_calls == list(validation.DEVELOPMENT_SEEDS)
    assert manifest["single_candidate_only"] is True
    assert manifest["candidate_grid_permitted"] is False
    assert manifest["candidate_gate_pass"] is True
    assert manifest["formal_certification_complete"] is True
    assert manifest["preregistered_hooks"]["background_state_upper_bound"] == {
        "implemented": True,
        "status": "implemented_after_candidate_freeze_and_required_for_certification",
        "required_before_locked_truth": False,
        "required_for_formal_certification": True,
    }
    for hook in ("frozen_v3_baseline", "production_fci_baseline"):
        assert manifest["preregistered_hooks"][hook] == {
            "implemented": True,
            "status": "implemented_after_candidate_freeze_diagnostic_only",
            "may_influence_v4_selection": False,
            "failures_and_timeouts_retained": True,
            "required_for_formal_certification": True,
        }
    assert manifest["post_candidate_baseline_diagnostics"]["pass"] is True
    assert manifest["truth_boundary"]["all_candidate_outputs_frozen_before_baselines"]
    assert manifest["truth_boundary"][
        "all_registered_discovery_attempts_finished_before_any_scorer_truth_open"
    ]
    assert not manifest["truth_boundary"]["baseline_outputs_available_to_v4_selection"]
    global_freeze_path = (
        tmp_path / "validation" / "global_pretruth_freeze_manifest.json"
    )
    global_freeze = json.loads(global_freeze_path.read_text())
    global_payload_hash = global_freeze.pop("manifest_payload_sha256")
    assert global_payload_hash == validation.canonical_sha256(global_freeze)
    assert global_freeze["registered_world_names"] == [
        world.name for world in worlds
    ]
    assert all(
        record["status"] == "ok" for record in global_freeze["world_records"]
    )
    saved = json.loads(Path(manifest["manifest_path"]).read_text())
    payload_hash = saved.pop("manifest_payload_sha256")
    assert payload_hash == validation.canonical_sha256(saved)
    results = pd.read_csv(
        tmp_path / "validation" / "world_validation_results.tsv",
        sep="\t",
    )
    assert len(results) == 16
    assert results["validation_world_success"].sum() == 16


def test_timeout_is_retained_as_a_failed_development_world(tmp_path: Path) -> None:
    audit, config, audit_hash, _ = _frozen_training_candidate(tmp_path / "freeze")
    worlds = _dummy_worlds(tmp_path / "worlds", validation.DEVELOPMENT_SEEDS)
    timed_out_seed = validation.DEVELOPMENT_SEEDS[0]

    scored_seeds: list[int] = []

    def discovery_executor(
        world: Path,
        name: str,
        _cfg: CABossV4Config,
        out: Path,
        _timeout: float,
    ) -> validation.DiscoveryExecution:
        if validation.extract_world_seed(world) == timed_out_seed:
            return validation.DiscoveryExecution(
                status="timeout",
                frozen=None,
                elapsed_seconds=60.0,
                error="synthetic timeout",
            )
        return validation.DiscoveryExecution(
            status="ok",
            frozen=_frozen_discovery_handle(world, name, out),
            elapsed_seconds=0.01,
        )

    def scoring_executor(
        frozen_world: validation.FrozenWorldPretruth,
        global_manifest: Path,
        timeout: float,
    ) -> validation.WorldExecution:
        scored_seeds.append(validation.extract_world_seed(frozen_world.world_dir))
        return _passing_scoring_executor(frozen_world, global_manifest, timeout)

    manifest = validation.run_validation(
        split="development",
        world_dirs=worlds,
        frozen_config=config,
        calibration_audit=audit,
        calibration_audit_sha256=audit_hash,
        out_dir=tmp_path / "validation",
        timeout_seconds=60.0,
        discovery_executor=discovery_executor,
        scoring_executor=scoring_executor,
        provenance_auditor=_passing_audit,
        baseline_runner=_passing_baselines,
        upper_bound_runner=_passing_upper_bound,
    )

    denominator = manifest["gates"]["execution_denominator"]
    assert denominator["registered_worlds"] == 16
    assert denominator["status_counts"] == {"ok": 15, "timeout": 1}
    assert manifest["gates"]["world_gate"]["successes"] == 15
    assert not manifest["gates"]["design_cell_gate"]["pass"]
    assert not manifest["candidate_gate_pass"]
    results = pd.read_csv(
        tmp_path / "validation" / "world_validation_results.tsv",
        sep="\t",
    )
    failed = results.loc[results["world_seed"].eq(timed_out_seed)].iloc[0]
    assert failed["execution_status"] == "timeout"
    assert failed["validation_world_success"] == 0
    assert timed_out_seed not in scored_seeds
    assert len(scored_seeds) == len(validation.DEVELOPMENT_SEEDS) - 1
    global_freeze = json.loads(
        (
            tmp_path / "validation" / "global_pretruth_freeze_manifest.json"
        ).read_text()
    )
    failed_record = next(
        record
        for record in global_freeze["world_records"]
        if record["world_name"] == f"world_seed{timed_out_seed}"
    )
    assert failed_record == {
        "world_name": f"world_seed{timed_out_seed}",
        "world_dir": str(worlds[0].resolve()),
        "status": "timeout",
    }


def test_global_barrier_retains_both_phase_failures_and_orders_baselines(
    tmp_path: Path,
) -> None:
    audit, config, audit_hash, _ = _frozen_training_candidate(tmp_path / "freeze")
    worlds = _dummy_worlds(tmp_path / "worlds", validation.DEVELOPMENT_SEEDS)
    discovery_exception_seed = validation.DEVELOPMENT_SEEDS[0]
    discovery_timeout_seed = validation.DEVELOPMENT_SEEDS[1]
    scoring_exception_seed = validation.DEVELOPMENT_SEEDS[2]
    discovery_attempts: list[int] = []
    scoring_attempts: list[int] = []
    manifest_paths: set[Path] = set()
    baselines_called = False

    def discovery_executor(
        world: Path,
        name: str,
        _config: CABossV4Config,
        out_dir: Path,
        _timeout: float,
    ) -> validation.DiscoveryExecution:
        seed = validation.extract_world_seed(world)
        discovery_attempts.append(seed)
        if seed == discovery_exception_seed:
            raise RuntimeError("synthetic discovery exception")
        if seed == discovery_timeout_seed:
            return validation.DiscoveryExecution(
                status="timeout",
                frozen=None,
                elapsed_seconds=60.0,
                error="synthetic discovery timeout",
            )
        return validation.DiscoveryExecution(
            status="ok",
            frozen=_frozen_discovery_handle(world, name, out_dir),
            elapsed_seconds=0.01,
        )

    def scoring_executor(
        frozen_world: validation.FrozenWorldPretruth,
        global_manifest_path: Path,
        timeout: float,
    ) -> validation.WorldExecution:
        assert discovery_attempts == list(validation.DEVELOPMENT_SEEDS)
        seed = validation.extract_world_seed(frozen_world.world_dir)
        scoring_attempts.append(seed)
        manifest_paths.add(global_manifest_path)
        global_manifest = json.loads(global_manifest_path.read_text())
        payload_hash = global_manifest.pop("manifest_payload_sha256")
        assert payload_hash == validation.canonical_sha256(global_manifest)
        assert global_manifest["all_registered_discovery_attempts_finished"] is True
        if seed == scoring_exception_seed:
            raise RuntimeError("synthetic scoring exception")
        return _passing_scoring_executor(
            frozen_world,
            global_manifest_path,
            timeout,
        )

    def baseline_runner(
        registered_worlds: dict[int, Path],
        split: str,
        evaluation_out: Path,
        out_dir: Path,
        timeout_seconds: float,
        candidate_freeze: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal baselines_called
        expected_scoring = [
            seed
            for seed in validation.DEVELOPMENT_SEEDS
            if seed not in (discovery_exception_seed, discovery_timeout_seed)
        ]
        assert scoring_attempts == expected_scoring
        assert candidate_freeze["all_registered_v4_execution_attempts_finished"]
        assert candidate_freeze["global_pretruth_freeze_manifest_sha256"] == (
            validation.file_sha256(next(iter(manifest_paths)))
        )
        baselines_called = True
        return _passing_baselines(
            registered_worlds,
            split,
            evaluation_out,
            out_dir,
            timeout_seconds,
            candidate_freeze,
        )

    manifest = validation.run_validation(
        split="development",
        world_dirs=worlds,
        frozen_config=config,
        calibration_audit=audit,
        calibration_audit_sha256=audit_hash,
        out_dir=tmp_path / "validation",
        timeout_seconds=60.0,
        discovery_executor=discovery_executor,
        scoring_executor=scoring_executor,
        provenance_auditor=_passing_audit,
        baseline_runner=baseline_runner,
        upper_bound_runner=_passing_upper_bound,
    )

    assert baselines_called
    assert len(manifest_paths) == 1
    assert scoring_exception_seed in scoring_attempts
    assert discovery_exception_seed not in scoring_attempts
    assert discovery_timeout_seed not in scoring_attempts
    assert manifest["gates"]["execution_denominator"] == {
        "registered_worlds": len(validation.DEVELOPMENT_SEEDS),
        "status_counts": {"executor_exception": 2, "ok": 13, "timeout": 1},
        "failures_are_retained": True,
    }
    results = pd.read_csv(
        tmp_path / "validation" / "world_validation_results.tsv",
        sep="\t",
    ).set_index("world_seed")
    assert results.loc[discovery_exception_seed, "execution_status"] == (
        "executor_exception"
    )
    assert results.loc[discovery_timeout_seed, "execution_status"] == "timeout"
    assert results.loc[scoring_exception_seed, "execution_status"] == (
        "executor_exception"
    )
    freeze = json.loads(next(iter(manifest_paths)).read_text())
    freeze_records = {
        record["world_name"]: record for record in freeze["world_records"]
    }
    assert freeze_records[f"world_seed{discovery_exception_seed}"]["status"] == (
        "executor_exception"
    )
    assert freeze_records[f"world_seed{discovery_timeout_seed}"]["status"] == (
        "timeout"
    )
    assert freeze_records[f"world_seed{scoring_exception_seed}"]["status"] == "ok"


def test_incomplete_baseline_diagnostics_do_not_change_candidate_gate(
    tmp_path: Path,
) -> None:
    audit, config, audit_hash, _ = _frozen_training_candidate(tmp_path / "freeze")
    worlds = _dummy_worlds(tmp_path / "worlds", validation.DEVELOPMENT_SEEDS)

    def incomplete_baselines(
        registered_worlds: dict[int, Path],
        split: str,
        _evaluation: Path,
        _out: Path,
        _timeout: float,
        freeze: dict[str, Any],
    ) -> dict[str, Any]:
        assert split == "development"
        assert tuple(registered_worlds) == validation.DEVELOPMENT_SEEDS
        assert freeze["all_registered_v4_execution_attempts_finished"]
        return {
            "pass": False,
            "diagnostic_only": True,
            "outcomes_enter_candidate_gate": False,
            "methods": {
                "frozen_v3": {"execution_status_counts": {"timeout": 1}},
                "production_fci": {"execution_status_counts": {"ok": 16}},
            },
        }

    manifest = validation.run_validation(
        split="development",
        world_dirs=worlds,
        frozen_config=config,
        calibration_audit=audit,
        calibration_audit_sha256=audit_hash,
        out_dir=tmp_path / "validation",
        timeout_seconds=60.0,
        discovery_executor=_passing_discovery_executor,
        scoring_executor=_passing_scoring_executor,
        provenance_auditor=_passing_audit,
        baseline_runner=incomplete_baselines,
        upper_bound_runner=_passing_upper_bound,
    )

    assert manifest["candidate_gate_pass"] is True
    assert manifest["gates"]["pass"] is True
    assert manifest["formal_certification_complete"] is False
    assert (
        manifest["formal_certification_status"]
        == "blocked_incomplete_preregistered_baseline_diagnostics"
    )


@pytest.mark.parametrize(
    ("candidate_pass", "formal_complete", "expected_exit"),
    ((True, False, 1), (False, False, 1), (True, True, 0)),
)
def test_cli_success_requires_formal_certification(
    monkeypatch: pytest.MonkeyPatch,
    candidate_pass: bool,
    formal_complete: bool,
    expected_exit: int,
) -> None:
    monkeypatch.setattr(
        validation,
        "parse_args",
        lambda: types.SimpleNamespace(
            split="development",
            world=[],
            frozen_config=Path("config.json"),
            calibration_audit=Path("audit.json"),
            calibration_audit_sha256="0" * 64,
            out_dir=Path("out"),
            timeout_seconds=1.0,
            development_manifest=None,
            development_manifest_sha256=None,
        ),
    )
    monkeypatch.setattr(
        validation,
        "run_validation",
        lambda **_kwargs: {
            "candidate_gate_pass": candidate_pass,
            "formal_certification_complete": formal_complete,
            "manifest_path": "validation_manifest.json",
            "manifest_file_sha256": "f" * 64,
        },
    )

    assert validation.main() == expected_exit


def test_killable_world_wrapper_reports_timeout_without_real_world(tmp_path: Path) -> None:
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("test uses fork to avoid pickling a pytest module evaluator")
    started = time.monotonic()
    execution = validation.evaluate_world_with_timeout(
        tmp_path / "synthetic_world",
        "candidate",
        CABossV4Config(),
        tmp_path / "out",
        0.05,
        evaluator=_sleeping_benchmark_evaluator,
    )
    assert execution.status == "timeout"
    assert "Exceeded" in execution.error
    assert time.monotonic() - started < 0.8


def test_killable_discovery_wrapper_reports_timeout_without_truth(tmp_path: Path) -> None:
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("test uses fork to avoid pickling a pytest module discoverer")
    started = time.monotonic()
    execution = validation.discover_world_with_timeout(
        tmp_path / "synthetic_world",
        "candidate",
        CABossV4Config(),
        tmp_path / "out",
        0.05,
        discoverer=_sleeping_pretruth_discoverer,
    )
    assert execution.status == "timeout"
    assert execution.frozen is None
    assert "Exceeded" in execution.error
    assert time.monotonic() - started < 0.8


def test_killable_frozen_scorer_reports_timeout(tmp_path: Path) -> None:
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("test uses fork to avoid pickling a pytest module scorer")
    frozen = validation.FrozenWorldPretruth(
        world_dir=str(tmp_path / "world_seed31001"),
        world_name="world_seed31001",
        world_out=str(tmp_path / "out" / "world_seed31001"),
        candidate_names=("candidate",),
        pretruth_manifest=str(tmp_path / "pretruth.json"),
        pretruth_manifest_sha256="0" * 64,
    )
    global_manifest = tmp_path / "global.json"
    global_manifest.write_text("{}")
    started = time.monotonic()
    execution = validation.score_world_with_timeout(
        frozen,
        global_manifest,
        0.05,
        scorer=_sleeping_frozen_scorer,
    )
    assert execution.status == "timeout"
    assert "scoring timeout" in execution.error
    assert time.monotonic() - started < 0.8


def test_killable_baseline_wrapper_reports_timeout(tmp_path: Path) -> None:
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("test uses fork to avoid pickling a pytest module evaluator")
    inputs = validation.BaselineObservableInputs(
        world_name="world_seed31001",
        world_seed=31_001,
        matrices=(),
        embedding=tmp_path / "embedding.tsv",
        chrom_cell_tables=(),
        file_sha256={},
    )
    started = time.monotonic()
    execution = validation.evaluate_baseline_with_timeout(
        "frozen_v3",
        inputs,
        tmp_path / "baseline",
        0.05,
        evaluator=_sleeping_baseline_evaluator,
    )
    assert execution.status == "timeout"
    assert "Exceeded" in execution.error
    assert time.monotonic() - started < 0.8


def test_baseline_consensus_votes_endpoints_independently() -> None:
    endpoint_pairs = [
        ("TAIL", "ARROW"),
        ("TAIL", "ARROW"),
        ("TAIL", "CIRCLE"),
        ("TAIL", "CIRCLE"),
        ("CIRCLE", "ARROW"),
        ("CIRCLE", "ARROW"),
    ]
    views = [
        pd.DataFrame(
            {
                "node1": ["a"],
                "endpoint1": [left],
                "node2": ["b"],
                "endpoint2": [right],
            }
        )
        for left, right in endpoint_pairs
    ]

    consensus = validation.consensus_baseline_graph(views)

    assert len(consensus) == 1
    assert consensus.iloc[0]["endpoint1"] == "TAIL"
    assert consensus.iloc[0]["endpoint2"] == "ARROW"


def test_baseline_consensus_rejects_unknown_endpoint() -> None:
    views = [
        pd.DataFrame(
            {
                "node1": ["a"],
                "endpoint1": ["STAR" if index == 0 else "CIRCLE"],
                "node2": ["b"],
                "endpoint2": ["CIRCLE"],
            }
        )
        for index in range(validation.BASELINE_EXPECTED_VIEWS)
    ]

    with pytest.raises(validation.ValidationError, match="invalid endpoints"):
        validation.consensus_baseline_graph(views)


def test_baseline_methods_reuse_exact_prepared_views_and_freeze_consensus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _dummy_worlds(tmp_path / "worlds", (31_001,))[0]
    inputs = _dummy_baseline_inputs(tmp_path, world)
    v3_matrices: list[Path] = []
    fci_matrices: list[Path] = []
    edge = pd.DataFrame(
        {
            "node1": ["peak_a"],
            "endpoint1": ["CIRCLE"],
            "node2": ["peak_b"],
            "endpoint2": ["CIRCLE"],
            "evidence_layers": ["synthetic"],
            "interpretation": ["synthetic"],
        }
    )

    def fake_v3(
        matrix: Path,
        cell_tables: list[Path],
        embedding: Path,
        config: validation.ChannelAwareConfig,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        v3_matrices.append(matrix)
        assert tuple(cell_tables) == inputs.chrom_cell_tables
        assert embedding.name == "v3_analysis_embeddings_compat.tsv"
        adapted = pd.read_csv(embedding, sep="\t")
        assert "retained_target" in adapted
        assert adapted["retained_target"].astype(bool).all()
        assert config == validation._frozen_v3_config(31_001)
        return edge.copy(), {"method": "synthetic_v3"}

    def fake_fci(matrix: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        fci_matrices.append(matrix)
        return edge.copy(), {"method": "synthetic_fci"}

    monkeypatch.setattr(validation, "discover_frozen_v3", fake_v3)
    monkeypatch.setattr(validation, "_run_production_fci_view", fake_fci)
    monkeypatch.setitem(sys.modules, "pytetrad", types.ModuleType("pytetrad"))
    v3 = validation.run_observable_baseline_world(
        "frozen_v3", inputs, tmp_path / "v3"
    )
    fci = validation.run_observable_baseline_world(
        "production_fci", inputs, tmp_path / "fci"
    )

    assert tuple(v3_matrices) == inputs.matrices
    assert tuple(fci_matrices) == inputs.matrices
    assert v3["n_consensus_edges"] == 1
    assert fci["n_consensus_edges"] == 1
    for root, result in ((tmp_path / "v3", v3), (tmp_path / "fci", fci)):
        manifest = json.loads((root / "baseline_pretruth_manifest.json").read_text())
        assert manifest["truth_opened_before_consensus_freeze"] is False
        assert manifest["same_prepared_matrices_as_v4"] is True
        if result is v3:
            assert manifest["v3_embedding_schema_adapter"]["coordinates_changed"] is False
        assert validation.file_sha256(Path(result["consensus"])) == result[
            "consensus_sha256"
        ]

    frozen, parent_audit = validation._load_and_validate_frozen_baseline_result(
        "production_fci", inputs, tmp_path / "fci", fci
    )
    assert len(frozen) == 1
    assert parent_audit["consensus_frame_parsed_before_truth"] is True

    manifest_path = Path(fci["manifest"])
    tampered = json.loads(manifest_path.read_text())
    tampered.pop("manifest_payload_sha256")
    tampered["algorithm_config"]["alpha"] = 0.99
    tampered["manifest_payload_sha256"] = validation.canonical_sha256(tampered)
    _write_json(manifest_path, tampered)
    tampered_result = {
        **fci,
        "manifest_sha256": validation.file_sha256(manifest_path),
        "manifest_payload_sha256": tampered["manifest_payload_sha256"],
    }
    with pytest.raises(validation.ValidationError, match="contract mismatch"):
        validation._load_and_validate_frozen_baseline_result(
            "production_fci", inputs, tmp_path / "fci", tampered_result
        )


def test_production_fci_adapts_observable_background_depth_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_dir = Path(__file__).resolve().parents[4] / "pipelines/empirical/analysis/scripts/commands"
    if str(command_dir) not in sys.path:
        sys.path.insert(0, str(command_dir))
    import run_pc_causallearn as production  # noqa: PLC0415

    monkeypatch.setattr(
        production,
        "run_fci_search",
        lambda *_args, **_kwargs: (None, []),
    )
    rows = 10
    matrix = pd.DataFrame(
        {
            "sample_id": [f"MC{i}" for i in range(rows)],
            "expr__GENE_log1p": list(range(rows)),
            "chr1:1-2__H3K27ac": [i % 3 for i in range(rows)],
            "chr2:1-2__H3K27me3": [(i * 2) % 5 for i in range(rows)],
            "chr3:1-2__H3K4me1": [(i * 3) % 7 for i in range(rows)],
            **{
                f"libsize_background__{assay}": [100 + i + j for i in range(rows)]
                for j, assay in enumerate(
                    ("rna", "H3K27ac", "H3K27me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3")
                )
            },
        }
    )
    path = tmp_path / "observed_metacell_matrix.tsv"
    matrix.to_csv(path, sep="\t", index=False)

    edges, summary = validation._run_production_fci_view(path)

    assert edges.empty
    assert set(summary["depth_aliases_from_observable_background"]) == {
        "libsize__rna",
        "libsize__H3K27ac",
        "libsize__H3K27me3",
        "libsize__H3K4me1",
        "libsize__H3K4me2",
        "libsize__H3K4me3",
        "libsize__H3K9me3",
    }
    assert set(summary["confounder_columns"]) == {
        "conf__rna_depth",
        "conf__depth_H3K27ac",
        "conf__depth_H3K27me3",
        "conf__depth_H3K4me1",
        "conf__depth_H3K4me2",
        "conf__depth_H3K4me3",
        "conf__depth_H3K9me3",
    }


def test_baseline_parent_freezes_frame_before_truth_and_scores_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _dummy_worlds(tmp_path / "worlds", (31_001,))[0]
    _dummy_baseline_inputs(tmp_path, world)
    edge = pd.DataFrame(
        {
            "node1": ["peak_a"],
            "endpoint1": ["TAIL"],
            "node2": ["peak_b"],
            "endpoint2": ["ARROW"],
        }
    )
    monkeypatch.setitem(sys.modules, "pytetrad", types.ModuleType("pytetrad"))
    monkeypatch.setattr(
        validation,
        "discover_frozen_v3",
        lambda *_args, **_kwargs: (edge.copy(), {"method": "synthetic_v3"}),
    )
    monkeypatch.setattr(
        validation,
        "_run_production_fci_view",
        lambda *_args, **_kwargs: (edge.copy(), {"method": "synthetic_fci"}),
    )

    def synchronous_executor(
        method: str,
        inputs: validation.BaselineObservableInputs,
        out_dir: Path,
        _timeout: float,
    ) -> validation.BaselineExecution:
        return validation.BaselineExecution(
            status="ok",
            elapsed_seconds=0.01,
            result=validation.run_observable_baseline_world(method, inputs, out_dir),
        )

    original_loader = validation._load_and_validate_frozen_baseline_result
    state: dict[str, Any] = {"loaded": False, "consensus": None, "scores": 0}

    def traced_loader(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame, audit = original_loader(*args, **kwargs)
        result = args[3]
        state["loaded"] = True
        state["consensus"] = Path(result["consensus"])
        return frame, audit

    original_read_csv = validation.pd.read_csv

    def guarded_read_csv(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(path, (str, Path)) and Path(path).name == "truth_typed_edges.tsv":
            assert state["loaded"] is True
            Path(state["consensus"]).write_text(
                "node1\tendpoint1\tnode2\tendpoint2\n"
                "tampered\tCIRCLE\tafter_truth\tCIRCLE\n"
            )
            return pd.DataFrame({"synthetic_truth": [1]})
        return original_read_csv(path, *args, **kwargs)

    def fake_score(
        frozen_edges: pd.DataFrame,
        _truth: pd.DataFrame,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        assert frozen_edges[["node1", "node2"]].iloc[0].tolist() == [
            "peak_a",
            "peak_b",
        ]
        state["scores"] += 1
        return (
            {"pairwise_diagnostic_success": 1, "diagnostic_only": True},
            pd.DataFrame({"label": ["synthetic"], "adjacency_present": [1]}),
        )

    monkeypatch.setattr(validation, "_load_and_validate_frozen_baseline_result", traced_loader)
    monkeypatch.setattr(validation.pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr(validation, "score_untyped_baseline", fake_score)

    result = validation.run_registered_baseline_diagnostics(
        {31_001: world},
        "development",
        tmp_path / "evaluation",
        tmp_path / "baseline_results",
        60.0,
        {"all_registered_v4_execution_attempts_finished": True},
        baseline_executor=synchronous_executor,
    )

    assert result["pass"] is True
    assert state["scores"] == 2
    assert all(
        record["parent_pretruth_audit"]["consensus_bytes_loaded_before_truth"]
        for record in result["worlds"]
    )


def test_registered_baselines_retain_failures_and_timeouts(
    tmp_path: Path,
) -> None:
    world = _dummy_worlds(tmp_path / "worlds", (31_001,))[0]
    _dummy_baseline_inputs(tmp_path, world)

    def failed_executor(
        method: str,
        _inputs: validation.BaselineObservableInputs,
        _out: Path,
        _timeout: float,
    ) -> validation.BaselineExecution:
        return validation.BaselineExecution(
            status="timeout" if method == "frozen_v3" else "crash",
            elapsed_seconds=1.0,
            result={},
            error=f"synthetic {method} failure",
        )

    result = validation.run_registered_baseline_diagnostics(
        {31_001: world},
        "development",
        tmp_path / "evaluation",
        tmp_path / "baseline_results",
        60.0,
        {"all_registered_v4_execution_attempts_finished": True},
        baseline_executor=failed_executor,
    )

    assert result["pass"] is False
    assert result["methods"]["frozen_v3"]["execution_status_counts"] == {
        "timeout": 1
    }
    assert result["methods"]["production_fci"]["execution_status_counts"] == {
        "crash": 1
    }
    rows = pd.read_csv(
        tmp_path / "baseline_results" / "baseline_world_results.tsv", sep="\t"
    )
    assert len(rows) == 2


def test_registered_baselines_retain_executor_exceptions_and_continue(
    tmp_path: Path,
) -> None:
    worlds = _dummy_worlds(tmp_path / "worlds", (31_001, 31_002))
    for world in worlds:
        _dummy_baseline_inputs(tmp_path, world)
    calls: list[tuple[str, int]] = []

    def raising_executor(
        method: str,
        inputs: validation.BaselineObservableInputs,
        _out: Path,
        _timeout: float,
    ) -> validation.BaselineExecution:
        calls.append((method, inputs.world_seed))
        raise RuntimeError(f"synthetic {method} {inputs.world_seed}")

    result = validation.run_registered_baseline_diagnostics(
        {validation.extract_world_seed(world): world for world in worlds},
        "development",
        tmp_path / "evaluation",
        tmp_path / "baseline_results",
        60.0,
        {"all_registered_v4_execution_attempts_finished": True},
        baseline_executor=raising_executor,
    )

    assert calls == [
        (method, seed)
        for method in validation.BASELINE_METHODS
        for seed in (31_001, 31_002)
    ]
    assert result["pass"] is False
    assert all(
        summary["execution_status_counts"] == {"executor_exception": 2}
        for summary in result["methods"].values()
    )


def test_locked_requires_immutable_dev_and_passing_upper_bound(
    tmp_path: Path,
) -> None:
    audit, config, audit_hash, _ = _frozen_training_candidate(tmp_path / "freeze")
    dev_worlds = _dummy_worlds(tmp_path / "dev_worlds", validation.DEVELOPMENT_SEEDS)
    dev = validation.run_validation(
        split="development",
        world_dirs=dev_worlds,
        frozen_config=config,
        calibration_audit=audit,
        calibration_audit_sha256=audit_hash,
        out_dir=tmp_path / "dev_validation",
        timeout_seconds=60.0,
        discovery_executor=_passing_discovery_executor,
        scoring_executor=_passing_scoring_executor,
        provenance_auditor=_passing_audit,
        baseline_runner=_passing_baselines,
        upper_bound_runner=_passing_upper_bound,
    )
    locked_worlds = _dummy_worlds(tmp_path / "locked_worlds", validation.FRESH_LOCKED_SEEDS)
    called = False

    def must_not_run(*_args: Any, **_kwargs: Any) -> validation.WorldExecution:
        nonlocal called
        called = True
        raise AssertionError("locked evaluator must not run before DEV validation")

    with pytest.raises(validation.ValidationError, match="requires a passing immutable DEV"):
        validation.run_validation(
            split="locked",
            world_dirs=locked_worlds,
            frozen_config=config,
            calibration_audit=audit,
            calibration_audit_sha256=audit_hash,
            out_dir=tmp_path / "locked_without_dev",
            timeout_seconds=60.0,
            discovery_executor=must_not_run,
            baseline_runner=_passing_baselines,
            upper_bound_runner=_passing_upper_bound,
        )

    def failing_upper_bound(
        _worlds: dict[int, Path],
        _split: str,
        _out: Path,
    ) -> dict[str, Any]:
        return {"pass": False}

    assert not called

    locked = validation.run_validation(
        split="locked",
        world_dirs=locked_worlds,
        frozen_config=config,
        calibration_audit=audit,
        calibration_audit_sha256=audit_hash,
        out_dir=tmp_path / "locked_validation",
        timeout_seconds=60.0,
        development_manifest=Path(dev["manifest_path"]),
        development_manifest_sha256=dev["manifest_file_sha256"],
        discovery_executor=_passing_discovery_executor,
        scoring_executor=_passing_scoring_executor,
        provenance_auditor=_passing_audit,
        baseline_runner=_passing_baselines,
        upper_bound_runner=failing_upper_bound,
    )
    assert locked["truth_boundary"]["all_candidate_outputs_frozen_before_upper_bound"]
    assert not locked["gates"]["upper_bound_gate"]["pass"]
    assert not locked["candidate_gate_pass"]


def test_locked_rejects_rehashed_dev_manifest_with_source_drift(tmp_path: Path) -> None:
    audit, config, audit_hash, _ = _frozen_training_candidate(tmp_path / "freeze")
    dev_worlds = _dummy_worlds(tmp_path / "dev_worlds", validation.DEVELOPMENT_SEEDS)
    dev = validation.run_validation(
        split="development",
        world_dirs=dev_worlds,
        frozen_config=config,
        calibration_audit=audit,
        calibration_audit_sha256=audit_hash,
        out_dir=tmp_path / "dev_validation",
        timeout_seconds=60.0,
        discovery_executor=_passing_discovery_executor,
        scoring_executor=_passing_scoring_executor,
        provenance_auditor=_passing_audit,
        baseline_runner=_passing_baselines,
        upper_bound_runner=_passing_upper_bound,
    )
    path = Path(dev["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest.pop("manifest_payload_sha256")
    manifest["source_freeze"]["files"]["algorithm"]["sha256"] = "f" * 64
    manifest["manifest_payload_sha256"] = validation.canonical_sha256(manifest)
    _write_json(path, manifest)

    frozen = validation.load_frozen_candidate(config, audit, audit_hash)
    with pytest.raises(validation.ValidationError, match="sources changed"):
        validation.validate_development_manifest(
            path,
            validation.file_sha256(path),
            frozen=frozen,
            current_source_hashes=validation.source_hash_bundle(),
        )


def test_world_provenance_audit_checks_single_pretruth_hash_and_safe_inputs(
    tmp_path: Path,
) -> None:
    world = _dummy_worlds(tmp_path / "worlds", (31_001,))[0]
    evaluation = tmp_path / "evaluation"
    world_out = evaluation / world.name
    candidate = "frozen"
    candidate_out = world_out / candidate
    candidate_out.mkdir(parents=True)
    consensus = candidate_out / "consensus_edges.tsv"
    consensus.write_text("node1\tendpoint1\tnode2\tendpoint2\tedge_type\n")
    pair_scores = pd.DataFrame(
        {
            "truth_type": ["sentinel_null"] * 15,
            "adjacency_present": [0] * 15,
        }
    )
    pair_scores.to_csv(candidate_out / "typed_pair_scores.tsv", sep="\t", index=False)
    projection = candidate_out / "loading_group_projection"
    projection.mkdir()
    hyperedges = projection / "typed_group_hyperedges.tsv"
    sparse_projection = projection / "sparse_pair_projection.tsv"
    hyperedges.write_text("hyperedge_id\tedge_type\n")
    sparse_projection.write_text("node1\tnode2\tedge_type\n")
    (candidate_out / "latent_group_metrics.json").write_text("{}\n")
    (candidate_out / "latent_axis_hyperedge_scores.tsv").write_text(
        "hyperedge_id\ttyped_valid\n"
    )
    axis_projection = candidate_out / "latent_axis_projection"
    axis_projection.mkdir()
    axis_files = {
        "group_pair_view_evidence.tsv": "group1\tgroup2\n",
        "group_pair_evidence.tsv": "group1\tgroup2\n",
        "axis_candidates.tsv": "axis_id\tselection_status\n",
        "group_axis_view_evidence.tsv": "axis_id\tloading_group\n",
        "group_axis_evidence.tsv": "axis_id\tloading_group\n",
        "typed_axis_hyperedges.tsv": "hyperedge_id\taxis_id\tedge_type\n",
        "representative_evidence.tsv": "axis_id\tfeature\n",
        "sparse_pair_projection.tsv": "node1\tnode2\tedge_type\n",
        "axis_provenance.json": "{}\n",
    }
    for filename, contents in axis_files.items():
        (axis_projection / filename).write_text(contents)
    axis_file_hashes = {
        filename: validation.file_sha256(axis_projection / filename)
        for filename in axis_files
    }
    projection_hash = validation.hashlib.sha256(
        (hyperedges.read_text() + sparse_projection.read_text()).encode("utf-8")
    ).hexdigest()
    _write_json(
        world_out / "pretruth_consensus_manifest.json",
        {
            "candidate_count": 1,
            "all_consensuses_built_before_truth_open": True,
            "consensus_sha256": {candidate: validation.file_sha256(consensus)},
            "loading_group_projection_sha256": {candidate: projection_hash},
            "latent_axis_projection_sha256": {candidate: "a" * 64},
            "latent_axis_projection_file_sha256": {
                candidate: axis_file_hashes,
            },
            "latent_axis_hyperedges_built_before_truth_open": True,
            "latent_axis_hyperedges_attached_once_post_consensus": True,
            "latent_axis_hyperedges_expanded_into_pairwise_skeleton": False,
            "ordinary_state_skeleton_contains_within_assay_groups_only": True,
            "pairwise_cross_group_matches_are_diagnostics_only": True,
            "loading_views_reuse_same_observed_cells": True,
            "loading_views_are_algorithmic_stability_repeats": True,
            "loading_views_are_independent_worlds": False,
            "loading_view_support_not_world_sample_size": True,
            "preliminary_pairwise_state_layer_replaced": True,
        },
    )
    prepared = world_out / "prepared_discovery"
    prepared.mkdir()
    embedding = prepared / "analysis_embeddings.tsv"
    embedding.write_text("x\n")
    for index in range(6):
        nested = prepared / "nested" / f"cells{index}_seed0"
        nested.mkdir(parents=True)
        matrix = nested / "observed_metacell_matrix.tsv"
        membership = nested / "analysis_membership.tsv"
        matrix.write_text("x\n")
        membership.write_text("x\n")
        summary_dir = world_out / f"cells{index}_seed0"
        summary_dir.mkdir()
        _write_json(
            summary_dir / "discovery_summary.json",
            {
                "method": "ca_boss_v4_explicit_state",
                "inputs": {
                    "metacell_matrix": str(matrix),
                    "embedding_table": str(embedding),
                    "membership_table": str(membership),
                    "cell_tables": [],
                },
                "provenance": {
                    "embedding_input": str(embedding),
                    "membership_input": str(membership),
                },
            },
        )

    audit = validation.audit_world_output(world, evaluation, candidate)
    assert audit["pass"]
    assert audit["null_adjacency_opportunities"] == 15

    axis_table = axis_projection / "typed_axis_hyperedges.tsv"
    original_axis_table = axis_table.read_text()
    axis_table.write_text(original_axis_table + "tampered\taxis\trow\n")
    axis_failed = validation.audit_world_output(world, evaluation, candidate)
    assert not axis_failed["pass"]
    assert any(
        "latent-axis artifact differs" in reason
        for reason in axis_failed["reasons"]
    )
    axis_table.write_text(original_axis_table)

    parameters_path = world / "scorer" / "world_parameters.json"
    parameters = json.loads(parameters_path.read_text())
    parameters["seed"] = 31_002
    _write_json(parameters_path, parameters)
    scorer_manifest_path = world / "scorer" / "manifest.json"
    scorer_manifest = json.loads(scorer_manifest_path.read_text())
    scorer_manifest["file_sha256"]["world_parameters.json"] = validation.file_sha256(
        parameters_path
    )
    _write_json(scorer_manifest_path, scorer_manifest)
    run_summary_path = world / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text())
    run_summary["scorer_manifest_sha256"] = validation.file_sha256(scorer_manifest_path)
    _write_json(run_summary_path, run_summary)
    identity_failed = validation.audit_world_output(world, evaluation, candidate)
    assert not identity_failed["pass"]
    assert any(
        "scorer-parameter seed" in reason
        for reason in identity_failed["reasons"]
    )
    parameters["seed"] = 31_001
    _write_json(parameters_path, parameters)
    scorer_manifest["file_sha256"]["world_parameters.json"] = validation.file_sha256(
        parameters_path
    )
    _write_json(scorer_manifest_path, scorer_manifest)
    run_summary["scorer_manifest_sha256"] = validation.file_sha256(scorer_manifest_path)
    _write_json(run_summary_path, run_summary)

    unsafe = world_out / "cells0_seed0" / "discovery_summary.json"
    value = json.loads(unsafe.read_text())
    value["inputs"]["embedding_table"] = str(world / "scorer" / "truth_typed_edges.tsv")
    _write_json(unsafe, value)
    failed = validation.audit_world_output(world, evaluation, candidate)
    assert not failed["pass"]
    assert any("forbidden discovery input" in reason for reason in failed["reasons"])
