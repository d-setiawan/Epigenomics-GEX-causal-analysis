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

import run_shared_state_subspace_block_v4 as runner  # noqa: E402
import score_shared_state_subspace_v4 as scorer  # noqa: E402


def _fake_successful_discovery(
    monkeypatch: pytest.MonkeyPatch,
    *,
    worlds_root: Path,
    output_root: Path,
    seed: int = 40_001,
    events: list[str] | None = None,
) -> dict[str, object]:
    events = events if events is not None else []
    world = worlds_root / runner.world_name(seed)
    (world / "discovery").mkdir(parents=True)
    registration_hash = "a" * 64

    def fake_validate(path: Path, *, verify_scorer_payload: bool) -> dict[str, object]:
        assert Path(path) == world
        assert verify_scorer_payload is False
        events.append("validate_discovery")
        return {"binding": True}

    def fake_local(
        world_dir: Path,
        candidates: list[tuple[str, object]],
        *,
        out_dir: Path,
        save_candidate_names: set[str],
        loading_embedding_path: Path | None = None,
        state_membership_path: Path | None = None,
    ) -> SimpleNamespace:
        assert Path(world_dir) == world
        assert len(candidates) == 1
        name, config = candidates[0]
        assert name == scorer.FROZEN_LOCAL_CANDIDATE_NAME
        assert config.alignment_mode == "hard"
        assert config.loading_min_abs_cosine == pytest.approx(0.85)
        assert asdict(config) == scorer.FROZEN_LOCAL_CONFIG
        assert save_candidate_names == {name}
        assert Path(loading_embedding_path) == (
            output_root
            / "artifacts"
            / world.name
            / "assay_local_state_embedding_v4"
            / "analysis_embeddings.tsv"
        )
        assert Path(state_membership_path) == (
            output_root
            / "artifacts"
            / world.name
            / "shared_state_subspace_v4"
            / "subspace_feature_membership.tsv"
        )
        world_out = out_dir / world.name
        (world_out / "prepared_discovery").mkdir(parents=True)
        (world_out / "prepared_discovery" / "analysis_embeddings.tsv").write_text(
            "modality\tcell\tintegration__dim0\nrna\tc0\t0.0\n"
        )
        candidate = world_out / name
        candidate.mkdir()
        (candidate / "consensus_edges.tsv").write_text(
            "node1\tendpoint1\tnode2\tendpoint2\tedge_type\n"
        )
        loading_projection = candidate / "loading_group_projection"
        loading_projection.mkdir()
        loading_groups = loading_projection / "loading_groups.tsv"
        loading_group_membership = (
            loading_projection / "loading_group_membership.tsv"
        )
        loading_groups.write_text("loading_group\tmodality\n")
        loading_group_membership.write_text(
            "loading_group\tmodality\tfeature\n"
        )
        axis_projection = candidate / "latent_axis_projection"
        axis_projection.mkdir()
        group_axis_evidence = axis_projection / "group_axis_evidence.tsv"
        axis_candidates = axis_projection / "axis_candidates.tsv"
        group_axis_evidence.write_text("loading_group\taxis_id\n")
        axis_candidates.write_text("axis_id\tselection_status\n")
        structure_anchor = candidate / "structure_anchored_loading_groups"
        structure_anchor.mkdir()
        structure_pair_evidence = (
            structure_anchor / "structure_anchor_pair_evidence.tsv"
        )
        structure_pair_evidence.write_text(
            "pair_key\topposite_parent_pair_key\tmodality\t"
            "parent_candidate\tnoncollider_sibling_candidate\t"
            "collider_center\tother_parent_candidate\tselected\t"
            "selection_stage\n"
        )
        manifest = world_out / "pretruth_consensus_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "loading_group_projection_file_sha256": {
                        name: {
                            "loading_groups.tsv": scorer.file_sha256(
                                loading_groups
                            ),
                            "loading_group_membership.tsv": scorer.file_sha256(
                                loading_group_membership
                            ),
                        }
                    },
                    "latent_axis_projection_file_sha256": {
                        name: {
                            "group_axis_evidence.tsv": scorer.file_sha256(
                                group_axis_evidence
                            ),
                            "axis_candidates.tsv": scorer.file_sha256(
                                axis_candidates
                            ),
                        }
                    },
                    "structure_anchored_loading_file_sha256": {
                        name: {
                            "structure_anchor_pair_evidence.tsv": (
                                scorer.file_sha256(structure_pair_evidence)
                            )
                        }
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        events.append("local_discovery")
        return SimpleNamespace(
            candidate_names=(name,),
            pretruth_manifest=str(manifest),
        )

    def fake_subspace(
        focal_discovery_dir: Path,
        safe_embedding_path: Path,
        out_dir: Path,
        *,
        config: object,
    ) -> dict[str, object]:
        assert Path(focal_discovery_dir) == world / "discovery"
        assert Path(safe_embedding_path) == (
            output_root
            / "artifacts"
            / world.name
            / "assay_local_state_embedding_v4"
            / "analysis_embeddings.tsv"
        )
        assert asdict(config) == scorer.FROZEN_DISCOVERY_CONFIG
        out = Path(out_dir)
        out.mkdir(parents=True)
        (out / "subspace_feature_membership.tsv").write_text("feature\n")
        (out / "typed_subspace_hyperedges.tsv").write_text("subspace_id\n")
        (out / "subspace_provenance.json").write_text("{}\n")
        (out / "frozen_discovery_manifest.json").write_text("{}\n")
        events.append("subspace_discovery")
        return {"truth_or_scorer_used": False}

    def fake_layered(
        membership_path: Path,
        subspace_hyperedges_path: Path,
        consensus_edges_path: Path,
        loading_groups_path: Path,
        loading_group_membership_path: Path,
        structure_anchor_pair_evidence_path: Path,
        out_dir: Path,
        *,
        config: object,
        empirical_null_calibrator_path: Path,
        multievidence_calibrator_path: Path,
        topology_consensus_calibrator_path: Path,
        group_axis_evidence_path: Path,
        axis_candidates_path: Path,
    ) -> dict[str, object]:
        paths = runner._artifact_paths(output_root, seed)
        expected_inputs = {
            "membership": paths["feature_membership"],
            "subspace_hyperedges": paths["typed_subspace_hyperedges"],
            "consensus_edges": paths["local_consensus_edges"],
            "loading_groups": paths["loading_groups"],
            "loading_group_membership": paths["loading_group_membership"],
            "structure_anchor_pair_evidence": paths[
                "structure_anchor_pair_evidence"
            ],
            "group_axis_evidence": paths["group_axis_evidence"],
            "axis_candidates": paths["axis_candidates"],
            "empirical_null_calibrator": Path(
                empirical_null_calibrator_path
            ).resolve(strict=True),
            "multievidence_calibrator": Path(
                multievidence_calibrator_path
            ).resolve(strict=True),
            "topology_consensus_calibrator": Path(
                topology_consensus_calibrator_path
            ).resolve(strict=True),
        }
        observed_inputs = {
            "membership": Path(membership_path),
            "subspace_hyperedges": Path(subspace_hyperedges_path),
            "consensus_edges": Path(consensus_edges_path),
            "loading_groups": Path(loading_groups_path),
            "loading_group_membership": Path(loading_group_membership_path),
            "structure_anchor_pair_evidence": Path(
                structure_anchor_pair_evidence_path
            ),
            "group_axis_evidence": Path(group_axis_evidence_path),
            "axis_candidates": Path(axis_candidates_path),
            "empirical_null_calibrator": Path(
                empirical_null_calibrator_path
            ).resolve(strict=True),
            "multievidence_calibrator": Path(
                multievidence_calibrator_path
            ).resolve(strict=True),
            "topology_consensus_calibrator": Path(
                topology_consensus_calibrator_path
            ).resolve(strict=True),
        }
        assert observed_inputs == expected_inputs
        assert asdict(config) == asdict(runner._frozen_layered_config())
        out = Path(out_dir)
        assert out == paths["layered_out"]
        out.mkdir()
        outputs = {
            "layered_feature_evidence.tsv": out
            / "layered_feature_evidence.tsv",
            "typed_layered_hyperedges.tsv": out
            / "typed_layered_hyperedges.tsv",
            "within_assay_direct_candidates.tsv": out
            / "within_assay_direct_candidates.tsv",
            "layered_provenance.json": out / "layered_provenance.json",
        }
        outputs["layered_feature_evidence.tsv"].write_text(
            "feature\tmodality\n"
        )
        outputs["typed_layered_hyperedges.tsv"].write_text("relation_id\n")
        outputs["within_assay_direct_candidates.tsv"].write_text("node1\n")
        outputs["layered_provenance.json"].write_text(
            json.dumps(
                {
                    "schema_version": runner.LAYERED_PROVENANCE_SCHEMA_VERSION,
                    "config": asdict(config),
                    "semantics": {
                        "layer_union_is_state_membership": False,
                        "assay_participation_implies_feature_membership": False,
                        "direct_layer_implies_state_membership": False,
                        "loading_group_implies_cross_assay_identity": False,
                        "state_route_loading_group_members_are_subset_of_state_v427_members": True,
                        "axis_route_loading_group_may_contain_one_unresolved_state_member": True,
                        "axis_route_changes_state_v427_membership": False,
                        "axis_route_implies_exact_cross_assay_feature_pair": False,
                        "loading_group_direct_structure_alone_is_insufficient": True,
                        "loading_group_guard_changes_state_v427_membership": False,
                        "cross_assay_pair_claimed": False,
                        "direct_cross_assay_effect_claimed": False,
                        "exact_latent_axis_claimed": False,
                        "state_equivalence_is_pairwise_structure": False,
                        "state_equivalence_implies_exact_cross_assay_pair": False,
                        "state_equivalence_implies_direct_causation": False,
                        "strict_membership_changed_by_equivalence_tier": False,
                        "empirical_null_membership_is_pairwise_structure": False,
                        "empirical_null_membership_implies_direct_causation": False,
                        "empirical_null_membership_implies_exact_cross_assay_pair": False,
                        "existing_state_membership_changed_by_empirical_null_tier": False,
                        "multievidence_membership_is_pairwise_structure": False,
                        "multievidence_membership_implies_direct_causation": False,
                        "multievidence_membership_implies_exact_cross_assay_pair": False,
                        "existing_v425_membership_changed_by_multievidence_tier": False,
                        "topology_consensus_membership_is_pairwise_structure": False,
                        "topology_consensus_membership_implies_direct_causation": False,
                        "topology_consensus_membership_implies_exact_cross_assay_pair": False,
                        "direct_structure_alone_implies_state_membership": False,
                        "existing_v426_membership_changed_by_topology_consensus_tier": False,
                    },
                    "information_boundary": {
                        "consumed": [
                            "frozen_observable_subspace_membership",
                            "frozen_observable_subspace_hyperedge",
                            "frozen_observable_typed_consensus_graph",
                            "frozen_observable_assay_local_loading_groups",
                            "frozen_observable_redundancy_demotion_annotations",
                            "optional_frozen_numeric_empirical_null_state_calibrator",
                            "optional_frozen_numeric_multievidence_state_calibrator",
                            "optional_frozen_topology_consensus_state_calibrator",
                            "frozen_v427_state_membership_for_loading_group_guard",
                            "optional_frozen_leave_one_group_out_group_axis_evidence",
                            "optional_frozen_selected_cross_assay_axis_candidates",
                        ],
                        "not_consumed": [
                            "true_latent_state",
                            "simulator_roles",
                            "truth_graph",
                            "scorer",
                            "paired_cells",
                            "external_data",
                            "row_level_calibrator_training_labels_at_application_time",
                        ],
                    },
                    "empirical_null_calibrator": {
                        **runner._frozen_empirical_null_calibrator(
                            empirical_null_calibrator_path
                        )[1],
                        "selection_threshold": runner._frozen_empirical_null_calibrator(
                            empirical_null_calibrator_path
                        )[0].selection_threshold,
                        "schema_version": runner.layered_evidence.empirical_null_state.SCHEMA,
                        "numeric_payload_sha256": runner._frozen_empirical_null_calibrator(
                            empirical_null_calibrator_path
                        )[1]["numeric_calibrator_payload_sha256"],
                        "creates_pairwise_or_direct_relations": False,
                    },
                    "multievidence_calibrator": {
                        **runner._frozen_multievidence_calibrator(
                            multievidence_calibrator_path
                        )[1],
                        "selection_threshold": runner._frozen_multievidence_calibrator(
                            multievidence_calibrator_path
                        )[0].selection_threshold,
                        "schema_version": runner.layered_evidence.multievidence_state.SCHEMA,
                        "numeric_payload_sha256": runner._frozen_multievidence_calibrator(
                            multievidence_calibrator_path
                        )[1]["numeric_calibrator_payload_sha256"],
                        "creates_pairwise_or_direct_relations": False,
                    },
                    "topology_consensus_calibrator": {
                        **runner._frozen_topology_consensus_calibrator(
                            topology_consensus_calibrator_path
                        )[1],
                        "schema_version": runner.layered_evidence.topology_consensus_state.SCHEMA,
                        "numeric_payload_sha256": runner._frozen_topology_consensus_calibrator(
                            topology_consensus_calibrator_path
                        )[1]["numeric_calibrator_payload_sha256"],
                        "minimum_direct_candidate_degree": 1,
                        "state_specific_gate": runner.layered_evidence.topology_consensus_state.STATE_SPECIFIC_GATE,
                        "direct_structure_alone_is_insufficient": True,
                        "creates_pairwise_or_direct_relations": False,
                    },
                    "loading_group_state_guard": {
                        "active": True,
                        "schema_version": runner.layered_evidence.loading_group_state_guard.SCHEMA,
                        "fixed_rule": runner.layered_evidence.loading_group_state_guard.FIXED_RULE,
                        "state_route": runner.layered_evidence.loading_group_state_guard.STATE_ROUTE,
                        "cross_assay_axis_route": runner.layered_evidence.loading_group_state_guard.AXIS_ROUTE,
                        "cross_assay_axis_evidence_active": True,
                        "leave_one_group_out_axis_evidence_required_for_axis_route": True,
                        "upstream_state_membership_schema": runner.layered_evidence.topology_consensus_state.SCHEMA,
                        "direct_structure_alone_is_insufficient": True,
                        "application_requires_truth": False,
                        "creates_cross_assay_or_direct_relations": False,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        output_hashes = {
            name: scorer.file_sha256(path)
            for name, path in sorted(outputs.items())
        }
        manifest = {
            "schema_version": runner.LAYERED_CANDIDATE_SCHEMA_VERSION,
            "algorithm_source_sha256": scorer.file_sha256(
                Path(runner.layered_evidence.__file__).resolve(strict=True)
            ),
            "dependency_source_sha256": {
                "empirical_null_state_calibration_v4.py": scorer.file_sha256(
                    Path(
                        runner.layered_evidence.empirical_null_state.__file__
                    ).resolve(strict=True)
                ),
                "shared_state_subspace_v4.py": scorer.file_sha256(
                    Path(runner.layered_evidence.state_subspace.__file__).resolve(
                        strict=True
                    )
                ),
                "loading_group_state_guard_v4.py": scorer.file_sha256(
                    Path(
                        runner.layered_evidence.loading_group_state_guard.__file__
                    ).resolve(strict=True)
                ),
                "loading_group_structure_uncertainty_guard_v4.py": scorer.file_sha256(
                    Path(
                        runner.layered_evidence
                        .loading_group_structure_guard.__file__
                    ).resolve(strict=True)
                ),
                "loading_group_competing_parent_observability_guard_v4.py": scorer.file_sha256(
                    Path(
                        runner.layered_evidence
                        .loading_group_competitor_guard.__file__
                    ).resolve(strict=True)
                ),
                "loading_group_structure_equivalence_v4.py": scorer.file_sha256(
                    Path(
                        runner.layered_evidence
                        .loading_group_structure_equivalence.__file__
                    ).resolve(strict=True)
                ),
                "multievidence_empirical_null_state_calibration_v4.py": scorer.file_sha256(
                    Path(
                        runner.layered_evidence.multievidence_state.__file__
                    ).resolve(strict=True)
                ),
                "state_equivalence_membership_v4.py": scorer.file_sha256(
                    Path(
                        runner.layered_evidence.state_equivalence.__file__
                    ).resolve(strict=True)
                ),
                "topology_consensus_state_calibration_v4.py": scorer.file_sha256(
                    Path(
                        runner.layered_evidence.topology_consensus_state.__file__
                    ).resolve(strict=True)
                ),
            },
            "config": asdict(config),
            "config_sha256": scorer.canonical_sha256(asdict(config)),
            "input_sha256": {
                label: scorer.file_sha256(path)
                for label, path in sorted(expected_inputs.items())
            },
            "output_sha256": output_hashes,
            "empirical_null_calibrator": runner._frozen_empirical_null_calibrator(
                empirical_null_calibrator_path
            )[1],
            "multievidence_calibrator": runner._frozen_multievidence_calibrator(
                multievidence_calibrator_path
            )[1],
            "topology_consensus_calibrator": runner._frozen_topology_consensus_calibrator(
                topology_consensus_calibrator_path
            )[1],
            "candidate_frozen": True,
            "scorer_access_chronology": (
                "not_asserted_by_this_standalone_module"
            ),
            "truth_or_scorer_consumed_by_algorithm": False,
        }
        manifest_path = out / "frozen_layered_candidate_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        events.append("layered_discovery")
        return {
            **manifest,
            "manifest_sha256": scorer.file_sha256(manifest_path),
        }

    def fake_assay_local_embedding(
        focal_discovery_dir: Path,
        out_dir: Path,
    ) -> dict[str, object]:
        assert Path(focal_discovery_dir) == world / "discovery"
        out = Path(out_dir)
        assert out.name == "assay_local_state_embedding_v4"
        out.mkdir(parents=True)
        (out / "analysis_embeddings.tsv").write_text(
            "modality\tcell\tintegration__dim0\tintegration__dim1\t"
            "integration__dim2\tintegration_uncertainty\n"
        )
        (out / "assay_local_state_provenance.json").write_text("{}\n")
        (out / "frozen_evidence_manifest.json").write_text("{}\n")
        (out / "frozen_evidence_manifest.sha256").write_text("0" * 64 + "\n")
        events.append("assay_local_embedding")
        return {"truth_or_scorer_used": False}

    def forbidden_scorer(*args: object, **kwargs: object) -> object:
        raise AssertionError("scorer domain was opened during Phase A")

    monkeypatch.setattr(
        runner.local_benchmark, "validate_world_input_binding", fake_validate
    )
    monkeypatch.setattr(
        runner.local_benchmark, "discover_world_pretruth", fake_local
    )
    monkeypatch.setattr(runner, "run_shared_state_subspace", fake_subspace)
    monkeypatch.setattr(
        runner.layered_evidence,
        "run_layered_state_evidence",
        fake_layered,
    )
    monkeypatch.setattr(
        runner.assay_local_embedding,
        "write_assay_local_state_embedding",
        fake_assay_local_embedding,
    )
    monkeypatch.setattr(
        runner.local_benchmark,
        "load_scorer_artifacts_after_pretruth_freeze",
        forbidden_scorer,
    )
    return runner._discover_one(
        seed=seed,
        worlds_root=worlds_root,
        output_root=output_root,
        registration_sha256=registration_hash,
        generate_missing=False,
    )


def test_exact_registered_seed_block_rejects_extra_worlds(tmp_path: Path) -> None:
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    resolved = runner.registered_world_paths("TRAIN", worlds)
    assert tuple(resolved) == scorer.FRESH_TRAIN_SEEDS
    assert runner.registered_seeds("DEV") == scorer.FRESH_DEVELOPMENT_SEEDS
    assert runner.registered_seeds("LOCKED") == scorer.FRESH_LOCKED_SEEDS

    first_train_seed = scorer.FRESH_TRAIN_SEEDS[0]
    (worlds / runner.world_name(first_train_seed)).mkdir()
    assert runner.registered_world_paths("TRAIN", worlds)[first_train_seed].is_dir()
    (worlds / "world_seed36001").mkdir()
    (worlds / "world_seed37001").mkdir()
    (worlds / "world_seed38001").mkdir()
    (worlds / "world_seed39001").mkdir()
    with pytest.raises(ValueError, match="outside the exact registered seed block"):
        runner.registered_world_paths("TRAIN", worlds)


def test_runner_v15_binds_layered_v432_structure_equivalence_contract() -> None:
    assert runner.RUNNER_SCHEMA_VERSION == "shared_state_subspace_v4.block_runner.v15"
    assert runner.REGISTRATION_SCHEMA_VERSION.endswith(".v15")
    assert runner.ATTEMPT_SCHEMA_VERSION.endswith(".v15")
    assert runner.PHASE_A_SCHEMA_VERSION.endswith(".v15")
    assert runner.RESULT_RECORDS_SCHEMA_VERSION.endswith(".v15")
    assert runner.RUN_SUMMARY_SCHEMA_VERSION.endswith(".v15")
    assert runner.LAYERED_GLOBAL_EXTENSION_SCHEMA_VERSION.endswith(".v13")
    assert runner.LAYERED_CANDIDATE_SCHEMA_VERSION.endswith(".v12")
    assert runner.LAYERED_PROVENANCE_SCHEMA_VERSION.endswith(".v12")
    assert (
        "loading_group_structure_equivalence_v4.py"
        in scorer.REQUIRED_JOINT_EXECUTABLE_SOURCES
    )
    assert asdict(runner._frozen_layered_config()) == {
        "maximum_confirmatory_feature_maxT_p": 0.10,
        "maximum_state_equivalence_maxT_p": 0.01,
        "maximum_assay_participation_error_probability": 0.10,
        "minimum_confirmed_modalities": 2,
    }
    assert runner._LAYERED_OUTPUT_NAMES == {
        "layered_feature_evidence.tsv",
        "typed_layered_hyperedges.tsv",
        "within_assay_direct_candidates.tsv",
        "layered_provenance.json",
        "frozen_layered_candidate_manifest.json",
    }


def test_phase_a_uses_hard_prep_and_never_opens_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worlds = tmp_path / "worlds"
    output = tmp_path / "output"
    events: list[str] = []
    record = _fake_successful_discovery(
        monkeypatch,
        worlds_root=worlds,
        output_root=output,
        events=events,
    )
    assert record["status"] == "ok", record
    assert record["scorer_domain_read_during_discovery"] is False
    assert events == [
        "validate_discovery",
        "assay_local_embedding",
        "subspace_discovery",
        "local_discovery",
        "layered_discovery",
    ]


def test_resume_rejects_layered_output_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worlds = tmp_path / "worlds"
    output = tmp_path / "output"
    record = _fake_successful_discovery(
        monkeypatch,
        worlds_root=worlds,
        output_root=output,
    )
    paths = runner._artifact_paths(output, 40_001)
    paths["layered_feature_evidence"].write_text("tampered\n")
    with pytest.raises(ValueError, match="frozen layered artifact changed"):
        runner._validate_attempt_record(
            record,
            seed=40_001,
            registration_sha256="a" * 64,
            output_root=output,
        )


def test_resume_rejects_retired_layered_v1_candidate_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worlds = tmp_path / "worlds"
    output = tmp_path / "output"
    record = _fake_successful_discovery(
        monkeypatch,
        worlds_root=worlds,
        output_root=output,
    )
    paths = runner._artifact_paths(output, 40_001)
    manifest = json.loads(paths["layered_candidate_manifest"].read_text())
    manifest["schema_version"] = "layered_state_evidence_v4.frozen_candidate.v1"
    paths["layered_candidate_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    stale_record = dict(record)
    stale_record.pop("attempt_payload_sha256")
    stale_record["layered_candidate_manifest_sha256"] = scorer.file_sha256(
        paths["layered_candidate_manifest"]
    )
    stale_record = runner._self_hashed(
        stale_record, "attempt_payload_sha256"
    )
    with pytest.raises(ValueError, match="unsupported or not truth-blind"):
        runner._validate_attempt_record(
            stale_record,
            seed=40_001,
            registration_sha256="a" * 64,
            output_root=output,
        )


def test_resume_rejects_formal_embedding_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worlds = tmp_path / "worlds"
    output = tmp_path / "output"
    record = _fake_successful_discovery(
        monkeypatch,
        worlds_root=worlds,
        output_root=output,
    )
    paths = runner._artifact_paths(output, 40_001)
    paths["subspace_embedding"].write_text("tampered\n")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        runner._validate_attempt_record(
            record,
            seed=40_001,
            registration_sha256="a" * 64,
            output_root=output,
        )


def test_result_records_require_every_frozen_base_and_layered_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = runner.registered_seeds("TRAIN")
    layered_outputs = {
        "layered_feature_evidence.tsv": "5" * 64,
        "layered_provenance.json": "6" * 64,
        "typed_layered_hyperedges.tsv": "7" * 64,
        "within_assay_direct_candidates.tsv": "8" * 64,
    }
    frozen_record = {
        "status": "ok",
        "candidate_manifest_sha256": "a" * 64,
        "subspace_embedding_manifest_sha256": "2" * 64,
        "subspace_embedding_sha256": "3" * 64,
        "local_pretruth_manifest_sha256": "b" * 64,
        "local_consensus_edges_sha256": "c" * 64,
        "layered_candidate_manifest_sha256": "4" * 64,
        "layered_output_sha256": layered_outputs,
    }
    audit = {
        "record_map": {seed: dict(frozen_record) for seed in seeds},
    }
    monkeypatch.setattr(
        runner,
        "_validate_layered_global_pretruth_freeze",
        lambda manifest, *, split_name: audit,
    )
    global_manifest = {
        "algorithm_source_sha256": "d" * 64,
        "algorithm_config_sha256": "e" * 64,
        "joint_candidate_identity_sha256": "f" * 64,
        "layered_candidate_identity_sha256": "1" * 64,
        "manifest_payload_sha256": "9" * 64,
    }

    def successful(seed: int) -> dict[str, object]:
        return {
            "world_seed": seed,
            "discovery_status": "ok",
            "scoring_status": "ok",
            "execution_status": "ok",
            "metrics": {},
            **{
                key: value
                for key, value in frozen_record.items()
                if key != "status"
            },
            "algorithm_source_sha256": global_manifest[
                "algorithm_source_sha256"
            ],
            "algorithm_config_sha256": global_manifest[
                "algorithm_config_sha256"
            ],
            "joint_candidate_identity_sha256": global_manifest[
                "joint_candidate_identity_sha256"
            ],
            "layered_candidate_identity_sha256": global_manifest[
                "layered_candidate_identity_sha256"
            ],
            "global_manifest_payload_sha256": global_manifest[
                "manifest_payload_sha256"
            ],
            "score_artifact_sha256": {
                "score_manifest.json": "7" * 64,
            },
        }

    records = [successful(seed) for seed in seeds]
    runner._validate_result_records_against_global_freeze(
        split_name="TRAIN", records=records, global_manifest=global_manifest
    )

    wrong_source = [dict(record) for record in records]
    wrong_source[0]["algorithm_source_sha256"] = "WRONG"
    with pytest.raises(ValueError, match="frozen joint identity"):
        runner._validate_result_records_against_global_freeze(
            split_name="TRAIN",
            records=wrong_source,
            global_manifest=global_manifest,
        )

    wrong_candidate = [dict(record) for record in records]
    wrong_candidate[0]["candidate_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="base candidate binding mismatch"):
        runner._validate_result_records_against_global_freeze(
            split_name="TRAIN",
            records=wrong_candidate,
            global_manifest=global_manifest,
        )

    missing_score_bundle = [dict(record) for record in records]
    missing_score_bundle[0].pop("score_artifact_sha256")
    with pytest.raises(ValueError, match="lacks its score bundle"):
        runner._validate_result_records_against_global_freeze(
            split_name="TRAIN",
            records=missing_score_bundle,
            global_manifest=global_manifest,
        )

    failed_score = [dict(record) for record in records]
    failed_score[0].update(
        {
            "scoring_status": "error",
            "execution_status": "error",
            "metrics": None,
            "score_artifact_sha256": {},
        }
    )
    with pytest.raises(ValueError, match="scoring error binds score artifacts"):
        runner._validate_result_records_against_global_freeze(
            split_name="TRAIN",
            records=failed_score,
            global_manifest=global_manifest,
        )

    audit["record_map"][seeds[0]] = {"status": "invalid"}
    failed = [dict(record) for record in records]
    failed[0] = {
        key: value
        for key, value in failed[0].items()
        if key
        not in {
            "candidate_manifest_sha256",
            "subspace_embedding_manifest_sha256",
            "subspace_embedding_sha256",
            "local_pretruth_manifest_sha256",
            "local_consensus_edges_sha256",
            "layered_candidate_manifest_sha256",
            "layered_output_sha256",
            "score_artifact_sha256",
        }
    }
    failed[0].update(
        {
            "discovery_status": "invalid",
            "scoring_status": "not_attempted",
            "execution_status": "invalid",
            "metrics": None,
        }
    )
    runner._validate_result_records_against_global_freeze(
        split_name="TRAIN", records=failed, global_manifest=global_manifest
    )
    failed[0]["candidate_manifest_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="failed result binds candidate outputs"):
        runner._validate_result_records_against_global_freeze(
            split_name="TRAIN", records=failed, global_manifest=global_manifest
        )


def test_global_barrier_loads_every_candidate_before_first_truth_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    worlds = tmp_path / "worlds"
    output.mkdir()
    worlds.mkdir()
    attempts = [
        {
            "world_seed": seed,
            "status": "ok",
            "candidate_manifest_sha256": "a" * 64,
            "subspace_embedding_manifest_sha256": "2" * 64,
            "subspace_embedding_sha256": "3" * 64,
            "local_pretruth_manifest_sha256": "b" * 64,
            "local_consensus_edges_sha256": "c" * 64,
            "layered_candidate_manifest_sha256": "4" * 64,
            "layered_output_sha256": {
                "layered_feature_evidence.tsv": "5" * 64,
                "layered_provenance.json": "6" * 64,
                "typed_layered_hyperedges.tsv": "7" * 64,
                "within_assay_direct_candidates.tsv": "8" * 64,
            },
            "layered_algorithm_source_sha256": "9" * 64,
            "layered_algorithm_config_sha256": "0" * 64,
        }
        for seed in runner.registered_seeds("TRAIN")
    ]
    global_path = output / runner.GLOBAL_FREEZE_NAME
    global_path.write_text("{}\n")
    events: list[tuple[str, int]] = []

    def fake_candidate_loader(**kwargs: object) -> tuple[object, ...]:
        seed = int(kwargs["world_seed"])
        events.append(("subspace_candidate", seed))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"seed": seed}

    def fake_layered_loader(**kwargs: object) -> tuple[object, ...]:
        seed = int(kwargs["world_seed"])
        assert events[-1] == ("subspace_candidate", seed)
        events.append(("layered_candidate", seed))
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"layered_seed": seed},
        )

    monkeypatch.setattr(
        runner.subspace_scorer,
        "load_frozen_subspace_candidate",
        fake_candidate_loader,
    )
    monkeypatch.setattr(
        runner,
        "_load_frozen_layered_candidate",
        fake_layered_loader,
    )
    monkeypatch.setattr(
        runner,
        "_validate_layered_global_pretruth_freeze",
        lambda manifest, *, split_name: {},
    )
    loaded = runner.load_all_candidates_before_scorer(
        split_name="TRAIN",
        attempts=attempts,
        output_root=output,
        global_freeze_path=global_path,
    )

    def fake_truth_loader(
        world_dir: Path, world_out: Path, **kwargs: object
    ) -> tuple[object, ...]:
        assert kwargs["expected_pretruth_manifest_sha256"] == "b" * 64
        assert kwargs["return_binding"] is True
        seed = int(Path(world_dir).name.removeprefix("world_seed"))
        assert sum(kind == "subspace_candidate" for kind, _ in events) == len(
            attempts
        )
        assert sum(kind == "layered_candidate" for kind, _ in events) == len(
            attempts
        )
        events.append(("truth", seed))
        return (
            pd.DataFrame(),
            {"seed": seed},
            pd.DataFrame(),
            {"scorer_manifest": {"sha256": "d" * 64}},
        )

    fake_score = scorer.SharedStateSubspaceScore(
        metrics={"schema_version": scorer.SCORER_SCHEMA_VERSION},
        feature_scores=pd.DataFrame(),
        assay_scores=pd.DataFrame(),
        hyperedge_scores=pd.DataFrame(),
        truth=SimpleNamespace(),
        local_pair_scores=pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner.local_benchmark,
        "load_scorer_artifacts_after_pretruth_freeze",
        fake_truth_loader,
    )
    monkeypatch.setattr(
        runner.subspace_scorer,
        "score_shared_state_subspace",
        lambda **kwargs: fake_score,
    )
    monkeypatch.setattr(
        runner.layered_scorer,
        "score_layered_state_evidence",
        lambda *args: {
            "schema_version": runner.layered_scorer.SCORER_SCHEMA_VERSION,
            "world_seed": int(args[-1]["seed"]),
        },
    )
    monkeypatch.setattr(
        runner, "_verify_global_sources_are_current", lambda manifest: None
    )
    global_manifest = {
        "algorithm_source_sha256": "e" * 64,
        "algorithm_config_sha256": "f" * 64,
        "joint_candidate_identity_sha256": "1" * 64,
        "layered_candidate_identity_sha256": "3" * 64,
        "manifest_payload_sha256": "2" * 64,
    }
    records = runner.score_loaded_candidates(
        attempts=attempts,
        loaded=loaded,
        worlds_root=worlds,
        output_root=output,
        global_manifest=global_manifest,
    )
    first_truth = next(index for index, event in enumerate(events) if event[0] == "truth")
    assert first_truth == 2 * len(attempts)
    assert len(records) == len(attempts)
    assert all(record["scoring_status"] == "ok" for record in records)


def test_completed_container_validators_reject_stale_wrapper_schemas() -> None:
    seeds = list(runner.registered_seeds("TRAIN"))
    artifact_hashes = {
        name: "a" * 64
        for name in (
            runner.REGISTRATION_NAME,
            runner.PHASE_A_NAME,
            runner.GLOBAL_FREEZE_NAME,
            runner.RESULT_RECORDS_NAME,
            runner.BLOCK_VALIDATION_NAME,
        )
    }
    summary = {
        "schema_version": runner.RUN_SUMMARY_SCHEMA_VERSION,
        "runner_schema_version": runner.RUNNER_SCHEMA_VERSION,
        "split": "TRAIN",
        "registered_world_seeds": seeds,
        "n_registered_worlds": len(seeds),
        "n_discovery_successes": 0,
        "n_scoring_successes": 0,
        "all_failures_retained_in_denominator": True,
        "all_candidates_loaded_before_any_scorer_read": True,
        "executable_source_unchanged_through_scoring": True,
        "maximum_workers": 2,
        "workers_used": 1,
        "block_certified": False,
        "artifact_sha256": artifact_hashes,
        "run_summary_payload_sha256": "b" * 64,
    }
    runner._validate_completed_summary_container(summary, split_name="TRAIN")
    stale_summary = dict(summary, schema_version="stale")
    with pytest.raises(ValueError, match="summary contract"):
        runner._validate_completed_summary_container(
            stale_summary, split_name="TRAIN"
        )

    results = {
        "schema_version": runner.RESULT_RECORDS_SCHEMA_VERSION,
        "split": "TRAIN",
        "registered_world_seeds": seeds,
        "global_manifest_payload_sha256": "c" * 64,
        "all_failures_retained_in_denominator": True,
        "records": [],
        "result_records_payload_sha256": "d" * 64,
    }
    runner._validate_completed_result_container(results, split_name="TRAIN")
    stale_results = dict(results, schema_version="stale")
    with pytest.raises(ValueError, match="result-record container"):
        runner._validate_completed_result_container(
            stale_results, split_name="TRAIN"
        )

    phase_a = {
        "schema_version": runner.PHASE_A_SCHEMA_VERSION,
        "split": "TRAIN",
        "registered_world_seeds": seeds,
        "registration_payload_sha256": "e" * 64,
        "all_registered_discovery_attempts_finished": True,
        "all_attempts_completed_before_any_scorer_read": True,
        "scorer_domain_opened": False,
        "attempts": [],
        "phase_a_payload_sha256": "f" * 64,
    }
    runner._validate_completed_phase_a_container(
        phase_a, split_name="TRAIN", registration_sha256="e" * 64
    )
    stale_phase_a = dict(phase_a, scorer_domain_opened=True)
    with pytest.raises(ValueError, match="Phase-A container"):
        runner._validate_completed_phase_a_container(
            stale_phase_a,
            split_name="TRAIN",
            registration_sha256="e" * 64,
        )


def test_missing_world_failures_remain_in_full_block_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worlds = tmp_path / "worlds"
    output = tmp_path / "output"

    def forbidden_scorer(*args: object, **kwargs: object) -> object:
        raise AssertionError("failed discovery block must not open scorer data")

    monkeypatch.setattr(
        runner.local_benchmark,
        "load_scorer_artifacts_after_pretruth_freeze",
        forbidden_scorer,
    )
    summary = runner.run_block(
        split_name="TRAIN",
        worlds_root=worlds,
        output_root=output,
        generate_missing=False,
        workers=1,
    )
    result_payload = json.loads((output / runner.RESULT_RECORDS_NAME).read_text())
    validation = json.loads((output / runner.BLOCK_VALIDATION_NAME).read_text())
    records = result_payload["records"]
    assert len(records) == 24
    assert {record["world_seed"] for record in records} == set(
        scorer.FRESH_TRAIN_SEEDS
    )
    assert all(record["discovery_status"] == "invalid" for record in records)
    assert all(record["scoring_status"] == "not_attempted" for record in records)
    assert validation["world_reliability_gate"]["registered_worlds"] == 24
    assert validation["n_scored_worlds"] == 0
    assert validation["execution_status_counts"] == {"invalid": 24}
    assert validation["DOE_cell_coverage"]["pass"] is True
    assert validation["block_certified"] is False
    assert summary["block_certified"] is False
    assert summary["all_failures_retained_in_denominator"] is True

    with pytest.raises(FileExistsError, match="already registered"):
        runner.run_block(
            split_name="TRAIN",
            worlds_root=worlds,
            output_root=output,
            generate_missing=False,
            workers=1,
        )
    assert runner.run_block(
        split_name="TRAIN",
        worlds_root=worlds,
        output_root=output,
        generate_missing=False,
        workers=1,
        resume=True,
    ) == summary

    (output / "scores").write_text("not a score directory\n")
    with pytest.raises(ValueError, match="score root must be a regular directory"):
        runner.run_block(
            split_name="TRAIN",
            worlds_root=worlds,
            output_root=output,
            generate_missing=False,
            workers=1,
            resume=True,
        )


def test_score_artifact_publication_is_atomic_on_serialization_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    seed = 40_001

    with pytest.raises(TypeError):
        runner._write_score_artifacts(
            output_root=output,
            seed=seed,
            layered_metrics={"not_json_serializable": object()},
            subspace_diagnostic=None,
            subspace_diagnostic_error=None,
            audit={},
            scorer_manifest_sha256="a" * 64,
        )

    score_parent = output / "scores"
    assert score_parent.is_dir()
    assert list(score_parent.iterdir()) == []
    assert not (score_parent / runner.world_name(seed)).exists()
    runner._validate_score_artifact_tree(
        output,
        [
            {
                "world_seed": seed,
                "execution_status": "error",
                "metrics": None,
            }
        ],
    )


def test_score_tree_validation_rejects_non_directory_or_partial_root(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    records = [
        {
            "world_seed": 40_001,
            "execution_status": "error",
            "metrics": None,
        }
    ]

    (output / "scores").write_text("not a directory\n")
    with pytest.raises(ValueError, match="regular directory"):
        runner._validate_score_artifact_tree(output, records)

    (output / "scores").unlink()
    with pytest.raises(ValueError, match="unexpectedly binds score artifacts"):
        runner._validate_score_artifact_tree(
            output,
            [{**records[0], "score_artifact_sha256": {}}],
        )

    partial = output / "scores" / ".world_seed40001.tmp-interrupted"
    partial.mkdir(parents=True)
    (partial / "metrics.json").write_text("{}\n")
    with pytest.raises(ValueError, match="missing, extra, or partial"):
        runner._validate_score_artifact_tree(output, records)
