"""Small twin-control contract fixtures; no production worlds or model fitting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "twin_controls.py"
SPEC = importlib.util.spec_from_file_location("recoverability_twin_controls", MODULE)
twins = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = twins
SPEC.loader.exec_module(twins)


def byte_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def fixture_worlds(root):
    worlds = [root / "rho0", root / "rho07"]
    rng = np.random.default_rng(912)
    candidates = [dict(name=f"{assay}_{role}", assay=assay, role=role, feature_id=f"f{a}_{r}")
                  for a, assay in enumerate(twins.ASSAYS) for r, role in enumerate(("core", "decoy", "sentinel"))]
    truth_by_cohort = {}
    for cohort in (*twins.PUBLIC_COHORTS, *twins.PRIVATE_COHORTS):
        assigned = np.repeat(twins.ASSAYS, 10) if cohort in twins.PUBLIC_COHORTS else np.repeat("paired", 30)
        truth_by_cohort[cohort] = dict(cell_ids=np.asarray([f"{cohort}_{i}" for i in range(30)]),
            assigned_assay=assigned, state=rng.normal(size=(30, 2)), activities=rng.normal(size=(30, 9)),
            candidate_names=np.asarray([row["name"] for row in candidates]),
            candidate_assays=np.asarray([row["assay"] for row in candidates]),
            candidate_roles=np.asarray([row["role"] for row in candidates]))
    for world, rho in zip(worlds, (0.0, 0.7)):
        files = {}
        for cohort, reference in truth_by_cohort.items():
            arrays = {key: value.copy() for key, value in reference.items()}
            if rho:
                rows = np.flatnonzero(arrays["assigned_assay"] != "RNA")
                arrays["activities"][rows, 0] += 0.7 * arrays["activities"][rows, 3]
            arrays["counts_realistic"] = np.floor(np.abs(arrays["activities"]) * 3).astype(int)
            path = world / "truth" / f"{cohort}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            files[str(path.relative_to(world))] = twins.file_hash(path)
            for assay in twins.ASSAYS:
                area = "observed" if cohort in twins.PUBLIC_COHORTS else "private_inputs"
                path = world / area / "realistic" / cohort / f"{assay}.h5ad"
                # Byte fixtures intentionally stand in for already validated
                # H5AD files; this test does not create a simulation dataset.
                payload = f"unit-only {cohort} {assay}" + (f" private rho {rho}" if area == "private_inputs" else "")
                byte_file(path, payload.encode())
                files[str(path.relative_to(world))] = twins.file_hash(path)
        truth_manifest = {
            "spec": {"design": "extended", "motif": "ambiguity", "rho": rho,
                     "world_id": world.name, "seed": 123, "visible_seed": 123},
            "candidates": candidates,
            "biological_dag_edges": [["S0", f"{assay}_core"] for assay in twins.ASSAYS]
                                    + ([["H3K27ac_core", "RNA_core"]] if rho else []),
        }
        twins.write_new(world / "truth/manifest.json", truth_manifest)
        files["truth/manifest.json"] = twins.file_hash(world / "truth/manifest.json")
        twins.write_new(world / "manifest.json", {"status": "complete", "files": files, "regimes": ["realistic"]})
    return worlds


def fixture_base_model(root, base_world, method="scglue"):
    run, prep = root / f"{method}_run", root / "preprocessing"
    for cohort in twins.PUBLIC_COHORTS:
        truth = twins.load_arrays(base_world / "truth" / f"{cohort}.npz")
        for assay in twins.ASSAYS:
            path = run / cohort / f"{assay}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            ids = truth["cell_ids"][truth["assigned_assay"] == assay]
            np.savez_compressed(path, cell_ids=ids, shared=np.arange(len(ids) * 3).reshape(len(ids), 3))
    if method == "scglue":
        source_inputs, training_files = {}, {}
        for assay in twins.ASSAYS:
            path = prep / "train" / f"{assay}.h5ad"
            byte_file(path, f"unit prepared {assay}".encode())
            training_files[assay] = {"sha256": twins.file_hash(path)}
            source_inputs[assay] = {"sha256": twins.file_hash(base_world / "observed/realistic/train" / f"{assay}.h5ad")}
        twins.write_new(prep / "manifest.json", {"status": "complete", "inputs": source_inputs})
        byte_file(prep / "guidance.graphml.gz", b"unit guidance")
        byte_file(run / "scglue_model.dill", b"unit fake checkpoint never deserialized")
        model = {"status": "complete", "label": "scglue_seed0", "seed": 0,
                 "checkpoint_sha256": twins.file_hash(run / "scglue_model.dill"),
                 "inputs": {"training_files": training_files, "guidance": {
                     "path": str(prep / "guidance.graphml.gz"), "sha256": twins.file_hash(prep / "guidance.graphml.gz")}}}
        twins.write_new(run / "model_manifest.json", model)
    else:
        byte_file(run / "model/checkpoint.pt", b"unit fake checkpoint never deserialized")
        twins.write_new(run / "gene_transform/manifest.json", {"unit_fixture": True})
        model = {"status": "complete", "method": "scmrdr_paper_faithful", "label": "scmrdr_seed0", "seed": 0,
                 "checkpoint_sha256": twins.file_hash(run / "model/checkpoint.pt"),
                 "gene_transform_dir": str(run / "gene_transform"),
                 "gene_manifest_sha256": twins.file_hash(run / "gene_transform/manifest.json")}
        for key, cohort in (("train_inputs", "train"), ("validation_inputs", "val")):
            model[key] = {assay: {"sha256": twins.file_hash(base_world / "observed/realistic" / cohort / f"{assay}.h5ad")}
                          for assay in twins.ASSAYS}
        twins.write_new(run / "manifest.json", model)
    receipt = {"status": "complete", "checkpoint_sha256": model["checkpoint_sha256"],
               "files": {f"{cohort}/{assay}.npz": twins.file_hash(run / cohort / f"{assay}.npz")
                         for cohort in twins.PUBLIC_COHORTS for assay in twins.ASSAYS}}
    twins.write_new(run / ("embedding_manifest.json" if method == "scglue" else "public_embedding_manifest.json"), receipt)
    return run, prep


def fake_private_encoding(base_run, preprocessing, counterpart, output, regime, device):
    destination = output / "unit_private_inference"
    for cohort in twins.PRIVATE_COHORTS:
        for assay in twins.ASSAYS:
            path = destination / cohort / f"{assay}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, cell_ids=np.asarray(["private0", "private1"]), shared=np.ones((2, 3)))
    return destination, None


class TwinValidationTests(unittest.TestCase):
    def test_exact_visible_equality_and_changed_joint(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            base, counterpart = fixture_worlds(Path(directory))
            result = twins.validate_twins(base, counterpart, "realistic")
            self.assertEqual(result["status"], "verified_exact_public_equivalence")
            self.assertTrue(result["joint_audits"]["assessment"]["joint_activities_differ"])
            self.assertEqual(result["independent_biological_trials"], 1)
            self.assertEqual(result["biological_edges_counterpart_only"], [["H3K27ac_core", "RNA_core"]])

    def test_mutated_public_file_rejected(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            base, counterpart = fixture_worlds(Path(directory))
            (counterpart / "observed/realistic/train/RNA.h5ad").write_bytes(b"changed unit fixture")
            with self.assertRaisesRegex(ValueError, "changed after freeze"):
                twins.validate_twins(base, counterpart, "realistic")

    def test_changed_public_input_even_with_updated_hash_rejected(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            base, counterpart = fixture_worlds(Path(directory))
            relative = "observed/realistic/train/RNA.h5ad"
            (counterpart / relative).write_bytes(b"changed unit fixture")
            manifest = twins.read_json(counterpart / "manifest.json")
            manifest["files"][relative] = twins.file_hash(counterpart / relative)
            (counterpart / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "not byte-identical"):
                twins.validate_twins(base, counterpart, "realistic")

    def test_reversed_pair_rejected(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            base, counterpart = fixture_worlds(Path(directory))
            with self.assertRaisesRegex(ValueError, "in that order"):
                twins.validate_twins(counterpart, base, "realistic")

    def test_frozen_checkpoint_mutation_rejected(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            root = Path(directory)
            base, _ = fixture_worlds(root)
            run, prep = fixture_base_model(root, base)
            (run / "scglue_model.dill").write_bytes(b"changed unit fixture")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                twins.validate_base_run(base, run, "scglue", "realistic", prep)

    def test_valid_but_changed_public_embedding_rejected_by_saved_receipt(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            root = Path(directory)
            base, _ = fixture_worlds(root)
            run, prep = fixture_base_model(root, base)
            path = run / "discovery/RNA.npz"
            values = twins.load_arrays(path)
            values["shared"] = values["shared"] + 1
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "frozen output receipt"):
                twins.validate_base_run(base, run, "scglue", "realistic", prep)

    def test_scmrdr_lineage_matches_train_and_validation(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            root = Path(directory)
            base, _ = fixture_worlds(root)
            run, _ = fixture_base_model(root, base, "scmrdr")
            self.assertEqual(twins.validate_base_run(base, run, "scmrdr", "realistic", None)["label"], "scmrdr_seed0")
            manifest = twins.read_json(run / "manifest.json")
            manifest["validation_inputs"]["RNA"]["sha256"] = "not_the_base_validation"
            (run / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "val did not use"):
                twins.validate_base_run(base, run, "scmrdr", "realistic", None)


class TwinReuseTests(unittest.TestCase):
    def test_scglue_only_transforms_and_encodes_private_cohorts(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            root = Path(directory)
            base, counterpart = fixture_worlds(root)
            run, prep = fixture_base_model(root, base)
            byte_file(prep / "transforms/unit_parameters.npz", b"frozen unit parameters")
            calls = []

            def transform(paths, fitted_dir, out_dir, cohort):
                self.assertEqual(fitted_dir, prep)
                self.assertIn(cohort, twins.PRIVATE_COHORTS)
                self.assertEqual(set(paths), set(twins.ASSAYS))
                self.assertTrue(all(str(path).startswith(str(counterpart / "private_inputs")) for path in paths.values()))
                calls.append(("transform", cohort))

            def encode(model_dir, prepared, encoded, cohorts, device):
                self.assertEqual(model_dir, run)
                self.assertEqual(tuple(cohorts), twins.PRIVATE_COHORTS)
                self.assertEqual(twins.file_hash(prepared / "transforms/unit_parameters.npz"),
                                 twins.file_hash(prep / "transforms/unit_parameters.npz"))
                self.assertNotEqual(encoded, run)
                calls.append(("encode", tuple(cohorts)))

            modules = {"preprocessing": SimpleNamespace(transform_preprocessing=transform),
                       "scglue_adapter": SimpleNamespace(encode_prepared=encode)}
            with patch.object(twins, "_module", side_effect=modules.__getitem__):
                _, native = twins._encode_private_scglue(run, prep, counterpart, root / "control", "realistic", "cpu")
            self.assertEqual(calls, [("transform", "probe"), ("transform", "assessment"),
                                     ("encode", twins.PRIVATE_COHORTS)])
            self.assertEqual(native, root / "control/counterpart_preprocessing/representations")

    def test_scmrdr_checkpoint_copy_private_encoding_and_verified_cache(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            root = Path(directory)
            base, counterpart = fixture_worlds(root)
            run, _ = fixture_base_model(root, base, "scmrdr")
            calls = []

            def encode(destination, input_dir, cohort, device):
                self.assertIn(cohort, twins.PRIVATE_COHORTS)
                self.assertEqual(input_dir, counterpart / "private_inputs/realistic" / cohort)
                self.assertEqual(twins.file_hash(destination / "model/checkpoint.pt"),
                                 twins.file_hash(run / "model/checkpoint.pt"))
                for assay in twins.ASSAYS:
                    path = destination / cohort / f"{assay}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, cell_ids=np.asarray(["unit_private"]), shared=np.ones((1, 2)))
                calls.append(cohort)

            with patch.object(twins, "_module", return_value=SimpleNamespace(encode_run=encode)):
                encoded, _ = twins._encode_private_scmrdr(run, counterpart, root / "control", "realistic", "cpu")
                twins._encode_private_scmrdr(run, counterpart, root / "control", "realistic", "cpu")
                self.assertEqual(calls, list(twins.PRIVATE_COHORTS))
                derived = twins.read_json(encoded / "manifest.json")
                self.assertFalse(derived["frozen_model_reuse"]["training_performed_here"])
                self.assertEqual(twins.file_hash(Path(derived["gene_transform_dir"]) / "manifest.json"),
                                 derived["gene_manifest_sha256"])
                (encoded / "probe/RNA.npz").write_bytes(b"changed unit private cache")
                with self.assertRaisesRegex(ValueError, "arrays changed"):
                    twins._encode_private_scmrdr(run, counterpart, root / "control", "realistic", "cpu")

    def test_reuse_copies_public_embeddings_and_never_refits(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            root = Path(directory)
            base, counterpart = fixture_worlds(root)
            run, prep = fixture_base_model(root, base)
            with patch.object(twins, "_encode_private_scglue", side_effect=fake_private_encoding) as encode:
                report = twins.prepare_counterpart(base, counterpart, run, root / "control",
                    method="scglue", regime="realistic", base_preprocessing=prep, device="cpu")
                cached = twins.prepare_counterpart(base, counterpart, run, root / "control",
                    method="scglue", regime="realistic", base_preprocessing=prep, device="cpu")
            self.assertEqual(encode.call_count, 1)
            self.assertFalse(report["training_performed_here"])
            self.assertTrue(report["public_embeddings_byte_identical"])
            self.assertEqual(report["input_fingerprint"], cached["input_fingerprint"])
            for cohort in twins.PUBLIC_COHORTS:
                for assay in twins.ASSAYS:
                    relative = Path(cohort) / f"{assay}.npz"
                    self.assertEqual(twins.file_hash(run / relative), twins.file_hash(root / "control/embeddings" / relative))
            (root / "control/embeddings/discovery/RNA.npz").write_bytes(b"changed unit output")
            with self.assertRaisesRegex(ValueError, "output changed"):
                twins.prepare_counterpart(base, counterpart, run, root / "control",
                    method="scglue", regime="realistic", base_preprocessing=prep, device="cpu")


def fixture_report(world, counterpart=False, failed=False):
    manifest = {"checkpoint_sha256": "unit_checkpoint"} if not counterpart else {
        "training_performed_here": False, "inputs": {"base_lineage": {"checkpoint_sha256": "unit_checkpoint"}}}
    provenance = {"unit_model": {"files": {assay: {"sha256": f"unit_hash_{assay}"} for assay in twins.ASSAYS}, "manifest": manifest}}
    graph = {"status": "failed" if failed else "ok", "edges": None if failed else []}
    assignment = {"cell_ids": {assay: [f"unit_{assay}"] for assay in twins.ASSAYS}}
    stress = {"truth_member": "counterpart" if counterpart else "base"}
    return {
        "inputs": {"world": str(world), "regime": "realistic", "world_manifest_sha256": twins.file_hash(world / "manifest.json"),
                   "configuration": {"test": "unit_only"}, "pool_seed": 0, "matching_seed": 0,
                   "cohort": "discovery", "evaluation_source_sha256": "unit_source", "provenance": provenance},
        "common_pools": assignment,
        "results": {"unit_model": {"assignment": assignment, "activities": graph, "counts": graph,
                                     "activity_graph_metrics": {"exact_cross_assay_stress_test": stress},
                                     "count_graph_metrics": {"exact_cross_assay_stress_test": stress}}},
    }


class TwinComparisonTests(unittest.TestCase):
    def test_same_output_scored_under_both_truths_without_discrimination_claim(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            root = Path(directory)
            base, counterpart = fixture_worlds(root)
            first, second = root / "base_report.json", root / "counterpart_report.json"
            twins.write_new(first, fixture_report(base))
            twins.write_new(second, fixture_report(counterpart, counterpart=True))
            result = twins.compare_evaluations(base, counterpart, first, second, regime="realistic")
            self.assertEqual(result["status"], "completed")
            score = result["comparisons"]["unit_model"]["scales"]["counts"]
            self.assertTrue(score["graph_output_identical"])
            self.assertNotEqual(score["base_truth_stress_score"], score["counterpart_truth_stress_score"])
            self.assertEqual(result["ambiguity_set_metric"]["status"], "not_applicable")

    def test_failed_graph_is_not_successful_empty_graph_control(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            root = Path(directory)
            base, counterpart = fixture_worlds(root)
            first, second = root / "base_report.json", root / "counterpart_report.json"
            twins.write_new(first, fixture_report(base, failed=True))
            twins.write_new(second, fixture_report(counterpart, counterpart=True, failed=True))
            result = twins.compare_evaluations(base, counterpart, first, second, regime="realistic")
            self.assertEqual(result["status"], "completed_with_failed_discovery")
            self.assertEqual(result["comparisons"]["unit_model"]["scales"]["counts"]["status"], "not_evaluable")

    def test_different_public_embedding_bytes_cannot_be_called_exact_twins(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_twins_unit_") as directory:
            root = Path(directory)
            base, counterpart = fixture_worlds(root)
            first, second = root / "base_report.json", root / "counterpart_report.json"
            twins.write_new(first, fixture_report(base))
            changed = fixture_report(counterpart, counterpart=True)
            changed["inputs"]["provenance"]["unit_model"]["files"]["RNA"]["sha256"] = "changed"
            twins.write_new(second, changed)
            with self.assertRaisesRegex(ValueError, "not byte-identical"):
                twins.compare_evaluations(base, counterpart, first, second, regime="realistic")


if __name__ == "__main__":
    unittest.main()
