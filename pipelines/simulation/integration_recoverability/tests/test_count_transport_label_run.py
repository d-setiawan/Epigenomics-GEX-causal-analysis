"""Followup boundaries and orchestration tests; no benchmark generation/fits."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("count_transport_label_run_tested", HERE / "count_transport_label_run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def model_fixture():
    p, mu = .2, .3
    sources = (np.array([0., 1.]) - p) / np.sqrt(p * (1 - p)) + mu
    matrices = {"a": np.array([[.4], [-.2]]), "b": np.array([[.3], [.7], [-.1]])}
    eta = {a: sources[:, None] @ m.T for a, m in matrices.items()}
    return {"dimension": 1, "variant": "strict", "probabilities": [p], "source_mean": [mu],
            "loadings": {a: m.tolist() for a, m in matrices.items()}, "state_bits": [[0], [1]],
            "state_weights": [1-p, p], "state_log_rates": {a: x.tolist() for a, x in eta.items()},
            "state_rates": {a: np.exp(x).tolist() for a, x in eta.items()}, "converged": True,
            "projected_gradient_inf_norm": 0., "convergence_tolerance": 1e-6,
            "stop_reason": "projected_gradient_convergence", "objective": 1.25, "iterations": 12,
            "singular_value_ratios": {"a": 1., "b": 1.}, "minimum_probability_gap": None}


class CalibrationSelection(unittest.TestCase):
    def test_all_failed_starts_keep_explicit_null_selection(self):
        name, status = runner.select_group({str(seed): {"model": None, "calibration_score": None} for seed in range(4)})
        self.assertIsNone(name)
        self.assertEqual(status, "no_finite_fit")

    def test_selection_uses_convergence_and_calibration_not_truth_or_acceptance(self):
        candidates = {
            "best": {"model": {"converged": True}, "calibration_score": -1., "dimension": 2, "seed": 1,
                     "acceptance": {"pass": False}},
            "lower": {"model": {"converged": True}, "calibration_score": -2., "dimension": 2, "seed": 0,
                      "acceptance": {"pass": True}},
            "not_converged": {"model": {"converged": False}, "calibration_score": 5., "dimension": 2, "seed": 2},
        }
        self.assertEqual(runner.select_group(candidates), ("best", "converged_calibration_selection"))
        candidates = {"not_converged": candidates["not_converged"]}
        self.assertEqual(runner.select_group(candidates), ("not_converged", "finite_nonconverged_diagnostic"))


class FitBoundary(unittest.TestCase):
    def setUp(self):
        self.data = {"a": np.array([[0., 1.], [2., 0.], [1., 1.]]),
                     "b": np.array([[1., 0., 1.], [3., 1., 0.], [2., 1., 1.]])}
        self.protocol = {"assays": ["a", "b"], "widths": {"a": 2, "b": 3},
                         "chunk_iterations": 600, "chunk_evaluations": 15000, "gtol": 1e-6}

    def test_fit_passes_only_public_count_arrays_and_known_dimension(self):
        fitted = {"model": model_fixture(), "raw_optimizer": {"probabilities": [.8]},
                  "continuation": {"total_iteration_limit": None}, "error": None}
        scales = {a: x.std(0).tolist() for a, x in self.data.items()}
        with patch.object(runner.matched, "training_arrays", return_value=(self.data, self.data, scales)) as loader, \
                patch.object(runner, "fit_label_model", return_value=fitted) as optimizer:
            result = runner.fit_one(Path("public_root"), self.protocol, "fixture", {"dimension": 1}, 2)
        loader.assert_called_once_with(Path("public_root"), {"dimension": 1}, self.protocol, 12000)
        self.assertIs(optimizer.call_args.args[0], self.data)
        self.assertEqual(optimizer.call_args.args[1:], (1, 2))
        self.assertEqual(result["raw_optimizer"], fitted["raw_optimizer"])
        self.assertEqual(result["continuation"], fitted["continuation"])
        expected = np.mean([runner.score_rows(self.data[a], fitted["model"]["state_weights"],
                                             fitted["model"]["state_rates"][a])[0].mean()/self.data[a].shape[1]
                            for a in self.data])
        self.assertAlmostEqual(result["calibration_score"], expected)
        self.assertTrue(result["acceptance"]["pass"])

    def test_canonicalization_failure_retains_raw_optimizer_and_continuation(self):
        fitted = {"model": None, "raw_optimizer": {"probabilities": [.5]},
                  "continuation": {"stop_reason": "symmetry"}, "error": "p equals one half"}
        with patch.object(runner.matched, "training_arrays", return_value=(self.data, self.data, {})), \
                patch.object(runner, "fit_label_model", return_value=fitted):
            result = runner.fit_one(Path("public_root"), self.protocol, "fixture", {"dimension": 1}, 0)
        self.assertIsNone(result["model"])
        self.assertIsNone(result["calibration_score"])
        self.assertFalse(result["acceptance"]["pass"])
        self.assertEqual(result["raw_optimizer"], fitted["raw_optimizer"])
        self.assertEqual(result["continuation"], fitted["continuation"])
        self.assertEqual(result["error"], fitted["error"])


class SealsAndReports(unittest.TestCase):
    def test_frozen_selection_is_replayed_from_calibration_and_all_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"worlds": ["fixture"], "assays": ["a", "b"], "widths": {"a": 2, "b": 3},
                        "seeds": [0, 1, 2, 3], "primary_generation_seal_sha256": "generation",
                        "primary_data_manifest_sha256": "manifest"}
            (root / "protocol.json").write_text(json.dumps(protocol))
            names, hashes = [], {}
            for seed in range(4):
                name = f"seed{seed}.json"
                artifact = {"world": "fixture", "dimension": 1, "n_fit": 12000, "optimizer_mode": runner.MODE,
                            "seed": seed, "model": model_fixture(), "calibration_score": -abs(seed-2),
                            "acceptance": {"pass": True}}
                (root / name).write_text(json.dumps(artifact))
                names.append(name)
                hashes[name] = runner.sha(root / name)
            group = runner.group_key("fixture")
            seal = {"protocol_sha256": runner.sha(root / "protocol.json"), "evaluator_read": False,
                    "exploratory": True, "primary_generation_seal_sha256": "generation",
                    "primary_data_manifest_sha256": "manifest", "artifacts": hashes,
                    "selected": {group: names[2]}, "selected_dimension_restarts": {group: names},
                    "selection_status": {group: "converged_calibration_selection"}}
            (root / "fit_seal.json").write_text(json.dumps(seal))
            with patch.object(runner, "verify_sources", return_value=protocol), \
                    patch.object(runner, "_primary", return_value=(root, {"worlds": {"fixture": {"dimension": 1}}}, {})):
                verified = runner.verify_frozen(root)
                self.assertEqual(len(verified[4]), 4)
                seal["selected"][group] = names[0]
                (root / "fit_seal.json").write_text(json.dumps(seal))
                with self.assertRaisesRegex(ValueError, "calibration-only rule"):
                    runner.verify_frozen(root)

    def test_fit_provenance_checks_do_not_read_private_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            primary.mkdir()
            (primary / "protocol.json").write_text("{}")
            (primary / "evaluation_seal.json").write_text("{}")
            inputs = {}
            for name in ("public.npz", "truth.json", "assessment.npz", "symmetry_control.json"):
                (primary / name).write_text("opaque test bytes")
                inputs[name] = runner.sha(primary / name)
            def entry(name):
                return {"path": name, "sha256": inputs[name]}
            manifest = {"worlds": {"fixture": {"dimension": 1, "public": {"a": entry("public.npz")},
                                               "truth": entry("truth.json"), "assessment": entry("assessment.npz")}}}
            (primary / "data_manifest.json").write_text(json.dumps(manifest))
            generation = {"protocol_sha256": runner.sha(primary / "protocol.json"),
                          "data_manifest_sha256": runner.sha(primary / "data_manifest.json"), "artifacts": inputs}
            (primary / "generation_seal.json").write_text(json.dumps(generation))
            protocol = {"primary_run": "primary", "worlds": ["fixture"], "assays": ["a"],
                        "primary_protocol_sha256": runner.sha(primary / "protocol.json"),
                        "primary_generation_seal_sha256": runner.sha(primary / "generation_seal.json"),
                        "primary_data_manifest_sha256": runner.sha(primary / "data_manifest.json"),
                        "primary_evaluation_seal_sha256": runner.sha(primary / "evaluation_seal.json")}
            (primary / "assessment.npz").write_text("changed private bytes")
            with patch.object(runner, "REPO", root):
                runner._primary(root, protocol, include_private=False)
                with self.assertRaisesRegex(ValueError, "Hash changed"):
                    runner._primary(root, protocol, include_private=True)

    def test_changed_protocol_fails_before_any_evaluator_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text("{}")
            (root / "fit_seal.json").write_text(json.dumps({"protocol_sha256": "wrong"}))
            with patch.object(runner.metrics, "load_evaluation_inputs") as loader:
                with self.assertRaisesRegex(ValueError, "Hash changed"):
                    runner.evaluate(root)
                loader.assert_not_called()

    def test_report_retains_an_all_failed_world_and_every_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text("{}")
            (root / "fit_seal.json").write_text("{}")
            protocol = {"worlds": ["good", "failed"], "assays": ["a", "b"], "widths": {"a": 2, "b": 3},
                        "assessment_reuse": "Exploratory reuse", "optimizer_change": "labels",
                        "parameter_domain_caveat": "broader", "selection": "calibration only", "uncertainty": "descriptive",
                        "recovery_thresholds": runner.metrics.DEFAULT_THRESHOLDS,
                        "primary_generation_seal_sha256": "generation", "primary_data_manifest_sha256": "manifest",
                        "primary_evaluation_seal_sha256": "evaluation"}
            model = model_fixture()
            counts = {world: {"a": np.array([[0, 1], [2, 0], [1, 1]]), "b": np.array([[1, 0, 1], [3, 1, 0], [2, 1, 1]])}
                      for world in protocol["worlds"]}
            artifacts, selected, restarts, statuses = {}, {}, {}, {}
            for world in protocol["worlds"]:
                names = []
                for seed in range(4):
                    name = f"{world}_{seed}.json"
                    artifacts[name] = {"world": world, "dimension": 1, "n_fit": 12000, "optimizer_mode": runner.MODE,
                                       "seed": seed, "model": model if world == "good" else None,
                                       "acceptance": {"pass": world == "good"}, "calibration_score": -1 if world == "good" else None}
                    names.append(name)
                group = runner.group_key(world)
                selected[group] = names[0] if world == "good" else None
                restarts[group] = names
                statuses[group] = "converged_calibration_selection" if world == "good" else "no_finite_fit"
            seal = {"selected": selected, "selected_dimension_restarts": restarts, "selection_status": statuses}
            with patch.object(runner, "verify_frozen", return_value=(protocol, seal, root, {}, artifacts, {})), \
                    patch.object(runner.metrics, "load_evaluation_inputs", return_value=(counts, {world: model for world in counts}, {})), \
                    patch.object(runner.metrics, "oracle_training_objectives", return_value={world: {12000: 1.25} for world in counts}), \
                    patch.object(runner, "bootstrap_indices", return_value=np.tile(np.arange(3), (10, 1))):
                report = runner.evaluate(root)
            self.assertTrue(report["exploratory"])
            self.assertEqual(len(report["fits"]), 8)
            self.assertIsNone(report["selected"][runner.group_key("failed")])
            self.assertEqual(report["restart_stability"][runner.group_key("failed")]["failed_fits"], 4)
            self.assertEqual(report["aggregate"][0]["worlds"], 2)
            self.assertEqual(report["aggregate"][0]["no_finite_fit_groups"], 1)
            self.assertTrue(all((root / name).exists() for name in runner.OUTPUTS))
            output_seal = json.loads((root / "evaluation_seal.json").read_text())
            self.assertTrue(output_seal["exploratory"])
            for name, digest in output_seal["outputs"].items():
                self.assertEqual(runner.sha(root / name), digest)


if __name__ == "__main__":
    unittest.main()
