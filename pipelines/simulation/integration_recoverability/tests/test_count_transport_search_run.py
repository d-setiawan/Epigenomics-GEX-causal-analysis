"""Search orchestration/seal tests; no benchmark optimization or generation."""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("count_transport_search_run_tested", HERE / "count_transport_search_run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def model_fixture():
    p, mu = .2, .3
    z = (np.array([0., 1.]) - p) / np.sqrt(p * (1 - p)) + mu
    matrices = {"a": np.array([[.4], [-.2]]), "b": np.array([[.3], [.7], [-.1]])}
    eta = {a: z[:, None] @ m.T for a, m in matrices.items()}
    return {"dimension": 1, "variant": "strict", "probabilities": [p], "source_mean": [mu],
            "loadings": {a: m.tolist() for a, m in matrices.items()}, "state_bits": [[0], [1]],
            "state_weights": [1-p, p], "state_log_rates": {a: x.tolist() for a, x in eta.items()},
            "state_rates": {a: np.exp(x).tolist() for a, x in eta.items()}, "converged": True,
            "projected_gradient_inf_norm": 0., "convergence_tolerance": 1e-6,
            "stop_reason": "projected_gradient_convergence", "objective": 1.25, "iterations": 12,
            "singular_value_ratios": {"a": 1., "b": 1.}, "minimum_probability_gap": None}


class SearchFitBoundary(unittest.TestCase):
    def setUp(self):
        self.data = {"a": np.array([[0., 1.], [2., 0.], [1., 1.]]),
                     "b": np.array([[1., 0., 1.], [3., 1., 0.], [2., 1., 1.]])}
        self.protocol = {"assays": ["a", "b"], "widths": {"a": 2, "b": 3},
                         "mixture_starts": 4, "refine_candidates": 4,
                         "chunk_iterations": 600, "chunk_evaluations": 15000, "gtol": 1e-6}

    def test_both_modes_receive_only_public_fit_counts_and_same_declared_settings(self):
        payload = {"model": model_fixture(), "candidates": [{"error": "inner failure"}, {"model": model_fixture()}],
                   "search_diagnostics": {"finite_combinations": 64}, "continuation": {"total_iteration_limit": None},
                   "additional_diagnostic": [1, 2], "error": None}
        scales = {a: x.std(0).tolist() for a, x in self.data.items()}
        for mode in runner.MODES:
            with patch.object(runner.prior.matched, "training_arrays", return_value=(self.data, self.data, scales)), \
                    patch.object(runner, "fit_source_search", return_value=payload) as optimizer:
                result = runner.fit_one(Path("public"), self.protocol, "fixture", {"dimension": 1}, mode, 3)
            self.assertIs(optimizer.call_args.args[0], self.data)
            self.assertEqual(optimizer.call_args.args[1:], (1, 3))
            self.assertEqual(optimizer.call_args.kwargs["mode"], mode)
            self.assertEqual(optimizer.call_args.kwargs["mixture_starts"], 4)
            self.assertEqual(optimizer.call_args.kwargs["refine_candidates"], 4)
            self.assertEqual(result["optimizer_mode"], mode)
            self.assertEqual(result["candidates"], payload["candidates"])
            self.assertEqual(result["additional_diagnostic"], [1, 2])
            self.assertTrue(result["acceptance"]["pass"])
            self.assertTrue(np.isfinite(result["calibration_score"]))

    def test_all_inner_failures_are_retained_when_outer_model_is_missing(self):
        payload = {"model": None, "candidates": [{"error": "failure one"}, {"error": "failure two"}],
                   "search_diagnostics": {"failed_refinements": 2}, "continuation": None, "error": "no finite candidate"}
        with patch.object(runner.prior.matched, "training_arrays", return_value=(self.data, self.data, {})), \
                patch.object(runner, "fit_source_search", return_value=payload):
            result = runner.fit_one(Path("public"), self.protocol, "fixture", {"dimension": 1}, "source_search", 0)
        self.assertIsNone(result["model"])
        self.assertIsNone(result["calibration_score"])
        self.assertEqual(result["candidates"], payload["candidates"])
        self.assertEqual(result["search_diagnostics"], payload["search_diagnostics"])
        self.assertEqual(result["error"], payload["error"])

    def test_scoring_failure_does_not_destroy_the_returned_model_or_candidates(self):
        payload = {"model": model_fixture(), "candidates": [{"model": model_fixture()}],
                   "search_diagnostics": {}, "continuation": {}, "error": None}
        with patch.object(runner.prior.matched, "training_arrays", return_value=(self.data, self.data, {})), \
                patch.object(runner, "fit_source_search", return_value=payload), \
                patch.object(runner.prior, "score_rows", side_effect=ValueError("score failure")):
            result = runner.fit_one(Path("public"), self.protocol, "fixture", {"dimension": 1}, "source_search", 0)
        self.assertIsNone(result["model"])
        self.assertEqual(result["model_before_scoring_failure"], payload["model"])
        self.assertEqual(result["candidates"], payload["candidates"])
        self.assertIn("score failure", result["error"])

    def test_optimizer_payload_cannot_replace_outer_seed_identity(self):
        payload = {"model": model_fixture(), "seed": 99, "candidates": [{"error": "retained"}]}
        with patch.object(runner.prior.matched, "training_arrays", return_value=(self.data, self.data, {})), \
                patch.object(runner, "fit_source_search", return_value=payload):
            result = runner.fit_one(Path("public"), self.protocol, "fixture", {"dimension": 1}, "source_search", 0)
        self.assertEqual(result["seed"], 0)
        self.assertIsNone(result["model"])
        self.assertEqual(result["rejected_search_payload"], payload)
        self.assertIn("outer identity", result["error"])


class SearchSealsAndReports(unittest.TestCase):
    def test_modes_have_separate_calibration_selection_and_complete_start_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"worlds": ["fixture"], "assays": ["a", "b"], "widths": {"a": 2, "b": 3},
                        "seeds": [0, 1, 2, 3], "optimizer_modes": list(runner.MODES),
                        "primary_generation_seal_sha256": "generation", "primary_data_manifest_sha256": "manifest"}
            (root / "protocol.json").write_text(json.dumps(protocol))
            artifacts, selected, restarts, statuses = {}, {}, {}, {}
            for mode in runner.MODES:
                names = []
                best = 2 if mode == "source_search" else 1
                for seed in range(4):
                    name = f"{mode}_{seed}.json"
                    point = {"world": "fixture", "dimension": 1, "n_fit": 12000, "optimizer_mode": mode,
                             "seed": seed, "model": model_fixture(), "calibration_score": -abs(seed-best)}
                    (root / name).write_text(json.dumps(point))
                    names.append(name)
                    artifacts[name] = runner.prior.sha(root / name)
                group = runner.group_key("fixture", mode)
                selected[group], restarts[group], statuses[group] = names[best], names, "converged_calibration_selection"
            seal = {"protocol_sha256": runner.prior.sha(root / "protocol.json"), "evaluator_read": False,
                    "exploratory": True, "primary_generation_seal_sha256": "generation",
                    "primary_data_manifest_sha256": "manifest", "artifacts": artifacts,
                    "selected": selected, "selected_dimension_restarts": restarts, "selection_status": statuses}
            (root / "fit_seal.json").write_text(json.dumps(seal))
            with patch.object(runner, "verify_sources", return_value=protocol), \
                    patch.object(runner.prior, "_primary", return_value=(root, {"worlds": {"fixture": {"dimension": 1}}}, {})):
                verified = runner.verify_frozen(root)
                self.assertEqual(len(verified[4]), 8)
                seal["selected"][runner.group_key("fixture", "source_search")] = "source_search_0.json"
                (root / "fit_seal.json").write_text(json.dumps(seal))
                with self.assertRaisesRegex(ValueError, "calibration-only selection"):
                    runner.verify_frozen(root)

    def test_changed_protocol_stops_before_any_truth_or_paired_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text("{}")
            (root / "fit_seal.json").write_text(json.dumps({"protocol_sha256": "wrong"}))
            with patch.object(runner.metrics, "load_evaluation_inputs") as loader:
                with self.assertRaisesRegex(ValueError, "Hash changed"):
                    runner.evaluate(root)
                loader.assert_not_called()

    def test_report_keeps_failed_mode_groups_and_unchanged_oracle_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text("{}")
            (root / "fit_seal.json").write_text("{}")
            protocol = {"worlds": ["fixture"], "assays": ["a", "b"], "widths": {"a": 2, "b": 3},
                        "optimizer_modes": list(runner.MODES), "exploratory": True, "assessment_reuse": "Exploratory",
                        "optimizer_change": "unbounded source search", "finite_search_inventory": "finite, explicit",
                        "inner_selection": "training only", "selection": "calibration only", "uncertainty": "descriptive",
                        "recovery_thresholds": runner.metrics.DEFAULT_THRESHOLDS,
                        "primary_generation_seal_sha256": "generation", "primary_data_manifest_sha256": "manifest",
                        "primary_evaluation_seal_sha256": "evaluation"}
            model = model_fixture()
            counts = {"fixture": {"a": np.array([[0, 1], [2, 0], [1, 1]]),
                                   "b": np.array([[1, 0, 1], [3, 1, 0], [2, 1, 1]])}}
            artifacts, selected, restarts, statuses = {}, {}, {}, {}
            for mode in runner.MODES:
                names = []
                for seed in range(4):
                    name = f"{mode}_{seed}.json"
                    artifacts[name] = {"world": "fixture", "dimension": 1, "n_fit": 12000, "optimizer_mode": mode,
                                       "seed": seed, "model": model if mode == "source_search" else None,
                                       "acceptance": {"pass": mode == "source_search"},
                                       "calibration_score": -1 if mode == "source_search" else None,
                                       "candidates": [{"error": "retained failure"}], "search_diagnostics": {"records": 1}}
                    names.append(name)
                group = runner.group_key("fixture", mode)
                selected[group] = names[0] if mode == "source_search" else None
                restarts[group] = names
                statuses[group] = "converged_calibration_selection" if mode == "source_search" else "no_finite_fit"
            seal = {"selected": selected, "selected_dimension_restarts": restarts, "selection_status": statuses}
            with patch.object(runner, "verify_frozen", return_value=(protocol, seal, root, {}, artifacts, {})), \
                    patch.object(runner.metrics, "load_evaluation_inputs", return_value=(counts, {"fixture": model}, {})), \
                    patch.object(runner.metrics, "oracle_training_objectives", return_value={"fixture": {12000: 1.25}}), \
                    patch.object(runner.prior, "bootstrap_indices", return_value=np.tile(np.arange(3), (10, 1))):
                report = runner.evaluate(root)
            self.assertEqual(len(report["fits"]), 8)
            self.assertEqual(len(report["aggregate"]), 2)
            by_mode = {row["optimizer_mode"]: row for row in report["aggregate"]}
            self.assertEqual(by_mode["plain_unbounded"]["no_finite_fit_groups"], 1)
            self.assertEqual(by_mode["source_search"]["no_finite_fit_groups"], 0)
            self.assertTrue(all(point["candidate_records_in_fit_artifact"] == 1 for point in report["fits"].values()))
            selected_report = report["fits"][selected[runner.group_key("fixture", "source_search")]]
            self.assertEqual(selected_report["optimization_oracle_check"]["fitted_minus_true_training_objective"], 0)
            self.assertTrue(all((root / name).exists() for name in runner.OUTPUTS))
            result_seal = json.loads((root / "evaluation_seal.json").read_text())
            for name, digest in result_seal["outputs"].items():
                self.assertEqual(runner.prior.sha(root / name), digest)


if __name__ == "__main__":
    unittest.main()
