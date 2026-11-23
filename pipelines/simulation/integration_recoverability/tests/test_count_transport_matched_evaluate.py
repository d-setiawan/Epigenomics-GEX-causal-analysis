"""Algebra and seal tests only; no benchmark counts or worlds are generated."""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import poisson

MODULE = Path(__file__).resolve().parents[1] / "count_transport_matched_evaluate.py"
SPEC = importlib.util.spec_from_file_location("count_transport_matched_evaluate", MODULE)
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def finite_model(probabilities=(.17, .36), matrices=None, mean=(.3, -.2)):
    p, mu = np.asarray(probabilities), np.asarray(mean)
    if matrices is None:
        matrices = {"a": np.array([[.7, .1], [.2, .8], [-.3, .6]]),
                    "b": np.array([[.2, .9], [.8, -.3], [.5, .4], [-.1, .3]])}
    bits = np.array(list(itertools.product((0, 1), repeat=len(p))))
    states = (bits - p) / np.sqrt(p * (1 - p)) + mu
    weights = np.prod(np.where(bits == 1, p, 1 - p), axis=1)
    eta = {a: states @ m.T for a, m in matrices.items()}
    return {"dimension": len(p), "probabilities": p.tolist(), "source_mean": mu.tolist(),
            "loadings": {a: m.tolist() for a, m in matrices.items()},
            "state_bits": bits.tolist(), "state_weights": weights.tolist(),
            "state_log_rates": {a: x.tolist() for a, x in eta.items()},
            "state_rates": {a: np.exp(x).tolist() for a, x in eta.items()},
            "converged": True, "projected_gradient_inf_norm": 0.,
            "convergence_tolerance": 1e-6, "stop_reason": "projected_gradient_convergence", "iterations": 12}


class MatchedTransportAlgebra(unittest.TestCase):
    def setUp(self):
        self.assays = ["a", "b"]
        self.widths = {"a": 3, "b": 4}
        self.truth = finite_model()
        self.parsed = ev.validate_model(self.truth, self.assays, self.widths)
        self.counts = {"a": np.array([[0, 1, 0], [3, 0, 1], [0, 2, 1], [4, 1, 3]]),
                       "b": np.array([[0, 2, 0, 1], [4, 0, 2, 0], [0, 3, 2, 1], [2, 1, 3, 1]])}

    def test_oracle_model_has_exact_covariance_transport_and_zero_score_gap(self):
        artifact = {"world": "fixture", "n_fit": 3000, "optimizer_mode": "convergence", "dimension": 2,
                    "seed": 0, "calibration_score": -1, "acceptance": {"pass": True}, "model": self.truth}
        report, rows = ev.evaluate_fit(artifact, self.counts, self.truth, self.assays, self.widths,
                                      ev.legacy.bootstrap_indices(4, "fixture", 50))
        self.assertTrue(report["fit_accepted"])
        for pair in report["pairs"].values():
            self.assertAlmostEqual(pair["exact_count_covariance_error"]["relative_frobenius_error"], 0)
            self.assertLess(pair["restricted_lograte_transport_error"]["relative_frobenius_error"], 1e-14)
            self.assertLess(pair["true_support_lograte_transport_rmse"], 1e-14)
            self.assertLess(pair["mean_lograte_transport_error"]["relative_frobenius_error"], 1e-14)
            self.assertEqual(pair["per_cell"]["oracle_conditional_logscore_gap"]["mean"], 0)
            self.assertEqual(pair["per_target_feature"]["oracle_conditional_logscore_gap"]["ci95"], [0, 0])
        np.testing.assert_allclose(rows["a__to__b"]["gain_over_same_model_target_marginal"],
                                   rows["b__to__a"]["gain_over_same_model_target_marginal"], atol=1e-14)

    def test_transport_restriction_is_invariant_to_one_shared_invertible_basis(self):
        source, target = self.parsed["loadings"]["a"], self.parsed["loadings"]["b"]
        change = np.array([[1., .8], [-.4, 2.]])
        report, operator = ev.transport_error(source @ change, target @ change, source, target)
        self.assertLess(report["relative_frobenius_error"], 1e-14)
        np.testing.assert_allclose(operator @ source, target, atol=1e-14)

    def test_global_source_permutation_is_aligned_once_for_every_assay(self):
        permutation = [1, 0]
        model = finite_model(self.parsed["probabilities"][permutation],
                             {a: m[:, permutation] for a, m in self.parsed["loadings"].items()},
                             self.parsed["mean"][permutation])
        parsed = ev.validate_model(model, self.assays, self.widths)
        aligned = ev.state_alignment(parsed, self.parsed, self.assays)
        self.assertEqual(aligned["source_permutation_true_to_fit"], permutation)
        self.assertLess(aligned["combined_log_rate_atom_rmse"], 1e-14)
        self.assertEqual(aligned["probability_rmse"], 0)
        for a in self.assays:
            self.assertLess(aligned["assays"][a]["loading_error"]["relative_frobenius_error"], 1e-14)

    def test_inconsistent_assay_permutations_cannot_be_hidden_by_alignment(self):
        matrices = dict(self.parsed["loadings"])
        matrices["b"] = matrices["b"][:, [1, 0]]
        model = ev.validate_model(finite_model(matrices=matrices), self.assays, self.widths)
        aligned = ev.state_alignment(model, self.parsed, self.assays)
        self.assertGreater(aligned["combined_log_rate_atom_rmse"], .1)
        error, _ = ev.transport_error(model["loadings"]["a"], model["loadings"]["b"],
                                       self.parsed["loadings"]["a"], self.parsed["loadings"]["b"])
        self.assertGreater(error["relative_frobenius_error"], .1)

    def test_strict_model_rejects_extra_intercept_or_nonindependent_weights(self):
        wrong = json.loads(json.dumps(self.truth))
        wrong["state_rates"]["a"][0][0] *= 1.1
        with self.assertRaisesRegex(ValueError, "strict log-linear"):
            ev.validate_model(wrong, self.assays, self.widths)
        wrong = json.loads(json.dumps(self.truth))
        wrong["state_weights"][0] += .01
        wrong["state_weights"][1] -= .01
        with self.assertRaisesRegex(ValueError, "independent-source probabilities"):
            ev.validate_model(wrong, self.assays, self.widths)

    def test_same_model_and_true_independence_baselines_remain_distinct(self):
        wrong = finite_model(probabilities=(.21, .39))
        fitted = ev.validate_model(wrong, self.assays, self.widths)
        rows = ev.predictive_rows(self.counts["a"], self.counts["b"], fitted, self.parsed, "a", "b")
        np.testing.assert_allclose(rows["gain_over_true_independent_target"] - rows["gain_over_same_model_target_marginal"],
                                   -rows["target_marginal_logscore_gap"])
        self.assertGreater(np.linalg.norm(rows["target_marginal_logscore_gap"]), .01)

    def test_failed_restart_is_retained_without_metrics_or_success(self):
        artifact = {"world": "fixture", "n_fit": 3000, "optimizer_mode": "convergence", "dimension": 2,
                    "seed": 0, "calibration_score": None, "acceptance": {"pass": False}, "model": None, "error": "failed"}
        report, rows = ev.evaluate_fit(artifact, self.counts, self.truth, self.assays, self.widths)
        self.assertEqual(report["evaluation_status"], "fit_failed")
        self.assertFalse(report["fit_accepted"])
        self.assertFalse(report["recovery_pass"])
        self.assertEqual(rows, {})
        summary = ev.restart_summary({"failed": report}, ["failed"])
        self.assertEqual(summary["failed_fits"], 1)
        self.assertEqual(summary["scored_fits"], 0)

    def test_reported_convergence_does_not_override_failed_gradient_check(self):
        model = {**self.truth, "projected_gradient_inf_norm": .1}
        artifact = {"world": "fixture", "n_fit": 3000, "optimizer_mode": "convergence", "dimension": 2,
                    "seed": 0, "calibration_score": -1, "acceptance": {"pass": True}, "model": model}
        report, _ = ev.evaluate_fit(artifact, self.counts, self.truth, self.assays, self.widths,
                                    ev.legacy.bootstrap_indices(4, "fixture", 10))
        self.assertFalse(report["gradient_converged"])
        self.assertFalse(report["fit_accepted"])
        self.assertFalse(report["recovery_pass"])

    def test_recovery_thresholds_are_explicit_arguments(self):
        artifact = {"world": "fixture", "n_fit": 3000, "optimizer_mode": "convergence", "dimension": 2,
                    "seed": 0, "calibration_score": -1, "acceptance": {"pass": True},
                    "model": finite_model(probabilities=(.21, .39))}
        strict = {**ev.DEFAULT_THRESHOLDS, "covariance_relative_error": 0.}
        loose = {**ev.DEFAULT_THRESHOLDS, "covariance_relative_error": 1e6}
        failed, _ = ev.evaluate_fit(artifact, self.counts, self.truth, self.assays, self.widths, thresholds=strict)
        passed, _ = ev.evaluate_fit(artifact, self.counts, self.truth, self.assays, self.widths, thresholds=loose)
        self.assertFalse(failed["pairs"]["a__to__b"]["criteria"]["exact_covariance_relative_error_pass"])
        self.assertTrue(passed["pairs"]["a__to__b"]["criteria"]["exact_covariance_relative_error_pass"])

    def test_oracle_training_comparison_certifies_only_the_positive_gap(self):
        worse = ev.optimization_oracle_check({"objective": 3.}, 2.)
        self.assertTrue(worse["known_feasible_truth_better"])
        self.assertEqual(worse["fitted_minus_true_training_objective"], 1)
        self.assertIn("suboptimality certificate", worse["interpretation"])
        better = ev.optimization_oracle_check({"objective": 1.}, 2.)
        self.assertFalse(better["known_feasible_truth_better"])
        self.assertIn("does not certify a global optimum", better["interpretation"])
        roundoff = ev.optimization_oracle_check({"objective": 2. + 5e-9}, 2.)
        self.assertFalse(roundoff["known_feasible_truth_better"])

    def test_oracle_training_objective_uses_same_prefix_equal_assay_feature_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = {}
            for assay in self.assays:
                path = f"{assay}.npz"
                np.savez(root / path, fit_counts=self.counts[assay])
                public[assay] = {"path": path}
            protocol = {"worlds": ["fixture"], "assays": self.assays,
                        "widths": self.widths, "fit_sizes": [2, 4]}
            manifest = {"worlds": {"fixture": {"public": public}}}
            actual = ev.oracle_training_objectives(root, protocol, manifest, {"fixture": self.truth})
            for size in protocol["fit_sizes"]:
                expected = []
                for assay in self.assays:
                    component_probability = poisson.pmf(self.counts[assay][:size, None, :],
                                                         self.parsed["rates"][assay][None, :, :]).prod(2)
                    expected.append(-np.log(component_probability @ self.parsed["weights"]).mean() / self.widths[assay])
                self.assertAlmostEqual(actual["fixture"][size], np.mean(expected))


class MatchedEvaluationBoundaries(unittest.TestCase):
    def test_report_pipeline_retains_failed_starts_and_writes_sealed_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("protocol.json", "fit_seal.json", "generation_seal.json", "data_manifest.json", "symmetry_control.json"):
                (root / name).write_text("{}")
            protocol = {"worlds": ["fixture"], "assays": ["a", "b"], "widths": {"a": 3, "b": 4},
                        "recovery_thresholds": ev.DEFAULT_THRESHOLDS}
            artifacts, selected, restarts = {}, {}, {}
            for size, mode in itertools.product(ev.SIZES, ev.MODES):
                group = f"fixture__n{size}__{mode}"
                names = []
                for seed in range(4):
                    name = f"{group}__s{seed}.json"
                    artifacts[name] = {"world": "fixture", "dimension": 2, "n_fit": size, "optimizer_mode": mode,
                                       "seed": seed, "model": None if seed == 3 else {**finite_model(), "objective": 1.25},
                                       "acceptance": {"pass": seed != 3}, "calibration_score": -1}
                    names.append(name)
                selected[group], restarts[group] = names[0], names
            seal = {"selected": selected, "selected_dimension_restarts": restarts}
            counts = {"fixture": {"a": np.array([[0, 1, 0], [3, 0, 1], [0, 2, 1], [4, 1, 3]]),
                                  "b": np.array([[0, 2, 0, 1], [4, 0, 2, 0], [0, 3, 2, 1], [2, 1, 3, 1]])}}
            with patch.object(ev, "verify_frozen", return_value=(protocol, seal, {}, artifacts, {})), \
                    patch.object(ev, "load_evaluation_inputs", return_value=(counts, {"fixture": finite_model()}, {})), \
                    patch.object(ev, "oracle_training_objectives", return_value={"fixture": {size: 1.25 for size in ev.SIZES}}), \
                    patch.object(ev, "BOOTSTRAPS", 10):
                output = ev.evaluate_run(root, root)
            self.assertEqual(len(output["fits"]), 16)
            self.assertEqual(len(output["mode_comparisons"]), 2)
            self.assertTrue(all(summary["failed_fits"] == 1 for summary in output["restart_stability"].values()))
            for comparison in output["mode_comparisons"].values():
                for pair in comparison["pairs"].values():
                    self.assertEqual(pair["conditional_logscore_improvement_per_target_feature"]["mean"], 0)
            self.assertTrue(all((root / name).exists() for name in ev.OUTPUTS))
            result_seal = json.loads((root / "evaluation_seal.json").read_text())
            for name, digest in result_seal["outputs"].items():
                self.assertEqual(ev.legacy.sha(root / name), digest)

    def test_full_generation_inventory_includes_symmetry_control_and_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"worlds": ["fixture"], "assays": ["a", "b"], "widths": {"a": 3, "b": 4},
                        "fit_sizes": list(ev.SIZES), "optimizer_modes": list(ev.MODES),
                        "recovery_thresholds": ev.DEFAULT_THRESHOLDS, "sources": {}}
            (root / "protocol.json").write_text(json.dumps(protocol))
            inputs = {}
            for name in ("public_a.npz", "public_b.npz", "assessment.npz", "truth.json", "symmetry_control.json"):
                (root / name).write_text("Opaque bookkeeping fixture; no count values.")
                inputs[name] = ev.legacy.sha(root / name)
            def entry(name):
                return {"path": name, "sha256": inputs[name]}
            manifest = {"worlds": {"fixture": {"dimension": 2,
                        "public": {"a": entry("public_a.npz"), "b": entry("public_b.npz")},
                        "assessment": entry("assessment.npz"), "truth": entry("truth.json")}}}
            (root / "data_manifest.json").write_text(json.dumps(manifest))
            protocol_hash, manifest_hash = ev.legacy.sha(root / "protocol.json"), ev.legacy.sha(root / "data_manifest.json")
            (root / "generation_seal.json").write_text(json.dumps({"protocol_sha256": protocol_hash,
                "data_manifest_sha256": manifest_hash, "artifacts": inputs}))
            artifacts, selected, restarts = {}, {}, {}
            for size, mode in itertools.product(ev.SIZES, ev.MODES):
                group = f"fixture__n{size}__{mode}"
                names = []
                for seed in range(4):
                    name = f"{group}__s{seed}.json"
                    artifact = {"world": "fixture", "dimension": 2, "n_fit": size, "optimizer_mode": mode,
                                "seed": seed, "model": finite_model()}
                    (root / name).write_text(json.dumps(artifact))
                    artifacts[name] = ev.legacy.sha(root / name)
                    names.append(name)
                selected[group], restarts[group] = names[0], names
            seal = {"protocol_sha256": protocol_hash, "data_manifest_sha256": manifest_hash,
                    "generation_seal_sha256": ev.legacy.sha(root / "generation_seal.json"),
                    "artifacts": artifacts, "selected": selected, "selected_dimension_restarts": restarts}
            (root / "fit_seal.json").write_text(json.dumps(seal))
            verified = ev.verify_frozen(root, root)
            self.assertEqual(verified[-1], inputs)
            seal["evaluator_read"] = True
            (root / "fit_seal.json").write_text(json.dumps(seal))
            with self.assertRaisesRegex(ValueError, "prior evaluator access"):
                ev.verify_frozen(root, root)
            seal["evaluator_read"] = False
            (root / "fit_seal.json").write_text(json.dumps(seal))
            (root / "symmetry_control.json").write_text("changed")
            with patch.object(ev, "load_evaluation_inputs") as loader:
                with self.assertRaisesRegex(ValueError, "Hash changed"):
                    ev.evaluate_run(root, root)
                loader.assert_not_called()

    def test_paired_fixture_feature_order_and_public_disjointness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"worlds": ["fixture"], "assays": ["a", "b"], "widths": {"a": 3, "b": 4},
                        "fit_sizes": [2, 4], "calibration_rows": 2, "assessment_rows": 4}
            public, assessment = {}, {"cell_ids": np.array([f"paired{i}" for i in range(4)])}
            for assay, width in protocol["widths"].items():
                path = f"public_{assay}.npz"
                features = np.array([f"{assay}:{i}" for i in range(width)])
                np.savez(root / path, feature_ids=features,
                         fit_ids=np.array([f"{assay}:fit{i}" for i in range(4)]),
                         calibration_ids=np.array([f"{assay}:cal{i}" for i in range(2)]))
                public[assay] = {"path": path}
                assessment[f"feature_ids_{assay}"] = features
                assessment[f"counts_{assay}"] = np.arange(4 * width).reshape(4, width)
            np.savez(root / "assessment.npz", **assessment)
            (root / "truth.json").write_text(json.dumps(finite_model()))
            manifest = {"worlds": {"fixture": {"dimension": 2, "public": public,
                        "assessment": {"path": "assessment.npz"}, "truth": {"path": "truth.json"}}}}
            counts, _, audits = ev.load_evaluation_inputs(root, protocol, manifest)
            self.assertEqual(audits["fixture"]["paired_rows"], 4)
            np.testing.assert_array_equal(counts["fixture"]["a"], assessment["counts_a"])
            assessment["cell_ids"][0] = "a:fit0"
            np.savez(root / "assessment.npz", **assessment)
            with self.assertRaisesRegex(ValueError, "overlap"):
                ev.load_evaluation_inputs(root, protocol, manifest)

    def test_changed_protocol_is_rejected_before_truth_or_assessment_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"changed": True}))
            (root / "fit_seal.json").write_text(json.dumps({"protocol_sha256": "wrong"}))
            with patch.object(ev, "load_evaluation_inputs") as loader:
                with self.assertRaisesRegex(ValueError, "Hash changed"):
                    ev.evaluate_run(root, root)
                loader.assert_not_called()

    def test_existing_table_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "comparison.tsv").write_text("frozen\n")
            with self.assertRaises(FileExistsError):
                ev.evaluate_run(root, root)
            self.assertEqual((root / "comparison.tsv").read_text(), "frozen\n")

    def test_plain_bootstrap_interval_is_descriptive_and_deterministic(self):
        values = np.array([-1., 0., 4., 5.])
        indices = ev.legacy.bootstrap_indices(4, "fixture", 100)
        report = ev.summarize_rows(values, indices)
        np.testing.assert_allclose(report["ci95"], np.quantile(values[indices].mean(1), [.025, .975]))
        self.assertEqual(report["mean"], 2)
        self.assertNotIn("ci_family_adjusted", report)


if __name__ == "__main__":
    unittest.main()
