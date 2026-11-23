"""Deterministic algebra and information-boundary tests; no benchmark worlds."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy import sparse
from scipy.stats import poisson

MODULE = Path(__file__).resolve().parents[1] / "count_transport_evaluate.py"
SPEC = importlib.util.spec_from_file_location("count_transport_evaluate", MODULE)
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


class PredictiveIdentities(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0.], [1.], [3.], [5.]])
        self.y = np.array([[1.], [0.], [4.], [7.]])
        self.w = np.array([.3, .7])
        self.rx = np.array([[.7], [3.]])
        self.ry = np.array([[1.2], [5.]])
        self.permutation = np.array([3, 0, 2, 1])

    def test_conditional_is_joint_over_marginal_and_shuffle_only_moves_source(self):
        rows = ev.prediction_rows(self.x, self.y, self.w, self.rx, self.ry, self.permutation)
        source_probability = poisson.pmf(self.x, self.rx.T)
        target_probability = poisson.pmf(self.y, self.ry.T)
        marginal_x = source_probability @ self.w
        marginal_y = target_probability @ self.w
        joint = (source_probability * target_probability) @ self.w
        np.testing.assert_allclose(rows["conditional_loglik"], np.log(joint / marginal_x))
        np.testing.assert_allclose(rows["target_marginal_loglik"], np.log(marginal_y))
        shuffled_joint = (source_probability[self.permutation] * target_probability) @ self.w
        np.testing.assert_allclose(rows["shuffled_conditional_loglik"],
                                   np.log(shuffled_joint / marginal_x[self.permutation]))
        expected_mean = (source_probability * self.w) @ self.ry / marginal_x[:, None]
        np.testing.assert_allclose(rows["conditional_mean"], expected_mean)

    def test_one_state_reduces_to_independence(self):
        rows = ev.prediction_rows(self.x, self.y, np.ones(1), self.rx[:1], self.ry[:1], self.permutation)
        np.testing.assert_allclose(rows["conditional_loglik"], rows["target_marginal_loglik"])
        np.testing.assert_allclose(rows["shuffled_conditional_loglik"], rows["target_marginal_loglik"])
        np.testing.assert_allclose(rows["conditional_mean"], np.tile(self.ry[0], (4, 1)))

    def test_paired_predictive_gain_is_symmetric_not_causal_direction(self):
        forward = ev.prediction_rows(self.x, self.y, self.w, self.rx, self.ry, self.permutation)
        reverse = ev.prediction_rows(self.y, self.x, self.w, self.ry, self.rx, self.permutation)
        np.testing.assert_allclose(
            forward["conditional_loglik"] - forward["target_marginal_loglik"],
            reverse["conditional_loglik"] - reverse["target_marginal_loglik"], atol=1e-14)

    def test_crosscovariance_matches_mixture_total_covariance(self):
        covariance, vx, vy = ev.cross_moments(self.w, self.rx, self.ry)
        expected = self.w @ (self.rx[:, 0] * self.ry[:, 0]) - (self.w @ self.rx[:, 0]) * (self.w @ self.ry[:, 0])
        self.assertAlmostEqual(covariance.item(), expected)
        self.assertAlmostEqual(vx.item(), self.w @ (self.rx[:, 0] ** 2 + self.rx[:, 0]) - (self.w @ self.rx[:, 0]) ** 2)
        self.assertGreater(vy.item(), 0)
        independent = ev.cross_moments(np.ones(1), self.rx[:1], self.ry[:1])[0]
        np.testing.assert_array_equal(independent, np.zeros((1, 1)))

    def test_zero_poisson_rate_is_valid_but_impossible_source_fails(self):
        actual = ev.poisson_components(np.array([[0.], [1.]]), np.array([[0.], [2.]]))
        self.assertEqual(actual[0, 0], 0)
        self.assertEqual(actual[1, 0], -np.inf)
        with self.assertRaisesRegex(ValueError, "zero probability"):
            ev.prediction_rows(self.x, self.y, np.ones(1), np.zeros((1, 1)), self.ry[:1], self.permutation)

    def test_repeated_or_noninteger_shuffle_indices_rejected(self):
        for perm in (np.array([0, 0, 2, 3]), np.array([0., 1., 2., 3.])):
            with self.assertRaisesRegex(ValueError, "permutation"):
                ev.prediction_rows(self.x, self.y, self.w, self.rx, self.ry, perm)

    def test_failed_fit_cannot_report_accepted_recovery(self):
        fit = {"world": "fixture", "control": "original", "variant": "E", "dimension": 1,
               "seed": 1, "calibration_score": -1, "acceptance": {"pass": False},
               "model": {"state_weights": self.w, "state_rates": {"a": self.rx, "b": self.ry}},
               "fit_scale": {"a": [1.], "b": [1.]}}
        report = ev.evaluate_fit(fit, {"a": self.x, "b": self.y}, ["a", "b"], {"a": 1, "b": 1},
                                 ev.bootstrap_indices(4, "fixture", 20), "fixture__original__E")
        self.assertFalse(report["fit_accepted"])
        self.assertFalse(report["all_directions_positive_family_adjusted_gain"])
        self.assertTrue(all(not r["accepted_fit_positive_family_adjusted_gain"] for r in report["pairs"].values()))

    def test_per_feature_gain_and_failed_restart_accounting(self):
        fit = {"world": "fixture", "control": "original", "variant": "E", "dimension": 1,
               "seed": 1, "calibration_score": -1, "acceptance": {"pass": True},
               "model": {"state_weights": self.w,
                         "state_rates": {"a": self.rx, "b": np.column_stack([self.ry[:, 0], self.ry[:, 0] + 1])}},
               "fit_scale": {"a": [1.], "b": [1., 1.]}}
        counts = {"a": self.x, "b": np.column_stack([self.y[:, 0], self.y[:, 0] + 1])}
        indices = ev.bootstrap_indices(4, "fixture", 20)
        scored = ev.evaluate_fit(fit, counts, ["a", "b"], {"a": 1, "b": 2}, indices, "fixture__original__E")
        pair = scored["pairs"]["a__to__b"]
        self.assertEqual(pair["conditional_loglik_gain_per_target_feature"]["mean"], pair["conditional_loglik_gain"]["mean"] / 2)
        np.testing.assert_allclose(pair["conditional_loglik_gain_per_target_feature"]["ci95"],
                                   np.asarray(pair["conditional_loglik_gain"]["ci95"]) / 2)
        failed = ev.evaluate_fit({**fit, "model": None, "error": "optimizer failed"}, counts,
                                 ["a", "b"], {"a": 1, "b": 2}, indices, "fixture__original__E")
        self.assertFalse(failed["fit_accepted"])
        self.assertEqual(failed["evaluation_status"], "fit_failed")
        self.assertEqual(failed["pairs"], {})
        summary = ev.restart_summary({"first": scored, "second": failed}, ["first", "second"])
        self.assertEqual(summary["scored_fits"], 1)
        self.assertEqual(summary["failed_fits"], 1)
        self.assertEqual(summary["accepted_fits"], 1)


class BoundariesAndUncertainty(unittest.TestCase):
    def _write_h5(self, path, ids, features):
        matrix = sparse.csr_matrix(np.arange(12, dtype=np.int64).reshape(4, 3))
        with h5py.File(path, "w") as handle:
            node = handle.create_group("X")
            node.attrs["encoding-type"] = "csr_matrix"
            node.attrs["shape"] = matrix.shape
            for key in ("data", "indices", "indptr"):
                node.create_dataset(key, data=getattr(matrix, key))
            for name, values in (("obs", ids), ("var", features)):
                group = handle.create_group(name)
                group.attrs["_index"] = "_index"
                group.create_dataset("_index", data=values, dtype=h5py.string_dtype())

    def test_assessment_preserves_fixed_column_order_and_rejects_pair_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ["c0", "c1", "c2", "c3"]
            for assay in ("a", "b"):
                self._write_h5(root / f"{assay}.h5ad", ids, ["f0", "f1", "f2"])
            protocol = {"worlds": ["fixture"], "assays": ["a", "b"], "widths": {"a": 2, "b": 2},
                        "source_indices": {"fixture": {a: [2, 0] for a in ("a", "b")}},
                        "feature_ids": {"fixture": {a: ["f2", "f0"] for a in ("a", "b")}},
                        "assessment_inputs": {"fixture": {a: {"path": f"{a}.h5ad", "sha256": ev.sha(root / f"{a}.h5ad")}
                                                           for a in ("a", "b")}}}
            counts, audits = ev.read_assessment(protocol, root, {"fixture": {"train0"}})
            np.testing.assert_array_equal(counts["fixture"]["a"], np.arange(12).reshape(4, 3)[:, [2, 0]])
            self.assertEqual(audits["fixture"]["paired_rows"], 4)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                ev.read_assessment(protocol, root, {"fixture": {"c0"}})
            self._write_h5(root / "b.h5ad", ids[::-1], ["f0", "f1", "f2"])
            protocol["assessment_inputs"]["fixture"]["b"]["sha256"] = ev.sha(root / "b.h5ad")
            with self.assertRaisesRegex(ValueError, "paired ID order"):
                ev.read_assessment(protocol, root, {"fixture": set()})

    def test_assessment_rejects_feature_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_h5(root / "a.h5ad", ["c0", "c1", "c2", "c3"], ["f0", "f1", "f2"])
            protocol = {"worlds": ["fixture"], "assays": ["a"], "widths": {"a": 2},
                        "source_indices": {"fixture": {"a": [2, 0]}},
                        "feature_ids": {"fixture": {"a": ["f0", "f2"]}},
                        "assessment_inputs": {"fixture": {"a": {"path": "a.h5ad", "sha256": ev.sha(root / "a.h5ad")}}}}
            with self.assertRaisesRegex(ValueError, "feature panel"):
                ev.read_assessment(protocol, root, {"fixture": set()})

    def test_bootstrap_is_deterministic_and_reports_wider_family_interval(self):
        first = ev.bootstrap_indices(4, "fixture", 1000)
        np.testing.assert_array_equal(first, ev.bootstrap_indices(4, "fixture", 1000))
        values = np.array([-3., 1., 2., 8.])
        report = ev.interval(values, first)
        estimates = values[first].mean(1)
        self.assertEqual(report["mean"], 2)
        np.testing.assert_allclose(report["ci95"], np.quantile(estimates, [.025, .975]))
        self.assertLessEqual(report["ci_family_adjusted"][0], report["ci95"][0])
        self.assertGreaterEqual(report["ci_family_adjusted"][1], report["ci95"][1])

    def test_changed_protocol_is_rejected_before_assessment_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"changed": True}))
            (root / "fit_seal.json").write_text(json.dumps({"protocol_sha256": "wrong"}))
            with patch.object(ev, "read_assessment") as loader:
                with self.assertRaisesRegex(ValueError, "Hash changed"):
                    ev.evaluate_run(root, root)
                loader.assert_not_called()

    def test_existing_evaluation_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation.json").write_text("original")
            with self.assertRaises(FileExistsError):
                ev.evaluate_run(root, root)
            self.assertEqual((root / "evaluation.json").read_text(), "original")

    def test_sealed_paths_cannot_escape_root(self):
        for path in ("../outside", "/tmp/outside"):
            with self.assertRaisesRegex(ValueError, "escapes"):
                ev.within(Path("/tmp/inside"), path)


if __name__ == "__main__":
    unittest.main()
