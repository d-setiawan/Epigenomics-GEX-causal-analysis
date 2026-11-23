"""Tiny arrays only: no model, world, discovery, or binary-file access."""

import importlib.util
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/integration_validation_metrics.py"
SPEC = importlib.util.spec_from_file_location("integration_validation_metrics", SCRIPT)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class TransportCertificateTests(unittest.TestCase):
    def test_squared_oracle_differs_from_unsquared_assignment(self):
        source = np.array([[0, 0], [3, 0]])
        target = np.array([[0, 0], [-7 / 6, 5 * np.sqrt(11) / 6]])
        result = metrics.transport_certificate(source, target, [0, 1])
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["matched_squared_cost"], 12.5)
        self.assertAlmostEqual(result["oracle_squared_cost"], 9)
        self.assertAlmostEqual(result["random_expected_squared_cost"], 10.75)
        self.assertAlmostEqual(result["normalized_gap"], 2)
        self.assertEqual(result["one_lipschitz_squared_error_bound"], result["matched_squared_cost"])

    def test_exact_random_expectation_matches_all_permutations(self):
        source = np.array([[-1, 2], [2, 1], [4, -2]])
        target = np.array([[2, 4], [0, 1], [-3, 3]])
        costs = [np.mean(np.sum((source - target[list(columns)]) ** 2, axis=1))
                 for columns in itertools.permutations(range(3))]
        result = metrics.transport_certificate(source, target, [0, 1, 2])
        self.assertAlmostEqual(result["random_expected_squared_cost"], np.mean(costs))
        self.assertAlmostEqual(result["oracle_squared_cost"], min(costs))

    def test_identical_and_duplicate_states_preserve_zero_cost(self):
        state = np.array([[0, 0], [0, 0], [1, 2]])
        result = metrics.transport_certificate(state, state, [1, 0, 2])
        self.assertEqual(result["matched_squared_cost"], 0)
        self.assertEqual(result["normalized_gap"], 0)

    def test_zero_reference_gap_keeps_nonzero_absolute_cost(self):
        for source, target, columns in [([[0]], [[3]], [0]),
                                        ([[0], [0]], [[1], [3]], [1, 0]),
                                        ([[0], [0]], [[0], [0]], [0, 1])]:
            with self.subTest(source=source, target=target):
                result = metrics.transport_certificate(source, target, columns)
                self.assertEqual(result["status"], "ok")
                self.assertIsNone(result["normalized_gap"])
                self.assertEqual(result["reason"], "uninformative_reference")
                self.assertAlmostEqual(result["matched_squared_cost"], result["oracle_squared_cost"])
                self.assertEqual(result["one_lipschitz_squared_error_bound"], result["matched_squared_cost"])

    def test_empty_pools_are_unavailable(self):
        result = metrics.transport_certificate(np.empty((0, 2)), np.empty((0, 2)), [])
        self.assertEqual(result["reason"], "empty_pools")
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["matched_squared_cost"])

    def test_rejects_invalid_permutations(self):
        for columns in ([0, 0], [0], [0, 2], [-1, 0], [0.0, 1.0], [False, True], [[0, 1]]):
            with self.subTest(columns=columns), self.assertRaisesRegex(ValueError, "permutation"):
                metrics.transport_certificate([[0], [1]], [[0], [1]], columns)

    def test_rejects_invalid_states(self):
        for source, target in [([0, 1], [[0], [1]]),
                               ([[0]], [[0], [1]]),
                               ([[0]], [[0, 1]]),
                               ([[np.nan]], [[0]]),
                               ([[0]], [[np.inf]]),
                               (np.array([[1j]]), [[0]]),
                               (np.empty((1, 0)), np.empty((1, 0)))]:
            with self.subTest(source=source, target=target), self.assertRaises(ValueError):
                metrics.transport_certificate(source, target, [0])

    def test_rejects_material_oracle_inconsistency(self):
        wrong = (np.arange(2), np.array([1, 0]))
        with patch.object(metrics, "linear_sum_assignment", return_value=wrong):
            with self.assertRaisesRegex(ValueError, "oracle"):
                metrics.transport_certificate([[0], [1]], [[0], [1]], [0, 1])
            with self.assertRaisesRegex(ValueError, "oracle"):
                metrics.transport_certificate([[0], [1]], [[0], [1]], [1, 0])

    def test_roundoff_sized_negative_excess_does_not_make_negative_gap(self):
        costs = np.array([[0, 1e-14, 10], [1e-14, 0, 10], [10, 10, 0]])
        with patch.object(metrics, "cdist", return_value=costs), patch.object(
                metrics, "linear_sum_assignment", return_value=(np.arange(3), np.array([1, 0, 2]))):
            result = metrics.transport_certificate([[0], [1], [2]], [[0], [1], [2]], [0, 1, 2])
        self.assertEqual(result["normalized_gap"], 0)
        self.assertEqual(result["matched_squared_cost"], 0)
        self.assertGreater(result["oracle_squared_cost"], 0)

    def test_rejects_distance_and_cost_overflow(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            metrics.transport_certificate([[1e200]], [[-1e200]], [0])
        with np.errstate(over="ignore"), self.assertRaisesRegex(ValueError, "nonfinite"):
            metrics.transport_certificate([[0], [0]], [[1e154], [1e154]], [0, 1])

    def test_certificate_bounds_lipschitz_function_error(self):
        source = np.array([[-2, 1], [0, 2], [3, -1]])
        target = np.array([[1, 4], [-1, 2], [2, -2]])
        columns = np.array([2, 0, 1])
        result = metrics.transport_certificate(source, target, columns)
        direction = np.array([0.6, 0.8])
        error = np.mean((np.tanh(source @ direction) - np.tanh(target[columns] @ direction)) ** 2)
        self.assertLessEqual(error, result["one_lipschitz_squared_error_bound"])


class NeighborhoodTests(unittest.TestCase):
    def test_overlap_identity_all_neighbors_and_dimension_change(self):
        values = np.array([[0], [1], [3], [8]])
        for k in (1, 15):
            result = metrics.neighborhood_overlap(values, np.column_stack((values, values)), k)
            self.assertEqual(result["mean_knn_jaccard"], 1)
            self.assertEqual(result["effective_k"], min(k, 3))

    def test_duplicate_vectors_exclude_self_by_identity(self):
        # All-zero ties would silently retain self for some rows if the first
        # nearest neighbor were removed instead of masking each row's diagonal.
        reference = np.zeros((3, 1))
        embedding = np.array([[0], [3], [1]])
        result = metrics.neighborhood_overlap(reference, embedding, k=1)
        self.assertAlmostEqual(result["mean_knn_jaccard"], 1 / 3)
        self.assertEqual(metrics.neighborhood_overlap(reference, reference, k=1)["mean_knn_jaccard"], 1)

    def test_overlap_with_different_neighbors_can_be_zero(self):
        result = metrics.neighborhood_overlap([[0], [1], [9], [10]], [[0], [9], [1], [10]], k=1)
        self.assertEqual(result["mean_knn_jaccard"], 0)

    def test_overlap_small_pools_unavailable(self):
        for n in (0, 1):
            result = metrics.neighborhood_overlap(np.zeros((n, 1)), np.zeros((n, 2)))
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["effective_k"], 0)
            self.assertIsNone(result["mean_knn_jaccard"])

    def test_hubness_uniform_and_concentrated(self):
        target = np.array([[0], [5], [10], [15]])
        uniform = metrics.neighbor_hubness(target, target, k=1)
        self.assertEqual(uniform["indegree_gini"], 0)
        self.assertEqual(uniform["coverage"], 1)
        self.assertEqual(uniform["largest_indegree_fraction"], 0.25)
        concentrated = metrics.neighbor_hubness(np.zeros((7, 1)), target, k=1)
        self.assertEqual(concentrated["indegree_gini"], 0.75)
        self.assertEqual(concentrated["coverage"], 0.25)
        self.assertEqual(concentrated["largest_indegree_fraction"], 1)

    def test_hubness_caps_k_and_counts_all_targets(self):
        result = metrics.neighbor_hubness([[0], [2]], [[0], [1], [2]], k=15)
        self.assertEqual(result["effective_k"], 3)
        self.assertEqual(result["coverage"], 1)
        self.assertEqual(result["indegree_gini"], 0)
        self.assertAlmostEqual(result["largest_indegree_fraction"], 1 / 3)

    def test_hubness_ranges_and_numpy_k_are_json_serializable(self):
        rng = np.random.default_rng(123)
        result = metrics.neighbor_hubness(rng.normal(size=(7, 2)), rng.normal(size=(5, 2)), np.int64(2))
        for key in ("coverage", "largest_indegree_fraction", "indegree_gini"):
            self.assertGreaterEqual(result[key], 0)
            self.assertLessEqual(result[key], 1)
        json.dumps(result, allow_nan=False)
        json.dumps(metrics.neighborhood_overlap([[0], [1]], [[0], [1]], np.int64(1)), allow_nan=False)

    def test_hubness_empty_pool_unavailable(self):
        for ns, nt in ((0, 2), (2, 0), (0, 0)):
            result = metrics.neighbor_hubness(np.zeros((ns, 1)), np.zeros((nt, 1)))
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["effective_k"], 0)
            self.assertIsNone(result["indegree_gini"])

    def test_rejects_invalid_neighbor_inputs(self):
        for function in (metrics.neighbor_hubness, metrics.neighborhood_overlap):
            for k in (0, -1, 1.5, True):
                with self.subTest(function=function.__name__, k=k), self.assertRaisesRegex(ValueError, "k"):
                    function([[0], [1]], [[0], [1]], k=k)
            with self.assertRaises(ValueError):
                function([[np.nan]], [[0]])
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                function([[1e200], [-1e200]], [[1e200], [-1e200]])
        with self.assertRaisesRegex(ValueError, "dimension"):
            metrics.neighbor_hubness([[0]], [[0, 1]])
        with self.assertRaisesRegex(ValueError, "aligned rows"):
            metrics.neighborhood_overlap([[0]], [[0], [1]])


if __name__ == "__main__":
    unittest.main()
