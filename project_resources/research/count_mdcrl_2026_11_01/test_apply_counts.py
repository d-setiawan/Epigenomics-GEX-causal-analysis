"""Checks for training boundaries, signed fingerprints and abstention rules."""
import unittest
import numpy as np

from apply_counts import (GRID, MARGINS, align_columns, characteristic_fingerprint,
                          match_pair, permute_columns, shared_cliques, split_rows)


class TestCountPilot(unittest.TestCase):
    def test_split_is_id_defined_and_permutation_stays_in_depth_strata(self):
        ids = np.asarray([f"opaque-{i}" for i in range(40)])
        fit, cal = split_rows(ids)
        rf, rc = split_rows(ids[::-1])
        self.assertEqual(set(ids[fit]), set(ids[::-1][rf]))
        self.assertFalse(set(fit) & set(cal))
        self.assertEqual(len(set(fit) | set(cal)), 40)
        x = np.column_stack([np.arange(40), np.arange(40) + 100])
        out = permute_columns(x, np.arange(1, 41), np.random.default_rng(7))
        for group in np.array_split(np.arange(40), 8):
            for j in range(2):
                np.testing.assert_array_equal(np.sort(x[group, j]), np.sort(out[group, j]))

    def test_cf_sign_and_exploding_noise_gate(self):
        x = np.tile([[0., 1.], [2., 0.], [1., 1.]], (40, 1))
        mean, w = x.mean(axis=0), np.array([[0.5, -0.2]])
        for correction in [True, False]:
            a = characteristic_fingerprint(x, np.ones(len(x)), w, mean, correction)[0]
            b = characteristic_fingerprint(x, np.ones(len(x)), -w, mean, correction)[0]
            np.testing.assert_allclose(a, np.conjugate(b), atol=1e-14)
        bad = characteristic_fingerprint(np.full((20, 2), 1000.), np.ones(20),
                                         np.full((1, 2), 100.), np.zeros(2), True)
        self.assertFalse(bad[-1][0])
        self.assertFalse(np.isfinite(bad[0]).all())

    def test_matching_requires_precision_unique_sign_and_unique_source(self):
        value = np.zeros((1, len(MARGINS)))
        value[0, 0] = 1.0
        value[0, 2 + len(GRID):] = 0.2
        fp = dict(value=value, bootstrap=np.repeat(value[None], 32, axis=0), eligible=np.array([True]))
        matches = match_pair("RNA", "H3K27ac", fp, fp)[0]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["sign"], 1)
        uncertain = {**fp, "bootstrap": fp["bootstrap"].copy()}
        uncertain["bootstrap"][::2, 0, 0] += 10
        self.assertEqual(match_pair("RNA", "H3K27ac", fp, uncertain)[0], [])
        duplicated = dict(value=np.repeat(value, 2, axis=0), bootstrap=np.repeat(fp["bootstrap"], 2, axis=1), eligible=np.ones(2, bool))
        self.assertEqual(match_pair("RNA", "H3K27ac", fp, duplicated)[0], [])
        self.assertEqual(match_pair("RNA", "H3K27ac", fp, {**fp, "eligible": np.array([False])})[0], [])

    def test_alignment_and_sign_consistent_cliques(self):
        a = np.array([[1., 0.], [0., 2.], [0.1, 0.1]])
        order, sign, cosine = align_columns(a, a[:, ::-1] * [-1., 1.], np.ones(3))
        np.testing.assert_array_equal(order, [1, 0])
        np.testing.assert_array_equal(sign, [1, -1])
        np.testing.assert_allclose(cosine, 1)
        pairs = [dict(a="RNA", b="H3K27ac", ia=0, ib=1, sign=-1),
                 dict(a="RNA", b="H3K4me3", ia=0, ib=0, sign=1),
                 dict(a="H3K27ac", b="H3K4me3", ia=1, ib=0, sign=-1)]
        self.assertEqual(len(shared_cliques(pairs)), 1)
        pairs[-1]["sign"] = 1
        self.assertEqual(shared_cliques(pairs), [])


if __name__ == "__main__":
    unittest.main()
