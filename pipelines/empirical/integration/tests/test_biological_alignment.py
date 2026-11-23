"""Small calculation-contract fixtures, not new biological datasets/worlds."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_biological_alignment.py"
spec = importlib.util.spec_from_file_location("biology_alignment", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AlignmentContractTests(unittest.TestCase):
    def test_incomplete_or_duplicate_manifest_rejected(self):
        valid = pd.DataFrame({"mark": sorted(module.MARKS)})
        module.validate_manifest(valid)
        with self.assertRaises(ValueError):
            module.validate_manifest(valid.iloc[:-1])
        with self.assertRaises(ValueError):
            module.validate_manifest(pd.concat([valid, valid.iloc[:1]]))

    def test_input_mutation_detected(self):
        with tempfile.TemporaryDirectory(prefix="biology-provenance-") as directory:
            path = Path(directory) / "input.txt"
            path.write_text("old")
            info = module.file_info(path)
            path.write_text("changed input")
            with self.assertRaises(RuntimeError):
                module.verify_unchanged([info])

    def test_loader_ignores_reserved_values_and_joins_reordered_columns(self):
        with tempfile.TemporaryDirectory(prefix="biology-loader-") as directory:
            root = Path(directory)
            barcodes = [f"cell{i:03d}" for i in range(100)]
            features = sum(module.PANELS.values(), []) + ["Isotype-control"]
            count = len(features)
            raw = np.column_stack([np.full(count, i + 1.) for i in range(100)])
            for j, barcode in enumerate(barcodes):
                if module.split_for("H3K27ac", barcode) == "assessment":
                    raw[:, j] = -100  # Would fail validation if reserved values were analyzed.
            pd.DataFrame(raw, index=features, columns=barcodes).iloc[:, ::-1].to_csv(
                root / "test_adt_clean_matrix.tsv", sep="\t", index_label="feature")
            meta = pd.DataFrame({"barcode": barcodes, "donor_id": ["HTO1"] * 100,
                                 "adt_total": np.arange(1, 101) * count,
                                 "n_cuttag_fragments": np.arange(100) + 100})
            meta.iloc[::-1].to_csv(root / "test_clean_cells.tsv", sep="\t", index=False)
            emb = pd.DataFrame({"cell": barcodes, "barcode_core": barcodes,
                                "id": [f"chrom_H3K27ac::{b}" for b in barcodes],
                                "modality_key": "chrom_H3K27ac", "mark": "H3K27ac", "GLUE_1": np.arange(100)})
            with h5py.File(root / "chrom_H3K27ac_preprocessed.h5ad", "w") as handle:
                handle.create_dataset("obs/barcode_core", data=np.asarray(barcodes, dtype=h5py.string_dtype()))
                handle.create_dataset("obsm/X_lsi", data=np.arange(200).reshape(100, 2))
            row = SimpleNamespace(mark="H3K27ac", clean_cells="test_clean_cells.tsv", prefix="test")
            result = module.load_assay(root, row, emb, ["GLUE_1"], "development", root)
            self.assertTrue(result["dev"].all())
            expected = [np.log1p(i + 1) for i, b in enumerate(barcodes)
                        if module.split_for("H3K27ac", b) == "development"]
            np.testing.assert_allclose(result["proteins"][:, 0], expected)
            with self.assertRaises(ValueError):
                module.load_assay(root, row, emb, ["GLUE_1"], "assessment", root)

    def test_split_is_stable_and_suffix_invariant(self):
        self.assertEqual(module.split_for("H3K27ac", "ABC-1"), module.split_for("H3K27ac", "ABC"))
        labels = [module.split_for("H3K27ac", str(i)) for i in range(1000)]
        self.assertTrue(700 < labels.count("development") < 800)
        other = [module.split_for("H3K4me3", str(i)) for i in range(1000)]
        self.assertNotEqual(labels, other)

    def test_query_sampling_is_order_invariant(self):
        ids = np.array(["a", "b", "c", "d", "e"])
        reversed_ids = ids[::-1]
        self.assertEqual(set(ids[module.sample_positions(ids, 3)]),
                         set(reversed_ids[module.sample_positions(reversed_ids, 3)]))

    def test_rank_correlations_match_scipy_with_ties(self):
        a = np.array([[0, 4], [0, 3], [2, 3], [4, 1], [5, 0]])
        b = np.array([[3, 1], [1, 2], [2, 2], [0, 4], [0, 5]])
        expected = [spearmanr(a[:, j], b[:, j]).statistic for j in range(2)]
        np.testing.assert_allclose(module.correlations(a, b), expected)

    def test_constant_correlations_are_undefined(self):
        self.assertTrue(np.isnan(module.correlations(np.ones((5, 1)), np.arange(5)[:, None])[0]))
        self.assertEqual(module.json_clean({"x": np.nan}), {"x": None})

    def test_self_excluded_even_with_tied_embeddings(self):
        x = np.zeros((6, 2))
        ids = np.array(list("abcdef"))
        indices, _ = module.neighbors(x, x, ids, ids, 3)
        self.assertEqual(indices.shape, (6, 3))
        for i, row in enumerate(indices):
            self.assertNotIn(i, row)

    def test_duplicate_and_nonfinite_identities_rejected(self):
        with self.assertRaises(ValueError):
            module.neighbors(np.zeros((3, 2)), np.zeros((1, 2)), ["a", "a", "b"], ["c"], 1)
        with self.assertRaises(ValueError):
            module.neighbors(np.zeros((3, 2)), np.full((1, 2), np.nan), ["a", "b", "c"], ["d"], 1)

    def test_neighbor_mean_matches_direct_indexing(self):
        values = np.arange(40).reshape(10, 4)
        indices = np.array([[2, 3], [7, 8], [0, 1]])
        np.testing.assert_allclose(module.neighbor_mean(values, indices), values[indices].mean(axis=1))

    def test_permutations_stay_within_strata_and_preserve_rows(self):
        groups = [np.arange(4), np.arange(4, 10), np.array([10])]
        order = module.permute_within(groups, 11, np.random.default_rng(4))
        for group in groups:
            self.assertEqual(set(order[group]), set(group))
        self.assertEqual(order[10], 10)
        values = np.column_stack([np.arange(11), np.arange(11) * 2])
        self.assertTrue(np.array_equal(values[order, 1], 2 * values[order, 0]))

    def test_assessment_values_do_not_change_development_adjustment(self):
        meta = pd.DataFrame({"adt_total": np.arange(1, 41),
                             "n_cuttag_fragments": np.arange(40) ** 2,
                             "donor_id": ["HTO1"] * 20 + ["HTO2"] * 20})
        scores = np.random.default_rng(5).normal(size=(40, 2))
        fit = np.arange(40) < 30
        before = module.adjusted_scores(scores, meta, fit)
        scores[~fit] += 1e6
        after = module.adjusted_scores(scores, meta, fit)
        np.testing.assert_allclose(before[fit], after[fit])

    def test_assessment_and_overwrite_guards(self):
        with tempfile.TemporaryDirectory(prefix="biology-eval-contract-") as directory:
            common = [sys.executable, str(SCRIPT), "--embedding", "missing", "--prefix", "GLUE_",
                      "--match-embedding", "missing", "--out-dir", directory]
            result = subprocess.run(common, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            result = subprocess.run(common + ["--partition", "assessment"], capture_output=True, text=True)
            self.assertIn("Assessment is reserved", result.stderr)


if __name__ == "__main__":
    unittest.main()
