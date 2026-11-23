import json
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from build_model_inputs_s1 import convert_one, file_sha256


class AnnDataBridgeTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root / "learner").mkdir()
        output = root / "model_inputs"
        (output / "train").mkdir(parents=True)
        sparse.save_npz(root / "learner/RNA_train_counts.npz", sparse.csr_matrix([[1, 0], [0, 3]], dtype=np.int32))
        pd.DataFrame({"cell_id": ["RNA.train.000001", "RNA.train.000000"], "library_depth": [1, 3]}).to_csv(root / "learner/RNA_train_cells.tsv", sep="\t", index=False)
        pd.DataFrame({"gene_id": ["ENSG2", "ENSG1"], "gene_name": ["B", "A"], "chrom": ["chr1", "chr2"]}).to_csv(root / "learner/RNA_features.tsv", sep="\t", index=False)
        summary = {"files": {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in (root / "learner").iterdir()}}
        return root, output, summary

    def test_exact_count_order_annotation_roundtrip_without_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output, summary = self.fixture(directory)
            record = convert_one(root, output, summary, "RNA", "train")
            data = ad.read_h5ad(output / record["path"])
            self.assertTrue(record["roundtrip_exact_counts_ids_gene_annotations"])
            self.assertEqual(data.obs_names.tolist(), ["RNA.train.000001", "RNA.train.000000"])
            self.assertEqual(data.var_names.tolist(), ["RNA|ENSG2", "RNA|ENSG1"])
            self.assertEqual(data.obs.columns.tolist(), [])
            self.assertEqual(dict(data.uns), {})
            np.testing.assert_array_equal(data.X.toarray(), [[1, 0], [0, 3]])

    def test_source_tampering_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output, summary = self.fixture(directory)
            with (root / "learner/RNA_train_cells.tsv").open("a") as handle:
                handle.write("\n")
            with self.assertRaises(ValueError):
                convert_one(root, output, summary, "RNA", "train")
            self.assertFalse((output / "train/RNA.h5ad").exists())


if __name__ == "__main__":
    unittest.main()
