"""Tiny storage-contract fixtures, not new biological benchmark datasets."""
import csv
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_biology_integration_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_biology_inputs", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="biology-storage-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "isolated"
        self.counts = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int64))
        self.rep = np.array([[.125, -.5], [1., 2.]], dtype=np.float32)
        self.rows = []
        for mark in ("RNA", *module.MARKS):
            name = "rna_preprocessed.h5ad" if mark == "RNA" else f"chrom_{mark}_preprocessed.h5ad"
            data = ad.AnnData(X=np.array([[.2, .3, -.7], [-.2, -.3, .7]], dtype=np.float64),
                             obs=pd.DataFrame({"group": pd.Categorical(["a", "b"])}, index=["cell1", "cell2"]),
                             var=pd.DataFrame({"highly_variable": [True, False, True]}, index=["f1", "f2", "f3"]))
            data.layers["counts"] = self.counts.copy()
            if mark != "RNA":
                data.X = self.counts.copy()
            data.obsm["X_pca" if mark == "RNA" else "X_lsi"] = self.rep.copy()
            data.uns["fixture"] = {"unchanged": "yes", "numeric": np.array([1., 2.])}
            data.write_h5ad(self.source / name)
            if mark != "RNA":
                self.rows.append({"mark": mark, "chrom_h5ad": f"/stale/integration/{name}", "n_features": "3"})
        self.write_manifest()

    def write_manifest(self):
        with (self.source / "chrom_preprocessed_manifest.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["mark", "chrom_h5ad", "n_features"], delimiter="\t")
            writer.writeheader()
            writer.writerows(self.rows)

    def test_round_trip_preserves_training_inputs_and_source_bytes(self):
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}
        result = module.prepare_inputs(self.source, self.output, chunk_bytes=16)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["files"]), 7)
        self.assertIn("RAW COUNTS", result["x_semantics"])
        self.assertTrue((self.output / module.COMPLETION_FILE).exists())
        for record in result["files"]:
            data = ad.read_h5ad(record["destination"])
            np.testing.assert_array_equal(data.X.toarray(), self.counts.toarray())
            np.testing.assert_array_equal(data.layers["counts"].toarray(), self.counts.toarray())
            np.testing.assert_array_equal(data.obsm[record["representation"]], self.rep)
            self.assertEqual(list(data.obs_names), ["cell1", "cell2"])
            self.assertEqual(list(data.var_names), ["f1", "f2", "f3"])
            self.assertEqual(data.uns["fixture"]["unchanged"], "yes")
            with h5py.File(record["destination"], "r") as handle:
                self.assertEqual(handle["X"].id, handle["layers/counts"].id)
                self.assertIn("raw counts", handle.attrs["biology-preparation-x-semantics"])
        for p in self.source.iterdir():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), source_hashes[p.name])
        with (self.output / "chrom_preprocessed_manifest.tsv").open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual({r["mark"] for r in rows}, set(module.MARKS))
        self.assertTrue(all(Path(r["chrom_h5ad"]).parent == self.output for r in rows))

    def test_never_reads_dense_source_x_values(self):
        original = h5py.Dataset.__getitem__

        def guarded(dataset, key, *args, **kwargs):
            if dataset.name == "/X" and Path(dataset.file.filename).parent == self.source:
                raise AssertionError("Dense source X was read")
            return original(dataset, key, *args, **kwargs)

        with patch.object(h5py.Dataset, "__getitem__", guarded):
            module.prepare_inputs(self.source, self.output)

    def test_refuses_existing_output_and_dangling_symlink(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            module.prepare_inputs(self.source, self.output)
        link = self.root / "dangling"
        link.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            module.prepare_inputs(self.source, link)

    def test_refuses_missing_or_duplicate_marks_before_creation(self):
        self.rows[-1] = self.rows[0].copy()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "six unique"):
            module.prepare_inputs(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_refuses_wrong_basename_or_foreign_existing_path(self):
        self.rows[0]["chrom_h5ad"] = "/stale/chrom_wrong_preprocessed.h5ad"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unexpected chromatin filename"):
            module.prepare_inputs(self.source, self.output)
        foreign = self.root / f"chrom_{module.MARKS[0]}_preprocessed.h5ad"
        foreign.write_bytes(b"do not read this")
        self.rows[0]["chrom_h5ad"] = str(foreign)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "different existing file"):
            module.prepare_inputs(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_refuses_missing_representation_before_creation(self):
        with h5py.File(self.source / "rna_preprocessed.h5ad", "r+") as handle:
            del handle["obsm/X_pca"]
        with self.assertRaisesRegex(ValueError, "Missing required"):
            module.prepare_inputs(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_refuses_destination_inside_source_and_missing_local_file(self):
        with self.assertRaisesRegex(ValueError, "inside the source"):
            module.prepare_inputs(self.source, self.source / "nested")
        (self.source / f"chrom_{module.MARKS[0]}_preprocessed.h5ad").rename(self.root / "moved_fixture.h5ad")
        with self.assertRaisesRegex(FileNotFoundError, "source-local"):
            module.prepare_inputs(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_refuses_duplicate_identities_before_creation(self):
        with h5py.File(self.source / "rna_preprocessed.h5ad", "r+") as handle:
            handle["obs/_index"][1] = "cell1"
        with self.assertRaisesRegex(ValueError, "Duplicate identities"):
            module.prepare_inputs(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_hash_is_chunk_invariant_and_detects_value_and_identity_changes(self):
        path = self.source / "rna_preprocessed.h5ad"
        with h5py.File(path, "r+") as handle:
            first = module.hash_h5_object(handle["layers/counts"], 8)
            self.assertEqual(first, module.hash_h5_object(handle["layers/counts"], 128))
            handle["layers/counts/data"][0] = 9
            self.assertNotEqual(first, module.hash_h5_object(handle["layers/counts"], 8))
            identity = module.hash_h5_object(handle["obs/_index"], 8)
            self.assertEqual(identity, module.hash_h5_object(handle["obs/_index"], 128))
            handle["obs/_index"][0] = "changed"
            self.assertNotEqual(identity, module.hash_h5_object(handle["obs/_index"], 8))

    def test_detects_source_change_and_does_not_mark_complete(self):
        original_stat = module._stat
        counts = {}

        def changing_stat(path):
            value = original_stat(path)
            if Path(path).name == "rna_preprocessed.h5ad":
                counts[str(path)] = counts.get(str(path), 0) + 1
                if counts[str(path)] > 1:
                    value["mtime_ns"] += 1
            return value

        with patch.object(module, "_stat", changing_stat):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                module.prepare_inputs(self.source, self.output)
        self.assertFalse((self.output / module.COMPLETION_FILE).exists())


if __name__ == "__main__":
    unittest.main()
