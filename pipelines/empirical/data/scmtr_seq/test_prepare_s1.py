"""Small scientific-integrity tests; no network or study input is needed."""
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

import acquire_s1
from prepare_s1 import canonical_id, collapse_rna, count_fragments, promoters_from_gtf
from build_cohort_s1 import ASSAYS, assign_cells, assay_qc, select_panel, windows_overlap


class PreparationTests(unittest.TestCase):
    def test_rna_tags_do_not_create_false_same_cell_sums(self):
        features = pd.DataFrame({"gene_name": ["GENE", "MT-TEST"]})
        matrix = sparse.csr_matrix([[8, 1], [1, 0], [5, 0], [5, 0], [0, 10]])
        labels = ["01:01:01:01", "01:01:01:02", "01:01:02:01", "01:01:02:02", "01:01:03:06"]
        result, audit = collapse_rna(matrix, labels, features)
        np.testing.assert_array_equal(result.toarray(), [[8, 1], [5, 0], [0, 10]])
        np.testing.assert_allclose(audit.dominant_RT_fraction, [.9, .5, 1])
        self.assertEqual(audit.dominant_RT_tag.tolist(), ["01", "01", "06"])
        self.assertEqual(audit.canonical_id.tolist(), ["01:01:01:S1", "01:01:02:S1", "01:01:03:S1"])
        with self.assertRaises(ValueError):
            canonical_id("49:01:01:S1")

    def test_strand_tss_conversion_and_exact_fragment_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with gzip.open(path / "genes.gtf.gz", "wt") as handle:
                handle.write('##description: GRCh38\nchr1\ttest\tgene\t3001\t4000\t.\t+\t.\tgene_id "ENSG1.1"; gene_name "A";\nchr1\ttest\tgene\t3001\t6000\t.\t-\t.\tgene_id "ENSG2.1"; gene_name "B";\n')
            genes = pd.DataFrame({"gene_id": ["ENSG1", "ENSG2"], "gene_name": ["A", "B"]})
            promoters, _ = promoters_from_gtf(path / "genes.gtf.gz", genes)
            self.assertEqual(promoters.tss.tolist(), [3000, 5999])
            self.assertEqual(promoters.start.tolist(), [1000, 3999])
            with gzip.open(path / "fragments.gz", "wt") as handle:
                handle.write("chr1\t999\t1000\t01:01:01:S1\t1\nchr1\t1000\t1001\t01:01:01:S1\t2\nchr1\t4500\t4501\t01:01:01:S1\t3\nchr1\t7999\t8000\t01:01:01:S1\t1\n")
            result, audit, summary = count_fragments(path / "fragments.gz", promoters)
            np.testing.assert_array_equal(result.toarray(), [[5, 3]])
            self.assertEqual(audit.counts.tolist(), [7])
            self.assertEqual(summary["multiple_promoter_overlap_rows"], 1)


class AcquisitionTests(unittest.TestCase):
    def test_complete_download_and_existing_file_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            response = io.BytesIO(b"complete")
            response.headers = {"Content-Length": "8"}
            with patch.object(acquire_s1, "BASE", Path(directory)), patch.object(acquire_s1.urllib.request, "urlopen", return_value=response):
                record = acquire_s1.acquire("input.gz", "https://example.test/input", 8)
                self.assertEqual(record["bytes"], 8)
                acquire_s1.acquire("input.gz", "https://example.test/input", 8)
                Path(record["path"]).write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    acquire_s1.acquire("input.gz", "https://example.test/input", 8)

    def test_truncated_input_stays_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            response = io.BytesIO(b"short")
            response.headers = {"Content-Length": "10"}
            with patch.object(acquire_s1, "BASE", Path(directory)), patch.object(acquire_s1.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    acquire_s1.acquire("input.gz", "https://example.test/input", 10)
            self.assertTrue((Path(directory) / "raw_s1/input.gz.partial").exists())
            self.assertFalse((Path(directory) / "raw_s1/input.gz").exists())


class CohortTests(unittest.TestCase):
    def test_assignment_precedes_qc_and_histone_qc_ignores_hidden_rna(self):
        ids = [f"01:01:{i:02d}:S1" for i in range(1, 13)]
        self.assertEqual(assign_cells(ids, reserve=3), assign_cells(ids[::-1], reserve=3))
        assignments = assign_cells(ids, reserve=3)
        self.assertEqual([list(assignments.values()).count(a) for a in ASSAYS], [3, 3, 3])
        cells = pd.DataFrame({"H3K27ac_counts": [249, 250], "RNA_counts": [100000, 0]})
        np.testing.assert_array_equal(assay_qc(cells, "H3K27ac"), [False, True])
        cells["RNA_counts"] = [0, 100000]
        np.testing.assert_array_equal(assay_qc(cells, "H3K27ac"), [False, True])

    def test_panel_uses_coverage_and_excludes_overlapping_loci(self):
        features = pd.DataFrame({"gene_id": [f"ENSG{i}" for i in range(5)], "gene_name": list("ABCDE"),
                                 "chrom": ["chr1"] * 5, "start": [0, 100, 2000, 4000, 6000], "end": [500, 600, 2500, 4500, 6500]})
        training = {assay: sparse.csr_matrix(np.ones((20, 5), dtype=int)) for assay in ASSAYS}
        panel, excluded, evidence = select_panel(features, training)
        self.assertEqual(len(panel), 3)
        self.assertTrue(evidence.eligible.all())
        for left in panel:
            for right in panel:
                self.assertFalse(left != right and windows_overlap(features.loc[left], features.loc[right]))
            overlapping = (features.start < features.loc[left, "end"]) & (features.end > features.loc[left, "start"])
            self.assertTrue(excluded[overlapping].all())
        training["H3K4me3"] = sparse.csr_matrix((20, 5))
        with self.assertRaises(ValueError):
            select_panel(features, training)


if __name__ == "__main__":
    unittest.main()
