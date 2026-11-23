from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gene_features import (GeneConfig, annotation_matrix, fit_gene_features,
                           transform_gene_features)


def _fixture(tmp_path):
    ad = pytest.importorskip("anndata")
    import pandas as pd
    paths = {}
    for assay, genes in {"RNA": ["a", "b", "c"],
                          "H3K27ac": ["a", "a", "d"],
                          "H3K4me3": ["b", "c", "d"]}.items():
        x = np.asarray([[1, 2, 0], [0, 1, 2], [3, 1, 0], [0, 0, 0]])
        data = ad.AnnData(sparse.csr_matrix(x),
                         obs=pd.DataFrame(index=[assay + str(i) for i in range(4)]),
                         var=pd.DataFrame({"gene_id": genes}, index=[assay + f"_f{i}" for i in range(3)]))
        paths[assay] = tmp_path / f"{assay}.h5ad"
        data.write_h5ad(paths[assay])
    return paths


def test_sum_mapping_and_duplicate_features():
    m = annotation_matrix(["x", "y", "z"], ["a", "a", "b"], ["a", "b", "c"])
    np.testing.assert_array_equal(np.array([[1, 2, 3]]) @ m, [[3, 3, 0]])
    with pytest.raises(ValueError, match="unique"):
        annotation_matrix(["x", "x"], ["a", "a"], ["a"])


def test_masks_measured_zero_not_unavailable(tmp_path):
    paths = _fixture(tmp_path)
    manifest = fit_gene_features(paths, tmp_path / "fitted", GeneConfig(n_top_genes=20))
    result = transform_gene_features(paths["RNA"], "RNA", manifest)
    assert manifest["feature_names"] == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(result["availability"], [True, True, True, False])
    assert result["counts"][0, 2] == 0  # Measured zero, mask remains true.
    assert result["counts"][0, 3] == 0  # Unavailable, mask false.
    assert result["cell_ids"].tolist() == ["RNA0", "RNA1", "RNA2"]
    assert result["dropped_cell_ids"].tolist() == ["RNA3"]
    ac = transform_gene_features(paths["H3K27ac"], "H3K27ac", manifest)
    np.testing.assert_array_equal(ac["counts"].toarray()[:, 0], [3, 1, 4])


def test_heldout_order_and_annotations_are_verified(tmp_path):
    ad = pytest.importorskip("anndata")
    paths = _fixture(tmp_path)
    manifest = fit_gene_features(paths, tmp_path / "fitted", GeneConfig(n_top_genes=20))
    data = ad.read_h5ad(paths["RNA"])
    target = tmp_path / "new.h5ad"
    data[:, [2, 1, 0]].copy().write_h5ad(target)
    with pytest.raises(ValueError, match="Feature order"):
        transform_gene_features(target, "RNA", manifest)
    data.var["gene_id"] = ["wrong", "b", "c"]
    data.write_h5ad(target)
    with pytest.raises(ValueError, match="annotations"):
        transform_gene_features(target, "RNA", manifest)


def test_fitted_transform_no_overwrite_and_exclusion_before_sums(tmp_path):
    paths = _fixture(tmp_path)
    cfg = GeneConfig(n_top_genes=20)
    manifest = fit_gene_features(paths, tmp_path / "fitted", cfg,
                                exclude_features=("H3K27ac_f0",))
    result = transform_gene_features(paths["H3K27ac"], "H3K27ac", manifest)
    np.testing.assert_array_equal(result["counts"].toarray()[:, 0], [2, 1, 1])
    with pytest.raises(FileExistsError):
        fit_gene_features(paths, tmp_path / "fitted", cfg)


def test_noninteger_counts_rejected(tmp_path):
    ad = pytest.importorskip("anndata")
    paths = _fixture(tmp_path)
    data = ad.read_h5ad(paths["RNA"])
    data.X = data.X.astype(float) * 0.2
    data.write_h5ad(paths["RNA"])
    with pytest.raises(ValueError, match="integer counts"):
        fit_gene_features(paths, tmp_path / "fitted", GeneConfig(n_top_genes=20))
