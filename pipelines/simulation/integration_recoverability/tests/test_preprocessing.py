from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import ASSAYS, file_sha256
from preprocessing import (PreprocessingConfig, _read_counts, fit_native_lsi,
                           fit_preprocessing, transform_native_lsi,
                           transform_preprocessing)


def test_native_lsi_matches_upstream_and_heldout_is_batch_independent():
    ad = pytest.importorskip("anndata")
    scglue = pytest.importorskip("scglue")
    rng = np.random.default_rng(14)
    counts = sparse.csr_matrix(rng.poisson(rng.uniform(1, 8, size=18), size=(80, 18)))
    representation, fitted, audit = fit_native_lsi(counts, 5, seed=17, n_iter=15)
    native = ad.AnnData(counts.astype(float))
    scglue.data.lsi(native, n_components=5, use_highly_variable=False, n_iter=15, random_state=17)
    np.testing.assert_allclose(representation, native.obsm["X_lsi"], atol=1e-10)
    heldout = sparse.csr_matrix(rng.poisson(rng.uniform(1, 8, size=18), size=(10, 18)))
    all_values = transform_native_lsi(heldout, fitted)
    one = transform_native_lsi(heldout[:1], fitted)
    np.testing.assert_allclose(one, all_values[:1], atol=1e-12)
    # Deliberately omitting D^-1 is not the implemented transformation.
    wrong = dict(fitted, singular_values=np.ones(5))
    assert not np.allclose(all_values, transform_native_lsi(heldout, wrong))
    assert audit["row_std_ddof"] == 1


def test_native_lsi_rejects_unseen_training_columns_and_capacity_reduction():
    pytest.importorskip("scglue")
    matrix = sparse.csr_matrix(np.column_stack((np.ones((10, 3)), np.zeros(10))))
    with pytest.raises(ValueError, match="Zero-total"):
        fit_native_lsi(matrix, 2)
    with pytest.raises(ValueError, match="Too few"):
        fit_native_lsi(sparse.csr_matrix(np.ones((4, 4))), 4)


def _write_inputs(directory: Path, seed: int, n: int = 180, p: int = 140):
    ad = pytest.importorskip("anndata")
    import pandas as pd

    directory.mkdir()
    rng = np.random.default_rng(seed)
    paths = {}
    for assay in ASSAYS:
        rates = np.geomspace(0.1, 15, p)
        counts = rng.negative_binomial(5, 5 / (5 + rates), size=(n, p))
        counts[:, -1] = 0
        var = pd.DataFrame({"gene_id": [f"g{i}" for i in range(p)]},
                           index=[f"{assay}_f{i}" for i in range(p)])
        obs = pd.DataFrame({"total_counts": 123456, "private_truth_forbidden": "never copy"},
                           index=[f"{seed}_{assay}_c{i}" for i in range(n)])
        data = ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=var)
        paths[assay] = directory / f"{assay}.h5ad"
        data.write_h5ad(paths[assay])
    return paths


def _config():
    return PreprocessingConfig(n_components=5, rna_min_genes=0, rna_min_cells=1,
                               rna_top_genes=40, chrom_encoder_features=110,
                               chrom_reconstruction_features=60, train_cells_per_assay=150,
                               discovery_cells_per_assay=50, seed=12)


def test_frozen_fit_transform_feature_order_and_masks(tmp_path):
    ad = pytest.importorskip("anndata")
    pytest.importorskip("skmisc")
    train = _write_inputs(tmp_path / "inputs", 4)
    heldout = _write_inputs(tmp_path / "new", 5, n=80)
    destination = tmp_path / "fitted"
    fit = fit_preprocessing(train, destination, _config())
    parameter_hashes = {a: file_sha256(destination / "transforms" / f"{a}.npz") for a in ASSAYS}
    transformed = transform_preprocessing(heldout, destination, cohort="discovery")
    assert transformed["status"] == "complete"
    for assay in ASSAYS:
        training = ad.read_h5ad(destination / "train" / f"{assay}.h5ad")
        new = ad.read_h5ad(destination / "cohorts/discovery" / f"{assay}.h5ad")
        assert training.var_names.tolist() == new.var_names.tolist()
        assert new.n_obs == 50
        assert training.n_obs == 150
        assert set(training.obs) == {"assay", "total_counts", "n_features"}
        assert all(training.var["highly_variable"])
        np.testing.assert_array_equal(training.X.toarray(), training.layers["counts"].toarray())
        assert parameter_hashes[assay] == file_sha256(destination / "transforms" / f"{assay}.npz")
        if assay != "RNA":
            assert fit["assays"][assay]["encoder_features"] == 110
            assert training.n_vars == 60
    # Reuse has a verified content fingerprint, not an untracked refit.
    assert fit_preprocessing(train, destination, _config())["fingerprint"] == fit["fingerprint"]
    assert transform_preprocessing(heldout, destination, cohort="discovery")["fingerprint"] == transformed["fingerprint"]


def test_train_rna_extension_reproduces_fitted_pca(tmp_path):
    ad = pytest.importorskip("anndata")
    pytest.importorskip("skmisc")
    train = _write_inputs(tmp_path / "inputs", 18)
    config = _config()
    config.train_cells_per_assay = None
    destination = tmp_path / "fitted"
    fit_preprocessing(train, destination, config)
    transform_preprocessing(train, destination, cohort="training_projection_check")
    original = ad.read_h5ad(destination / "train/RNA.h5ad")
    projected = ad.read_h5ad(destination / "cohorts/training_projection_check/RNA.h5ad")
    np.testing.assert_allclose(original.obsm["X_pca"], projected.obsm["X_pca"], rtol=1e-5, atol=1e-5)


def test_exclusions_precede_normalization_and_metadata_does_not_leak(tmp_path):
    inputs = _write_inputs(tmp_path / "inputs", 11, n=30)
    data, audit = _read_counts(inputs["RNA"], "RNA", ["RNA_f138"], None)
    assert "RNA_f138" not in data.var_names
    assert "private_truth_forbidden" not in data.obs
    assert "total_counts" not in data.obs  # Recomputed from allowed outputs only.
    assert audit["excluded_present"] == ["RNA_f138"]


def test_guidance_has_no_excluded_cross_links_and_self_loops(tmp_path):
    pytest.importorskip("skmisc")
    import networkx as nx

    inputs = _write_inputs(tmp_path / "inputs", 35)
    config = _config()
    config.guidance_excluded_feature_ids = [f"H3K27ac_f{i}" for i in range(140)]
    root = tmp_path / "fitted"
    manifest = fit_preprocessing(inputs, root, config)
    graph = nx.read_graphml(root / "guidance.graphml.gz")
    assert graph.number_of_nodes() == sum(x["n_features"] for x in manifest["assays"].values())
    for source, target, attrs in graph.edges(data=True):
        assert attrs == {"weight": 1.0, "sign": 1}
        if source.startswith("H3K27ac") or target.startswith("H3K27ac"):
            assert source == target
    assert all(graph.has_edge(node, node) for node in graph)
