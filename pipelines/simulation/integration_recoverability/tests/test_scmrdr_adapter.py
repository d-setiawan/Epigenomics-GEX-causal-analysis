"""CPU end-to-end contract tests for the benchmark's scMRDR wrapper.

These use deliberately tiny count fixtures, not additional benchmark worlds.
They test provenance and data isolation, not causal recovery performance.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ad = pytest.importorskip("anndata")
torch = pytest.importorskip("torch")
pytest.importorskip("scanpy")
pytest.importorskip("skmisc")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scmrdr_adapter as wrapper
from gene_features import GeneConfig, transform_gene_features
from paper_faithful.upstream.loss import isometric_loss


def _write_cohort(root, cohort, n=64, truth_variant=0):
    """Only nonmodel metadata changes between truth_variant values."""
    root.mkdir(parents=True)
    mapping = {
        "RNA": ["g0", "g1", "g2", "g3"],
        "H3K27ac": ["g0", "g0", "g1", "g4"],
        "H3K4me3": ["g1", "g2", "g4", "g5"],
    }
    for assay_i, (assay, genes) in enumerate(mapping.items()):
        rng = np.random.default_rng(121 + assay_i + n)
        counts = rng.poisson(2, (n, 4)).astype(np.float32)
        counts[1:, 0] += 1
        counts[0] = 0
        data = ad.AnnData(
            sparse.csr_matrix(counts),
            obs=pd.DataFrame(
                {"latent_state": np.arange(n) + 1000 * truth_variant,
                 "paired_id": [f"private_{truth_variant}_{i}" for i in range(n)]},
                index=[f"{cohort}_{assay}_{i}" for i in range(n)]),
            var=pd.DataFrame({"gene_id": genes,
                              "true_role": [f"hidden_role_{truth_variant}"] * 4},
                             index=[f"{assay}_feature_{i}" for i in range(4)]),
        )
        data.uns["private_graph"] = {"variant": truth_variant}
        data.obsm["true_state"] = np.full((n, 2), truth_variant, dtype=np.float32)
        data.write_h5ad(root / f"{assay}.h5ad")
    return root


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    # Only this CPU test process is affected; parent GPU training is separate.
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("scmrdr_wrapper_fixture_")
    train = _write_cohort(root / "train", "train", 64)
    validation = _write_cohort(root / "validation", "validation", 9)
    heldout = _write_cohort(root / "heldout", "heldout", 11)
    run, transform = root / "run", root / "gene_transform"
    result = wrapper.fit_run(
        train, run, validation_dir=validation, gene_transform_dir=transform,
        seed=2, device="cpu", max_epochs=1, train_limit=64, n_top_genes=32,
    )
    return {"root": root, "train": train, "validation": validation,
            "heldout": heldout, "run": run, "transform": transform,
            "manifest": result}


def test_fit_preserves_ids_masks_and_explicit_validation(completed_run):
    fixture = completed_run
    manifest = fixture["manifest"]
    assert manifest["status"] == "complete"
    assert manifest["config"]["dispersion_strategy"] == "gene"
    gene = json.loads((fixture["transform"] / "manifest.json").read_text())
    assert gene["feature_names"] == [f"g{i}" for i in range(6)]
    np.testing.assert_array_equal(gene["assays"]["RNA"]["availability"],
                                  [True, True, True, True, False, False])
    for assay in wrapper.ASSAYS:
        with np.load(fixture["run"] / "train" / f"{assay}.npz") as z:
            assert z["cell_ids"].tolist() == [f"train_{assay}_{i}" for i in range(1, 64)]
            assert z["shared"].shape == (63, 20)
            assert z["private"].shape == (63, 20)
            assert np.isfinite(z["shared"]).all()
        assert manifest["train_inputs"][assay]["dropped_cell_ids"] == [f"train_{assay}_0"]
    fit_summary = json.loads((fixture["run"] / "model/fit_summary.json").read_text())
    assert set(fit_summary["training_rows"].values()) == {63}
    assert set(fit_summary["validation_rows"].values()) == {8}
    split = json.loads((fixture["run"] / "model/split_indices.json").read_text())
    assert all(s == {"training_rows": "all", "validation_rows": "external"}
               for s in split.values())


def test_encode_keeps_fitted_manifest_and_checkpoint_fixed(completed_run):
    fixture = completed_run
    before_checkpoint = wrapper.sha256(fixture["run"] / "model/checkpoint.pt")
    before_gene = wrapper.sha256(fixture["transform"] / "manifest.json")
    result = wrapper.encode_run(fixture["run"], fixture["heldout"], "heldout", device="cpu")
    assert result["post_fit_only"] is True
    for assay in wrapper.ASSAYS:
        with np.load(fixture["run"] / "heldout" / f"{assay}.npz") as z:
            assert z["cell_ids"].tolist() == [f"heldout_{assay}_{i}" for i in range(1, 11)]
            assert z["shared"].shape == (10, 20)
        assert result["inputs"][assay]["dropped_cell_ids"] == [f"heldout_{assay}_0"]
    assert wrapper.sha256(fixture["run"] / "model/checkpoint.pt") == before_checkpoint
    assert wrapper.sha256(fixture["transform"] / "manifest.json") == before_gene
    with pytest.raises(FileExistsError, match="already exists"):
        wrapper.encode_run(fixture["run"], fixture["heldout"], "heldout", device="cpu")


def test_fit_and_encode_refuse_overwrite_or_unsafe_cohort(completed_run):
    fixture = completed_run
    with pytest.raises(FileExistsError, match="already exists"):
        wrapper.fit_run(fixture["train"], fixture["run"], device="cpu", max_epochs=1)
    for bad in ("../escape", "train", "", ".", ".."):
        with pytest.raises(ValueError, match="cohort"):
            wrapper.encode_run(fixture["run"], fixture["heldout"], bad, device="cpu")


def test_reusable_transform_is_same_and_rejects_changed_configuration(completed_run, tmp_path):
    fixture = completed_run
    train = wrapper._paths(fixture["train"])
    manifest = wrapper._reuse_gene_manifest(fixture["transform"], train,
                                          GeneConfig(n_top_genes=32), (), 64)
    for assay in wrapper.ASSAYS:
        values = transform_gene_features(train[assay], assay, manifest, limit=64)
        assert values["cell_ids"].tolist() == [f"train_{assay}_{i}" for i in range(1, 64)]
    with pytest.raises(ValueError, match="configuration"):
        wrapper._reuse_gene_manifest(fixture["transform"], train,
                                     GeneConfig(n_top_genes=31), (), 64)
    # A new seed run can reuse this transform without refitting it. Mock only
    # feature fitting to prove the cache path is taken; model fitting is real.
    from unittest.mock import patch
    with patch.object(wrapper, "fit_gene_features", side_effect=AssertionError("must not refit transform")):
        result = wrapper.fit_run(
            fixture["train"], tmp_path / "reused_run", validation_dir=fixture["validation"],
            gene_transform_dir=fixture["transform"], seed=2, device="cpu",
            max_epochs=1, train_limit=64, n_top_genes=32,
        )
    assert result["gene_manifest_sha256"] == fixture["manifest"]["gene_manifest_sha256"]
    for assay in wrapper.ASSAYS:
        with np.load(fixture["run"] / "train" / f"{assay}.npz") as a, \
             np.load(tmp_path / "reused_run/train" / f"{assay}.npz") as b:
            np.testing.assert_array_equal(a["shared"], b["shared"])


def test_truth_metadata_does_not_change_fit(completed_run, tmp_path):
    fixture = completed_run
    train = _write_cohort(tmp_path / "train", "train", 64, truth_variant=97)
    validation = _write_cohort(tmp_path / "validation", "validation", 9, truth_variant=97)
    result = wrapper.fit_run(train, tmp_path / "run", validation_dir=validation,
                             seed=2, device="cpu", max_epochs=1,
                             train_limit=64, n_top_genes=32)
    assert result["status"] == "complete"
    for assay in wrapper.ASSAYS:
        with np.load(fixture["run"] / "train" / f"{assay}.npz") as original, \
             np.load(tmp_path / "run/train" / f"{assay}.npz") as changed:
            np.testing.assert_array_equal(original["cell_ids"], changed["cell_ids"])
            np.testing.assert_array_equal(original["shared"], changed["shared"])
            np.testing.assert_array_equal(original["private"], changed["private"])


def test_tampered_checkpoint_is_rejected_before_encoding(completed_run, tmp_path):
    fixture = completed_run
    copied = tmp_path / "copied_run"
    shutil.copytree(fixture["run"], copied)
    path = copied / "model/checkpoint.pt"
    # Deliberately corrupt only a private test copy, never a benchmark artifact.
    path.write_bytes(path.read_bytes() + b"invalid_test_suffix")
    with pytest.raises(ValueError, match="checkpoint changed"):
        wrapper.encode_run(copied, fixture["heldout"], "tampered", device="cpu")
    assert not (copied / "tampered").exists()


def test_heldout_feature_reordering_is_rejected(completed_run, tmp_path):
    fixture = completed_run
    modified = tmp_path / "reordered"
    shutil.copytree(fixture["heldout"], modified)
    path = modified / "RNA.h5ad"
    data = ad.read_h5ad(path)
    data[:, [3, 2, 1, 0]].copy().write_h5ad(path)
    with pytest.raises(ValueError, match="Feature order"):
        wrapper.encode_run(fixture["run"], modified, "reordered_failure", device="cpu")


def test_changed_input_cannot_reuse_fitted_transform(completed_run, tmp_path):
    fixture = completed_run
    modified = tmp_path / "modified"
    shutil.copytree(fixture["train"], modified)
    path = modified / "RNA.h5ad"
    data = ad.read_h5ad(path)
    data.X.data[0] += 1
    data.write_h5ad(path)
    with pytest.raises(ValueError, match="different training counts"):
        wrapper._reuse_gene_manifest(fixture["transform"], wrapper._paths(modified),
                                     GeneConfig(n_top_genes=32), (), 64)


def test_native_geometry_matches_pair_formula_and_gradients():
    shared = torch.tensor([[0.], [1.], [5.]], requires_grad=True)
    specific = torch.tensor([[0.], [2.], [17.]], requires_grad=True)
    modality = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    actual = isometric_loss(torch.cat([shared, specific], dim=1), shared, modality)
    # Only ordered pairs (0,1) and (1,0) contribute; self-pairs and cross-assay
    # pairs do not. Native code divides the paper's pair sum by minibatch size.
    expected = 2 / 3 * (torch.sqrt((shared[1] - shared[0]) ** 2 +
                                   (specific[1] - specific[0]) ** 2) -
                        torch.abs(shared[1] - shared[0])).square().sum()
    actual_grads = torch.autograd.grad(actual, (shared, specific), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (shared, specific))
    torch.testing.assert_close(actual, expected)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want)
    assert actual_grads[1][1].item() != 0  # Full/private geometry is NOT detached.
    assert actual_grads[1][2].item() == 0  # Singleton second assay contributes no pair.


def test_geometry_alone_does_not_prove_biological_preservation():
    # A constant private representation yields zero geometry loss. This valid
    # native optimum is a caution for interpreting the objective, not a defect
    # to "fix" by silently substituting a different integration algorithm.
    shared = torch.tensor([[0., 0.], [1., 2.], [3., -1.]])
    specific = torch.ones(3, 2)
    modality = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
    torch.testing.assert_close(isometric_loss(torch.cat([shared, specific], dim=1),
                                             shared, modality), torch.tensor(0.))
