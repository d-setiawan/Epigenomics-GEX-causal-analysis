import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pilot_adapter as adapter
from loss_patch import corrected_loss


def toy_views(directory, prefix="train"):
    directory.mkdir()
    rng = np.random.default_rng(1792)
    for assay in adapter.ASSAYS:
        counts = rng.poisson(3, (8, 12)).astype(np.float32)
        counts = np.column_stack((counts, np.zeros(8))).astype(np.float32)
        data = ad.AnnData(sparse.csr_matrix(counts),
                         obs=pd.DataFrame({"forbidden_state": np.arange(8)},
                                          index=[f"{prefix}_{assay}_{i}" for i in range(8)]),
                         var=pd.DataFrame({"gene_id": pd.Categorical([f"g{i}" for i in range(13)])},
                                          index=[f"{assay}_{i}" for i in range(13)]))
        data.write_h5ad(directory / f"{assay}.h5ad")
    return directory


def test_three_pair_sum_gradients_and_pair_order():
    def calculate(order):
        z = {a: torch.tensor([[float(i + 1), float(i + 2)]], requires_grad=True)
             for i, a in enumerate(adapter.ASSAYS)}
        pairs = [f"{a}+{b}" for a, b in itertools.combinations(adapter.ASSAYS, 2)]
        x = {a: {"input": v} for a, v in z.items()}
        x["cross"] = {p: torch.tensor(float(i + 1)) for i, p in enumerate(pairs)}
        class AE:
            metrics_to_log = []
            def loss(self, data, **kwargs):
                return {"reconstruction_loss": data["input"].square().sum()}
        model = SimpleNamespace(modalities=adapter.ASSAYS, modality_pairs=order,
                                aes={a: AE() for a in z}, metrics_to_log=[],
                                iot_loss_weight=0.01, sinkhorn_loss_weight=0.1,
                                get_iot_loss=lambda z_1, z_2, c_cross: c_cross * (z_1.sum() + 2 * z_2.sum()),
                                get_sinkhorn_reg=lambda z_1, z_2: z_1.square().sum() + 3 * z_2.square().sum())
        losses = corrected_loss(model, x, {a: {"z": v} for a, v in z.items()}, {a: {} for a in z})
        expected_iot, expected_sink = 0.0, 0.0
        for pair in order:
            a, b = pair.split("+")
            expected_iot += x["cross"][pair] * (z[a].sum() + 2 * z[b].sum())
            expected_sink += z[a].square().sum() + 3 * z[b].square().sum()
        torch.testing.assert_close(losses["iot_loss"].squeeze(), expected_iot * 0.01)
        torch.testing.assert_close(losses["sinkhorn_reg"].squeeze(), expected_sink * 0.1)
        actual = sum(losses.values())
        reference = sum(v.square().sum() for v in z.values()) + expected_iot * 0.01 + expected_sink * 0.1
        gradients = torch.autograd.grad(actual, list(z.values()), retain_graph=True)
        expected = torch.autograd.grad(reference, list(z.values()))
        for left, right in zip(gradients, expected):
            torch.testing.assert_close(left, right)
        return float(actual.detach()), gradients
    pairs = [f"{a}+{b}" for a, b in itertools.combinations(adapter.ASSAYS, 2)]
    first, gradients = calculate(pairs)
    for order in itertools.permutations(pairs):
        value, other = calculate(order)
        assert value == pytest.approx(first)
        for left, right in zip(gradients, other):
            torch.testing.assert_close(left, right)
    calculate(pairs[:1])  # A single pair is weighted once, not doubled.


def test_preparation_freezes_features_and_rejects_leakage(tmp_path):
    train = toy_views(tmp_path / "train")
    heldout = toy_views(tmp_path / "val", "val")
    prepared, contract, dropped = adapter.prepare(train)
    assert all(v.n_vars == 12 and not len(v.obs.columns) for v in prepared.values())
    assert all(not ids for ids in dropped.values())
    before = json.dumps(contract, sort_keys=True)
    transformed, same, _ = adapter.prepare(heldout, contract)
    assert same is contract and json.dumps(contract, sort_keys=True) == before
    assert all(v.n_vars == 12 for v in transformed.values())
    with pytest.raises(ValueError, match="overlaps training"):
        adapter.prepare(train, contract)
    data = ad.read_h5ad(heldout / "RNA.h5ad")
    data[:, ::-1].write_h5ad(heldout / "RNA.h5ad")
    with pytest.raises(ValueError, match="feature order"):
        adapter.prepare(heldout, contract)


def test_reference_selects_exact_ids_and_features_without_embeddings(tmp_path):
    train = toy_views(tmp_path / "train")
    references = tmp_path / "reference"
    references.mkdir()
    paths = {}
    for assay in adapter.ASSAYS:
        data = ad.read_h5ad(train / f"{assay}.h5ad")[3::-1, 2:10].copy()
        data.obsm["forbidden_embedding"] = np.full((4, 2), np.nan)
        paths[assay] = references / f"{assay}.h5ad"
        data.write_h5ad(paths[assay])
    prepared, contract, dropped = adapter.prepare(train, selection_paths=paths)
    for assay, data in prepared.items():
        assert data.obs_names.tolist() == [f"train_{assay}_{i}" for i in range(3, -1, -1)]
        assert data.var_names.tolist() == [f"{assay}_{i}" for i in range(2, 10)]
        assert not data.obsm and not dropped[assay]
        assert contract["selection_references"][assay]["sha256"] == adapter.sha256(paths[assay])


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 0.5])
def test_invalid_counts_rejected(bad):
    data = ad.AnnData(sparse.csr_matrix([[1.0, bad]]))
    with pytest.raises(ValueError, match="integer counts"):
        adapter.counts(data)


def test_zero_libraries_and_degenerate_guides_rejected():
    with pytest.raises(ValueError, match="zero count"):
        adapter.normalize(sparse.csr_matrix([[0, 0], [1, 0]]))
    prepared = {}
    for assay in adapter.ASSAYS:
        data = ad.AnnData(sparse.csr_matrix(np.ones((4, 3))),
                         var=pd.DataFrame({"gene_id": ["a", "b", "c"]}))
        data.layers["counts"] = data.X.copy()
        prepared[assay] = data
    with pytest.raises(ValueError, match="variable common"):
        adapter.cross_costs(prepared)


def test_native_minibatch_with_an_absent_assay_is_finite(tmp_path):
    import mudata as md
    prepared, _, _ = adapter.prepare(toy_views(tmp_path / "train"))
    costs, _ = adapter.cross_costs(prepared)
    data = md.MuData(prepared)
    data.uns["cross_keys"] = [f"cross_{pair}" for pair in costs]
    for pair, values in costs.items():
        data.uns[f"cross_{pair}"] = values
    Model, _ = adapter.native()
    config = adapter.Config(device="cpu", n_latent=3, n_hidden=8)
    model = Model(data, adapter.make_encoders(prepared, config))
    x = model.dataset[np.array([0, 1, 8, 9])]
    inferred = model.inference(x)
    reconstructed = model.generative(inferred)
    from scconfluence.model import ScConfluence
    original = ScConfluence.loss(model, x, inferred, reconstructed)
    assert not torch.isfinite(sum(original.values())).all()
    model.reset_log_metrics()
    actual = model.loss(x, inferred, reconstructed)
    assert torch.isfinite(sum(actual.values())).all()
    sum(actual.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.reset_log_metrics()
    model.log_latent_norms(inferred)
    assert len(model.metrics_to_log) == 2
    assert all(torch.isfinite(m["value"]).all() for m in model.metrics_to_log)


def test_real_native_cpu_fit_frozen_export_and_checkpoint_guard(tmp_path):
    train = toy_views(tmp_path / "train")
    val = toy_views(tmp_path / "val", "val")
    config = adapter.Config(device="cpu", n_latent=3, n_hidden=8, batch_size=4,
                            max_epochs=2, patience=1)
    output = tmp_path / "fit"
    record = adapter.fit_run(train, output, config)
    assert record["status"] == "complete"
    assert all(error == 0 for error in record["checkpoint_roundtrip_max_abs_error"].values())
    result = adapter.encode_run(output, val, "val", device="cpu")
    for assay, entry in result["exports"].items():
        with np.load(entry["path"]) as data:
            assert set(data.files) == {"cell_ids", "shared"}
            assert data["shared"].shape == (8, 3)
            assert entry["cell_ids_sha256"] == hashlib.sha256("\n".join(data["cell_ids"]).encode()).hexdigest()
    with pytest.raises(FileExistsError):
        adapter.encode_run(output, val, "val", device="cpu")
    with (output / "checkpoint.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed after fit"):
        adapter.encode_run(output, val, "discovery", device="cpu")
