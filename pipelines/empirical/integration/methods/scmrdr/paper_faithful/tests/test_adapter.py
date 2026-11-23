"""CPU-only equation, update-isolation, checkpoint and held-out tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_faithful import SCmRDRConfig, ScMRDRAdapter, fit
from paper_faithful.adapter import _SparseDataset, training_step, validate_model
from paper_faithful.upstream.loss import ZINBLoss, isometric_loss, klLoss, klLoss_prior


class PaperEquations(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_mask_scaling_and_no_direct_unobserved_gradient(self):
        x = torch.tensor([[0., 2., 1.], [1., 0., 3.]], dtype=torch.float64)
        rho = torch.tensor([[.2, .3, .5], [.3, .2, .5]], dtype=torch.float64, requires_grad=True)
        pi = torch.full_like(rho, .2, requires_grad=True)
        r = torch.full_like(rho, 2., requires_grad=True)
        s = torch.tensor([[5.], [4.]], dtype=torch.float64)
        mask = torch.tensor([[1., 0., 1.], [1., 0., 1.]], dtype=torch.float64)
        objective = ZINBLoss()(x, rho, r, pi, s, mask, eps=1e-10)
        mu = rho * s
        nb = torch.distributions.NegativeBinomial(total_count=r, probs=mu / (mu + r))
        log_nb = nb.log_prob(x)
        log_probability = torch.where(x == 0,
            torch.log(pi + (1 - pi) * torch.exp(log_nb) + 1e-10),
            log_nb + torch.log(1 - pi + 1e-10))
        expected = -(log_probability * mask).sum(dim=1).mul(3 / 2).mean()
        torch.testing.assert_close(objective, expected)
        objective.backward()
        for value in (rho, r, pi):
            torch.testing.assert_close(value.grad[:, 1], torch.zeros(2, dtype=torch.float64))

    def test_unmasked_loss_equals_all_ones_mask(self):
        x = torch.tensor([[1., 0., 3.], [0., 2., 1.]])
        rho = torch.full_like(x, 1 / 3)
        r, pi = torch.ones_like(x), torch.full_like(x, .15)
        s = x.sum(1, keepdim=True)
        torch.testing.assert_close(ZINBLoss()(x, rho, r, pi, s),
                                   ZINBLoss()(x, rho, r, pi, s, torch.ones_like(x)))

    def test_kl_sums_over_latent_dimensions(self):
        mu = torch.ones(2, 4)
        logvar = torch.zeros_like(mu)
        self.assertAlmostEqual(klLoss(mu, logvar).item(), 2.0)
        torch.testing.assert_close(klLoss(mu, logvar),
                                   klLoss_prior(mu, logvar, torch.zeros(2, 1), torch.zeros(2, 1)))

    def test_geometry_uses_within_modality_sum_divided_by_batch(self):
        shared = torch.tensor([[0.], [1.], [9.], [10.]])
        full = torch.cat([shared, torch.tensor([[0.], [2.], [0.], [2.]])], dim=1)
        m = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        expected = (5 ** .5 - 1) ** 2
        self.assertAlmostEqual(isometric_loss(full, shared, m).item(), expected, places=5)

    def test_zero_counts_have_finite_masked_objective(self):
        x = torch.zeros(3, 4)
        loss = ZINBLoss()(x, torch.full_like(x, .25), torch.ones_like(x),
                         torch.full_like(x, .5), x.sum(1, keepdim=True), torch.ones_like(x))
        self.assertTrue(torch.isfinite(loss))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.features = [f"feature_{i}" for i in range(8)]
        self.names = ("mark", "rna")
        self.masks = {name: np.ones(8, dtype=bool) for name in self.names}
        self.config = SCmRDRConfig(hidden_layers=(8, 8), latent_dim_shared=3,
                                  latent_dim_specific=3, batch_size=16,
                                  max_epochs=2, patience=2, seed=7)
        rng = np.random.default_rng(23)
        self.counts = {name: sparse.csr_matrix(rng.poisson(1., (48, 8)).astype(np.float32))
                       for name in self.names}

    def adapter(self):
        return ScMRDRAdapter(self.features, self.names, self.masks, self.config)

    def test_unified_architecture_and_paper_dispersion(self):
        model = self.adapter().model
        self.assertEqual(model.encoder_shared.zxy_encoder[0].in_features, 8)
        self.assertEqual(model.encoder_specific.zxy_encoder[0].in_features, 10)
        model.eval()
        _, r, _ = model.decoder(torch.zeros(4, 6), torch.zeros(4, 0),
                                torch.tensor([[1., 0.]] * 4))
        self.assertEqual(r.shape, (8,))

    def test_adversarial_updates_do_not_leave_discriminator_gradients(self):
        adapter = self.adapter()
        model = adapter.model
        ds = _SparseDataset(self.counts, self.names,
                            {name: np.zeros((48, 0), dtype=np.float32) for name in self.names}, self.names)
        params = [p for name, p in model.named_parameters() if not name.startswith("discriminator.")]
        ov = torch.optim.Adam(params, lr=1e-3)
        od = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
        before_d = [p.detach().clone() for p in model.discriminator.parameters()]
        before_v = model.encoder_shared.mu_layer.weight.detach().clone()
        parts = training_step(model, ds.batch(np.arange(16), "cpu", 2), ov, od)
        self.assertTrue(np.isfinite(list(parts.values())).all())
        self.assertTrue(all(p.grad is None for p in model.discriminator.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_d, model.discriminator.parameters())))
        self.assertFalse(torch.equal(before_v, model.encoder_shared.mu_layer.weight))

    def test_validation_uses_every_batch_and_cell_weights(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.placeholder = torch.nn.Parameter(torch.tensor(1.))
                self.modality_num = 2
            def forward(self, x, b, m, i, w, stage):
                loss = x.mean()
                return None, None, loss, {"total_loss": float(loss)}
        data = {"mark": sparse.csr_matrix([[1.], [2.]]), "rna": sparse.csr_matrix([[9.]])}
        cov = {"mark": np.zeros((2, 0), np.float32), "rna": np.zeros((1, 0), np.float32)}
        dataset = _SparseDataset(data, self.names, cov, self.names)
        result = validate_model(FakeModel(), dataset, batch_size=2, seed=1)
        self.assertAlmostEqual(result["total_loss"], 4.)

    def test_checkpoint_and_frozen_heldout_encoding(self):
        adapter = self.adapter()
        x = self.counts["rna"][:7]
        before = {k: v.clone() for k, v in adapter.model.state_dict().items()}
        encoded = adapter.encode(x, "rna", self.features, batch_size=2)
        chunked = adapter.encode(x, "rna", self.features, batch_size=7)
        np.testing.assert_allclose(encoded["shared"], chunked["shared"], rtol=1e-5, atol=1e-6)
        for k, v in adapter.model.state_dict().items():
            torch.testing.assert_close(v, before[k])
        with tempfile.TemporaryDirectory(prefix="scmrdr_checkpoint_") as tmp:
            checkpoint = Path(tmp) / "model.pt"
            adapter.save(checkpoint)
            loaded = ScMRDRAdapter.load(checkpoint)
            for key, value in loaded.encode(x, "rna", self.features).items():
                np.testing.assert_allclose(value, encoded[key], rtol=1e-5, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "Feature identity/order"):
            adapter.encode(x, "rna", list(reversed(self.features)))

    def test_raw_count_and_mask_contract(self):
        adapter = self.adapter()
        with self.assertRaisesRegex(ValueError, "raw counts"):
            adapter.encode(np.full((3, 8), .3), "rna", self.features)
        self.masks["rna"][0] = False
        adapter = self.adapter()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            adapter.encode(np.ones((3, 8)), "rna", self.features)

    def test_end_to_end_cpu_fit_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="scmrdr_fit_") as tmp:
            adapter = fit(self.counts, self.masks, self.features, tmp, self.config)
            self.assertEqual(adapter.summary["epochs_completed"], 2)
            self.assertEqual(adapter.summary["n_features"], 8)
            self.assertEqual(adapter.encode(self.counts["rna"], "rna", self.features)["full"].shape, (48, 6))
            loaded = ScMRDRAdapter.load(Path(tmp) / "checkpoint.pt")
            np.testing.assert_allclose(adapter.encode(self.counts["rna"], "rna", self.features)["shared"],
                                       loaded.encode(self.counts["rna"], "rna", self.features)["shared"])
            with self.assertRaises(FileExistsError):
                fit(self.counts, self.masks, self.features, tmp, self.config)


if __name__ == "__main__":
    unittest.main()
