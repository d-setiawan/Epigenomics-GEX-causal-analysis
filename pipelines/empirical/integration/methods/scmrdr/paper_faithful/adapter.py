"""Pinned scMRDR core with paper-consistent, leakage-safe training.

This adapter is GPL-3.0, as is the authors' computational core that it uses.
See README.md for the precise paper/upstream differences and their resolutions.
The input is RAW, feature-aligned count data, not total-normalized/logged values.
No AnnData metadata, paired IDs, cell types, or simulation truth are consumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
import torch
from torch import nn

from .upstream.model import EmbeddingNet

UPSTREAM_COMMIT = "7280e64ae19f7391c6f5adf00d9d353f79e96603"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class SCmRDRConfig:
    """Paper Appendix A.1 / three-modality weights, with disclosed source choices.

    Hidden widths and dropout are taken from the authors' BMMC script because
    the supplied paper does not specify them. Change configurations before
    assessment, not in response to structure-recovery scores.
    """

    hidden_layers: tuple[int, ...] = (128, 128)
    latent_dim_shared: int = 20
    latent_dim_specific: int = 20
    dropout_rate: float = 0.2
    beta: float = 5.0
    gamma: float = 10.0
    lambda_adv: float = 10.0
    dispersion_strategy: str = "gene"
    max_epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    cosine_learning_rate: bool = True
    validation_fraction: float = 0.1
    patience: int = 10
    min_delta: float = 0.0
    validation_mc_samples: int = 1
    gradient_clip_norm: float = 1.0
    seed: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        if not self.hidden_layers or min(self.hidden_layers) < 1:
            raise ValueError("hidden_layers must contain positive widths")
        if min(self.latent_dim_shared, self.latent_dim_specific) < 1:
            raise ValueError("Both latent dimensions must be positive")
        if self.max_epochs < 1 or self.batch_size < 2 or self.patience < 1:
            raise ValueError("Invalid epochs, batch size, or patience")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not 0 <= self.dropout_rate < 1 or self.validation_mc_samples < 1:
            raise ValueError("Invalid dropout or validation sample count")
        if self.dispersion_strategy not in ("gene", "gene-modality", "gene-cell"):
            raise ValueError("Unsupported upstream dispersion strategy")
        if min(self.learning_rate, self.gradient_clip_norm) <= 0:
            raise ValueError("Learning rate and clipping norm must be positive")
        if min(self.beta, self.gamma, self.lambda_adv, self.min_delta) < 0:
            raise ValueError("Loss weights and min_delta must be nonnegative")


def _counts(matrix, width: int) -> sparse.csr_matrix:
    x = sparse.csr_matrix(matrix, dtype=np.float32, copy=True)
    x.sum_duplicates()
    x.sort_indices()
    x.eliminate_zeros()
    if x.ndim != 2 or x.shape[1] != width or x.shape[0] < 1:
        raise ValueError(f"Expected a nonempty count matrix with {width} features")
    if not np.isfinite(x.data).all() or (x.data < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if not np.allclose(x.data, np.rint(x.data), rtol=0, atol=1e-5):
        raise ValueError("ZINB requires raw counts; normalized/logged input rejected")
    return x


def _matrix_digest(x: sparse.csr_matrix) -> str:
    h = hashlib.sha256(str(x.shape).encode())
    for values in (x.indptr, x.indices, x.data):
        h.update(values.tobytes())
    return h.hexdigest()


def source_provenance() -> dict:
    root = Path(__file__).parent
    return {
        "repository": "https://github.com/sjl-sjtu/scMRDR",
        "commit": UPSTREAM_COMMIT,
        "license": "GPL-3.0",
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vendored_sha256": {
            name: hashlib.sha256((root / "upstream" / name).read_bytes()).hexdigest()
            for name in ("model.py", "loss.py", "LICENSE")
        },
        "local_core_change": "Removed unused NumPy/matplotlib/SciPy/POT imports from model.py only",
    }


class _SparseDataset:
    """Keep the feature universe sparse until a minibatch is requested."""

    def __init__(self, matrices, modalities, covariates, names):
        self.counts = sparse.vstack([matrices[m] for m in names], format="csr")
        self.labels = np.concatenate([
            np.full(matrices[m].shape[0], modalities.index(m), dtype=np.int64)
            for m in names
        ])
        self.covariates = np.concatenate([covariates[m] for m in names], axis=0)

    def __len__(self):
        return self.counts.shape[0]

    def batch(self, indices, device, n_modalities):
        x = torch.as_tensor(self.counts[indices].toarray(), device=device)
        labels = torch.as_tensor(self.labels[indices], device=device)
        m = nn.functional.one_hot(labels, n_modalities).to(x.dtype)
        b = torch.as_tensor(self.covariates[indices], device=device)
        # No cell-type or truth supervision; mask group is the modality itself.
        w = torch.zeros((len(indices), 0), dtype=x.dtype, device=device)
        return x, b, m, m, w


def _set_discriminator_grad(model, enabled: bool) -> None:
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(enabled)


def training_step(model, batch, optimizer_vae, optimizer_d, clip_norm=1.0):
    """One true alternating min/max step, without stale discriminator gradients."""
    vae_parameters = [p for name, p in model.named_parameters()
                      if not name.startswith("discriminator.")]
    _set_discriminator_grad(model, True)
    model.eval()
    model.discriminator.train()
    optimizer_d.zero_grad(set_to_none=True)
    optimizer_vae.zero_grad(set_to_none=True)
    discriminator_loss = model(*batch, stage="discriminator")
    if not torch.isfinite(discriminator_loss):
        raise FloatingPointError("Nonfinite discriminator objective")
    discriminator_loss.backward()
    nn.utils.clip_grad_norm_(model.discriminator.parameters(), clip_norm,
                             error_if_nonfinite=True)
    optimizer_d.step()
    optimizer_d.zero_grad(set_to_none=True)

    model.train()
    model.discriminator.eval()
    _set_discriminator_grad(model, False)
    try:
        _, _, loss, parts = model(*batch, stage="vae")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite scMRDR objective")
        loss.backward()
        nn.utils.clip_grad_norm_(vae_parameters, clip_norm, error_if_nonfinite=True)
        optimizer_vae.step()
    finally:
        _set_discriminator_grad(model, True)
    optimizer_vae.zero_grad(set_to_none=True)
    parts["discriminator_loss"] = float(discriminator_loss.detach())
    return parts


def validate_model(model, dataset, batch_size, seed, mc_samples=1):
    """Cell-weighted mean across ALL validation batches, with fixed MC draws.

    Like upstream, geometry is estimated within minibatches. A fixed seeded
    permutation mixes modalities instead of validating one modality block at a
    time. Fixed batches and MC draws make epochs comparable without consuming
    training RNG state. No validation row updates weights or BatchNorm stats.
    """
    model.eval()
    aggregate = {}
    device = next(model.parameters()).device
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    order = np.random.default_rng(seed).permutation(len(dataset))
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for _ in range(mc_samples):
            for start in range(0, len(dataset), batch_size):
                idx = order[start:min(start + batch_size, len(dataset))]
                batch = dataset.batch(idx, device, model.modality_num)
                _, _, loss, parts = model(*batch, stage="vae")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation objective")
                for key, value in parts.items():
                    aggregate[key] = aggregate.get(key, 0.0) + float(value) * len(idx)
    return {key: value / (len(dataset) * mc_samples)
            for key, value in aggregate.items()}


class ScMRDRAdapter:
    """Fit the pinned architecture and encode unseen rows without refitting."""

    def __init__(self, feature_names: Sequence[str], modalities: Sequence[str],
                 availability_by_modality: Mapping[str, np.ndarray],
                 config: SCmRDRConfig | None = None, covariate_dim: int = 0):
        self.config = config or SCmRDRConfig()
        self.config.validate()
        self.feature_names = tuple(map(str, feature_names))
        self.modalities = tuple(map(str, modalities))
        if len(set(self.feature_names)) != len(self.feature_names) or not self.feature_names:
            raise ValueError("Feature names must be nonempty and unique")
        if len(set(self.modalities)) != len(self.modalities) or len(self.modalities) < 2:
            raise ValueError("At least two unique modality names are required")
        if set(availability_by_modality) != set(self.modalities):
            raise ValueError("Availability masks must cover exactly the modalities")
        self.availability = {}
        for name in self.modalities:
            mask = np.asarray(availability_by_modality[name])
            if mask.shape != (len(self.feature_names),) or not np.isin(mask, [0, 1]).all() or not mask.any():
                raise ValueError(f"Invalid nonempty binary feature mask for {name}")
            self.availability[name] = mask.astype(np.float32)
        if covariate_dim < 0:
            raise ValueError("covariate_dim must be nonnegative")
        self.covariate_dim = covariate_dim
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        c = self.config
        masks = torch.as_tensor(np.stack([self.availability[m] for m in self.modalities]))
        self.model = EmbeddingNet(
            self.device, len(self.feature_names), len(self.modalities),
            covariate_dim=covariate_dim, celltype_num=0,
            layer_dims=list(c.hidden_layers), latent_dim_shared=c.latent_dim_shared,
            latent_dim_specific=c.latent_dim_specific, dropout_rate=c.dropout_rate,
            beta=c.beta, gamma=c.gamma, lambda_adv=c.lambda_adv,
            feat_mask=masks, distribution="ZINB", encoder_covariates=False,
        ).to(self.device)
        # Upstream implements all three dispersion options, but its enclosing
        # forward hardcodes the decoder default. Select explicitly, without
        # changing the equations/network implementation.
        self.model.decoder.forward = partial(
            self.model.decoder.forward, dispersion_strategy=c.dispersion_strategy)
        self.history = []
        self.summary = {}

    def _validate_input(self, counts, modality, feature_names, covariates=None):
        if tuple(map(str, feature_names)) != self.feature_names:
            raise ValueError("Feature identity/order differs from the fitted feature manifest")
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}")
        x = _counts(counts, len(self.feature_names))
        unavailable = self.availability[modality] == 0
        if unavailable.any() and x[:, unavailable].nnz:
            raise ValueError("Nonzero counts supplied for structurally unavailable features")
        if covariates is None:
            if self.covariate_dim:
                raise ValueError("Fitted decoder covariates must be provided explicitly")
            b = np.zeros((x.shape[0], 0), dtype=np.float32)
        else:
            b = np.asarray(covariates, dtype=np.float32)
            if b.shape != (x.shape[0], self.covariate_dim) or not np.isfinite(b).all():
                raise ValueError("Invalid covariate shape/values")
        return x, b

    def fit(self, counts_by_modality, out_dir, validation_counts_by_modality=None,
            covariates_by_modality=None, validation_covariates_by_modality=None):
        """Fit once; refuse to overwrite an existing checkpoint/history.

        Explicit validation matrices are preferred for the extended benchmark.
        Otherwise a stratified, seeded fraction of the supplied observations is
        held out internally (the simple benchmark remains transductive overall).
        """
        if self.history:
            raise RuntimeError("This adapter was already fitted; create a new adapter")
        if set(counts_by_modality) != set(self.modalities):
            raise ValueError("Training data must cover exactly the fitted modalities")
        if validation_counts_by_modality is not None and set(validation_counts_by_modality) != set(self.modalities):
            raise ValueError("Validation must cover exactly the fitted modalities")
        out_dir = Path(out_dir)
        protected = ("checkpoint.pt", "last_checkpoint.pt", "history.json", "fit_summary.json")
        if any((out_dir / name).exists() for name in protected):
            raise FileExistsError(f"Refusing to overwrite an existing scMRDR fit: {out_dir}")
        train, valid, train_cov, valid_cov, split_info = {}, {}, {}, {}, {}
        rng = np.random.default_rng(self.config.seed)
        for name in self.modalities:
            cov = None if covariates_by_modality is None else covariates_by_modality[name]
            x, b = self._validate_input(counts_by_modality[name], name, self.feature_names, cov)
            if validation_counts_by_modality is None:
                n_valid = max(1, int(math.ceil(x.shape[0] * self.config.validation_fraction)))
                if self.config.validation_fraction == 0:
                    raise ValueError("Supply validation data or a positive validation_fraction")
                if x.shape[0] - n_valid < 2:
                    raise ValueError("Too few training rows after validation split")
                order = rng.permutation(x.shape[0])
                vi, ti = np.sort(order[:n_valid]), np.sort(order[n_valid:])
                train[name], valid[name] = x[ti], x[vi]
                train_cov[name], valid_cov[name] = b[ti], b[vi]
                split_info[name] = {"training_rows": ti.tolist(), "validation_rows": vi.tolist()}
            else:
                v_cov = None if validation_covariates_by_modality is None else validation_covariates_by_modality[name]
                v, vb = self._validate_input(validation_counts_by_modality[name], name, self.feature_names, v_cov)
                train[name], valid[name], train_cov[name], valid_cov[name] = x, v, b, vb
                split_info[name] = {"training_rows": "all", "validation_rows": "external"}
        train_data = _SparseDataset(train, self.modalities, train_cov, self.modalities)
        valid_data = _SparseDataset(valid, self.modalities, valid_cov, self.modalities)
        if len(train_data) < self.config.batch_size:
            raise ValueError("Training rows are fewer than batch_size; no implicit batch-size change")
        if train_data.counts.nnz == 0:
            raise ValueError("All training counts are zero")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "split_indices.json").write_text(json.dumps(split_info, indent=2) + "\n")
        c = self.config
        vae_parameters = [p for name, p in self.model.named_parameters()
                          if not name.startswith("discriminator.")]
        optimizer_vae = torch.optim.Adam(vae_parameters, lr=c.learning_rate)
        optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=c.learning_rate)
        batches = len(train_data) // c.batch_size
        schedulers = []
        if c.cosine_learning_rate:
            schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, c.max_epochs * batches)
                          for o in (optimizer_vae, optimizer_d)]
        best, best_epoch, stale = float("inf"), 0, 0
        started = time.monotonic()
        for epoch in range(1, c.max_epochs + 1):
            order = rng.permutation(len(train_data))
            aggregate = {}
            for start in range(0, batches * c.batch_size, c.batch_size):
                indices = order[start:start + c.batch_size]
                batch = train_data.batch(indices, self.device, len(self.modalities))
                parts = training_step(self.model, batch, optimizer_vae, optimizer_d, c.gradient_clip_norm)
                for key, value in parts.items():
                    aggregate[key] = aggregate.get(key, 0.0) + value / batches
                for scheduler in schedulers:
                    scheduler.step()
            validation = validate_model(self.model, valid_data, c.batch_size,
                                        c.seed + 100003, c.validation_mc_samples)
            score = validation["total_loss"]
            self.history.append({"epoch": epoch, "train": aggregate, "validation": validation,
                                 "learning_rate": optimizer_vae.param_groups[0]["lr"]})
            improved = score < best - c.min_delta
            if improved:
                best, best_epoch, stale = score, epoch, 0
                self.save(out_dir / "checkpoint.pt", {"epoch": epoch, "validation_loss": score})
            else:
                stale += 1
            (out_dir / "history.json").write_text(json.dumps(self.history, indent=2) + "\n")
            print(json.dumps({"method": "scmrdr_paper_faithful", "epoch": epoch,
                              "train_loss": aggregate["total_loss"], "validation_loss": score,
                              "best_epoch": best_epoch}), flush=True)
            if stale >= c.patience:
                break
        self.save(out_dir / "last_checkpoint.pt", {"epoch": epoch, "validation_loss": score})
        best_checkpoint = torch.load(out_dir / "checkpoint.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(best_checkpoint["model_state"])
        self.model.eval()
        self.summary = {
            "method": "scmrdr_paper_faithful", "format_version": FORMAT_VERSION,
            "config": asdict(c), "source": source_provenance(),
            "n_features": len(self.feature_names), "modalities": list(self.modalities),
            "training_rows": {m: train[m].shape[0] for m in self.modalities},
            "validation_rows": {m: valid[m].shape[0] for m in self.modalities},
            "zero_library_training_rows": {m: int((np.asarray(train[m].sum(axis=1)).ravel() == 0).sum()) for m in self.modalities},
            "training_digests": {m: _matrix_digest(train[m]) for m in self.modalities},
            "validation_digests": {m: _matrix_digest(valid[m]) for m in self.modalities},
            "epochs_completed": len(self.history), "best_epoch": best_epoch,
            "best_validation_loss": best, "early_stopped": stale >= c.patience,
            "runtime_seconds": time.monotonic() - started,
            "training_drop_last": True,
            "validation_order_seed": c.seed + 100003,
            "rows_per_epoch": batches * c.batch_size,
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "readout": "deterministic shared and specific posterior means",
        }
        (out_dir / "fit_summary.json").write_text(json.dumps(self.summary, indent=2) + "\n")
        return self

    def encode(self, counts, modality, feature_names, batch_size=512, covariates=None):
        """Return shared/private posterior means; no refitting or decoder inputs.

        This uses exactly the native count-encoder transform log1p(raw counts).
        Fitted BatchNorm buffers and feature order are held fixed on unseen cells.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        x, _ = self._validate_input(counts, modality, feature_names, covariates)
        self.model.eval()
        shared, specific = [], []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                values = torch.as_tensor(x[start:start + batch_size].toarray(), device=self.device)
                values = torch.log1p(values)
                labels = torch.full((values.shape[0],), self.modalities.index(modality), device=self.device)
                m = nn.functional.one_hot(labels, len(self.modalities)).to(values.dtype)
                # Call deterministic network components directly: native forward
                # additionally samples z, but returns the same posterior means.
                h_u = self.model.encoder_shared.zxy_encoder(values)
                h_s = self.model.encoder_specific.zxy_encoder(torch.cat([values, m], dim=1))
                shared.append(self.model.encoder_shared.mu_layer(h_u).cpu().numpy())
                specific.append(self.model.encoder_specific.mu_layer(h_s).cpu().numpy())
        result = {"shared": np.concatenate(shared), "specific": np.concatenate(specific)}
        result["full"] = np.concatenate([result["shared"], result["specific"]], axis=1)
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError("Nonfinite posterior means")
        return result

    def save(self, path, extra=None):
        torch.save({
            "format_version": FORMAT_VERSION, "config": asdict(self.config),
            "feature_names": self.feature_names, "modalities": self.modalities,
            "availability": self.availability, "covariate_dim": self.covariate_dim,
            "model_state": self.model.state_dict(), "source": source_provenance(),
            "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load only trusted local checkpoints (torch serialization uses pickle)."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint format")
        if checkpoint["source"]["commit"] != UPSTREAM_COMMIT:
            raise ValueError("Checkpoint upstream version does not match this adapter")
        config = dict(checkpoint["config"])
        config["device"] = device
        config["hidden_layers"] = tuple(config["hidden_layers"])
        result = cls(checkpoint["feature_names"], checkpoint["modalities"],
                     checkpoint["availability"], SCmRDRConfig(**config),
                     covariate_dim=checkpoint["covariate_dim"])
        result.model.load_state_dict(checkpoint["model_state"])
        result.model.eval()
        return result


def fit(counts_by_modality, availability_by_modality, feature_names, out_dir,
        config=None, validation_counts_by_modality=None, covariates_by_modality=None,
        validation_covariates_by_modality=None):
    """Convenience entry point used by the simulation integration runner."""
    modalities = tuple(sorted(counts_by_modality))
    covariate_dim = (0 if covariates_by_modality is None else
                     np.asarray(covariates_by_modality[modalities[0]]).shape[1])
    adapter = ScMRDRAdapter(feature_names, modalities, availability_by_modality,
                           config, covariate_dim=covariate_dim)
    return adapter.fit(counts_by_modality, out_dir, validation_counts_by_modality,
                       covariates_by_modality, validation_covariates_by_modality)
