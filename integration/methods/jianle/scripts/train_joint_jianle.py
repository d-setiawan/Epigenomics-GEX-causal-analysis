#!/usr/bin/env python3
"""Train a joint Jianle-track model with scMRDR-like objective terms.

Objective (paper-aligned terms):
- ZINB reconstruction loss
- beta-VAE KL with modality-specific prior for z_s
- adversarial alignment on shared latent z_u
- isometric structure preservation from z=(z_u,z_s) to z_u
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModalityData:
    key: str
    mark: str
    adata: ad.AnnData
    counts: sp.csr_matrix
    feature_mask: np.ndarray | None
    obs_names: pd.Index
    n_obs: int
    n_vars: int
    dropped_zero_count_rows: int


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def build_embedding_df(mat: np.ndarray, index: pd.Index, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_{i+1}" for i in range(mat.shape[1])]
    return pd.DataFrame(mat, index=index, columns=cols)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_normal_std(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL(q || N(0, I))
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def kl_normal_general(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    # KL(q || p) for diagonal Gaussians
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8) - 1.0
    )
    return kl.mean()


def isometric_preserve_loss(mu_u: torch.Tensor, mu_full: torch.Tensor) -> torch.Tensor:
    # L_preserve = ||D(mu_u) - D(mu_full)||^2
    if mu_u.shape[0] < 3:
        return torch.zeros((), dtype=mu_u.dtype, device=mu_u.device)
    du = torch.cdist(mu_u, mu_u, p=2)
    df = torch.cdist(mu_full, mu_full, p=2)
    return ((du - df) ** 2).mean()


def zinb_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked ZINB NLL averaged over cells.

    x, mu, theta, pi_logits: shape (B, G) except theta may be (G,).
    mask: binary feature-availability mask with shape (B, G) or (G,)
    """
    eps = 1e-8
    theta = theta.unsqueeze(0) if theta.dim() == 1 else theta

    # NB log-likelihood
    log_nb = (
        torch.lgamma(x + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    )

    log_pi = -F.softplus(-pi_logits)
    log_1m_pi = -F.softplus(pi_logits)

    zero_case = torch.logaddexp(log_pi, log_1m_pi + log_nb)
    nonzero_case = log_1m_pi + log_nb
    log_prob = torch.where(x < 1e-8, zero_case, nonzero_case)

    if mask is None:
        nll = -log_prob.mean(dim=1)
    else:
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=log_prob.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        nll = -(log_prob * mask).sum(dim=1) / denom
    return nll.mean()


def sample_indices(n: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= batch_size:
        return np.arange(n)
    return rng.integers(0, n, size=batch_size)


def extract_batch(
    m: ModalityData,
    idx: np.ndarray,
    norm_target_sum: float,
    use_log_input: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = m.counts[idx]
    if sp.issparse(sub):
        x_counts = sub.toarray().astype(np.float32)
    else:
        x_counts = np.asarray(sub, dtype=np.float32)

    lib = x_counts.sum(axis=1, keepdims=True).astype(np.float32)
    lib = np.where(lib <= 0.0, 1.0, lib)

    if use_log_input:
        x_norm = np.log1p(x_counts * (norm_target_sum / lib))
    else:
        x_norm = x_counts * (norm_target_sum / lib)

    if m.feature_mask is not None:
        feature_mask = np.asarray(m.feature_mask, dtype=np.float32).reshape(1, -1)
    else:
        feature_mask = np.ones((1, x_counts.shape[1]), dtype=np.float32)
    return x_norm, x_counts, feature_mask


class SharedEncoder(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int,
        hidden_depth: int,
        dropout: float,
        shared_latent_dim: int,
        specific_latent_dim: int,
        modality_keys: list[str],
    ):
        super().__init__()
        self.modality_keys = modality_keys
        self.input_adapters = nn.ModuleDict(
            {k: nn.Linear(d, hidden_dim) for k, d in modality_dims.items()}
        )
        self.modality_embed = nn.Embedding(len(modality_keys), hidden_dim)

        trunk = []
        for _ in range(max(1, hidden_depth)):
            trunk += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*trunk)

        self.mu_u = nn.Linear(hidden_dim, shared_latent_dim)
        self.logvar_u = nn.Linear(hidden_dim, shared_latent_dim)
        self.mu_s = nn.Linear(hidden_dim, specific_latent_dim)
        self.logvar_s = nn.Linear(hidden_dim, specific_latent_dim)

    def forward(self, key: str, x: torch.Tensor, modality_idx: int):
        h = F.relu(self.input_adapters[key](x)) + self.modality_embed.weight[modality_idx].unsqueeze(0)
        h = self.trunk(h)
        return self.mu_u(h), self.logvar_u(h), self.mu_s(h), self.logvar_s(h)


class SharedDecoder(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int,
        hidden_depth: int,
        dropout: float,
        shared_latent_dim: int,
        specific_latent_dim: int,
        modality_keys: list[str],
    ):
        super().__init__()
        self.modality_keys = modality_keys

        in_dim = shared_latent_dim + specific_latent_dim
        trunk = []
        for _ in range(max(1, hidden_depth)):
            trunk += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk)

        self.rho_heads = nn.ModuleDict({k: nn.Linear(hidden_dim, d) for k, d in modality_dims.items()})
        self.pi_heads = nn.ModuleDict({k: nn.Linear(hidden_dim, d) for k, d in modality_dims.items()})

        # Per-modality dispersion parameter (log theta_g)
        self.log_theta = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(d, dtype=torch.float32)) for k, d in modality_dims.items()}
        )

    def forward(self, key: str, z_u: torch.Tensor, z_s: torch.Tensor):
        z = torch.cat([z_u, z_s], dim=1)
        h = self.trunk(z)

        rho_logits = self.rho_heads[key](h)
        pi_logits = self.pi_heads[key](h)
        theta = F.softplus(self.log_theta[key]) + 1e-4
        return rho_logits, pi_logits, theta


class ModalityDiscriminator(nn.Module):
    def __init__(self, shared_latent_dim: int, hidden_dim: int, n_modalities: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_modalities),
        )

    def forward(self, z_u: torch.Tensor) -> torch.Tensor:
        return self.net(z_u)


class JianleParityModel(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        modality_keys: list[str],
        hidden_dim: int,
        hidden_depth: int,
        dropout: float,
        shared_latent_dim: int,
        specific_latent_dim: int,
    ):
        super().__init__()
        self.modality_keys = modality_keys
        self.key_to_idx = {k: i for i, k in enumerate(modality_keys)}

        self.encoder = SharedEncoder(
            modality_dims,
            hidden_dim,
            hidden_depth,
            dropout,
            shared_latent_dim,
            specific_latent_dim,
            modality_keys,
        )
        self.decoder = SharedDecoder(
            modality_dims,
            hidden_dim,
            hidden_depth,
            dropout,
            shared_latent_dim,
            specific_latent_dim,
            modality_keys,
        )
        self.discriminator = ModalityDiscriminator(
            shared_latent_dim,
            hidden_dim,
            len(modality_keys),
            dropout,
        )

        # Modality-specific prior parameters for z_s: p(z_s|m)=N(mu_m, sigma_m^2 I)
        self.prior_mu_s = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(specific_latent_dim, dtype=torch.float32)) for k in modality_keys}
        )
        self.prior_logvar_s = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(specific_latent_dim, dtype=torch.float32)) for k in modality_keys}
        )

    def encode(self, key: str, x: torch.Tensor):
        idx = self.key_to_idx[key]
        return self.encoder(key, x, idx)

    def decode(self, key: str, z_u: torch.Tensor, z_s: torch.Tensor):
        return self.decoder(key, z_u, z_s)


def load_modalities(
    preprocess_dir: Path,
    rna_layer: str,
    chrom_layer: str,
    use_highly_variable: bool,
) -> Dict[str, ModalityData]:
    rna_path = preprocess_dir / "rna_preprocessed.h5ad"
    chrom_manifest_path = preprocess_dir / "chrom_preprocessed_manifest.tsv"
    if not rna_path.exists() or not chrom_manifest_path.exists():
        raise FileNotFoundError("Missing required preprocess inputs for joint Jianle training")

    out: Dict[str, ModalityData] = {}

    def _prepare_counts(
        adata: ad.AnnData,
        layer_name: str,
    ) -> Tuple[sp.csr_matrix, int, ad.AnnData, np.ndarray | None]:
        x = adata.layers[layer_name] if layer_name in adata.layers else adata.X
        if not sp.issparse(x):
            x = sp.csr_matrix(np.asarray(x, dtype=np.float32))
        else:
            x = x.tocsr().astype(np.float32)

        feature_mask = None
        if "feature_available" in adata.var:
            feature_mask = adata.var["feature_available"].to_numpy(dtype=np.float32)

        shared_feature_universe = bool(adata.uns.get("shared_feature_universe", False))
        if use_highly_variable and (not shared_feature_universe) and "highly_variable" in adata.var:
            hv = adata.var["highly_variable"].to_numpy(dtype=bool)
            if hv.sum() > 0:
                adata = adata[:, hv].copy()
                x = x[:, hv]
                if feature_mask is not None:
                    feature_mask = feature_mask[hv]

        row_sums = np.asarray(x.sum(axis=1)).ravel()
        keep = row_sums > 0
        dropped = int((~keep).sum())
        if dropped:
            adata = adata[keep].copy()
            x = x[keep]

        return x.tocsr(), dropped, adata, feature_mask

    rna = ad.read_h5ad(rna_path)
    x, dropped, rna, feature_mask = _prepare_counts(rna, rna_layer)
    if rna.n_obs < 2:
        raise RuntimeError("Too few RNA cells after filtering zero-count rows")
    out["rna"] = ModalityData(
        key="rna",
        mark="RNA",
        adata=rna,
        counts=x,
        feature_mask=feature_mask,
        obs_names=rna.obs_names.copy(),
        n_obs=int(rna.n_obs),
        n_vars=int(rna.n_vars),
        dropped_zero_count_rows=dropped,
    )

    chrom_manifest = pd.read_csv(chrom_manifest_path, sep="\t")
    for row in chrom_manifest.to_dict(orient="records"):
        mark = str(row["mark"])
        key = str(row.get("modality_key", f"chrom_{mark}"))
        cpath = Path(str(row["chrom_h5ad"]))
        if not cpath.is_absolute():
            cpath = (preprocess_dir / cpath).resolve() if (preprocess_dir / cpath).exists() else cpath.resolve()

        adata = ad.read_h5ad(cpath)
        x, dropped, adata, feature_mask = _prepare_counts(adata, chrom_layer)
        if adata.n_obs < 2:
            raise RuntimeError(f"Too few cells in {key} after filtering zero-count rows")
        out[key] = ModalityData(
            key=key,
            mark=mark,
            adata=adata,
            counts=x,
            feature_mask=feature_mask,
            obs_names=adata.obs_names.copy(),
            n_obs=int(adata.n_obs),
            n_vars=int(adata.n_vars),
            dropped_zero_count_rows=dropped,
        )

    return out


def train(
    mod_data: Dict[str, ModalityData],
    args: argparse.Namespace,
    device: torch.device,
):
    modality_keys = list(mod_data.keys())
    modality_dims = {k: int(mod_data[k].n_vars) for k in modality_keys}

    model = JianleParityModel(
        modality_dims=modality_dims,
        modality_keys=modality_keys,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        dropout=args.dropout,
        shared_latent_dim=args.shared_latent_dim,
        specific_latent_dim=args.specific_latent_dim,
    ).to(device)

    vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.prior_mu_s.parameters()) + list(model.prior_logvar_s.parameters())
    opt_vae = torch.optim.Adam(vae_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.discriminator_lr, weight_decay=args.weight_decay)

    if args.steps_per_epoch is not None:
        steps_per_epoch = max(1, int(args.steps_per_epoch))
    else:
        max_n = max(int(v.n_obs) for v in mod_data.values())
        steps_per_epoch = max(1, math.ceil(max_n / max(1, args.batch_size)))

    rng = np.random.default_rng(args.random_seed)
    history = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        adv_coeff = args.lam_alignment if epoch > args.adv_warmup_epochs else 0.0

        meters = {
            "loss_total": 0.0,
            "loss_recon": 0.0,
            "loss_kl": 0.0,
            "loss_align": 0.0,
            "loss_preserve": 0.0,
            "loss_disc": 0.0,
            "adv_coeff": adv_coeff,
        }

        for _ in range(steps_per_epoch):
            batches = {}
            for key in modality_keys:
                idx = sample_indices(mod_data[key].n_obs, args.batch_size, rng)
                x_in_np, x_counts_np, mask_np = extract_batch(
                    mod_data[key], idx, args.norm_target_sum, args.use_log_input
                )
                batches[key] = {
                    "x_in": torch.from_numpy(x_in_np).to(device),
                    "x_counts": torch.from_numpy(x_counts_np).to(device),
                    "mask": torch.from_numpy(mask_np).to(device),
                    "lib": torch.from_numpy(x_counts_np.sum(axis=1, keepdims=True)).to(device),
                    "mod_idx": model.key_to_idx[key],
                }

            # 1) update discriminator
            opt_d.zero_grad(set_to_none=True)
            disc_losses = []
            for key in modality_keys:
                x_in = batches[key]["x_in"]
                mu_u, logvar_u, _, _ = model.encode(key, x_in)
                z_u = reparameterize(mu_u, logvar_u).detach()
                logits = model.discriminator(z_u)
                labels = torch.full(
                    (z_u.shape[0],),
                    batches[key]["mod_idx"],
                    dtype=torch.long,
                    device=device,
                )
                disc_losses.append(F.cross_entropy(logits, labels))
            loss_disc = torch.stack(disc_losses).mean()
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), args.grad_clip)
            opt_d.step()

            # 2) update VAE (paper Eq.12)
            opt_vae.zero_grad(set_to_none=True)

            recon_losses = []
            kl_losses = []
            align_losses = []
            preserve_losses = []

            for key in modality_keys:
                x_in = batches[key]["x_in"]
                x_counts = batches[key]["x_counts"]
                mask = batches[key]["mask"]
                lib = batches[key]["lib"].clamp_min(1.0)

                mu_u, logvar_u, mu_s, logvar_s = model.encode(key, x_in)
                z_u = reparameterize(mu_u, logvar_u)
                z_s = reparameterize(mu_s, logvar_s)

                rho_logits, pi_logits, theta = model.decode(key, z_u, z_s)

                rho = F.softmax(rho_logits, dim=1)
                mu_nb = lib * rho

                recon = zinb_nll(
                    x_counts,
                    mu=mu_nb,
                    theta=theta,
                    pi_logits=pi_logits,
                    mask=mask,
                )

                kl_u = kl_normal_std(mu_u, logvar_u)
                prior_mu = model.prior_mu_s[key].unsqueeze(0)
                prior_logvar = model.prior_logvar_s[key].unsqueeze(0)
                kl_s = kl_normal_general(mu_s, logvar_s, prior_mu, prior_logvar)
                kl = kl_u + kl_s

                logits = model.discriminator(z_u)
                labels = torch.full(
                    (z_u.shape[0],),
                    batches[key]["mod_idx"],
                    dtype=torch.long,
                    device=device,
                )
                # q minimizes discriminator confidence in true modality
                align = -F.cross_entropy(logits, labels)

                mu_full = torch.cat([mu_u, mu_s], dim=1)
                preserve = isometric_preserve_loss(mu_u, mu_full)

                recon_losses.append(recon)
                kl_losses.append(kl)
                align_losses.append(align)
                preserve_losses.append(preserve)

            loss_recon = torch.stack(recon_losses).mean()
            loss_kl = torch.stack(kl_losses).mean()
            loss_align = torch.stack(align_losses).mean()
            loss_preserve = torch.stack(preserve_losses).mean()

            loss_total = (
                loss_recon
                + args.beta * loss_kl
                + adv_coeff * loss_align
                + args.lam_preserve * loss_preserve
            )

            if not torch.isfinite(loss_total):
                raise RuntimeError(
                    "Non-finite Jianle training loss. "
                    "Try lower learning rate, lower batch size, or enable --cpu-only."
                )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(vae_params, args.grad_clip)
            opt_vae.step()

            meters["loss_total"] += float(loss_total.item())
            meters["loss_recon"] += float(loss_recon.item())
            meters["loss_kl"] += float(loss_kl.item())
            meters["loss_align"] += float(loss_align.item())
            meters["loss_preserve"] += float(loss_preserve.item())
            meters["loss_disc"] += float(loss_disc.item())

        for k in ["loss_total", "loss_recon", "loss_kl", "loss_align", "loss_preserve", "loss_disc"]:
            meters[k] /= steps_per_epoch
        meters["epoch"] = epoch
        history.append(meters)

        if epoch == 1 or epoch % max(1, args.log_every) == 0 or epoch == args.max_epochs:
            print(
                "[JianleParity][Epoch {epoch}] total={loss_total:.4f} recon={loss_recon:.4f} "
                "kl={loss_kl:.4f} align={loss_align:.4f} preserve={loss_preserve:.4f} disc={loss_disc:.4f}"
                .format(**meters)
            )

    return model, history, modality_keys


def encode_shared_mu(
    model: JianleParityModel,
    modality_keys: list[str],
    mod_data: Dict[str, ModalityData],
    batch_size: int,
    norm_target_sum: float,
    use_log_input: bool,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    out = {}
    with torch.no_grad():
        for key in modality_keys:
            m = mod_data[key]
            embs = []
            for i in range(0, m.n_obs, batch_size):
                idx = np.arange(i, min(i + batch_size, m.n_obs))
                x_in_np, _, _ = extract_batch(m, idx, norm_target_sum, use_log_input)
                x_in = torch.from_numpy(x_in_np).to(device)
                mu_u, _, _, _ = model.encode(key, x_in)
                embs.append(mu_u.detach().cpu().numpy())
            out[key] = np.vstack(embs).astype(np.float32)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Train joint Jianle-track model (parity-upgraded objective)")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--preprocess-dir", required=True, help="Joint preprocess directory")
    p.add_argument("--out-dir", default=None, help="Default: <preprocess-parent>/train")

    p.add_argument("--rna-layer", default="counts")
    p.add_argument("--chrom-layer", default="counts")
    p.add_argument("--no-use-highly-variable", action="store_true")

    p.add_argument("--shared-latent-dim", type=int, default=20)
    p.add_argument("--specific-latent-dim", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--hidden-depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--lam-alignment", type=float, default=1.0)
    p.add_argument("--lam-preserve", type=float, default=1.0)
    p.add_argument("--adv-warmup-epochs", type=int, default=5)

    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--discriminator-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument("--norm-target-sum", type=float, default=1e4)
    p.add_argument("--use-log-input", action="store_true", default=True)
    p.add_argument("--no-use-log-input", action="store_false", dest="use_log_input")

    p.add_argument("--cpu-only", action="store_true", default=False)
    p.add_argument("--no-cpu-only", action="store_false", dest="cpu_only")
    p.add_argument("--encode-batch-size", type=int, default=512)
    p.add_argument("--random-seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.random_seed)

    repo_root = Path(args.repo_root).resolve()
    preprocess_dir = resolve_path(repo_root, args.preprocess_dir)
    out_dir = resolve_path(repo_root, args.out_dir) if args.out_dir else preprocess_dir.parent / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    modalities_dir = out_dir / "modalities"
    modalities_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Runtime: {'CPU-only' if device.type == 'cpu' else 'GPU-preferred'} (device={device})")

    mod_data = load_modalities(
        preprocess_dir,
        rna_layer=args.rna_layer,
        chrom_layer=args.chrom_layer,
        use_highly_variable=(not args.no_use_highly_variable),
    )

    model, history, modality_keys = train(mod_data, args, device)
    emb_dict = encode_shared_mu(
        model,
        modality_keys,
        mod_data,
        batch_size=args.encode_batch_size,
        norm_target_sum=args.norm_target_sum,
        use_log_input=args.use_log_input,
        device=device,
    )

    history_tsv = out_dir / "train_history.tsv"
    pd.DataFrame(history).to_csv(history_tsv, sep="\t", index=False)

    model_path = out_dir / "jianle_joint_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "modality_keys": modality_keys,
            "modality_to_mark": {k: mod_data[k].mark for k in modality_keys},
            "input_dims": {k: int(mod_data[k].n_vars) for k in modality_keys},
            "shared_latent_dim": int(args.shared_latent_dim),
            "specific_latent_dim": int(args.specific_latent_dim),
            "params": vars(args),
        },
        model_path,
    )

    modality_rows = []
    all_emb_frames = []
    for key in modality_keys:
        adata = mod_data[key].adata.copy()
        emb = emb_dict[key]
        adata.obsm["X_jianle"] = emb

        emb_df = build_embedding_df(emb, mod_data[key].obs_names, "JIANLE")
        emb_tsv = modalities_dir / f"{key}_jianle_embeddings.tsv"
        emb_df.to_csv(emb_tsv, sep="\t", index_label="cell")

        out_h5ad = modalities_dir / f"{key}_with_jianle.h5ad"
        adata.write_h5ad(out_h5ad, compression="gzip")

        frame = emb_df.copy()
        frame.insert(0, "cell", frame.index.astype(str))
        frame.insert(1, "modality_key", key)
        frame.insert(2, "mark", mod_data[key].mark)
        all_emb_frames.append(frame.reset_index(drop=True))

        modality_rows.append(
            {
                "modality_key": key,
                "mark": mod_data[key].mark,
                "h5ad": str(out_h5ad),
                "embeddings_tsv": str(emb_tsv),
                "n_cells": int(adata.n_obs),
                "n_features": int(adata.n_vars),
                "dropped_zero_count_rows": int(mod_data[key].dropped_zero_count_rows),
            }
        )

    all_cells_df = pd.concat(all_emb_frames, axis=0, ignore_index=True)
    all_cells_tsv = out_dir / "all_cells_jianle_embeddings.tsv"
    all_cells_df.to_csv(all_cells_tsv, sep="\t", index=False)

    modality_manifest_tsv = out_dir / "modality_outputs.tsv"
    write_tsv(
        modality_manifest_tsv,
        modality_rows,
        [
            "modality_key",
            "mark",
            "h5ad",
            "embeddings_tsv",
            "n_cells",
            "n_features",
            "dropped_zero_count_rows",
        ],
    )

    summary = {
        "run_type": "joint",
        "inputs": {
            "preprocess_dir": str(preprocess_dir),
        },
        "modalities": modality_rows,
        "params": vars(args),
        "runtime": {
            "device": str(device),
            "n_modalities": len(modality_keys),
            "max_epochs": int(args.max_epochs),
            "objective": "L_recon(ZINB) + beta*L_KL + lambda*L_alignment + gamma*L_preserve",
        },
        "outputs": {
            "model_pt": str(model_path),
            "all_cells_jianle_embeddings_tsv": str(all_cells_tsv),
            "modality_outputs_tsv": str(modality_manifest_tsv),
            "train_history_tsv": str(history_tsv),
        },
    }
    summary_path = out_dir / "train_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {model_path}")
    print(f"Wrote: {all_cells_tsv}")
    print(f"Wrote: {modality_manifest_tsv}")
    print(f"Wrote: {history_tsv}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
