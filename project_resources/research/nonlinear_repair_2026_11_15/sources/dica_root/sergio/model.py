import torch
import torch.nn as nn
from torch import vmap
from torch.func import jacfwd
import numpy as np

# Autoencoder with custom loss
class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
# Vol reg coefficient scheduler
def alpha(epoch, lam_vol, warmup):
    t_clamped = np.minimum(np.maximum(epoch, 0), warmup)
    return (lam_vol / warmup) * t_clamped

# J-VolMax criterion
def dica_loss(x_hat, x, z, model, epoch, rho, warmup, lam_vol, lam_norm):
    recon_loss = nn.functional.mse_loss(x_hat, x)
    jac = vmap(jacfwd(model.decoder))(z)
    vol = torch.logdet(jac.transpose(-1, -2) @ jac).mean()
    l1norm = jac.abs().sum(dim=(1, 2))
    alpha_coeff = alpha(epoch, lam_vol, warmup)
    if epoch <= warmup:
        return recon_loss - alpha_coeff * vol + lam_norm * l1norm.mean(), recon_loss.item(), vol.item(), l1norm.mean().item(), alpha_coeff, 0
    else:
        l1norm_reg = torch.nn.functional.softplus(l1norm - rho).mean()
        return recon_loss - alpha_coeff * vol + lam_norm * l1norm_reg, recon_loss.item(), vol.item(), l1norm.mean().item(), alpha_coeff, 0

# Vanilla loss
def base_loss(x_hat, x, z, model):
    recon_loss = nn.functional.mse_loss(x_hat, x)
    jac = vmap(jacfwd(model.decoder))(z)
    vol = torch.logdet(jac.transpose(-1, -2) @ jac).mean()
    l1norm = jac.abs().sum(dim=(1, 2))
    return recon_loss, recon_loss.item(), vol.item(), l1norm.mean().item()

# Jacobian L1-sparsity loss
def sparse_loss(x_hat, x, z, model, lam_sparse):
    recon_loss = nn.functional.mse_loss(x_hat, x)
    jac = vmap(jacfwd(model.decoder))(z)
    vol = torch.logdet(jac.transpose(-1, -2) @ jac).mean()
    l1norm = jac.abs().sum(dim=(1, 2))
    return recon_loss + lam_sparse * l1norm.mean(), recon_loss.item(), vol.item(), l1norm.mean().item()