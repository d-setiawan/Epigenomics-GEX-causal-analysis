import torch
import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.func import jacfwd
from torch import vmap, nn
import time

from data import get_data
from eval import eval_model

# Init weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)

# Autoencoder using MLPs
class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        sh = self.encoder(x)
        xh = self.decoder(sh)
        return sh, xh

# Schedulers for regularizer coefficients
def alpha(epoch, lam_vol, warmup):
    return np.minimum(lam_vol, lam_vol * epoch / warmup)

# Parse arguments
parser = argparse.ArgumentParser(description="DICA for Mixtures")
parser.add_argument("--cuda", type=int, default=0, help="CUDA device number")
parser.add_argument("--type", type=str, required=True, help="Type of model to train")
parser.add_argument("--exp", type=str, required=True, help="Experiment name")
parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension")
parser.add_argument("--input_dim", type=int, required=True, help="Input dimension")
args = parser.parse_args()

assert args.type in ["dica", "dica_trace", "sparse", "base", "ima"], "Invalid model type, must be one of ['dica', 'sparse', 'base']"
assert args.exp in ["a", "b", "c"], "Invalid experiment name, must be one of ['a', 'b', 'c']"

config = {
    "lam_vol": 1e-4,
    "lam_trace": 1e-6,
    "lam_norm": 1e-4,
    "lam_sparse": 1e-4,
    "lam_ima": 1e-4,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_iters": 200,
    "eval_iter": 10,
    "nobs": 30000,
    "latent_dim": args.latent_dim,
    "input_dim": args.input_dim,
    "exp": args.exp,
    "cuda": args.cuda,
    "warmup": 20,
    "type": args.type
}

# W&B init
wandb.init(project=f"dica_rebuttal_ima", config=config)

# Set device
device = torch.device(f"cuda:{config['cuda']}")

# Create model
model = MLPAutoencoder(config['input_dim'], config['latent_dim']).to(device)
model.encoder.apply(init_weights)
model.decoder.apply(init_weights)

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

# Load data
train_loader, val_loader = get_data(
    config['exp'], config['nobs'], config['input_dim'], config['latent_dim'], config['batch_size']
)

# Train loop
run_recon, run_vol, run_trace_vol, run_jacnorm, run_jacnorm_reg = 0, 0, 0, 0, 0
run_grad_time = 0.0 # <-- 1. Initialize accumulator
run_ima_contrast = 0.0
d = args.latent_dim

# Hyperparameters
estimated_rho = 0
rho_history = []
lam_norm = config['lam_norm']
lam_vol = config['lam_vol']
lam_sparse = config['lam_sparse']
lam_ima = config['lam_ima']
lam_trace = config['lam_trace']

for epoch in range(config['num_iters']):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x, _ = batch
        x = x.to(device)
        sh, xh = model(x)
        sh = sh.reshape(sh.shape[0], config['latent_dim'], -1)

        # Recon loss
        recon = (x - xh).square().mean()
        
        # Get Jacobian
        jac = vmap(jacfwd(model.decoder))(sh.flatten(1))

        # Volume reg
        vol = torch.logdet(jac.transpose(-1, -2) @ jac).mean()
        G = d * torch.eye(d, device=device) - torch.ones(d, d, device=device)
        trace_vol = torch.diagonal(G @ jac.transpose(-1, -2) @ jac, dim1=-2, dim2=-1).sum(-1).mean()

        # IMA
        ima_contrast = (-0.5 * torch.logdet(jac.transpose(-1, -2) @ jac) + torch.sum(torch.log(jac.norm(dim=1)), dim=1)).mean()

        # L1-norm reg
        jacnorm = jac.abs().sum(dim=(1, 2))
        jacnorm_reg = torch.nn.functional.softplus(jacnorm - estimated_rho).mean()
        if epoch <= config['warmup']:
            rho_history.append(jacnorm.mean().item())

        # Total loss
        if config['type'] == "ima":
            total_loss = recon + lam_ima * ima_contrast
        elif config['type'] == "dica":
            alpha_coeff = alpha(epoch, lam_vol, config['warmup'])
            if epoch <= config['warmup']:
                total_loss = recon - alpha_coeff * vol + lam_norm * jacnorm.mean()
            else:
                total_loss = recon - alpha_coeff * vol + lam_norm * jacnorm_reg
        elif config['type'] == "dica_trace":
            alpha_coeff = alpha(epoch, lam_trace, config['warmup'])
            if epoch <= config['warmup']:
                total_loss = recon - alpha_coeff * trace_vol + lam_norm * jacnorm.mean()
            else:
                total_loss = recon - alpha_coeff * trace_vol + lam_norm * jacnorm_reg
        elif config['type'] == "sparse":
            total_loss = recon + lam_sparse * jacnorm.mean()
        elif config['type'] == "base":
            total_loss = recon

        # Logging
        run_recon += recon.item()
        run_vol += vol
        run_trace_vol += trace_vol.item()
        run_jacnorm += jacnorm.mean().item()
        run_jacnorm_reg += jacnorm_reg.item()
        run_ima_contrast += ima_contrast.item()

        # Optimize
        optimizer.zero_grad()

        # Time the backprop
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        total_loss.backward()

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        run_grad_time += (end_time - start_time)

        optimizer.step()

    # Get rho
    if epoch == config['warmup'] and config['warmup'] >= 0:
        estimated_rho = np.mean(rho_history[-10:])
        wandb.config["rho"] = estimated_rho

    # Eval model
    if epoch % config['eval_iter'] == 0 or epoch == config['num_iters'] - 1:
        model.eval()
        val_recon, val_vol, val_trace_vol, val_jacnorm, val_jacnorm_reg, val_ima_contrast,\
            val_r2, val_r2_mat, val_mcc, val_mcc_mat = eval_model(model, val_loader, device, estimated_rho)

        train_recon = run_recon / len(train_loader)
        train_vol = run_vol / len(train_loader)
        train_jacnorm = run_jacnorm / len(train_loader)
        train_jacnorm_reg = run_jacnorm_reg / len(train_loader)
        train_trace_vol = run_trace_vol / len(train_loader)
        train_grad_time = run_grad_time / len(train_loader) # <-- 3. Calculate average
        train_ima_contrast = run_ima_contrast / len(train_loader)

        # Log to W&B
        wandb.log({
            "train/loss": total_loss.item(),
            "train/recon": train_recon,
            "train/vol": train_vol,
            "train/trace_vol": train_trace_vol,
            "train/ima_contrast": train_ima_contrast,
            "train/jacnorm": train_jacnorm_reg,
            "train/jacnorm_reg": jacnorm_reg,
            "train/epoch": epoch,
            "train/rho": estimated_rho,
            "train/grad_time_ms": train_grad_time * 1000, # <-- 3. Log to W&B in ms
            "val/recon": val_recon,
            "val/ima_contrast": val_ima_contrast,
            "val/vol": val_vol,
            "val/trace_vol": val_trace_vol,
            "val/jacnorm": val_jacnorm,
            "val/jacnorm_reg": val_jacnorm_reg,
            "val/r2": val_r2,
            "val/mcc": val_mcc,
            "val/epoch": epoch,
            "val/rho": estimated_rho
        })

        # Visualize results
        fig_mcc, ax_mcc = plt.subplots()
        sns.heatmap(val_mcc_mat.cpu().numpy(), annot=True, cmap="viridis",
                        xticklabels=False, yticklabels=False)
        ax_mcc.set_title("Absolute Correlation Matrix (TFs vs Latents)")
        ax_mcc.set_xlabel("Latent dimensions")
        ax_mcc.set_ylabel("True TFs")
        wandb.log({"MCC": wandb.Image(fig_mcc)})
        plt.close(fig_mcc)

        fig_r2, ax_r2 = plt.subplots()
        sns.heatmap(val_r2_mat.cpu().numpy(), annot=True, cmap="viridis",
                        xticklabels=False, yticklabels=False)
        ax_r2.set_title("R2 Matrix (TFs vs Latents)")
        ax_r2.set_xlabel("Latent dimensions")
        ax_r2.set_ylabel("True TFs")
        wandb.log({"R2": wandb.Image(fig_r2)})
        plt.close(fig_r2)

        # Reset logging losses
        run_recon, run_vol, run_jacnorm = 0.0, 0.0, 0.0
        run_grad_time = 0.0
        run_ima_contrast = 0.0

# W&B summary and finish
wandb.run.summary["final_r2"] = val_r2
wandb.run.summary["final_mcc"] = val_mcc

wandb.finish()
