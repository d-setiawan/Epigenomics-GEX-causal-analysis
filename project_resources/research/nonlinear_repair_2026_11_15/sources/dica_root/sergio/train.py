import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import argparse

from eval import mcc, r2
from model import MLPAutoencoder, base_loss, sparse_loss, dica_loss

# Init weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

# Parse arguments
parser = argparse.ArgumentParser(description="DICA for SERGIO")
parser.add_argument("--cuda", type=int, required=True, help="CUDA device ID")
parser.add_argument("--type", type=str, required=True, help="Criterion type: 'sparse', 'base', or 'dica'")
parser.add_argument("--save", type=str, default=None, help="Path to save the model")
args = parser.parse_args()

# Ensure arguments are valid
if args.type not in ['sparse', 'base', 'dica']:
    raise ValueError("Invalid type. Choose from 'sparse', 'base', or 'dica'.")

# Config
config = {
    "learning_rate": 1e-4,
    "num_epochs": 4000,
    "batch_size": 64,
    "lam_vol": 1e-3,
    "lam_norm": 1e-4,
    "lam_sparse": 1e-4,  
    "val_epoch": 10,
    "warmup": 1000,
    "hidden_dim": 64,
    "type": args.type,
}

# Load data
X = np.loadtxt(f"X.csv", delimiter=",")
true_tfs = np.loadtxt(f"true_tfs.csv", delimiter=",")
config["latent_dim"] = true_tfs.shape[1]
config["input_dim"] = X.shape[1]
config["number_sc"] = X.shape[0]

# Initialize wandb
wandb.init(project="dica_sergio", config=config)

# Get device
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# Get model and move to device 
model = MLPAutoencoder(
    input_dim=X.shape[1],
    latent_dim=true_tfs.shape[1],
    hidden_dim=config['hidden_dim']
)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
model = model.to(device)

# Initialize weights
model.apply(init_weights)

# Set model to train
model.train()

# True TFs as Tensor
Z = torch.tensor(true_tfs, dtype=torch.float32).to(device)

# Get data and move to device
X_tensor = torch.tensor(X, dtype=torch.float32)
X_tensor = X_tensor.to(device)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# Set up hyperparameters
estimated_rho = 0 # Means rho is not used
rho_history = []
lam_vol = config['lam_vol']
lam_sparse = config['lam_sparse']
lam_norm = config['lam_norm']
warmup = config['warmup']

# Training loop
for epoch in range(config['num_epochs']):
    total_loss = 0
    for (batch,) in loader:
        # Get data
        x_hat, z = model(batch)

        # Get loss
        if config['type'] == 'dica':
            loss, recon_loss, vol, jac_norm, alpha, beta = dica_loss(
                x_hat, batch, z, model, epoch, estimated_rho, warmup,
                lam_vol, lam_norm
            )
            if estimated_rho == 0:
                rho_history.append(jac_norm)
        elif config['type'] == 'base':
            loss, recon_loss, vol, jac_norm = base_loss(x_hat, batch, z, model)
        elif config['type'] == 'sparse':
            loss, recon_loss, vol, jac_norm = sparse_loss(x_hat, batch, z, model, lam_sparse)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        total_loss += loss.item()
    
    if config['type'] == 'dica' and epoch == config['warmup'] and config['warmup'] >= 0:
        estimated_rho = np.mean(rho_history[-10:])

    if epoch % config['val_epoch'] == 0 or epoch == config['num_epochs'] - 1:
        model.eval()
        with torch.no_grad():
            Zh = model.encoder(X_tensor)

        # MCC score
        mcc_mat, mcc_score = mcc(Z, Zh)

        # R2 score
        r2_mat, r2_score = r2(Z, Zh)

        # Logging
        wandb.log({
            "mcc": mcc_score,
            "r2": r2_score,
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "vol": vol,
            "jac_norm": jac_norm,
            "rho": estimated_rho,
            "epoch": epoch
        })

        if config['type'] == 'dica':
            wandb.log({"alpha": alpha, "beta": beta})
        
        # Visualize results
        fig_mcc, ax_mcc = plt.subplots()
        sns.heatmap(mcc_mat.cpu().numpy(), annot=True, cmap="viridis", xticklabels=False, yticklabels=False)
        ax_mcc.set_title("Absolute Correlation Matrix (TFs vs Latents)")
        ax_mcc.set_xlabel("Latent dimensions")
        ax_mcc.set_ylabel("True TFs")
        wandb.log({"MCC": wandb.Image(fig_mcc)})
        plt.close(fig_mcc)

        fig_r2, ax_r2 = plt.subplots()
        sns.heatmap(r2_mat.cpu().numpy(), annot=True, cmap="viridis", xticklabels=False, yticklabels=False)
        ax_r2.set_title("R2 Matrix (TFs vs Latents)")
        ax_r2.set_xlabel("Latent dimensions")
        ax_r2.set_ylabel("True TFs")
        wandb.log({"R2": wandb.Image(fig_r2)})
        plt.close(fig_r2)


# Logging
wandb.summary['final_mcc'] = mcc_score
wandb.summary['final_r2'] = r2_score

# Save model
if args.save is not None:
    torch.save(model.state_dict(), args.save)
    print(f"Model saved to {args.save}")

wandb.finish()
