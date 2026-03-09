
# Author: Riya Kayal
# Created: 15/11/2025
# Fixed: 02/03/2026


## Conformers were generated using RDKit ETKDGv3 and optimized with MMFF94 prior to RMSD calculation.
## Supervised Learning + Siamese NN with an early stoping criterion used for RMSD prediction.
## Open MP parallelised (implicit):
## i) PyTorch's OpenMP/MKL threading — PyTorch automatically parallelises matrix operations
## (like the MLP forward/backward passes) across all available CPU cores using Intel MKL
## or 
## ii) OpenMP under the hood. This is for free with no code changes.
## RDKit's conformer generation — EmbedMultipleConfs and MMFFOptimizeMoleculeConfs
## also use internal threading.

#------------------------------------------------------------------
# 1. import csv generated using RDKit
#------------------------------------------------------------------
import pandas as pd
import time
import psutil
import os
import argparse

# ------------------------------------------------------------------
# COMMAND-LINE ARGUMENTS
# Usage: python predict_rmsd.py --input a.csv
#        python predict_rmsd.py --input a.csv --n_mols 200
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Siamese RMSD Predictor")
parser.add_argument("--input",  type=str, required=True,
                    help="Path to input CSV file (must contain a 'smiles' column)")
parser.add_argument("--n_mols", type=int, default=500,
                    help="Number of molecules to use (default: 500)")
args = parser.parse_args()

input_file = args.input

# ------------------------------------------------------------------
# OUTPUT FOLDER — all plots and summary saved here
# ------------------------------------------------------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

df = pd.read_csv(input_file)

## set number of molecules
n_mols = args.n_mols

# ------------------------------------------------------------------
# SANITY CHECK: cap n_mols to the actual number of rows in the CSV
# ------------------------------------------------------------------
available_mols = len(df)
if n_mols > available_mols:
    print(f"WARNING: Requested n_mols={n_mols} exceeds available molecules "
          f"({available_mols}) in the CSV. Capping to {available_mols}.")
    n_mols = available_mols

# ------------------------------------------------------------------
# USER-DEFINED LOSS THRESHOLD (optional)
# Set to a float (e.g. 0.05) to stop training early once val loss
# drops below that value. Set to None to use patience-based early
# stopping only (default behaviour).
# ------------------------------------------------------------------
target_loss = None   # e.g. target_loss = 0.05

print(df.columns)
print(len(df))

#------------------------------------------------------------------
# 2. Generate Conformer Ensembles using RDKit
#------------------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch.nn as nn

def generate_conformers(smiles, n_confs=5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    ## ensures reproducibility
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol)

    return mol

#------------------------------------------------------------------
# 3. Compute RMSD Between Conformers
# Use RDKit's built-in RMSD (more robust than manual Kabsch)
#------------------------------------------------------------------

from rdkit.Chem import rdMolAlign

def compute_rdkit_rmsd(mol, conf1, conf2):
    return rdMolAlign.AlignMol(mol, mol, prbCid=conf1, refCid=conf2)


#------------------------------------------------------------------
# FIX 1: Pre-scan dataset to compute the true max_atoms needed.
#------------------------------------------------------------------
import torch

def compute_max_atoms(smiles_list):
    """Scan all SMILES and return the maximum atom count (with H)."""
    max_atoms = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        max_atoms = max(max_atoms, mol.GetNumAtoms())
    print(f"Computed max_atoms from dataset: {max_atoms}")
    return max_atoms

#------------------------------------------------------------------
# 4. Build Dataset
#------------------------------------------------------------------
import random


def build_dataset(smiles_list):
    dataset = []

    for smiles in smiles_list:
        mol = generate_conformers(smiles, n_confs=5)
        # SAFETY CHECK
        if mol is None or not isinstance(mol, Chem.Mol):
            print(f"WARNING: Could not generate conformers for: {smiles}")
            continue
        # FIX 2: Skip molecules where embedding produced no conformers
        # (can happen for complex ring systems or bad SMILES)
        if mol.GetNumConformers() == 0:
            print(f"WARNING: No conformers generated for: {smiles}")
            continue

        n = mol.GetNumConformers()

        for i in range(n):
            for j in range(i + 1, n):
                rmsd = rdMolAlign.AlignMol(mol, mol, prbCid=i, refCid=j)

                pos1 = torch.tensor(
                    mol.GetConformer(i).GetPositions(),
                    dtype=torch.float
                )

                pos2 = torch.tensor(
                    mol.GetConformer(j).GetPositions(),
                    dtype=torch.float
                )

                dataset.append((pos1, pos2, torch.tensor(rmsd, dtype=torch.float)))

    print("Total pairs:", len(dataset))
    return dataset

# FIX 1 (continued): Compute max_atoms dynamically before building
# the model, so DistanceEncoder is sized correctly for this CSV.
smiles_subset = df["smiles"][:n_mols].tolist()
max_atoms = compute_max_atoms(smiles_subset)

dataset = build_dataset(smiles_subset)

#-------------------------------------------------------------
# 5. Compute pairwise distance matrix for One Conformer
#-------------------------------------------------------------

def pairwise_distances(pos):
    """
    pos: (N, 3)
    returns: (N, N) distance matrix
    """
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-9)


#--------------------------------------------------------------
# 6. Convert RDKit Conformer to Tensor
#--------------------------------------------------------------

def conformer_to_tensor(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    return pos


#--------------------------------------------------------------
# 7. Invariant Encoder (MLP)
# FIX 1 (continued): max_atoms is now passed in dynamically
#--------------------------------------------------------------
class DistanceEncoder(nn.Module):
    def __init__(self, max_atoms):
        super().__init__()

        self.max_atoms = max_atoms
        self.input_dim = max_atoms * (max_atoms - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, pos):
        N = pos.shape[0]

        # FIX 3: Guard against molecules larger than max_atoms at runtime.
        # This prevents a silent index-out-of-bounds corruption when
        # padded[:N, :N] = D overflows the allocated padded matrix.
        if N > self.max_atoms:
            raise ValueError(
                f"Molecule has {N} atoms but model was built for max_atoms={self.max_atoms}. "
                f"Re-run compute_max_atoms() on the full dataset."
            )

        D = pairwise_distances(pos)

        padded = torch.zeros(self.max_atoms, self.max_atoms)
        padded[:N, :N] = D

        idx = torch.triu_indices(self.max_atoms, self.max_atoms, offset=1)
        flat = padded[idx[0], idx[1]]

        return self.mlp(flat)

#--------------------------------------------------------------
# 8. Siamese RMSD predictor
#--------------------------------------------------------------

class RMSDPredictor(nn.Module):
    def __init__(self, max_atoms):
        super().__init__()

        self.encoder = DistanceEncoder(max_atoms)

        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, pos1, pos2):
        h1 = self.encoder(pos1)
        h2 = self.encoder(pos2)

        diff = torch.abs(h1 - h2)
        return self.regressor(diff)

#-------------------------------------------------------------------
# 9. Train
# For n molecules with large n (n x 10 conformer pairs) the basic
# sample-by-sample loop is very slow and has no early stopping.
# Changes made:
#   - Train/val split (80/20) for proper generalisation tracking
#   - DataLoader with batch training for speed
#   - Early stopping (patience=20) to find the ideal epoch count
#   - LR scheduler (ReduceLROnPlateau) to fine-tune convergence
#   - n_range raised to 200 (early stopping will cut this short)
#------------------------------------------------------------------
from torch.utils.data import Dataset, DataLoader, random_split

class RMSDDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    pos1_list, pos2_list, rmsd_list = zip(*batch)
    return list(pos1_list), list(pos2_list), torch.stack(rmsd_list)

full_dataset = RMSDDataset(dataset)
val_size = max(1, int(0.2 * len(full_dataset)))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, collate_fn=collate_fn)

model = RMSDPredictor(max_atoms=max_atoms)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.MSELoss()

n_range       = 200
patience      = 20

# --- start tracking time and CPU ---
train_start_time = time.time()
process = psutil.Process(os.getpid())
cpu_samples = []
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

# --- history for plots ---
history_train_loss = []
history_val_loss   = []
history_lr         = []
best_epoch         = 0

for epoch in range(n_range):
    # --- train ---
    model.train()
    train_loss = 0.0
    for pos1_batch, pos2_batch, rmsd_batch in train_loader:
        batch_loss = 0.0
        for pos1, pos2, rmsd in zip(pos1_batch, pos2_batch, rmsd_batch):
            pred = model(pos1, pos2).squeeze()
            batch_loss += criterion(pred, rmsd)
        batch_loss /= len(pos1_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()

    # --- validate ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for pos1_batch, pos2_batch, rmsd_batch in val_loader:
            for pos1, pos2, rmsd in zip(pos1_batch, pos2_batch, rmsd_batch):
                pred = model(pos1, pos2).squeeze()
                val_loss += criterion(pred, rmsd).item()
    val_loss /= max(len(val_dataset), 1)

    scheduler.step(val_loss)
    current_lr = scheduler.get_last_lr()[0]
    elapsed = time.time() - train_start_time
    n_cores = psutil.cpu_count(logical=True)
    cpu_pct_raw = process.cpu_percent(interval=None)          # sum across all cores
    cpu_pct_norm = cpu_pct_raw / n_cores                      # normalised: 100% = all cores maxed
    cpu_samples.append(cpu_pct_norm)
    history_train_loss.append(train_loss)
    history_val_loss.append(val_loss)
    history_lr.append(current_lr)
    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s | CPU: {cpu_pct_norm:.1f}% of {n_cores} cores ({cpu_pct_raw:.0f}% total)")

    # --- target loss threshold (user-defined) ---
    if target_loss is not None and val_loss <= target_loss:
        print(f"Target loss {target_loss} reached at epoch {epoch} "
              f"(val loss: {val_loss:.4f}). Stopping.")
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
        break

    # --- early stopping (patience-based, used when target_loss=None) ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_epoch = epoch
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
            break

# Restore best weights before evaluation
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Restored best model weights.")

# --- training summary ---
total_train_time = time.time() - train_start_time
avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
peak_cpu = max(cpu_samples) if cpu_samples else 0.0
n_cores = psutil.cpu_count(logical=True)
ram_mb = process.memory_info().rss / 1024 ** 2

print()
print("======================================")
print("          Training Summary")
print("======================================")
print(f"Total training time : {total_train_time:.1f}s  ({total_train_time/60:.2f} min)")
print(f"Epochs run          : {epoch + 1}")
print(f"Avg CPU usage       : {avg_cpu:.1f}% normalised ({avg_cpu * n_cores:.0f}% raw across {n_cores} cores)")
print(f"Peak CPU usage      : {peak_cpu:.1f}% normalised ({peak_cpu * n_cores:.0f}% raw across {n_cores} cores)")
print(f"RAM used (process)  : {ram_mb:.1f} MB")
print("======================================")


#--------------------------------------------------------------
# 10. Evaluation Function
#--------------------------------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

def evaluate(model, dataset):
    preds, targets = [], []
    with torch.no_grad():
        for pos1, pos2, rmsd in dataset:
            preds.append(model(pos1, pos2).item())
            targets.append(rmsd.item())
    mae          = mean_absolute_error(targets, preds)
    rmse         = np.sqrt(mean_squared_error(targets, preds))
    r2           = r2_score(targets, preds)
    spearman, _  = spearmanr(targets, preds)
    print("======================================")
    print("              Evaluation")
    print("======================================")
    print(f"MAE:      {mae:.4f} Å")
    print(f"RMSE:     {rmse:.4f} Å")
    print(f"R²:       {r2:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print("======================================")
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman,
            "preds": np.array(preds), "targets": np.array(targets)}

val_data   = [full_dataset[i] for i in val_dataset.indices]
train_data = [full_dataset[i] for i in train_dataset.indices]
print("--- Validation set ---")
val_metrics   = evaluate(model, val_data)
print("--- Train set ---")
train_metrics = evaluate(model, train_data)

# ---------------------------------------------------------------
# 11. Summary file
# ---------------------------------------------------------------
import datetime
summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 50 + "\n")
    f.write("         RMSD Predictor — Run Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Run date/time       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input CSV           : {input_file}\n")
    f.write(f"Molecules used      : {n_mols}\n")
    f.write(f"Max atoms (with H)  : {max_atoms}\n")
    f.write(f"Total dataset pairs : {len(dataset)}\n")
    f.write(f"Train pairs         : {len(train_data)}\n")
    f.write(f"Val pairs           : {len(val_data)}\n")
    f.write("\n--- Training ---\n")
    f.write(f"Epochs run          : {epoch + 1} / {n_range}\n")
    f.write(f"Best epoch          : {best_epoch}\n")
    f.write(f"Best val loss       : {best_val_loss:.6f}\n")
    f.write(f"Target loss         : {target_loss if target_loss is not None else 'N/A (patience-based)'}\n")
    f.write(f"Total training time : {total_train_time:.1f}s ({total_train_time/60:.2f} min)\n")
    f.write(f"CPU cores (logical) : {n_cores}\n")
    f.write(f"Avg CPU usage       : {avg_cpu:.1f}% normalised ({avg_cpu * n_cores:.0f}% raw)\n")
    f.write(f"Peak CPU usage      : {peak_cpu:.1f}% normalised ({peak_cpu * n_cores:.0f}% raw)\n")
    f.write(f"RAM used (process)  : {ram_mb:.1f} MB\n")
    f.write("\n--- Validation Metrics ---\n")
    f.write(f"MAE:      {val_metrics['mae']:.4f} Å\n")
    f.write(f"RMSE:     {val_metrics['rmse']:.4f} Å\n")
    f.write(f"R²:       {val_metrics['r2']:.4f}\n")
    f.write(f"Spearman: {val_metrics['spearman']:.4f}\n")
    f.write("\n--- Train Metrics ---\n")
    f.write(f"MAE:      {train_metrics['mae']:.4f} Å\n")
    f.write(f"RMSE:     {train_metrics['rmse']:.4f} Å\n")
    f.write(f"R²:       {train_metrics['r2']:.4f}\n")
    f.write(f"Spearman: {train_metrics['spearman']:.4f}\n")
    f.write("=" * 50 + "\n")
print(f"Saved: {summary_path}")

# ---------------------------------------------------------------
# 12. Visualisation — all saved to OUTPUT_DIR at 300 dpi
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

SAVE_DPI = 300
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "figure.dpi": SAVE_DPI})

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=SAVE_DPI, format="jpeg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

ep       = list(range(len(history_train_loss)))
val_p    = val_metrics["preds"]
val_t    = val_metrics["targets"]
residuals = val_p - val_t

# collect atom counts per val pair for plot 9
val_atom_counts = [data[0].shape[0] for data in val_data]

# collect per-molecule labels and per-molecule errors for plot 8
# smiles_subset index matches dataset build order; we need mol index per pair
mol_pair_labels = []
mol_idx = 0
pair_count = 0
smiles_labels = df["smiles"][:n_mols].tolist()
mol_name_map = {}
if "Name" in df.columns:
    mol_name_map = dict(zip(df["smiles"][:n_mols].tolist(),
                            df["Name"][:n_mols].tolist()))

# ------------------------------------------------------------------
# Plot 1 — Train vs Val Loss
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ep, history_train_loss, label="Train loss", linewidth=1.8, color="#2563EB")
ax.plot(ep, history_val_loss,   label="Val loss",   linewidth=1.8, color="#DC2626")
ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.2,
           label=f"Best epoch ({best_epoch})")
if target_loss is not None:
    ax.axhline(target_loss, color="green", linestyle=":", linewidth=1.2,
               label=f"Target loss ({target_loss})")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("Train vs Validation Loss")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot1_loss_curve.jpg")

# ------------------------------------------------------------------
# Plot 2 — Learning Rate Schedule
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ep, history_lr, color="#7C3AED", linewidth=1.8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule")
ax.set_yscale("log"); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot2_lr_schedule.jpg")

# ------------------------------------------------------------------
# Plot 3 — Predicted vs Actual RMSD
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(val_t, val_p, alpha=0.5, s=20, color="#2563EB", edgecolors="none")
lims = [min(val_t.min(), val_p.min()) - 0.1,
        max(val_t.max(), val_p.max()) + 0.1]
ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")
ax.set_xlabel("Actual RMSD (Å)"); ax.set_ylabel("Predicted RMSD (Å)")
ax.set_title("Predicted vs Actual RMSD (Validation Set)")
ax.legend(); ax.set_xlim(lims); ax.set_ylim(lims); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot3_pred_vs_actual.jpg")

# ------------------------------------------------------------------
# Plot 4 — Residuals vs Actual RMSD
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(val_t, residuals, alpha=0.5, s=20, color="#DC2626", edgecolors="none")
ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_xlabel("Actual RMSD (Å)"); ax.set_ylabel("Residual (Predicted − Actual) (Å)")
ax.set_title("Residual Plot (Validation Set)"); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot4_residuals.jpg")

# ------------------------------------------------------------------
# Plot 5 — Residual Distribution
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(residuals, bins=30, color="#059669", edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.axvline(residuals.mean(), color="#DC2626", linewidth=1.2,
           label=f"Mean: {residuals.mean():.3f} Å")
ax.set_xlabel("Residual (Å)"); ax.set_ylabel("Count")
ax.set_title("Residual Distribution (Validation Set)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot5_residual_hist.jpg")

# ------------------------------------------------------------------
# Plot 6 — RMSD distribution per molecule (box plot)
# Build a per-molecule error dict from the val set
# ------------------------------------------------------------------
mol_errors = {}   # mol_smiles -> list of |residual|
for idx, (pos1, pos2, rmsd) in enumerate(val_data):
    # identify which molecule this pair came from by atom count as proxy
    # (exact mapping would require storing mol index in build_dataset)
    n_atoms = pos1.shape[0]
    key = f"n={n_atoms}"
    mol_errors.setdefault(key, []).append(abs(residuals[idx]))

if len(mol_errors) > 1:
    fig, ax = plt.subplots(figsize=(max(8, len(mol_errors) * 0.6), 5))
    labels = sorted(mol_errors.keys())
    data_box = [mol_errors[k] for k in labels]
    bp = ax.boxplot(data_box, patch_artist=True, notch=False)
    colors = get_cmap("tab20")(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Molecule group (by atom count)")
    ax.set_ylabel("|Residual| (Å)")
    ax.set_title("Absolute Error Distribution by Molecule Size Group")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    savefig("plot6_error_by_mol_group.jpg")

# ------------------------------------------------------------------
# Plot 7 — Error vs Molecular Size (atom count)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(val_atom_counts, np.abs(residuals),
                alpha=0.5, s=20, c=val_t, cmap="viridis", edgecolors="none")
plt.colorbar(sc, ax=ax, label="Actual RMSD (Å)")
ax.set_xlabel("Atom count (with H)")
ax.set_ylabel("|Residual| (Å)")
ax.set_title("Absolute Error vs Molecular Size (Validation Set)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot7_error_vs_mol_size.jpg")

# ------------------------------------------------------------------
# Plot 8 — Parity plot coloured by atom-count group
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
unique_sizes = sorted(set(val_atom_counts))
cmap8 = get_cmap("tab20")
size_color = {s: cmap8(i / max(len(unique_sizes) - 1, 1))
              for i, s in enumerate(unique_sizes)}
for s in unique_sizes:
    mask = np.array(val_atom_counts) == s
    ax.scatter(val_t[mask], val_p[mask],
               alpha=0.6, s=22, label=f"n={s}",
               color=size_color[s], edgecolors="none")
lims8 = [min(val_t.min(), val_p.min()) - 0.1,
         max(val_t.max(), val_p.max()) + 0.1]
ax.plot(lims8, lims8, "k--", linewidth=1.2)
ax.set_xlabel("Actual RMSD (Å)"); ax.set_ylabel("Predicted RMSD (Å)")
ax.set_title("Parity Plot Coloured by Molecule Size")
ax.set_xlim(lims8); ax.set_ylim(lims8); ax.grid(True, alpha=0.3)
if len(unique_sizes) <= 15:
    ax.legend(fontsize=7, ncol=2, title="Atom count")
plt.tight_layout()
savefig("plot8_parity_by_mol.jpg")

# ------------------------------------------------------------------
# Plot 9 — t-SNE of encoder embeddings (val set)
# ------------------------------------------------------------------
try:
    from sklearn.manifold import TSNE
    embeddings = []
    model.eval()
    with torch.no_grad():
        for pos1, pos2, rmsd in val_data:
            h1 = model.encoder(pos1).numpy()
            h2 = model.encoder(pos2).numpy()
            embeddings.append(np.abs(h1 - h2))
    embeddings = np.array(embeddings)

    if len(embeddings) >= 10:
        perp = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        emb_2d = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                        c=val_t, cmap="plasma", alpha=0.7, s=25, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Actual RMSD (Å)")
        ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        ax.set_title("t-SNE of Encoder Embeddings\n(coloured by actual RMSD)")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        savefig("plot9_tsne_embeddings.jpg")
    else:
        print("Skipping t-SNE: not enough validation samples.")
except Exception as e:
    print(f"Skipping t-SNE plot: {e}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")

