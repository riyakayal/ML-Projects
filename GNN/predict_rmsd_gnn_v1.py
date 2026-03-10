
# Author: Riya Kayal
# Created: 15/11/2025
# GNN version v1: 05/02/2026

## Conformers were generated using RDKit ETKDGv3 and optimized with MMFF94 prior to RMSD calculation.
##
## IMPROVEMENTS over predict_rmsd_gnn_v0.py:
##   #7  — More conformers: n_confs=10 (was 5) → 45 pairs/mol (was 10)
##   #9  — Huber loss instead of MSE → less sensitive to outlier RMSD pairs
##   #1  — RBF edge features: 16-bin Gaussian expansion of distance (was scalar)
##   #3  — Richer node features: +formal charge, hybridisation, H count,
##          in-ring flag, ring size (25-dim total, was 13-dim)
##   NEW — Non-bonded edges: atom pairs within NONBOND_CUTOFF Å are connected
##          with edges even if not bonded, giving the GNN access to
##          through-space 3D geometry (crucial for conformer RMSD)
##   NEW — Batched training loop: graphs in a batch are processed together
##          via concatenated node/edge tensors and a batch index vector,
##          replacing the slow per-sample Python for-loop

#------------------------------------------------------------------
# 1. Imports and command-line arguments
#------------------------------------------------------------------
import pandas as pd
import time
import psutil
import os
import argparse
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdchem

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

# ------------------------------------------------------------------
# COMMAND-LINE ARGUMENTS
# Usage: python predict_rmsd_gnn.py --input a.csv
#        python predict_rmsd_gnn.py --input a.csv --n_mols 200
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Siamese GNN RMSD Predictor v1")
parser.add_argument("--input",  type=str, required=True,
                    help="Path to input CSV file (must contain a 'smiles' column)")
parser.add_argument("--n_mols", type=int, default=500,
                    help="Number of molecules to use (default: 500)")
args = parser.parse_args()

input_file = args.input

# ------------------------------------------------------------------
# OUTPUT FOLDER
# ------------------------------------------------------------------
OUTPUT_DIR = "output_gnn_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

df = pd.read_csv(input_file)
print(df.columns)
print(len(df))

n_mols = args.n_mols

# ------------------------------------------------------------------
# SANITY CHECK
# ------------------------------------------------------------------
available_mols = len(df)
if n_mols > available_mols:
    print(f"WARNING: Requested n_mols={n_mols} exceeds available molecules "
          f"({available_mols}) in the CSV. Capping to {available_mols}.")
    n_mols = available_mols

# ------------------------------------------------------------------
# USER-DEFINED LOSS THRESHOLD (optional)
# ------------------------------------------------------------------
target_loss = None   # e.g. target_loss = 0.05

# ------------------------------------------------------------------
# IMPROVEMENT: Filter near-zero RMSD pairs and log-transform labels
#   RMSD_MIN : pairs below this threshold are degenerate (conformers
#              collapsed to same geometry) and are dropped.
#   LOG_RMSD : predict log(RMSD) instead of RMSD directly to compress
#              the right-skewed distribution into a symmetric range.
#              Predictions are exp()-transformed back at evaluation.
# ------------------------------------------------------------------
RMSD_MIN  = 0.1    # Å — drop near-identical conformer pairs
LOG_RMSD  = True   # set False to revert to raw RMSD prediction

# ------------------------------------------------------------------
# NEW: Non-bonded edges
#   Atom pairs within this distance (Å) are connected with edges even
#   if they share no covalent bond. This lets the GNN see through-space
#   geometry — e.g. a ring flip changes many non-bonded distances but
#   NO bond distances, so without this the GNN is blind to it.
#   Set to None to disable (bonded edges only, original behaviour).
# ------------------------------------------------------------------
NONBOND_CUTOFF = 5.0   # Å  (None = bonded edges only)

#------------------------------------------------------------------
# 2. Generate Conformer Ensembles using RDKit
# IMPROVEMENT #7: n_confs raised from 5 → 10
#   5 confs → 10 pairs/mol
#  10 confs → 45 pairs/mol  (4.5x more training data, same molecules)
#------------------------------------------------------------------
N_CONFS = 10   # <-- change here to adjust

def generate_conformers(smiles, n_confs=N_CONFS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    return mol

#------------------------------------------------------------------
# 3. Featurisation
#
# IMPROVEMENT #1: RBF edge features
#   Replace scalar distance with 16-bin Gaussian radial basis expansion.
#   Each distance d → vector of 16 Gaussians centred at evenly spaced
#   values between 0.5 Å and 5.0 Å. This gives the GNN a smooth,
#   rich description of how close/far bonded atoms are.
#
# IMPROVEMENT #3: Richer node features (26-dim, was 13-dim)
#   Added: formal charge, hybridisation (sp/sp2/sp3/other),
#          number of attached H, in-ring flag, ring size bucket.
#------------------------------------------------------------------

# --- RBF parameters ---
RBF_BINS    = 16
RBF_D_MIN   = 0.5
RBF_D_MAX   = 8.0    # extended upper bound to cover non-bonded distances up to 5 Å
RBF_GAMMA   = 0.5
RBF_CENTRES = torch.linspace(RBF_D_MIN, RBF_D_MAX, RBF_BINS)  # (16,)

def rbf_expand(distances):
    """
    distances: list of scalar float values (Å)
    returns  : (len(distances), RBF_BINS) tensor
    """
    d = torch.tensor(distances, dtype=torch.float).unsqueeze(1)   # (E, 1)
    c = RBF_CENTRES.unsqueeze(0)                                   # (1, 16)
    return torch.exp(-RBF_GAMMA * (d - c) ** 2)                   # (E, 16)

# EDGE_DIM = RBF (16) + is_bonded flag (1) = 17
# The is_bonded flag lets the GNN distinguish covalent from non-bonded edges.
EDGE_DIM = RBF_BINS + 1  # 17

# --- Node features ---
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H", "other"]

HYBRIDISATION_MAP = {
    rdchem.HybridizationType.SP:  [1, 0, 0, 0],
    rdchem.HybridizationType.SP2: [0, 1, 0, 0],
    rdchem.HybridizationType.SP3: [0, 0, 1, 0],
}

RING_SIZE_BUCKETS = [3, 4, 5, 6, 7]   # one-hot: is atom in ring of size k?

def atom_features(atom, ring_info):
    """
    Return 25-dim node feature vector:
      [0:11]  one-hot atom type           (11)
      [11]    normalised degree            (1)
      [12]    is aromatic                  (1)
      [13]    normalised formal charge     (1)
      [14:18] hybridisation one-hot        (4)  sp / sp2 / sp3 / other
      [18]    normalised H count           (1)
      [19]    in any ring                  (1)
      [20:25] in ring of size 3/4/5/6/7   (5)  multi-hot ring size buckets
    Total = 25 dims
    """
    symbol  = atom.GetSymbol()
    one_hot = [1.0 if symbol == t else 0.0 for t in ATOM_TYPES[:-1]]
    one_hot.append(1.0 if symbol not in ATOM_TYPES[:-1] else 0.0)

    degree   = [atom.GetDegree() / 6.0]
    aromatic = [float(atom.GetIsAromatic())]
    charge   = [atom.GetFormalCharge() / 4.0]

    hyb      = HYBRIDISATION_MAP.get(atom.GetHybridization(), [0, 0, 0, 1])

    h_count  = [atom.GetTotalNumHs() / 4.0]

    idx       = atom.GetIdx()
    in_ring   = [float(ring_info.NumAtomRings(idx) > 0)]
    ring_size = [float(ring_info.IsAtomInRingOfSize(idx, s))
                 for s in RING_SIZE_BUCKETS]

    feats = one_hot + degree + aromatic + charge + hyb + h_count + in_ring + ring_size
    return torch.tensor(feats, dtype=torch.float)   # dim = 26

NODE_DIM = 25   # must match atom_features output

def mol_conformer_to_graph(mol, conf_id):
    """
    Convert one RDKit conformer to a graph dict:
      x         : (N, NODE_DIM=25)   node features
      edge_index: (2, E)             COO edge list (both directions)
      edge_attr : (E, EDGE_DIM=17)   RBF distance (16) + is_bonded flag (1)
      pos       : (N, 3)             raw 3D coordinates

    Edges include:
      - All covalent bonds (is_bonded=1)
      - All atom pairs within NONBOND_CUTOFF Å that are NOT bonded (is_bonded=0)
        These through-space edges encode 3D geometry invisible to bond-only graphs.
    """
    conf      = mol.GetConformer(conf_id)
    pos       = torch.tensor(conf.GetPositions(), dtype=torch.float)  # (N, 3)
    ring_info = mol.GetRingInfo()
    atoms     = list(mol.GetAtoms())
    N         = len(atoms)

    x = torch.stack([atom_features(a, ring_info) for a in atoms])  # (N, 25)

    src, dst, raw_dist, is_bonded = [], [], [], []

    # --- covalent bond edges ---
    bonded_set = set()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        d = torch.norm(pos[i] - pos[j]).item()
        src      += [i, j]; dst      += [j, i]
        raw_dist += [d, d]; is_bonded += [1.0, 1.0]
        bonded_set.add((i, j)); bonded_set.add((j, i))

    # --- non-bonded edges within cutoff ---
    if NONBOND_CUTOFF is not None:
        # Compute all pairwise distances efficiently with broadcasting
        diff     = pos.unsqueeze(0) - pos.unsqueeze(1)          # (N, N, 3)
        dist_mat = torch.sqrt((diff ** 2).sum(-1) + 1e-9)       # (N, N)
        pairs    = (dist_mat < NONBOND_CUTOFF).nonzero(as_tuple=False)
        for p in pairs:
            i, j = p[0].item(), p[1].item()
            if i == j:
                continue
            if (i, j) in bonded_set:
                continue
            d = dist_mat[i, j].item()
            src      += [i, j]; dst      += [j, i]
            raw_dist += [d, d]; is_bonded += [0.0, 0.0]

    edge_index  = torch.tensor([src, dst], dtype=torch.long)      # (2, E)
    rbf         = rbf_expand(raw_dist)                             # (E, 16)
    bonded_flag = torch.tensor(is_bonded, dtype=torch.float).unsqueeze(1)  # (E, 1)
    edge_attr   = torch.cat([rbf, bonded_flag], dim=1)             # (E, 17)

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "pos": pos}

#------------------------------------------------------------------
# 4. Build Dataset
#------------------------------------------------------------------
def build_dataset(smiles_list):
    dataset = []
    for smiles in smiles_list:
        mol = generate_conformers(smiles, n_confs=N_CONFS)
        if mol is None or not isinstance(mol, Chem.Mol):
            print(f"WARNING: Could not generate conformers for: {smiles}")
            continue
        if mol.GetNumConformers() == 0:
            print(f"WARNING: No conformers generated for: {smiles}")
            continue

        n = mol.GetNumConformers()
        for i in range(n):
            for j in range(i + 1, n):
                rmsd = rdMolAlign.AlignMol(mol, mol, prbCid=i, refCid=j)

                # IMPROVEMENT: drop near-identical conformer pairs
                if rmsd < RMSD_MIN:
                    continue

                g1 = mol_conformer_to_graph(mol, i)
                g2 = mol_conformer_to_graph(mol, j)

                # IMPROVEMENT: log-transform label if enabled
                label = np.log(rmsd) if LOG_RMSD else rmsd
                dataset.append((g1, g2, torch.tensor(label, dtype=torch.float)))

    print("Total pairs:", len(dataset))
    return dataset

smiles_subset = df["smiles"][:n_mols].tolist()
dataset       = build_dataset(smiles_subset)

#------------------------------------------------------------------
# 5. GNN building block: Edge-Conditioned Message Passing
#    Edge features are now RBF vectors (16-dim) instead of scalars.
#------------------------------------------------------------------
class EdgeConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=EDGE_DIM):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )
        self.self_lin  = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)
        self.act       = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        w   = self.edge_mlp(edge_attr)        # (E, in_dim)
        msg = w * x[src]                      # (E, in_dim)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)            # (N, in_dim)
        return self.act(self.self_lin(x) + self.neigh_lin(agg))

#------------------------------------------------------------------
# 6. GNN Encoder
#    Supports both single-graph (evaluation) and batched (training).
#    When graph["batch"] is present, global mean pooling returns one
#    embedding per graph in the batch using scatter_mean logic.
#------------------------------------------------------------------
class GNNEncoder(nn.Module):
    def __init__(self, node_dim=NODE_DIM, hidden=64, out_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden)
        self.conv1      = EdgeConvLayer(hidden, hidden)
        self.conv2      = EdgeConvLayer(hidden, hidden)
        self.conv3      = EdgeConvLayer(hidden, out_dim)
        self.out_dim    = out_dim

    def forward(self, graph):
        x          = graph["x"]
        edge_index = graph["edge_index"]
        edge_attr  = graph["edge_attr"]
        batch      = graph.get("batch", None)

        x = torch.relu(self.input_proj(x))
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)

        if batch is None:
            # single graph: mean over all nodes → (out_dim,)
            return x.mean(dim=0)
        else:
            # batched graphs: scatter mean per graph → (B, out_dim)
            B   = batch.max().item() + 1
            out = torch.zeros(B, self.out_dim, dtype=x.dtype, device=x.device)
            cnt = torch.zeros(B, 1,            dtype=x.dtype, device=x.device)
            out.index_add_(0, batch, x)
            cnt.index_add_(0, batch, torch.ones(x.shape[0], 1,
                                                dtype=x.dtype, device=x.device))
            return out / cnt.clamp(min=1)

#------------------------------------------------------------------
# 7. Siamese RMSD Predictor
# IMPROVEMENT: richer comparison head.
#   Old: abs(h1-h2)           → 64-dim input to regressor
#   New: cat(h1, h2, |h1-h2|, h1*h2) → 256-dim input
#   The product h1*h2 captures feature correlation (similarity signal).
#   Concatenating h1 and h2 directly preserves individual conformer info.
#------------------------------------------------------------------
class RMSDPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = GNNEncoder()
        self.regressor = nn.Sequential(
            nn.Linear(64 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, g1, g2):
        h1 = self.encoder(g1)   # (B, 64) if batched, (64,) if single
        h2 = self.encoder(g2)
        if h1.dim() == 1:
            # single sample (evaluation)
            combined = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2])
        else:
            # batched: operate row-wise
            combined = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=1)
        return self.regressor(combined)   # (B, 1) or (1,)

#------------------------------------------------------------------
# 8. Train
# IMPROVEMENT #9: HuberLoss (delta=1.0) replaces MSELoss.
#
# SPEED FIX: Batched GNN forward pass.
#   Old approach: Python for-loop over each sample in the batch,
#     calling model(g1, g2) once per pair → pure Python overhead
#     dominates, GPU/CPU sits idle between calls. With 4455 pairs,
#     this is ~4455 separate forward passes per epoch.
#   New approach: all graphs in a batch are concatenated into a single
#     large disconnected graph using a batch_index vector. One single
#     forward pass processes the entire batch, then global mean pooling
#     uses batch_index to aggregate each graph separately. This is the
#     standard approach in PyG and reduces epoch time by ~5–10x.
#------------------------------------------------------------------

def batch_graphs(graph_list):
    """
    Concatenate a list of graph dicts into one large disconnected graph.
    Returns:
      x          : (sum_N, NODE_DIM)
      edge_index : (2, sum_E)   — node indices offset per graph
      edge_attr  : (sum_E, EDGE_DIM)
      batch_idx  : (sum_N,)     — which graph each node belongs to
    """
    x_list, ei_list, ea_list, batch_list = [], [], [], []
    node_offset = 0
    for b_idx, g in enumerate(graph_list):
        n = g["x"].shape[0]
        x_list.append(g["x"])
        ei_list.append(g["edge_index"] + node_offset)
        ea_list.append(g["edge_attr"])
        batch_list.append(torch.full((n,), b_idx, dtype=torch.long))
        node_offset += n
    return {
        "x":         torch.cat(x_list,   dim=0),
        "edge_index": torch.cat(ei_list,  dim=1),
        "edge_attr":  torch.cat(ea_list,  dim=0),
        "batch":      torch.cat(batch_list, dim=0),
    }

class RMSDDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self):        return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate_fn(batch):
    g1_list, g2_list, rmsd_list = zip(*batch)
    return list(g1_list), list(g2_list), torch.stack(rmsd_list)

full_dataset  = RMSDDataset(dataset)
val_size      = max(1, int(0.2 * len(full_dataset)))
train_size    = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, collate_fn=collate_fn)

model     = RMSDPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.HuberLoss(delta=1.0)   # IMPROVEMENT #9

n_range          = 200
patience         = 20
best_val_loss    = float("inf")
patience_counter = 0
best_model_state = None
best_epoch       = 0

# --- tracking ---
train_start_time   = time.time()
process            = psutil.Process(os.getpid())
cpu_samples        = []
history_train_loss = []
history_val_loss   = []
history_lr         = []

for epoch in range(n_range):
    # --- train (batched) ---
    model.train()
    train_loss = 0.0
    for g1_list, g2_list, rmsd_batch in train_loader:
        bg1  = batch_graphs(g1_list)
        bg2  = batch_graphs(g2_list)
        pred = model(bg1, bg2).squeeze(1)   # (B,)
        loss = criterion(pred, rmsd_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- validate (batched) ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for g1_list, g2_list, rmsd_batch in val_loader:
            bg1       = batch_graphs(g1_list)
            bg2       = batch_graphs(g2_list)
            pred      = model(bg1, bg2).squeeze(1)
            val_loss += criterion(pred, rmsd_batch).item()
    val_loss /= max(len(val_loader), 1)

    scheduler.step(val_loss)
    current_lr   = scheduler.get_last_lr()[0]
    elapsed      = time.time() - train_start_time
    n_cores      = psutil.cpu_count(logical=True)
    cpu_pct_raw  = process.cpu_percent(interval=None)
    cpu_pct_norm = cpu_pct_raw / n_cores
    cpu_samples.append(cpu_pct_norm)
    history_train_loss.append(train_loss)
    history_val_loss.append(val_loss)
    history_lr.append(current_lr)
    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
          f"| CPU: {cpu_pct_norm:.1f}% of {n_cores} cores ({cpu_pct_raw:.0f}% total)")

    # --- target loss threshold ---
    if target_loss is not None and val_loss <= target_loss:
        print(f"Target loss {target_loss} reached at epoch {epoch} "
              f"(val loss: {val_loss:.4f}). Stopping.")
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
        break

    # --- early stopping ---
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        best_epoch       = epoch
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Restored best model weights.")

# --- training summary (console) ---
total_train_time = time.time() - train_start_time
avg_cpu  = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
peak_cpu = max(cpu_samples) if cpu_samples else 0.0
ram_mb   = process.memory_info().rss / 1024 ** 2
n_cores  = psutil.cpu_count(logical=True)

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

#------------------------------------------------------------------
# 9. Evaluation
# When LOG_RMSD=True, model outputs log(RMSD). We exp()-transform
# both predictions and targets back to Å space before computing metrics
# so that MAE/RMSE/R² are always interpretable in Ångströms.
#------------------------------------------------------------------
def evaluate(model, dataset):
    preds, targets = [], []
    with torch.no_grad():
        for g1, g2, label in dataset:
            pred = model(g1, g2).item()
            preds.append(pred)
            targets.append(label.item())

    if LOG_RMSD:
        preds   = np.exp(np.array(preds))
        targets = np.exp(np.array(targets))
    else:
        preds   = np.array(preds)
        targets = np.array(targets)

    mae         = mean_absolute_error(targets, preds)
    rmse        = np.sqrt(mean_squared_error(targets, preds))
    r2          = r2_score(targets, preds)
    spearman, _ = spearmanr(targets, preds)
    print("======================================")
    print("              Evaluation")
    print("======================================")
    print(f"MAE:      {mae:.4f} Å")
    print(f"RMSE:     {rmse:.4f} Å")
    print(f"R²:       {r2:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print("======================================")
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman,
            "preds": preds, "targets": targets}

val_data   = [full_dataset[i] for i in val_dataset.indices]
train_data = [full_dataset[i] for i in train_dataset.indices]
print("--- Validation set ---")
val_metrics   = evaluate(model, val_data)
print("--- Train set ---")
train_metrics = evaluate(model, train_data)

#------------------------------------------------------------------
# 10. Summary file  (unchanged)
#------------------------------------------------------------------
summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 50 + "\n")
    f.write("   GNN v1 RMSD Predictor — Run Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Run date/time       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model               : Siamese GNN v1 (EdgeConv, 3 layers)\n")
    f.write(f"Improvements        : #7 more conformers ({N_CONFS}), #9 Huber loss,\n")
    f.write(f"                      #1 RBF edges ({RBF_BINS}-bin), #3 rich node features (25-dim),\n")
    f.write(f"                      non-bonded edges (cutoff={NONBOND_CUTOFF}Å),\n")
    f.write(f"                      filter RMSD<{RMSD_MIN}Å, log-transform={LOG_RMSD},\n")
    f.write(f"                      richer Siamese head (cat h1,h2,|h1-h2|,h1*h2),\n")
    f.write(f"                      batched GNN training loop\n")
    f.write(f"Input CSV           : {input_file}\n")
    f.write(f"Molecules used      : {n_mols}\n")
    f.write(f"Conformers/mol      : {N_CONFS}\n")
    f.write(f"Total dataset pairs : {len(dataset)}\n")
    f.write(f"Train pairs         : {len(train_data)}\n")
    f.write(f"Val pairs           : {len(val_data)}\n")
    f.write("\n--- Training ---\n")
    f.write(f"Loss function       : HuberLoss (delta=1.0)\n")
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

#------------------------------------------------------------------
# 11. Visualisation  (unchanged)
#------------------------------------------------------------------
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

ep        = list(range(len(history_train_loss)))
val_p     = val_metrics["preds"]
val_t     = val_metrics["targets"]
residuals = val_p - val_t

val_atom_counts = [data[0]["x"].shape[0] for data in val_data]

# Plot 1 — Train vs Val Loss
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ep, history_train_loss, label="Train loss", linewidth=1.8, color="#2563EB")
ax.plot(ep, history_val_loss,   label="Val loss",   linewidth=1.8, color="#DC2626")
ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.2,
           label=f"Best epoch ({best_epoch})")
if target_loss is not None:
    ax.axhline(target_loss, color="green", linestyle=":", linewidth=1.2,
               label=f"Target loss ({target_loss})")
ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
ax.set_title("Train vs Validation Loss (GNN v1)")
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
savefig("plot1_loss_curve.jpg")

# Plot 2 — Learning Rate Schedule
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ep, history_lr, color="#7C3AED", linewidth=1.8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule"); ax.set_yscale("log")
ax.grid(True, alpha=0.3); plt.tight_layout()
savefig("plot2_lr_schedule.jpg")

# Plot 3 — Predicted vs Actual RMSD
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(val_t, val_p, alpha=0.5, s=20, color="#2563EB", edgecolors="none")
lims = [min(val_t.min(), val_p.min()) - 0.1, max(val_t.max(), val_p.max()) + 0.1]
ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")
ax.set_xlabel("Actual RMSD (Å)"); ax.set_ylabel("Predicted RMSD (Å)")
ax.set_title("Predicted vs Actual RMSD — GNN v1 (Validation Set)")
ax.legend(); ax.set_xlim(lims); ax.set_ylim(lims); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot3_pred_vs_actual.jpg")

# Plot 4 — Residuals vs Actual RMSD
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(val_t, residuals, alpha=0.5, s=20, color="#DC2626", edgecolors="none")
ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_xlabel("Actual RMSD (Å)"); ax.set_ylabel("Residual (Predicted − Actual) (Å)")
ax.set_title("Residual Plot — GNN v1 (Validation Set)"); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot4_residuals.jpg")

# Plot 5 — Residual Distribution
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(residuals, bins=30, color="#059669", edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.axvline(residuals.mean(), color="#DC2626", linewidth=1.2,
           label=f"Mean: {residuals.mean():.3f} Å")
ax.set_xlabel("Residual (Å)"); ax.set_ylabel("Count")
ax.set_title("Residual Distribution — GNN v1 (Validation Set)")
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
savefig("plot5_residual_hist.jpg")

# Plot 6 — Error by molecule size group
mol_errors = {}
for idx, (g1, g2, rmsd) in enumerate(val_data):
    key = f"n={g1['x'].shape[0]}"
    mol_errors.setdefault(key, []).append(abs(residuals[idx]))

if len(mol_errors) > 1:
    fig, ax = plt.subplots(figsize=(max(8, len(mol_errors) * 0.6), 5))
    labels   = sorted(mol_errors.keys())
    data_box = [mol_errors[k] for k in labels]
    bp       = ax.boxplot(data_box, patch_artist=True)
    colors   = get_cmap("tab20")(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Molecule group (by atom count)")
    ax.set_ylabel("|Residual| (Å)")
    ax.set_title("Absolute Error by Molecule Size Group — GNN v1")
    ax.grid(True, alpha=0.3, axis="y"); plt.tight_layout()
    savefig("plot6_error_by_mol_group.jpg")

# Plot 7 — Error vs Molecular Size
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(val_atom_counts, np.abs(residuals),
                alpha=0.5, s=20, c=val_t, cmap="viridis", edgecolors="none")
plt.colorbar(sc, ax=ax, label="Actual RMSD (Å)")
ax.set_xlabel("Atom count (with H)"); ax.set_ylabel("|Residual| (Å)")
ax.set_title("Absolute Error vs Molecular Size — GNN v1"); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot7_error_vs_mol_size.jpg")

# Plot 8 — Parity plot coloured by atom-count group
fig, ax = plt.subplots(figsize=(7, 6))
unique_sizes = sorted(set(val_atom_counts))
cmap8        = get_cmap("tab20")
size_color   = {s: cmap8(i / max(len(unique_sizes) - 1, 1))
                for i, s in enumerate(unique_sizes)}
for s in unique_sizes:
    mask = np.array(val_atom_counts) == s
    ax.scatter(val_t[mask], val_p[mask], alpha=0.6, s=22,
               label=f"n={s}", color=size_color[s], edgecolors="none")
lims8 = [min(val_t.min(), val_p.min()) - 0.1, max(val_t.max(), val_p.max()) + 0.1]
ax.plot(lims8, lims8, "k--", linewidth=1.2)
ax.set_xlabel("Actual RMSD (Å)"); ax.set_ylabel("Predicted RMSD (Å)")
ax.set_title("Parity Plot Coloured by Molecule Size — GNN v1")
ax.set_xlim(lims8); ax.set_ylim(lims8); ax.grid(True, alpha=0.3)
if len(unique_sizes) <= 15:
    ax.legend(fontsize=7, ncol=2, title="Atom count")
plt.tight_layout()
savefig("plot8_parity_by_mol.jpg")

# Plot 9 — t-SNE of GNN embeddings
try:
    from sklearn.manifold import TSNE
    embeddings = []
    model.eval()
    with torch.no_grad():
        for g1, g2, rmsd in val_data:
            h1 = model.encoder(g1).numpy()
            h2 = model.encoder(g2).numpy()
            embeddings.append(np.abs(h1 - h2))
    embeddings = np.array(embeddings)

    if len(embeddings) >= 10:
        perp   = min(30, len(embeddings) - 1)
        tsne   = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        emb_2d = tsne.fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                        c=val_t, cmap="plasma", alpha=0.7, s=25, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Actual RMSD (Å)")
        ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        ax.set_title("t-SNE of GNN v1 Embeddings\n(coloured by actual RMSD)")
        ax.grid(True, alpha=0.2); plt.tight_layout()
        savefig("plot9_tsne_embeddings.jpg")
    else:
        print("Skipping t-SNE: not enough validation samples.")
except Exception as e:
    print(f"Skipping t-SNE plot: {e}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")

