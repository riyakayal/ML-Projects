
# An RMSD prediction code for organic molecules with multiple conformers
## Uses Graph Neural Network
#### NOTE: Must give at least 500 molecules.

## To run:
1) First generate n no. of unique organic molecules using RDKit.
   This generates two files n_molecules.csv and n_molecules_unique.csv.
```

python molecules_gen.py
```
2a) Use the n_molecules_unique.csv (e.g. 1000_molecules_unique.csv) as the input file for the next job.
   Create conformer set, calculate RMSD, train, evaluate and finally print and store results.. 
```   
# basic usage 
python predict_rmsd_gnn_v1.py --input 1000_molecules_unique.csv

# with custom molecule count
python predict_rmsd_gnn_v1.py --input 1000_molecules_unique.csv --n_mols 200
```

2b) advanced code with dropout option, faster 
```
# 99 mols, default dropout 0.1
python predict_rmsd_gnn_v2.py --input 1000_molecules_unique.csv --n_mols 99

# 200 mols, same dropout
python predict_rmsd_gnn_v2.py --input 1000_molecules_unique.csv --n_mols 200

# 500 mols, stronger dropout for larger dataset
python predict_rmsd_gnn_v2.py --input 1000_molecules_unique.csv --n_mols 500 --dropout 0.2

# No dropout (revert to original behaviour)
python predict_rmsd_gnn_v2.py --input 1000_molecules_unique.csv --n_mols 99 --dropout 0.0

```

###### If you pass a file that doesn't exist, it exits immediately with a clear error rather than crashing later. The `--input` argument is required — omitting it prints a usage message automatically:

```
usage: predict_rmsd_gnn_v1.py --input INPUT [--n_mols N_MOLS]
predict_rmsd_gnn_v1.py: error: the following arguments are required: --input
```
The outputs will be in the ***output_\**** directory.
###### NOTE: Look into the python files to change defaults as needed. predict_rmsd_with_plot.py is Open MP parallelised.

**What changed vs `predict_rmsd_with_plot.py`**?
If we want a basic implementation (predict_rmsd_gnn_v0.py), only sections 3, 4, 7, and 8 will change wrt predict_rmsd_with_plot.py.  Everything else (training loop, early stopping, CPU tracking, evaluation, all 9 plots, summary file) will be identical.

| | MLP baseline | GNN version (basic) |
|---|---|---|
| Input representation | Flattened distance matrix (fixed size, padded) | Molecular graph (nodes=atoms, edges=bonds) |
| Node features | None (geometry only) | Atom type, degree, aromaticity (13-dim) |
| Edge features | None | Interatomic distance (1-dim) |
| `max_atoms` needed | Yes — hardcoded/padded | No — graphs are variable size naturally |
| Encoder | 3-layer MLP | 3-layer EdgeConv message passing + mean pooling |
| Siamese head | `abs(h1−h2)` → MLP → scalar | Same |
| No. of new dependencies | — | None — pure PyTorch, no PyG needed |

The GNN can be implemented from scratch using plain PyTorch (`EdgeConvLayer`) so you don't need `torch_geometric` or any extra install beyond what you already have.
However, the caveat is that, the performance is poor with such a basic code. So we add improvements.

# Siamese GNN RMSD Predictor v2 — Code Summary


---

## Purpose

Predicts the RMSD (Root Mean Square Deviation, Å) between pairs of molecular conformers using a Siamese Graph Neural Network. Each conformer is encoded independently into a 64-dim embedding; the two embeddings are then combined in a regression head to predict their pairwise RMSD. Conformers are generated and optimized with RDKit prior to training.

---

## Usage

```bash
# Minimal
python predict_rmsd_gnn_v2.py --input molecules.csv

# With molecule count and dropout override
python predict_rmsd_gnn_v2.py --input molecules.csv --n_mols 200 --dropout 0.2
```

The input CSV must contain a `smiles` column. All other settings are controlled by constants at the top of the script.

---

## Settings Block

| Setting | Default | Effect |
|---|---|---|
| `N_CONFS` | `10` | Conformers generated per molecule; 10 confs → 45 pairs/mol |
| `RMSD_MIN` | `0.1` Å | Drop conformer pairs below this RMSD — removes degenerate collapsed pairs |
| `LOG_RMSD` | `True` | Predict log(RMSD) to compress the right-skewed distribution; exp()-transformed back at evaluation |
| `NONBOND_CUTOFF` | `5.0` Å | Atom pairs within this distance are connected with non-bonded edges; set `None` for bonded-only |
| `DROPOUT` | `0.1` | Dropout rate applied throughout the GNN and regressor head; overrideable via `--dropout` |
| `target_loss` | `None` | Optional early-stopping loss threshold; `None` = patience-based only |

---

## Pipeline Overview

### 1. Conformer Generation
Each SMILES is embedded into `N_CONFS=10` 3D conformers using RDKit's `ETKDGv3` algorithm with a fixed random seed, then geometry-optimized with MMFF94. All pairwise RMSD values between conformers of the same molecule are computed with `rdMolAlign.AlignMol`. Pairs below `RMSD_MIN` are discarded as degenerate.

### 2. Graph Construction
Each conformer is converted to a graph with:

- **Node features** (25-dim per atom):

| Dimension | Feature |
|---|---|
| 0–10 | One-hot atom type (C, N, O, S, F, Cl, Br, I, P, H, other) |
| 11 | Normalised degree |
| 12 | Is aromatic |
| 13 | Normalised formal charge |
| 14–17 | Hybridisation one-hot (SP / SP2 / SP3 / other) |
| 18 | Normalised H count |
| 19 | In any ring |
| 20–24 | In ring of size 3 / 4 / 5 / 6 / 7 (multi-hot) |

- **Edge features** (17-dim per edge):
  - 16-bin Gaussian RBF expansion of interatomic distance (centres spaced 0.5–8.0 Å, γ=0.5)
  - 1 binary `is_bonded` flag distinguishing covalent bonds from non-bonded through-space edges

- **Non-bonded edges**: all atom pairs within `NONBOND_CUTOFF=5.0 Å` that are not covalently bonded receive an edge. This is the key architectural improvement — a ring flip or chain rotation changes many non-bonded distances while changing no bond distances, so without these edges the GNN cannot distinguish such conformers.

### 3. Label Transform
If `LOG_RMSD=True`, training labels are stored as log(RMSD). All predictions are exp()-transformed back to Å before metric calculation, so MAE/RMSE/R²/Spearman are always reported in Ångströms.

---

## Model Architecture

### EdgeConvLayer (message passing block)
```
h_i' = Dropout( ReLU( W_self · h_i  +  W_neigh · AGG_j[ MLP_edge(e_ij) · h_j ] ) )
```
The edge MLP gates each neighbour's message by the edge feature vector, so the
aggregation is distance- and bond-type-aware.

### GNNEncoder
```
Input (25-dim) → Linear → Dropout → ReLU
               → EdgeConvLayer × 2 (with dropout)
               → EdgeConvLayer × 1 (no dropout on final layer)
               → Global Mean Pooling → 64-dim embedding
```
Supports both single-graph (evaluation) and batched (training) forward passes via a `batch` index vector.

### RMSDPredictor (Siamese head)
```
g1, g2  →  GNNEncoder  →  h1, h2  (64-dim each)
combined = cat(h1, h2, |h1−h2|, h1⊙h2)   →  256-dim
         → Linear(256→128) → ReLU → Dropout
         → Linear(128→64)  → ReLU
         → Linear(64→1)    → scalar RMSD prediction
```

---

## Training

| Setting | Value |
|---|---|
| Loss function | `HuberLoss(delta=1.0)` — less sensitive to outlier pairs than MSE |
| Optimiser | Adam, lr=1e-3 |
| LR scheduler | `ReduceLROnPlateau` (patience=10, factor=0.5) |
| Early stopping | patience=20 epochs |
| Max epochs | 200 |
| Batch size | 32 pairs |
| Train / val split | 80 / 20 |

**Batched forward pass:** all graphs in a batch are concatenated into one large disconnected graph with a `batch` index vector. A single matrix multiply processes the whole batch, then `index_add_` scatter-pools per graph. This replaced a slow per-sample Python for-loop and reduces epoch time by ~5–10×.

---

## Evaluation Metrics

Reported on both validation and train sets in Å-space (after exp()-transform):

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of determination
- **Spearman ρ** — Rank correlation (robust to non-linearity)

---

## Scaling Guide

| `--n_mols` | Expected val R² | Approx. runtime |
|---|---|---|
| 99 | ~0.64 | ~20 min |
| 200 | ~0.68–0.72 | ~45 min |
| 500 | ~0.72–0.80 | ~110 min |

Recommended dropout by scale:

- `--dropout 0.1` for 99–200 mols (default)
- `--dropout 0.2` for 200–500 mols
- `--dropout 0.0` only if val R² < train R² by less than 0.05

---

## Output Files

All outputs are saved to `output_gnn_v2/`:

| File | Description |
|---|---|
| `plot1_loss_curve.jpg` | Train vs validation Huber loss per epoch |
| `plot2_lr_schedule.jpg` | Learning rate schedule (log scale) |
| `plot3_pred_vs_actual.jpg` | Predicted vs actual RMSD — parity plot (val set) |
| `plot4_residuals.jpg` | Residuals vs actual RMSD (val set) |
| `plot5_residual_hist.jpg` | Residual distribution (val set) |
| `plot6_error_by_mol_group.jpg` | Absolute error boxplot by atom-count group |
| `plot7_error_vs_mol_size.jpg` | Absolute error vs atom count, coloured by RMSD |
| `plot8_parity_by_mol.jpg` | Parity plot coloured by molecule size |
| `plot9_tsne_embeddings.jpg` | t-SNE of GNN embeddings, coloured by actual RMSD |
| `summary.txt` | Full run summary: settings, metrics, runtime, hardware |

---

## Dependencies

```
pandas
numpy
torch
rdkit
scikit-learn
scipy
matplotlib
psutil
```


