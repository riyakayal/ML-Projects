# Author: Riya Kayal
# Created: 15/11/2025
# Fixed: 03/02/2026

## Conformers were generated using RDKit ETKDGv3 and optimized with MMFF94 prior to RMSD calculation.

#------------------------------------------------------------------
# 1. import csv generated using RDKit
#------------------------------------------------------------------
import pandas as pd

df = pd.read_csv("a.csv")
#df = pd.read_csv("1000_molecules_multiple_conformers.csv")

## set number of molecules
n_mols = 100
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
# The original code hardcoded max_atoms=29. Molecules in a.csv
# (e.g. Molecule_8, a steroid; Molecule_9, a nucleotide) have
# significantly more atoms with hydrogens added, causing the
# padded distance matrix to be undersized. This silently
# truncated atom positions, corrupting the input to the MLP.
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
# instead of hardcoded to 29.
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

#--------------------------------------------------------------
# 9. Train
# FIX 1 (continued): Pass computed max_atoms to model constructor.
#--------------------------------------------------------------
model = RMSDPredictor(max_atoms=max_atoms)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# n_range = 100
n_range = 25
for epoch in range(n_range):
    total_loss = 0

    for pos1, pos2, rmsd in dataset:

        pred = model(pos1, pos2).squeeze()

        loss = criterion(pred, rmsd)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: {total_loss:.4f}")


#--------------------------------------------------------------
# 10. Evaluation Function 
#--------------------------------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

def evaluate(model, dataset):
    preds = []
    targets = []

    with torch.no_grad():
        for pos1, pos2, rmsd in dataset:
            pred = model(pos1, pos2).item()
            preds.append(pred)
            targets.append(rmsd.item())

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    spearman_corr, _ = spearmanr(targets, preds)


    print("======================================")
    print("              Evaluation")
    print("======================================")
    print(f"MAE: {mae:.4f} Å")
    print(f"RMSE: {rmse:.4f} Å")
    print(f"R²: {r2:.4f}")
    print(f"Spearman: {spearman_corr:.4f}")
    print("======================================")

evaluate(model, dataset)
