
# An RMSD prediction code for organic molecules with multiple conformers
## Uses Graph Neural Network


## To run:
1) First generate n no. of unique organic molecules using RDKit.
   This generates two files n_molecules.csv and n_molecules_unique.csv.
```

python molecules_gen.py
```
2) Use the n_molecules_unique.csv (e.g. 1000_molecules_unique.csv) as the input file for the next job.
   Create conformer set, calculate RMSD, train, evaluate and finally print and store results.. 
```   
# basic usage 
python predict_rmsd_gnn.py --input 1000_molecules_unique.csv

# with custom molecule count
python predict_rmsd_gnn.py --input 1000_molecules_unique.csv --n_mols 200

```

###### If you pass a file that doesn't exist, it exits immediately with a clear error rather than crashing later. The `--input` argument is required — omitting it prints a usage message automatically:

```
usage: predict_rmsd_gnn.py --input INPUT [--n_mols N_MOLS]
predict_rmsd_gnn.py: error: the following arguments are required: --input
```
The outputs will be in the ***output*** directory.
###### NOTE: Look into the python files to change defaults as needed. predict_rmsd_with_plot.py is Open MP parallelised.

**What changed vs `predict_rmsd_with_plot.py`** — only sections 3, 4, 7, and 8. Everything else (training loop, early stopping, CPU tracking, evaluation, all 9 plots, summary file) is identical.

| | MLP baseline | GNN version |
|---|---|---|
| Input representation | Flattened distance matrix (fixed size, padded) | Molecular graph (nodes=atoms, edges=bonds) |
| Node features | None (geometry only) | Atom type, degree, aromaticity (13-dim) |
| Edge features | None | Interatomic distance (1-dim) |
| `max_atoms` needed | Yes — hardcoded/padded | No — graphs are variable size naturally |
| Encoder | 3-layer MLP | 3-layer EdgeConv message passing + mean pooling |
| Siamese head | `abs(h1−h2)` → MLP → scalar | Same |
| No. of new dependencies | — | None — pure PyTorch, no PyG needed |

The GNN is implemented from scratch using plain PyTorch (`EdgeConvLayer`) so you don't need `torch_geometric` or any extra install beyond what you already have.

