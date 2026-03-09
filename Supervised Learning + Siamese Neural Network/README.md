# An RMSD prediction code for organic molecules with multiple conformers

## Requirements:
Based on the imports throughout the scripts, here are all the requirements:
```bash
pip install pandas numpy torch rdkit scikit-learn scipy matplotlib psutil
```
If you're on a system where RDKit isn't available via pip directly (older environments), use:
```bash
pip install pandas numpy torch scikit-learn scipy matplotlib psutil
conda install -c conda-forge rdkit
```
And if you're on a **shared HPC cluster** (for path like `/usr/users/username/miniconda3/`), you likely need:
```bash
pip install pandas numpy torch scikit-learn scipy matplotlib psutil --break-system-packages
conda install -c conda-forge rdkit
```

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
python predict_rmsd_with_plot.py --input 1000_molecules_unique.csv

# with custom molecule count
python predict_rmsd_with_plot.py --input 1000_molecules_unique.csv --n_mols 200

```

###### If you pass a file that doesn't exist, it exits immediately with a clear error rather than crashing later. The `--input` argument is required — omitting it prints a usage message automatically:

```
usage: predict_rmsd_with_plot.py --input INPUT [--n_mols N_MOLS]
predict_rmsd_with_plot.py: error: the following arguments are required: --input
```
The outputs will be in the ***output*** directory.
###### NOTE: Look into the python files to change defaults as needed. predict_rmsd_with_plot.py is Open MP parallelised.

## The ML approach
<p>The code uses <strong>supervised regression</strong> with a <strong>Siamese neural network</strong> architecture. Here's the explanation:</p>

* <p><strong>Siamese Network</strong> — the core idea is that both conformers are passed through the <em>same</em> encoder (<code>DistanceEncoder</code>) with shared weights, and then the difference between their embeddings is used to predict RMSD. This is a classic Siamese setup, originally designed for similarity learning.</p>

* <p><strong>The encoder is an MLP (Multi-Layer Perceptron)</strong> — it takes a flattened upper-triangle of the pairwise distance matrix as input (a rotation/translation-invariant representation of 3D structure) and compresses it to a 64-dim embedding.</p>

* <p><strong>The regressor is also an MLP</strong> — it takes <code>|h1 - h2|</code> (element-wise absolute difference of the two embeddings) and outputs a single scalar: the predicted RMSD in Ångströms.</p>

* <p><strong>Training signal</strong> — MSE loss against RDKit-computed RMSD values, so it's fully supervised.</p>

---

<p>A few things worth knowing about this specific approach:</p>

* <p>The <strong>distance matrix input</strong> makes it invariant to rotation and translation, which is the right inductive bias for molecular geometry — but it loses all <em>directionality</em> (two mirror-image conformers would look identical to the encoder).</p>

* <p>It's a relatively <strong>simple baseline</strong>. More modern approaches for conformer comparison use graph neural networks (GNNs) or equivariant networks (like SchNet, DimeNet, or EGNN) that operate directly on atom graphs rather than flattened distance matrices.</p>

* <p>The <strong>Siamese design</strong> is a good fit here because RMSD is symmetric — <code>RMSD(A,B) == RMSD(B,A)</code> — and shared encoder weights enforce that naturally.</p>

