

# Author: Riya Kayal
# Created: 10/11/2025


import pandas as pd
import numpy as np
import os
import datetime
import time
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# ------------------------------------------------------------------
# OUTPUT FOLDER
# ------------------------------------------------------------------
OUTPUT_DIR = "output_krr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_DPI = 300
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "figure.dpi": SAVE_DPI})

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=SAVE_DPI, format="jpeg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ------------------------------------------------------------------
# 1. Data Loading
# ------------------------------------------------------------------
f = open('homolumo.csv', "r")
smiles, gaps = [], []
skipped = 0
for line in f.readlines()[1:]:
    line = line.split()
    if line[1].count("<"):   # skip problematic SMILES (e.g. line 269)
        skipped += 1
        continue
    smiles.append(line[1])
    gaps.append(float(line[2]))

gaps = np.array(gaps, dtype=float)
print(f"Loaded {len(smiles)} molecules  |  Skipped: {skipped}")
print(f"HOMO-LUMO gap range: {gaps.min():.4f} – {gaps.max():.4f} eV  |  Mean: {gaps.mean():.4f} eV")

# ------------------------------------------------------------------
# 2. Data Distribution Plot  (equivalent to rmsd_dist.jpg)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(gaps, bins=20, color="#2563EB", edgecolor="white", linewidth=0.6)
axes[0].axvline(gaps.mean(), color="#DC2626", linewidth=1.5, linestyle="--",
                label=f"Mean: {gaps.mean():.3f} eV")
axes[0].axvline(np.median(gaps), color="#059669", linewidth=1.5, linestyle="-.",
                label=f"Median: {np.median(gaps):.3f} eV")
axes[0].set_xlabel("HOMO-LUMO Gap (eV)")
axes[0].set_ylabel("Count")
axes[0].set_title("HOMO-LUMO Gap Distribution")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(gaps, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#2563EB", alpha=0.6),
                medianprops=dict(color="#DC2626", linewidth=2))
axes[1].set_ylabel("HOMO-LUMO Gap (eV)")
axes[1].set_title("Box Plot of HOMO-LUMO Gaps")
axes[1].grid(True, alpha=0.3, axis="y")

plt.suptitle("Data Distribution — HOMO-LUMO Gap", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("plot0_data_distribution.jpg")

# ------------------------------------------------------------------
# 3. Feature Extraction (Mordred)
# ------------------------------------------------------------------
print("\nComputing Mordred descriptors...")
t0 = time.time()
calc = Calculator(descriptors, ignore_3D=True)
mols = [Chem.MolFromSmiles(s) for s in smiles]
df_desc = calc.pandas(mols, quiet=True)
df_desc = df_desc.apply(pd.to_numeric, errors='coerce').fillna(0)

# Clip extreme values to prevent numpy overflow in StandardScaler/KRR.
# Mordred can produce astronomically large descriptor values for some
# molecules; values beyond ±1e6 are numerically meaningless and cause
# overflow warnings during reduce operations.
df_desc = df_desc.clip(lower=-1e6, upper=1e6)

# Drop any columns that are constant after clipping (zero variance)
# — these contribute nothing to the model and can cause scaler issues.
df_desc = df_desc.loc[:, df_desc.std() > 0]
print(f"Descriptors after cleaning: {df_desc.shape[1]}")
descriptor_time = time.time() - t0
print(f"Descriptors computed in {descriptor_time:.1f}s  |  Shape: {df_desc.shape}")

X = df_desc.values
y = gaps

# ------------------------------------------------------------------
# 4. Train / Test Split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ------------------------------------------------------------------
# 5. Quick KRR baseline (fixed alpha/gamma)
# ------------------------------------------------------------------
print("\nFitting baseline KRR...")
t1 = time.time()
scaler_base = StandardScaler()
X_train_sc  = scaler_base.fit_transform(X_train)
X_test_sc   = scaler_base.transform(X_test)

krr_base = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)
krr_base.fit(X_train_sc, y_train)
y_pred_base = krr_base.predict(X_test_sc)
baseline_time = time.time() - t1

mae_base  = mean_absolute_error(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
r2_base   = r2_score(y_test, y_pred_base)
sp_base, _= spearmanr(y_test, y_pred_base)

print(f"  Baseline MAE:  {mae_base:.4f} eV")
print(f"  Baseline RMSE: {rmse_base:.4f} eV")
print(f"  Baseline R²:   {r2_base:.4f}")

# ------------------------------------------------------------------
# 6. GridSearchCV for optimal alpha & gamma
# ------------------------------------------------------------------
print("\nRunning GridSearchCV...")
t2 = time.time()
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('krr', KernelRidge(kernel='rbf'))
])

param_grid = {
    'krr__alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0],
    'krr__gamma': np.logspace(-4, -1, 5)
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - t2
total_time = time.time() - t0

best_params  = grid_search.best_params_
best_cv_mae  = -grid_search.best_score_
best_model   = grid_search.best_estimator_
y_pred_best  = best_model.predict(X_test)

mae_best   = mean_absolute_error(y_test, y_pred_best)
rmse_best  = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best    = r2_score(y_test, y_pred_best)
sp_best, _ = spearmanr(y_test, y_pred_best)
residuals  = y_pred_best - y_test

print("\n#==================================================================#")
print(f"  Best Parameters : {best_params}")
print(f"  Best CV MAE     : {best_cv_mae:.4f} eV")
print(f"  Test MAE        : {mae_best:.4f} eV")
print(f"  Test RMSE       : {rmse_best:.4f} eV")
print(f"  Test R²         : {r2_best:.4f}")
print(f"  Test Spearman   : {sp_best:.4f}")
print("#==================================================================#\n")

# ------------------------------------------------------------------
# 7. Summary file
# ------------------------------------------------------------------
summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 55 + "\n")
    f.write("   KRR HOMO-LUMO Gap Predictor — Run Summary\n")
    f.write("=" * 55 + "\n")
    f.write(f"Run date/time         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input file            : homolumo.csv\n")
    f.write(f"Molecules loaded      : {len(smiles)}\n")
    f.write(f"Molecules skipped     : {skipped}\n")
    f.write(f"\n--- Data Statistics ---\n")
    f.write(f"Gap min               : {gaps.min():.4f} eV\n")
    f.write(f"Gap max               : {gaps.max():.4f} eV\n")
    f.write(f"Gap mean              : {gaps.mean():.4f} eV\n")
    f.write(f"Gap std               : {gaps.std():.4f} eV\n")
    f.write(f"Gap median            : {np.median(gaps):.4f} eV\n")
    f.write(f"\n--- Features ---\n")
    f.write(f"Descriptor type       : Mordred (2D, ignore_3D=True)\n")
    f.write(f"Number of descriptors : {df_desc.shape[1]}\n")
    f.write(f"Descriptor calc time  : {descriptor_time:.1f}s\n")
    f.write(f"\n--- Train / Test Split ---\n")
    f.write(f"Train size            : {len(X_train)}\n")
    f.write(f"Test size             : {len(X_test)}\n")
    f.write(f"Test fraction         : 0.2\n")
    f.write(f"\n--- Baseline KRR (alpha=0.1, gamma=0.01) ---\n")
    f.write(f"MAE                   : {mae_base:.4f} eV\n")
    f.write(f"RMSE                  : {rmse_base:.4f} eV\n")
    f.write(f"R²                    : {r2_base:.4f}\n")
    f.write(f"Spearman              : {sp_base:.4f}\n")
    f.write(f"Fit time              : {baseline_time:.1f}s\n")
    f.write(f"\n--- GridSearchCV (5-fold CV) ---\n")
    f.write(f"alpha grid            : [1e-4, 1e-3, 1e-2, 0.1, 1.0]\n")
    f.write(f"gamma grid            : logspace(-4, -1, 5)\n")
    f.write(f"Best alpha            : {best_params['krr__alpha']}\n")
    f.write(f"Best gamma            : {best_params['krr__gamma']:.6f}\n")
    f.write(f"Best CV MAE           : {best_cv_mae:.4f} eV\n")
    f.write(f"Grid search time      : {grid_time:.1f}s\n")
    f.write(f"\n--- Best Model — Test Set Metrics ---\n")
    f.write(f"MAE                   : {mae_best:.4f} eV\n")
    f.write(f"RMSE                  : {rmse_best:.4f} eV\n")
    f.write(f"R²                    : {r2_best:.4f}\n")
    f.write(f"Spearman              : {sp_best:.4f}\n")
    f.write(f"\n--- Runtime ---\n")
    f.write(f"Total time            : {total_time:.1f}s ({total_time/60:.2f} min)\n")
    f.write("=" * 55 + "\n")
print(f"Saved: {summary_path}")

# ------------------------------------------------------------------
# 8. Plots
# ------------------------------------------------------------------

# Plot 1 — Predicted vs Actual (best model)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred_best, alpha=0.7, s=40, color="#2563EB", edgecolors="none")
lims = [min(y_test.min(), y_pred_best.min()) - 0.1,
        max(y_test.max(), y_pred_best.max()) + 0.1]
ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")
ax.set_xlabel("Actual HOMO-LUMO Gap (eV)")
ax.set_ylabel("Predicted HOMO-LUMO Gap (eV)")
ax.set_title("Predicted vs Actual — Best KRR (Test Set)")
ax.legend(); ax.set_xlim(lims); ax.set_ylim(lims); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot1_pred_vs_actual.jpg")

# Plot 2 — Residuals vs Actual
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, residuals, alpha=0.7, s=40, color="#DC2626", edgecolors="none")
ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_xlabel("Actual HOMO-LUMO Gap (eV)")
ax.set_ylabel("Residual (Predicted − Actual) (eV)")
ax.set_title("Residual Plot — Best KRR (Test Set)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot2_residuals.jpg")

# Plot 3 — Residual Distribution
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(residuals, bins=15, color="#059669", edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.axvline(residuals.mean(), color="#DC2626", linewidth=1.5,
           label=f"Mean: {residuals.mean():.4f} eV")
ax.set_xlabel("Residual (eV)"); ax.set_ylabel("Count")
ax.set_title("Residual Distribution — Best KRR (Test Set)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot3_residual_hist.jpg")

# Plot 4 — Baseline vs Best model comparison (bar chart)
fig, ax = plt.subplots(figsize=(7, 5))
metrics  = ["MAE (eV)", "RMSE (eV)", "R²", "Spearman"]
baseline = [mae_base, rmse_base, r2_base, sp_base]
best     = [mae_best, rmse_best, r2_best, sp_best]
x        = np.arange(len(metrics))
width    = 0.35
bars1 = ax.bar(x - width/2, baseline, width, label="Baseline (fixed α,γ)",
               color="#94A3B8", edgecolor="white")
bars2 = ax.bar(x + width/2, best,     width, label="Best (GridSearchCV)",
               color="#2563EB", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_title("Baseline vs Tuned KRR — Test Set Metrics")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
savefig("plot4_baseline_vs_best.jpg")

# Plot 5 — GridSearchCV heatmap (alpha vs gamma, mean CV MAE)
cv_results = pd.DataFrame(grid_search.cv_results_)
alphas  = param_grid['krr__alpha']
gammas  = param_grid['krr__gamma']
mae_mat = np.zeros((len(alphas), len(gammas)))
for i, a in enumerate(alphas):
    for j, g in enumerate(gammas):
        mask = ((cv_results['param_krr__alpha'] == a) &
                (cv_results['param_krr__gamma'].astype(float).round(8) == round(g, 8)))
        if mask.any():
            mae_mat[i, j] = -cv_results.loc[mask, 'mean_test_score'].values[0]

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(mae_mat, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, label="CV MAE (eV)")
ax.set_xticks(range(len(gammas)))
ax.set_xticklabels([f"{g:.1e}" for g in gammas], rotation=45, ha="right")
ax.set_yticks(range(len(alphas)))
ax.set_yticklabels([str(a) for a in alphas])
ax.set_xlabel("gamma"); ax.set_ylabel("alpha")
ax.set_title("GridSearchCV — Mean CV MAE Heatmap")
for i in range(len(alphas)):
    for j in range(len(gammas)):
        ax.text(j, i, f"{mae_mat[i,j]:.3f}", ha="center", va="center",
                fontsize=7, color="black")
plt.tight_layout()
savefig("plot5_gridsearch_heatmap.jpg")

# Plot 6 — Absolute error per test molecule
abs_errors = np.abs(residuals)
mol_indices = np.arange(len(y_test))
fig, ax = plt.subplots(figsize=(10, 4))
colors_err = get_cmap("RdYlGn_r")(abs_errors / abs_errors.max())
bars = ax.bar(mol_indices, abs_errors, color=colors_err, edgecolor="none")
ax.axhline(abs_errors.mean(), color="#2563EB", linewidth=1.5, linestyle="--",
           label=f"Mean |error|: {abs_errors.mean():.4f} eV")
ax.set_xlabel("Test molecule index")
ax.set_ylabel("|Residual| (eV)")
ax.set_title("Absolute Error per Test Molecule")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
savefig("plot6_per_molecule_error.jpg")

# Plot 7 — Actual gap vs absolute error (does error correlate with gap size?)
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(y_test, abs_errors, c=abs_errors, cmap="plasma",
                alpha=0.7, s=40, edgecolors="none")
plt.colorbar(sc, ax=ax, label="|Error| (eV)")
ax.set_xlabel("Actual HOMO-LUMO Gap (eV)")
ax.set_ylabel("|Residual| (eV)")
ax.set_title("Absolute Error vs Gap Size (Test Set)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot7_error_vs_gap_size.jpg")

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
