

# Author: Riya Kayal
# Created: 16/12/2025

import pandas as pd
import numpy as np
import os
import datetime
import time
import warnings
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
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
    print(f"      Saved: {path}")

# ------------------------------------------------------------------
# IMPROVEMENT SETTINGS  ← tune these at the top
# ------------------------------------------------------------------
LOG_TRANSFORM   = True    # predict log(gap) instead of gap directly
OUTLIER_CAP_PCT = 95      # cap training labels at this percentile (None = no cap)
K_FEATURES      = 200     # number of best Mordred features to keep (None = keep all)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 1. Data Loading
# ------------------------------------------------------------------
print("[1/8] Loading data...")
f = open('homolumo.csv', "r")
smiles, gaps = [], []
skipped = 0
for line in f.readlines()[1:]:
    parts = line.split()
    if len(parts) < 3:
        continue
    if parts[1].count("<"):
        skipped += 1
        continue
    smiles.append(parts[1])
    gaps.append(float(parts[2]))

gaps = np.array(gaps, dtype=float)
mols = [Chem.MolFromSmiles(s) for s in smiles]
print(f"      {len(smiles)} molecules loaded  |  {skipped} skipped")
print(f"      Gap range: {gaps.min():.3f} – {gaps.max():.3f} eV  "
      f"|  Mean: {gaps.mean():.3f}  Std: {gaps.std():.3f}  Median: {np.median(gaps):.3f} eV")

# ------------------------------------------------------------------
# 2. Data Distribution Plot
# ------------------------------------------------------------------
print("\n[2/8] Generating data distribution plot...")
cap_val = np.percentile(gaps, OUTLIER_CAP_PCT) if OUTLIER_CAP_PCT else None

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(gaps, bins=30, color="#2563EB", edgecolor="white", linewidth=0.6)
axes[0].axvline(gaps.mean(),    color="#DC2626", linewidth=1.5, linestyle="--",
                label=f"Mean: {gaps.mean():.3f} eV")
axes[0].axvline(np.median(gaps), color="#059669", linewidth=1.5, linestyle="-.",
                label=f"Median: {np.median(gaps):.3f} eV")
if cap_val:
    axes[0].axvline(cap_val, color="#F59E0B", linewidth=1.5, linestyle=":",
                    label=f"{OUTLIER_CAP_PCT}th pct cap: {cap_val:.3f} eV")
axes[0].set_xlabel("HOMO-LUMO Gap (eV)"); axes[0].set_ylabel("Count")
axes[0].set_title("HOMO-LUMO Gap Distribution"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].boxplot(gaps, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#2563EB", alpha=0.6),
                medianprops=dict(color="#DC2626", linewidth=2),
                flierprops=dict(marker="o", markerfacecolor="#DC2626", markersize=4, alpha=0.5))
axes[1].set_ylabel("HOMO-LUMO Gap (eV)"); axes[1].set_title("Box Plot"); axes[1].grid(True, alpha=0.3, axis="y")

plt.suptitle("Data Distribution — HOMO-LUMO Gap", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("plot0_data_distribution.jpg")

# ------------------------------------------------------------------
# 3. Feature Extraction (Mordred)
# ------------------------------------------------------------------
print("\n[3/8] Computing Mordred descriptors...")
t0 = time.time()
calc = Calculator(descriptors, ignore_3D=True)

clean_mols, clean_smiles, clean_gaps = [], [], []
for idx, (smi, gap, mol) in enumerate(zip(smiles, gaps, mols)):
    print(f"      Sanitizing {idx+1}/{len(smiles)}: {smi[:50]}", end="\r")
    if mol is None:
        print(f"\n      WARNING: Invalid SMILES skipped: {smi}")
        continue
    try:
        Chem.SanitizeMol(mol)
        clean_mols.append(mol)
        clean_smiles.append(smi)
        clean_gaps.append(gap)
    except Exception as e:
        print(f"\n      WARNING: Sanitization failed for {smi}: {e}")

print(f"\n      {len(clean_mols)} molecules after sanitization")
smiles = clean_smiles
gaps   = np.array(clean_gaps, dtype=float)

print("      Running Mordred...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    df_desc = calc.pandas(clean_mols, quiet=True)

df_desc = df_desc.apply(pd.to_numeric, errors='coerce').fillna(0)
df_desc = df_desc.clip(lower=-1e6, upper=1e6)
df_desc = df_desc.loc[:, df_desc.std() > 0]
descriptor_time = time.time() - t0
print(f"      Done in {descriptor_time:.1f}s  |  {df_desc.shape[1]} descriptors after cleaning")

X = df_desc.values
y = gaps

# ------------------------------------------------------------------
# 4. Stratified Train / Test Split
# IMPROVEMENT: bin gap values into 10 quantile-based strata so that
# every region of the gap distribution (including the sparse tails)
# is proportionally represented in both train and test sets.
# ------------------------------------------------------------------
print("\n[4/8] Stratified train/test split...")
n_strata  = min(10, len(y) // 20)
y_bins    = pd.qcut(y, q=n_strata, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_bins)
print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}  |  Strata used: {n_strata}")

# ------------------------------------------------------------------
# 5. Outlier Capping
# IMPROVEMENT: molecules with very large gaps (>95th percentile) are
# chemically unusual and distort both the kernel and the loss.
# We cap their training labels at the 95th percentile value so the
# model is not penalised heavily for slightly underestimating outliers.
# Test labels are NOT capped — we evaluate on the full true distribution.
# ------------------------------------------------------------------
y_train_fit = y_train.copy()
if OUTLIER_CAP_PCT is not None:
    cap_val     = np.percentile(y_train, OUTLIER_CAP_PCT)
    n_capped    = (y_train_fit > cap_val).sum()
    y_train_fit = np.clip(y_train_fit, None, cap_val)
    print(f"\n[5/8] Outlier capping at {OUTLIER_CAP_PCT}th pct "
          f"({cap_val:.3f} eV)  →  {n_capped} training labels capped")
else:
    cap_val  = None
    n_capped = 0
    print("\n[5/8] Outlier capping: disabled")

# ------------------------------------------------------------------
# 6. Log-Transform Labels
# IMPROVEMENT: the right-skewed distribution makes regression harder.
# log(gap) compresses the tail and makes the target more symmetric.
# All predictions are exp()-transformed back to eV for evaluation.
# ------------------------------------------------------------------
if LOG_TRANSFORM:
    # guard: gaps must be > 0 (HOMO-LUMO gaps always are)
    y_train_fit = np.log(np.clip(y_train_fit, 1e-6, None))
    print(f"\n[6/8] Log-transform: ON  "
          f"|  log(gap) range: {y_train_fit.min():.3f} – {y_train_fit.max():.3f}")
else:
    print(f"\n[6/8] Log-transform: OFF")

# ------------------------------------------------------------------
# Helper: inverse-transform predictions back to eV
# ------------------------------------------------------------------
def to_ev(arr):
    return np.exp(arr) if LOG_TRANSFORM else arr

def compute_metrics(y_true, y_pred):
    mae      = mean_absolute_error(y_true, y_pred)
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    r2       = r2_score(y_true, y_pred)
    spear, _ = spearmanr(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spear}

# ------------------------------------------------------------------
# 7. Feature Selection
# IMPROVEMENT: SelectKBest with f_regression keeps the K descriptors
# most correlated with the (possibly log-transformed) target.
# Reducing from ~1200 to 200 features directly improves KRR kernel
# quality and speeds up GridSearch significantly.
# ------------------------------------------------------------------
print(f"\n[7/8] Feature selection (SelectKBest, K={K_FEATURES})...")
t_fs = time.time()
scaler_fs = StandardScaler()
X_train_sc = scaler_fs.fit_transform(X_train)
X_test_sc  = scaler_fs.transform(X_test)

if K_FEATURES and K_FEATURES < X_train_sc.shape[1]:
    selector   = SelectKBest(f_regression, k=K_FEATURES)
    X_train_fs = selector.fit_transform(X_train_sc, y_train_fit)
    X_test_fs  = selector.transform(X_test_sc)
    print(f"      {X_train_sc.shape[1]} → {X_train_fs.shape[1]} features  "
          f"in {time.time()-t_fs:.1f}s")
else:
    X_train_fs = X_train_sc
    X_test_fs  = X_test_sc
    selector   = None
    print(f"      Feature selection skipped (K_FEATURES=None)")

# ------------------------------------------------------------------
# 8. Models: Baseline KRR, Tuned KRR (extended grid + Laplacian),
#            Random Forest benchmark
# ------------------------------------------------------------------
print(f"\n[8/8] Training models...")

# ---- 8a. Baseline KRR ----
print("      [8a] Baseline KRR (alpha=0.1, gamma=0.01, rbf)...")
t1 = time.time()
krr_base        = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)
krr_base.fit(X_train_fs, y_train_fit)
y_pred_base_raw = krr_base.predict(X_test_fs)
y_pred_base     = to_ev(y_pred_base_raw)
baseline_time   = time.time() - t1
m_base          = compute_metrics(y_test, y_pred_base)
print(f"           Done in {baseline_time:.1f}s  |  MAE: {m_base['mae']:.4f}  "
      f"R²: {m_base['r2']:.4f}  Spearman: {m_base['spearman']:.4f}")

# ---- 8b. GridSearchCV — extended grid, both RBF and Laplacian kernels ----
# IMPROVEMENT: extended alpha/gamma range catches the optimal at the
# edge of the old grid. Adding Laplacian kernel often outperforms RBF
# for molecular descriptor data.
alphas_grid = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
gammas_grid = np.logspace(-5, 0, 7)
n_combos    = len(alphas_grid) * len(gammas_grid) * 2 * 5  # kernels x cv
print(f"\n      [8b] GridSearchCV  "
      f"({len(alphas_grid)} alphas × {len(gammas_grid)} gammas × 2 kernels × 5-fold "
      f"= {n_combos} fits)...")
print("           sklearn verbose output:")

t2 = time.time()
param_grid_ext = {
    'krr__kernel': ['rbf', 'laplacian'],
    'krr__alpha':  alphas_grid,
    'krr__gamma':  gammas_grid,
}
pipeline_gs = Pipeline([('krr', KernelRidge())])

# Use StratifiedKFold on binned y_train_fit so each fold sees the full range
y_train_bins = pd.qcut(y_train_fit, q=min(5, len(y_train_fit)//10),
                       labels=False, duplicates="drop")
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline_gs, param_grid_ext,
    cv=cv_strat.split(X_train_fs, y_train_bins),
    scoring='neg_mean_absolute_error',
    n_jobs=-1, verbose=1, refit=False
)
grid_search.fit(X_train_fs, y_train_fit)
grid_time = time.time() - t2

best_params  = grid_search.best_params_
best_cv_mae  = -grid_search.best_score_
print(f"\n           GridSearch done in {grid_time:.1f}s")
print(f"           Best params: {best_params}  |  CV MAE: {best_cv_mae:.4f} (log-space)")

# Refit best model manually (refit=False above avoids redundant refits)
best_krr = KernelRidge(kernel=best_params['krr__kernel'],
                       alpha=best_params['krr__alpha'],
                       gamma=best_params['krr__gamma'])
best_krr.fit(X_train_fs, y_train_fit)
y_pred_krr_raw = best_krr.predict(X_test_fs)
y_pred_krr     = to_ev(y_pred_krr_raw)
m_krr          = compute_metrics(y_test, y_pred_krr)
residuals_krr  = y_pred_krr - y_test
print(f"           Test  →  MAE: {m_krr['mae']:.4f}  RMSE: {m_krr['rmse']:.4f}  "
      f"R²: {m_krr['r2']:.4f}  Spearman: {m_krr['spearman']:.4f}")

# ---- 8c. Random Forest benchmark ----
print("\n      [8c] Random Forest benchmark (200 trees)...")
t3 = time.time()
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train_fs, y_train_fit)
y_pred_rf_raw = rf.predict(X_test_fs)
y_pred_rf     = to_ev(y_pred_rf_raw)
rf_time       = time.time() - t3
m_rf          = compute_metrics(y_test, y_pred_rf)
residuals_rf  = y_pred_rf - y_test
print(f"           Done in {rf_time:.1f}s  |  MAE: {m_rf['mae']:.4f}  "
      f"RMSE: {m_rf['rmse']:.4f}  R²: {m_rf['r2']:.4f}  Spearman: {m_rf['spearman']:.4f}")

total_time = time.time() - t0

# ------------------------------------------------------------------
# Summary to console
# ------------------------------------------------------------------
print("\n" + "="*60)
print("                  FINAL RESULTS")
print("="*60)
print(f"{'Model':<28} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Spearman':>9}")
print("-"*60)
print(f"{'Baseline KRR (rbf)':<28} {m_base['mae']:>7.4f} {m_base['rmse']:>7.4f} "
      f"{m_base['r2']:>7.4f} {m_base['spearman']:>9.4f}")
print(f"{'Tuned KRR ('+best_params['krr__kernel']+')':<28} {m_krr['mae']:>7.4f} "
      f"{m_krr['rmse']:>7.4f} {m_krr['r2']:>7.4f} {m_krr['spearman']:>9.4f}")
print(f"{'Random Forest':<28} {m_rf['mae']:>7.4f} {m_rf['rmse']:>7.4f} "
      f"{m_rf['r2']:>7.4f} {m_rf['spearman']:>9.4f}")
print("="*60)

# ------------------------------------------------------------------
# Summary file
# ------------------------------------------------------------------
summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("   KRR HOMO-LUMO Gap Predictor — Run Summary\n")
    f.write("=" * 60 + "\n")
    f.write(f"Run date/time         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input file            : homolumo.csv\n")
    f.write(f"Molecules loaded      : {len(smiles)}\n")
    f.write(f"Molecules skipped     : {skipped}\n")
    f.write(f"\n--- Data Statistics (raw gaps) ---\n")
    f.write(f"Gap min               : {gaps.min():.4f} eV\n")
    f.write(f"Gap max               : {gaps.max():.4f} eV\n")
    f.write(f"Gap mean              : {gaps.mean():.4f} eV\n")
    f.write(f"Gap std               : {gaps.std():.4f} eV\n")
    f.write(f"Gap median            : {np.median(gaps):.4f} eV\n")
    f.write(f"\n--- Improvements Applied ---\n")
    f.write(f"Log-transform labels  : {LOG_TRANSFORM}\n")
    f.write(f"Outlier cap pct       : {OUTLIER_CAP_PCT}th pct ({cap_val:.4f} eV)  → {n_capped} labels capped\n" if cap_val else f"Outlier cap           : disabled\n")
    f.write(f"Feature selection     : SelectKBest (k={K_FEATURES})\n")
    f.write(f"Stratified split      : Yes ({n_strata} strata)\n")
    f.write(f"Extended grid         : 7 alphas × 7 gammas × 2 kernels\n")
    f.write(f"RF benchmark          : Yes (200 trees)\n")
    f.write(f"\n--- Features ---\n")
    f.write(f"Descriptor type       : Mordred (2D, ignore_3D=True)\n")
    f.write(f"Raw descriptors       : {df_desc.shape[1]}\n")
    f.write(f"After SelectKBest     : {X_train_fs.shape[1]}\n")
    f.write(f"Descriptor calc time  : {descriptor_time:.1f}s\n")
    f.write(f"\n--- Train / Test Split ---\n")
    f.write(f"Train size            : {len(X_train)}\n")
    f.write(f"Test size             : {len(X_test)}\n")
    f.write(f"Test fraction         : 0.2  (stratified)\n")
    f.write(f"\n--- Baseline KRR (alpha=0.1, gamma=0.01, rbf) ---\n")
    f.write(f"MAE                   : {m_base['mae']:.4f} eV\n")
    f.write(f"RMSE                  : {m_base['rmse']:.4f} eV\n")
    f.write(f"R²                    : {m_base['r2']:.4f}\n")
    f.write(f"Spearman              : {m_base['spearman']:.4f}\n")
    f.write(f"Fit time              : {baseline_time:.1f}s\n")
    f.write(f"\n--- GridSearchCV (extended, stratified 5-fold) ---\n")
    f.write(f"Best kernel           : {best_params['krr__kernel']}\n")
    f.write(f"Best alpha            : {best_params['krr__alpha']}\n")
    f.write(f"Best gamma            : {best_params['krr__gamma']:.6f}\n")
    f.write(f"Best CV MAE (log-sp.) : {best_cv_mae:.4f}\n")
    f.write(f"Grid search time      : {grid_time:.1f}s\n")
    f.write(f"\n--- Tuned KRR — Test Set ---\n")
    f.write(f"MAE                   : {m_krr['mae']:.4f} eV\n")
    f.write(f"RMSE                  : {m_krr['rmse']:.4f} eV\n")
    f.write(f"R²                    : {m_krr['r2']:.4f}\n")
    f.write(f"Spearman              : {m_krr['spearman']:.4f}\n")
    f.write(f"\n--- Random Forest — Test Set ---\n")
    f.write(f"MAE                   : {m_rf['mae']:.4f} eV\n")
    f.write(f"RMSE                  : {m_rf['rmse']:.4f} eV\n")
    f.write(f"R²                    : {m_rf['r2']:.4f}\n")
    f.write(f"Spearman              : {m_rf['spearman']:.4f}\n")
    f.write(f"Fit time              : {rf_time:.1f}s\n")
    f.write(f"\n--- Runtime ---\n")
    f.write(f"Total time            : {total_time:.1f}s ({total_time/60:.2f} min)\n")
    f.write("=" * 60 + "\n")
print(f"\nSaved: {summary_path}")

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
print("\nGenerating plots...")

# Best model for main plots is whichever has better R²
if m_rf['r2'] >= m_krr['r2']:
    y_pred_best = y_pred_rf
    residuals   = residuals_rf
    best_label  = "Random Forest"
    best_m      = m_rf
else:
    y_pred_best = y_pred_krr
    residuals   = residuals_krr
    best_label  = f"Tuned KRR ({best_params['krr__kernel']})"
    best_m      = m_krr

# Plot 1 — Predicted vs Actual: all 3 models side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_data = [
    ("Baseline KRR", y_pred_base, m_base, "#94A3B8"),
    (f"Tuned KRR ({best_params['krr__kernel']})", y_pred_krr, m_krr, "#2563EB"),
    ("Random Forest", y_pred_rf, m_rf, "#059669"),
]
for ax, (label, y_pred, m, color) in zip(axes, model_data):
    ax.scatter(y_test, y_pred, alpha=0.6, s=25, color=color, edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()) - 0.2,
            max(y_test.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=1.2)
    ax.set_xlabel("Actual (eV)"); ax.set_ylabel("Predicted (eV)")
    ax.set_title(f"{label}\nMAE={m['mae']:.3f}  R²={m['r2']:.3f}")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.grid(True, alpha=0.3)
plt.suptitle("Predicted vs Actual HOMO-LUMO Gap — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("plot1_pred_vs_actual_all.jpg")

# Plot 2 — Residuals vs Actual (best model)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, residuals, alpha=0.7, s=35,
           c=np.abs(residuals), cmap="RdYlGn_r", edgecolors="none")
ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_xlabel("Actual HOMO-LUMO Gap (eV)")
ax.set_ylabel("Residual (Predicted − Actual) (eV)")
ax.set_title(f"Residual Plot — {best_label} (Test Set)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot2_residuals.jpg")

# Plot 3 — Residual Distribution (best model)
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(residuals, bins=20, color="#059669", edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.axvline(residuals.mean(), color="#DC2626", linewidth=1.5,
           label=f"Mean: {residuals.mean():.4f} eV")
ax.set_xlabel("Residual (eV)"); ax.set_ylabel("Count")
ax.set_title(f"Residual Distribution — {best_label} (Test Set)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot3_residual_hist.jpg")

# Plot 4 — Model comparison bar chart (all 4 metrics)
fig, ax = plt.subplots(figsize=(9, 5))
metric_names = ["MAE (eV)", "RMSE (eV)", "R²", "Spearman"]
vals_base = [m_base['mae'], m_base['rmse'], m_base['r2'], m_base['spearman']]
vals_krr  = [m_krr['mae'],  m_krr['rmse'],  m_krr['r2'],  m_krr['spearman']]
vals_rf   = [m_rf['mae'],   m_rf['rmse'],   m_rf['r2'],   m_rf['spearman']]
x     = np.arange(len(metric_names))
width = 0.26
for bars, vals, label, color in [
    (ax.bar(x - width, vals_base, width, label="Baseline KRR",  color="#94A3B8", edgecolor="white"), vals_base, "Baseline KRR",  "#94A3B8"),
    (ax.bar(x,         vals_krr,  width, label=f"Tuned KRR ({best_params['krr__kernel']})", color="#2563EB", edgecolor="white"), vals_krr, "Tuned KRR",   "#2563EB"),
    (ax.bar(x + width, vals_rf,   width, label="Random Forest", color="#059669", edgecolor="white"), vals_rf,   "Random Forest", "#059669"),
]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(metric_names)
ax.set_title("Model Comparison — Test Set Metrics")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
savefig("plot4_model_comparison.jpg")

# Plot 5 — Extended GridSearch heatmap (one panel per kernel)
cv_results = pd.DataFrame(grid_search.cv_results_)
fig, axes  = plt.subplots(1, 2, figsize=(14, 5))
for ax, kernel in zip(axes, ['rbf', 'laplacian']):
    mae_mat = np.full((len(alphas_grid), len(gammas_grid)), np.nan)
    for i, a in enumerate(alphas_grid):
        for j, g in enumerate(gammas_grid):
            mask = ((cv_results['param_krr__kernel'] == kernel) &
                    (cv_results['param_krr__alpha'].astype(float) == a) &
                    (cv_results['param_krr__gamma'].astype(float).round(10) == round(g, 10)))
            if mask.any():
                mae_mat[i, j] = -cv_results.loc[mask, 'mean_test_score'].values[0]
    im = ax.imshow(mae_mat, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="CV MAE (log-space)")
    ax.set_xticks(range(len(gammas_grid)))
    ax.set_xticklabels([f"{g:.0e}" for g in gammas_grid], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(alphas_grid)))
    ax.set_yticklabels([str(a) for a in alphas_grid], fontsize=7)
    ax.set_xlabel("gamma"); ax.set_ylabel("alpha")
    ax.set_title(f"GridSearch Heatmap — {kernel.upper()} kernel")
    for i in range(len(alphas_grid)):
        for j in range(len(gammas_grid)):
            if not np.isnan(mae_mat[i, j]):
                ax.text(j, i, f"{mae_mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=6, color="black")
plt.suptitle("GridSearchCV CV MAE Heatmap (log-transformed labels)", fontweight="bold")
plt.tight_layout()
savefig("plot5_gridsearch_heatmap.jpg")

# Plot 6 — Absolute error per test molecule (best model)
abs_errors  = np.abs(residuals)
mol_indices = np.arange(len(y_test))
fig, ax     = plt.subplots(figsize=(12, 4))
colors_err  = get_cmap("RdYlGn_r")(abs_errors / abs_errors.max())
ax.bar(mol_indices, abs_errors, color=colors_err, edgecolor="none")
ax.axhline(abs_errors.mean(), color="#2563EB", linewidth=1.5, linestyle="--",
           label=f"Mean |error|: {abs_errors.mean():.4f} eV")
ax.set_xlabel("Test molecule index"); ax.set_ylabel("|Residual| (eV)")
ax.set_title(f"Absolute Error per Test Molecule — {best_label}")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
savefig("plot6_per_molecule_error.jpg")

# Plot 7 — Error vs Gap Size, coloured by outlier status
fig, ax = plt.subplots(figsize=(7, 5))
outlier_mask = y_test > (cap_val if cap_val else np.inf)
ax.scatter(y_test[~outlier_mask], abs_errors[~outlier_mask],
           c=abs_errors[~outlier_mask], cmap="plasma",
           alpha=0.7, s=35, edgecolors="none", label="Normal")
if outlier_mask.any():
    ax.scatter(y_test[outlier_mask], abs_errors[outlier_mask],
               color="#DC2626", s=60, marker="*", zorder=5,
               label=f"Outliers (gap > {cap_val:.1f} eV)")
ax.set_xlabel("Actual HOMO-LUMO Gap (eV)"); ax.set_ylabel("|Residual| (eV)")
ax.set_title(f"Absolute Error vs Gap Size — {best_label}")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("plot7_error_vs_gap_size.jpg")

# Plot 8 — RF feature importances (top 20)
if selector is not None:
    selected_names = np.array(df_desc.columns)[selector.get_support()]
else:
    selected_names = np.array(df_desc.columns)

importances  = rf.feature_importances_
top_idx      = np.argsort(importances)[-20:][::-1]
top_names    = [selected_names[i] if i < len(selected_names) else f"feat_{i}"
                for i in top_idx]
top_imp      = importances[top_idx]

fig, ax = plt.subplots(figsize=(9, 6))
colors_imp = get_cmap("Blues")(np.linspace(0.4, 0.9, len(top_imp)))[::-1]
ax.barh(range(len(top_imp)), top_imp[::-1], color=colors_imp[::-1], edgecolor="none")
ax.set_yticks(range(len(top_imp)))
ax.set_yticklabels(top_names[::-1], fontsize=8)
ax.set_xlabel("Feature Importance")
ax.set_title("Top 20 Mordred Descriptors — Random Forest Feature Importance")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
savefig("plot8_rf_feature_importance.jpg")

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
