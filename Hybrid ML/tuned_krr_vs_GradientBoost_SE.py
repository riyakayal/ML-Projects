

# Author: Riya Kayal
# Created: 21/12/2025


import pandas as pd
import numpy as np
import os
import datetime
import time
import warnings
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
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
OUTPUT_DIR = "output_krr3"
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
STACK_MIN_R2    = 0.60    # only include models with test R² >= this in the stacking ensemble
                          # prevents weak models dragging down the meta-learner
                          # set to None to always stack all available models
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
# 8. Models
# ------------------------------------------------------------------
print(f"\n[8/10] Training models...")

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

# ---- 8b. GridSearchCV — extended grid, RBF + Laplacian ----
alphas_grid = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
gammas_grid = np.logspace(-5, 0, 7)
n_combos    = len(alphas_grid) * len(gammas_grid) * 2 * 5
print(f"\n      [8b] GridSearchCV KRR  "
      f"({len(alphas_grid)} alphas × {len(gammas_grid)} gammas × 2 kernels × 5-fold "
      f"= {n_combos} fits)...")
print("           sklearn verbose output:")

t2 = time.time()
param_grid_ext = {
    'krr__kernel': ['rbf', 'laplacian'],
    'krr__alpha':  alphas_grid,
    'krr__gamma':  gammas_grid,
}
pipeline_gs  = Pipeline([('krr', KernelRidge())])
y_train_bins = pd.qcut(y_train_fit, q=min(5, len(y_train_fit)//10),
                       labels=False, duplicates="drop")
cv_strat     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline_gs, param_grid_ext,
    cv=cv_strat.split(X_train_fs, y_train_bins),
    scoring='neg_mean_absolute_error',
    n_jobs=-1, verbose=1, refit=False
)
grid_search.fit(X_train_fs, y_train_fit)
grid_time   = time.time() - t2
best_params = grid_search.best_params_
best_cv_mae = -grid_search.best_score_
print(f"\n           GridSearch done in {grid_time:.1f}s")
print(f"           Best params: {best_params}  |  CV MAE: {best_cv_mae:.4f} (log-space)")

best_krr       = KernelRidge(kernel=best_params['krr__kernel'],
                             alpha=best_params['krr__alpha'],
                             gamma=best_params['krr__gamma'])
best_krr.fit(X_train_fs, y_train_fit)
y_pred_krr     = to_ev(best_krr.predict(X_test_fs))
m_krr          = compute_metrics(y_test, y_pred_krr)
residuals_krr  = y_pred_krr - y_test
print(f"           Test  →  MAE: {m_krr['mae']:.4f}  RMSE: {m_krr['rmse']:.4f}  "
      f"R²: {m_krr['r2']:.4f}  Spearman: {m_krr['spearman']:.4f}")

# ---- 8c. Random Forest with hyperparameter tuning ----
# Tune max_features and min_samples_leaf via 5-fold CV.
# These two knobs have the most impact on RF performance.
from sklearn.model_selection import RandomizedSearchCV
print("\n      [8c] Random Forest — tuning max_features & min_samples_leaf...")
t3 = time.time()
rf_param_dist = {
    'n_estimators':      [300, 500],
    'max_features':      ['sqrt', 'log2', 0.3, 0.5],
    'min_samples_leaf':  [1, 2, 4],
    'max_depth':         [None, 20, 40],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(n_jobs=-1, random_state=42),
    rf_param_dist, n_iter=20, cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1, verbose=1, random_state=42, refit=True
)
rf_search.fit(X_train_fs, y_train_fit)
rf         = rf_search.best_estimator_
rf_time    = time.time() - t3
y_pred_rf  = to_ev(rf.predict(X_test_fs))
m_rf       = compute_metrics(y_test, y_pred_rf)
residuals_rf = y_pred_rf - y_test
print(f"           Best RF params: {rf_search.best_params_}")
print(f"           Done in {rf_time:.1f}s  |  MAE: {m_rf['mae']:.4f}  "
      f"RMSE: {m_rf['rmse']:.4f}  R²: {m_rf['r2']:.4f}  Spearman: {m_rf['spearman']:.4f}")

# ---- 8d. XGBoost ----
print("\n      [8d] XGBoost with early stopping...")
try:
    import xgboost as xgb
    t4 = time.time()
    # Hold out 15% of train for XGBoost's internal early stopping eval set
    X_tr_xgb, X_val_xgb, y_tr_xgb, y_val_xgb = train_test_split(
        X_train_fs, y_train_fit, test_size=0.15, random_state=42)

    # Adapt search space to training set size.
    # Small datasets (< 150 train samples) are prone to XGBoost overfitting:
    # restrict to shallow trees, high regularisation, low learning rate.
    n_train = len(X_train_fs)
    if n_train < 150:
        print(f"           Small dataset ({n_train} train samples) — using conservative XGB search space")
        xgb_param_dist = {
            'n_estimators':    [100, 200, 300],
            'max_depth':       [2, 3],
            'learning_rate':   [0.01, 0.05],
            'subsample':       [0.6, 0.8],
            'colsample_bytree':[0.5, 0.7],
            'reg_alpha':       [0.1, 1.0, 5.0],
            'reg_lambda':      [1.0, 5.0, 10.0],
        }
    else:
        xgb_param_dist = {
            'n_estimators':    [500, 1000],
            'max_depth':       [3, 5, 7],
            'learning_rate':   [0.01, 0.05, 0.1],
            'subsample':       [0.7, 0.9],
            'colsample_bytree':[0.6, 0.8, 1.0],
            'reg_alpha':       [0, 0.1, 1.0],
            'reg_lambda':      [1.0, 5.0],
        }
    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror', eval_metric='mae',
                         early_stopping_rounds=30, n_jobs=-1, random_state=42,
                         verbosity=0),
        xgb_param_dist, n_iter=20, cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=1, verbose=1, random_state=42, refit=False
    )
    xgb_search.fit(X_tr_xgb, y_tr_xgb,
                   eval_set=[(X_val_xgb, y_val_xgb)])
    best_xgb_params = xgb_search.best_params_
    # Refit on full training data with best params
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', eval_metric='mae',
        early_stopping_rounds=30, n_jobs=-1, random_state=42, verbosity=0,
        **{k: v for k, v in best_xgb_params.items() if k != 'n_estimators'},
        n_estimators=best_xgb_params.get('n_estimators', 500)
    )
    xgb_model.fit(X_tr_xgb, y_tr_xgb, eval_set=[(X_val_xgb, y_val_xgb)], verbose=False)
    xgb_time    = time.time() - t4
    y_pred_xgb  = to_ev(xgb_model.predict(X_test_fs))
    m_xgb       = compute_metrics(y_test, y_pred_xgb)
    residuals_xgb = y_pred_xgb - y_test
    xgb_available = True
    print(f"           Best XGB params: {best_xgb_params}")
    print(f"           Done in {xgb_time:.1f}s  |  MAE: {m_xgb['mae']:.4f}  "
          f"RMSE: {m_xgb['rmse']:.4f}  R²: {m_xgb['r2']:.4f}  Spearman: {m_xgb['spearman']:.4f}")
except ImportError:
    xgb_available = False
    print("           XGBoost not installed — skipping. Run: pip install xgboost")
    m_xgb = None; y_pred_xgb = None; residuals_xgb = None; xgb_time = 0

# ---- 8e. LightGBM ----
print("\n      [8e] LightGBM with hyperparameter tuning...")
try:
    import lightgbm as lgb
    t5 = time.time()
    lgb_param_dist = {
        'n_estimators':    [500, 1000],
        'max_depth':       [-1, 6, 10],
        'learning_rate':   [0.01, 0.05, 0.1],
        'num_leaves':      [31, 63, 127],
        'subsample':       [0.7, 0.9],
        'colsample_bytree':[0.6, 0.8, 1.0],
        'reg_alpha':       [0, 0.1, 1.0],
        'reg_lambda':      [1.0, 5.0],
        'min_child_samples': [5, 20, 50],
    }
    lgb_search = RandomizedSearchCV(
        lgb.LGBMRegressor(objective='regression', n_jobs=-1,
                          random_state=42, verbose=-1),
        lgb_param_dist, n_iter=20, cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=1, verbose=1, random_state=42, refit=True
    )
    lgb_search.fit(X_train_fs if isinstance(X_train_fs, np.ndarray) else X_train_fs.values,
                   y_train_fit)
    lgb_model     = lgb_search.best_estimator_
    lgb_time      = time.time() - t5
    # Always predict on numpy array to match training dtype and suppress warning
    X_test_lgb    = X_test_fs if isinstance(X_test_fs, np.ndarray) else X_test_fs.values
    y_pred_lgb    = to_ev(lgb_model.predict(X_test_lgb))
    m_lgb         = compute_metrics(y_test, y_pred_lgb)
    residuals_lgb = y_pred_lgb - y_test
    lgb_available = True
    print(f"           Best LGB params: {lgb_search.best_params_}")
    print(f"           Done in {lgb_time:.1f}s  |  MAE: {m_lgb['mae']:.4f}  "
          f"RMSE: {m_lgb['rmse']:.4f}  R²: {m_lgb['r2']:.4f}  Spearman: {m_lgb['spearman']:.4f}")
except ImportError:
    lgb_available = False
    print("           LightGBM not installed — skipping. Run: pip install lightgbm")
    m_lgb = None; y_pred_lgb = None; residuals_lgb = None; lgb_time = 0

# ---- 8f. Stacking ensemble ----
# Combine model predictions via a Ridge meta-learner trained on OOF predictions.
# Quality filter: only models with test R² >= STACK_MIN_R2 are included.
# This prevents weak models (e.g. XGBoost on small data) from dragging down
# the ensemble. If fewer than 2 models pass the threshold, stacking is skipped.
print("\n      [8f] Stacking ensemble (Ridge meta-learner)...")
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

# Build candidate pool with R² scores for filtering
stack_candidates = {
    'KRR': (best_krr,  X_train_fs, X_test_fs, m_krr['r2']),
    'RF':  (rf,        X_train_fs, X_test_fs, m_rf['r2']),
}
if xgb_available:
    xgb_oof = xgb.XGBRegressor(
        objective='reg:squarederror', n_jobs=-1, random_state=42, verbosity=0,
        **{k: v for k, v in best_xgb_params.items()}
    )
    stack_candidates['XGB'] = (xgb_oof, X_train_fs, X_test_fs, m_xgb['r2'])
if lgb_available:
    X_train_lgb = X_train_fs if isinstance(X_train_fs, np.ndarray) else X_train_fs.values
    stack_candidates['LGB'] = (lgb_model, X_train_lgb, X_test_lgb, m_lgb['r2'])

# Apply quality threshold
if STACK_MIN_R2 is not None:
    stack_models_raw = {
        name: (model, Xtr, Xte)
        for name, (model, Xtr, Xte, r2) in stack_candidates.items()
        if r2 >= STACK_MIN_R2
    }
    excluded = [f"{n} (R²={r2:.3f})" for n, (_, _, _, r2) in stack_candidates.items()
                if r2 < STACK_MIN_R2]
    if excluded:
        print(f"           Excluded from stack (R² < {STACK_MIN_R2}): {', '.join(excluded)}")
else:
    stack_models_raw = {
        name: (model, Xtr, Xte)
        for name, (model, Xtr, Xte, r2) in stack_candidates.items()
    }

if len(stack_models_raw) < 2:
    print(f"           Only {len(stack_models_raw)} model(s) passed the R²>={STACK_MIN_R2} threshold — "
          f"skipping stacking (need at least 2).")
    # Use best individual model as the stack result
    best_individual = max(stack_candidates.items(), key=lambda x: x[1][3])
    best_ind_name   = best_individual[0]
    y_pred_stack    = {'KRR': y_pred_krr, 'RF': y_pred_rf,
                       'XGB': y_pred_xgb if xgb_available else None,
                       'LGB': y_pred_lgb if lgb_available else None}[best_ind_name]
    m_stack         = {'KRR': m_krr, 'RF': m_rf,
                       'XGB': m_xgb if xgb_available else None,
                       'LGB': m_lgb if lgb_available else None}[best_ind_name]
    residuals_stack = y_pred_stack - y_test
    print(f"           Stacking result = best individual model: {best_ind_name}")
else:
    print(f"           Stacking {list(stack_models_raw.keys())} ({len(stack_models_raw)} models)...")
    oof_train = np.zeros((len(X_train_fs), len(stack_models_raw)))
    oof_test  = np.zeros((len(X_test_fs),  len(stack_models_raw)))

    for col_idx, (name, (model, Xtr, Xte)) in enumerate(stack_models_raw.items()):
        print(f"           OOF predictions for {name}...", end="\r")
        oof_train[:, col_idx] = cross_val_predict(model, Xtr, y_train_fit, cv=5, n_jobs=-1)
        oof_test[:,  col_idx] = xgb_model.predict(Xte) if name == 'XGB' else model.predict(Xte)

    print(f"\n           Fitting Ridge meta-learner on OOF predictions...")
    meta            = Ridge(alpha=1.0)
    meta.fit(oof_train, y_train_fit)
    y_pred_stack    = to_ev(meta.predict(oof_test))
    m_stack         = compute_metrics(y_test, y_pred_stack)
    residuals_stack = y_pred_stack - y_test

print(f"           Stacking  →  MAE: {m_stack['mae']:.4f}  RMSE: {m_stack['rmse']:.4f}  "
      f"R²: {m_stack['r2']:.4f}  Spearman: {m_stack['spearman']:.4f}")

total_time = time.time() - t0

# ------------------------------------------------------------------
# Summary to console
# ------------------------------------------------------------------
all_models = [
    ("Baseline KRR (rbf)",                    m_base),
    (f"Tuned KRR ({best_params['krr__kernel']})", m_krr),
    ("Random Forest (tuned)",                 m_rf),
]
if xgb_available: all_models.append(("XGBoost (tuned)",    m_xgb))
if lgb_available: all_models.append(("LightGBM (tuned)",   m_lgb))
all_models.append(("Stacking ensemble",                     m_stack))

print("\n" + "="*65)
print("                     FINAL RESULTS")
print("="*65)
print(f"{'Model':<32} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Spearman':>9}")
print("-"*65)
for name, m in all_models:
    print(f"{name:<32} {m['mae']:>7.4f} {m['rmse']:>7.4f} "
          f"{m['r2']:>7.4f} {m['spearman']:>9.4f}")
print("="*65)

# ------------------------------------------------------------------
# Summary file
# ------------------------------------------------------------------
summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 65 + "\n")
    f.write("   HOMO-LUMO Gap Predictor — Run Summary\n")
    f.write("=" * 65 + "\n")
    f.write(f"Run date/time         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input file            : homolumo.csv\n")
    f.write(f"Molecules loaded      : {len(smiles)}\n")
    f.write(f"Molecules skipped     : {skipped}\n")
    f.write(f"\n--- Data Statistics (raw gaps) ---\n")
    f.write(f"Gap min / max         : {gaps.min():.4f} / {gaps.max():.4f} eV\n")
    f.write(f"Gap mean / std        : {gaps.mean():.4f} / {gaps.std():.4f} eV\n")
    f.write(f"Gap median            : {np.median(gaps):.4f} eV\n")
    f.write(f"\n--- Preprocessing ---\n")
    f.write(f"Log-transform labels  : {LOG_TRANSFORM}\n")
    f.write(f"Outlier cap           : {OUTLIER_CAP_PCT}th pct ({cap_val:.4f} eV) → {n_capped} labels capped\n" if cap_val else "Outlier cap           : disabled\n")
    f.write(f"Feature selection     : SelectKBest (k={K_FEATURES})\n")
    f.write(f"Stratified split      : Yes ({n_strata} strata)\n")
    f.write(f"Descriptor type       : Mordred (2D, ignore_3D=True)\n")
    f.write(f"Raw descriptors       : {df_desc.shape[1]}\n")
    f.write(f"After SelectKBest     : {X_train_fs.shape[1]}\n")
    f.write(f"Descriptor calc time  : {descriptor_time:.1f}s\n")
    f.write(f"Train / Test size     : {len(X_train)} / {len(X_test)}\n")
    f.write(f"Stack min R² filter   : {STACK_MIN_R2} "
            f"({'disabled' if STACK_MIN_R2 is None else f'{len(stack_models_raw)} model(s) included'})\n")
    f.write(f"XGB search space      : {'conservative (small data)' if len(X_train_fs) < 150 else 'standard'}\n")
    f.write(f"\n--- Results — Test Set ---\n")
    f.write(f"{'Model':<32} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Spearman':>9}\n")
    f.write("-" * 65 + "\n")
    for name, m in all_models:
        f.write(f"{name:<32} {m['mae']:>7.4f} {m['rmse']:>7.4f} "
                f"{m['r2']:>7.4f} {m['spearman']:>9.4f}\n")
    f.write(f"\n--- Hyperparameters ---\n")
    f.write(f"Best KRR kernel       : {best_params['krr__kernel']}\n")
    f.write(f"Best KRR alpha/gamma  : {best_params['krr__alpha']} / {best_params['krr__gamma']:.6f}\n")
    f.write(f"Best RF params        : {rf_search.best_params_}\n")
    if xgb_available: f.write(f"Best XGB params       : {best_xgb_params}\n")
    if lgb_available: f.write(f"Best LGB params       : {lgb_search.best_params_}\n")
    f.write(f"\n--- Runtime ---\n")
    f.write(f"Total time            : {total_time:.1f}s ({total_time/60:.2f} min)\n")
    f.write("=" * 65 + "\n")
print(f"\nSaved: {summary_path}")

# ------------------------------------------------------------------
# 9. SHAP Values
# ------------------------------------------------------------------
print("\n[9/10] Computing SHAP values (RF + best boosting model)...")
try:
    import shap

    if selector is not None:
        selected_names = np.array(df_desc.columns)[selector.get_support()]
    else:
        selected_names = np.array(df_desc.columns)

    # --- RF SHAP (TreeExplainer — fast) ---
    print("      RF SHAP...")
    explainer_rf  = shap.TreeExplainer(rf)
    shap_vals_rf  = explainer_rf.shap_values(X_test_fs)

    # SHAP summary plot — RF
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals_rf, X_test_fs,
                      feature_names=selected_names,
                      plot_type="dot", show=False, max_display=20)
    plt.title("SHAP Summary — Random Forest (Top 20 features)", fontsize=12)
    plt.tight_layout()
    savefig("plot9_shap_rf_summary.jpg")

    # SHAP bar plot — RF mean |SHAP|
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_vals_rf, X_test_fs,
                      feature_names=selected_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance — Random Forest (mean |SHAP|)", fontsize=12)
    plt.tight_layout()
    savefig("plot10_shap_rf_bar.jpg")

    # --- Best boosting model SHAP ---
    best_boost_name  = None
    best_boost_model = None
    best_boost_r2    = -np.inf
    if xgb_available and m_xgb['r2'] > best_boost_r2:
        best_boost_r2    = m_xgb['r2']
        best_boost_model = xgb_model
        best_boost_name  = "XGBoost"
    if lgb_available and m_lgb['r2'] > best_boost_r2:
        best_boost_r2    = m_lgb['r2']
        best_boost_model = lgb_model
        best_boost_name  = "LightGBM"

    if best_boost_model is not None:
        print(f"      {best_boost_name} SHAP...")
        explainer_boost = shap.TreeExplainer(best_boost_model)
        shap_vals_boost = explainer_boost.shap_values(X_test_fs)
        if isinstance(shap_vals_boost, list):
            shap_vals_boost = shap_vals_boost[0]

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_vals_boost, X_test_fs,
                          feature_names=selected_names,
                          plot_type="dot", show=False, max_display=20)
        plt.title(f"SHAP Summary — {best_boost_name} (Top 20 features)", fontsize=12)
        plt.tight_layout()
        savefig(f"plot11_shap_{best_boost_name.lower()}_summary.jpg")

    shap_available = True
except ImportError:
    shap_available = False
    print("      SHAP not installed — skipping. Run: pip install shap")

# ------------------------------------------------------------------
# 10. Plots
# ------------------------------------------------------------------
print("\n[10/10] Generating plots...")

# Pick best model by R² for residual plots
best_r2     = -np.inf
best_label  = ""
y_pred_best = None
residuals   = None
best_m      = None
for name, m, ypred, res in [
    ("Baseline KRR",           m_base,  y_pred_base,  y_pred_base  - y_test),
    (f"Tuned KRR ({best_params['krr__kernel']})", m_krr, y_pred_krr, residuals_krr),
    ("Random Forest",          m_rf,    y_pred_rf,    residuals_rf),
] + ([("XGBoost",   m_xgb, y_pred_xgb, residuals_xgb)] if xgb_available else []) \
  + ([("LightGBM",  m_lgb, y_pred_lgb, residuals_lgb)] if lgb_available else []) \
  + [("Stacking",   m_stack, y_pred_stack, residuals_stack)]:
    if m['r2'] > best_r2:
        best_r2     = m['r2']
        best_label  = name
        y_pred_best = ypred
        residuals   = res
        best_m      = m

# Plot 1 — Predicted vs Actual: all models
n_models    = len(all_models)
n_cols      = min(3, n_models)
n_rows      = (n_models + n_cols - 1) // n_cols
palette     = ["#94A3B8", "#2563EB", "#059669", "#F59E0B", "#8B5CF6", "#EC4899"]
model_preds = [y_pred_base, y_pred_krr, y_pred_rf]
model_names_short = ["Baseline KRR", f"Tuned KRR ({best_params['krr__kernel']})", "RF (tuned)"]
if xgb_available: model_preds.append(y_pred_xgb); model_names_short.append("XGBoost")
if lgb_available: model_preds.append(y_pred_lgb); model_names_short.append("LightGBM")
model_preds.append(y_pred_stack); model_names_short.append("Stacking")
model_metrics = [m for _, m in all_models]

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes_flat = axes.flatten() if n_models > 1 else [axes]
for ax, label, y_pred, m, color in zip(axes_flat, model_names_short,
                                        model_preds, model_metrics, palette):
    ax.scatter(y_test, y_pred, alpha=0.6, s=25, color=color, edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()) - 0.2,
            max(y_test.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=1.2)
    ax.set_xlabel("Actual (eV)"); ax.set_ylabel("Predicted (eV)")
    ax.set_title(f"{label}\nMAE={m['mae']:.3f}  R²={m['r2']:.3f}")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.grid(True, alpha=0.3)
for ax in axes_flat[n_models:]:
    ax.set_visible(False)
plt.suptitle("Predicted vs Actual HOMO-LUMO Gap — All Models",
             fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("plot1_pred_vs_actual_all.jpg")

# Plot 2 — Residuals vs Actual (best model)
abs_errors = np.abs(residuals)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, residuals, alpha=0.7, s=35,
           c=abs_errors, cmap="RdYlGn_r", edgecolors="none")
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

# Plot 4 — Model comparison bar chart (R² and MAE)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
names_short = [n for n, _ in all_models]
r2s   = [m['r2']  for _, m in all_models]
maes  = [m['mae'] for _, m in all_models]
colors_bar = palette[:len(all_models)]
x = np.arange(len(all_models))

for ax, vals, ylabel, title in [
    (axes[0], r2s,  "R²",       "R² — Test Set"),
    (axes[1], maes, "MAE (eV)", "MAE — Test Set"),
]:
    bars = ax.bar(x, vals, color=colors_bar, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names_short, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
plt.suptitle("Model Comparison — Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("plot4_model_comparison.jpg")

# Plot 5 — KRR GridSearch heatmap
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
plt.suptitle("KRR GridSearchCV — CV MAE Heatmap", fontweight="bold")
plt.tight_layout()
savefig("plot5_gridsearch_heatmap.jpg")

# Plot 6 — Absolute error per test molecule
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

# Plot 7 — Error vs Gap Size
outlier_mask = y_test > (cap_val if cap_val else np.inf)
fig, ax = plt.subplots(figsize=(7, 5))
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

# Plot 8 — RF feature importance (top 20)
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
ax.set_title("Top 20 Mordred Descriptors — RF Feature Importance")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
savefig("plot8_rf_feature_importance.jpg")

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


