# HOMO-LUMO Gap Predictor 
##### We combine multiple ML techniques (Baseline KRR, Tuned KRR, Random Forest, XGBoost, LightGBM, Stacking ensemble) and compare the results.
### Requirement:
```
pip install xgboost lightgbm shap
```
 ### Code Summary

---

## Purpose

Predicts HOMO-LUMO gaps (eV) from SMILES strings using Mordred 2D molecular descriptors and a suite of machine learning models, with full hyperparameter tuning, ensemble stacking, SHAP interpretability, and diagnostic plots.

---

## Pipeline Overview

### 1. Data Loading & Cleaning
Reads `homolumo.csv` containing SMILES and DFT-computed gaps. Molecules with invalid or problematic SMILES (e.g. containing `<`) are skipped. All molecules are sanitized with RDKit before descriptor calculation.

### 2. Data Distribution Plot
Generates a histogram and box plot of the raw gap distribution before any modelling, saved as `plot0_data_distribution.jpg`. The 95th percentile cap threshold is marked on the histogram.

### 3. Mordred Descriptor Calculation
Computes ~1600 2D molecular descriptors. Descriptors are cleaned by converting errors to NaN, filling with zero, clipping to ±1×10⁶ to prevent overflow, and dropping zero-variance columns. NumPy overflow warnings from Mordred's internals are suppressed during calculation.

### 4. Preprocessing

Three settings at the top of the script control all preprocessing:

| Setting | Default | Effect |
|---|---|---|
| `LOG_TRANSFORM` | `True` | Predict log(gap) to compress the right-skewed distribution; predictions are exp()-transformed back to eV for evaluation |
| `OUTLIER_CAP_PCT` | `95` | Cap training labels at the 95th percentile to reduce the influence of chemically unusual high-gap molecules; test labels are never capped |
| `K_FEATURES` | `200` | Keep the top 200 Mordred descriptors by F-statistic correlation with the target (SelectKBest), reducing ~1600 → 200 features |

The train/test split (80/20) is **stratified** by gap value — gaps are binned into quantile strata so both sets contain proportional coverage of the full distribution, including the sparse tails.

---

## Models Trained

| Step | Model | Tuning Method |
|---|---|---|
| 8a | Baseline KRR (RBF, α=0.1, γ=0.01) | Fixed — reference point |
| 8b | Tuned KRR (RBF + Laplacian kernels) | GridSearchCV: 7 α × 7 γ × 2 kernels × 5-fold CV |
| 8c | Random Forest | RandomizedSearchCV: `n_estimators`, `max_features`, `min_samples_leaf`, `max_depth` |
| 8d | XGBoost | RandomizedSearchCV + early stopping on held-out eval set |
| 8e | LightGBM | RandomizedSearchCV: `num_leaves`, `learning_rate`, `subsample`, regularisation |
| 8f | Stacking ensemble | Ridge meta-learner trained on out-of-fold predictions from all above models |

XGBoost and LightGBM are wrapped in `try/except` blocks — if not installed, they are skipped gracefully with a `pip install` message.

---

## Evaluation Metrics

All models are evaluated on the held-out test set (20%) using four metrics, all reported in eV-space after inverse log-transform where applicable:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of determination
- **Spearman ρ** — Rank correlation (robust to non-linearity)

---

## SHAP Interpretability

After training, `shap.TreeExplainer` computes feature contributions for both the Random Forest and the best-performing boosting model (whichever of XGBoost/LightGBM has the higher R²). Three plots are produced:

- **Dot summary plot** — one point per test molecule per feature; colour = feature value, x-axis = SHAP impact on prediction
- **Bar importance plot** — mean |SHAP| per feature, ranked
- Both RF and best booster get their own dot summary

Install with: `pip install shap`

---

## Output Files

All outputs are saved to `output_krr/`:

| File | Description |
|---|---|
| `plot0_data_distribution.jpg` | Gap histogram + box plot |
| `plot1_pred_vs_actual_all.jpg` | Parity plots for all models |
| `plot2_residuals.jpg` | Residuals vs actual (best model) |
| `plot3_residual_hist.jpg` | Residual distribution (best model) |
| `plot4_model_comparison.jpg` | R² and MAE bar chart across all models |
| `plot5_gridsearch_heatmap.jpg` | KRR CV MAE heatmap (RBF and Laplacian) |
| `plot6_per_molecule_error.jpg` | Absolute error per test molecule |
| `plot7_error_vs_gap_size.jpg` | Error vs gap value, outliers flagged |
| `plot8_rf_feature_importance.jpg` | Top 20 RF feature importances |
| `plot9_shap_rf_summary.jpg` | SHAP dot summary — Random Forest |
| `plot10_shap_rf_bar.jpg` | SHAP bar importance — Random Forest |
| `plot11_shap_*_summary.jpg` | SHAP dot summary — best boosting model |
| `summary.txt` | Full run summary: all metrics, hyperparameters, runtime |

---

## Dependencies

```
rdkit
mordred
scikit-learn
scipy
numpy<2.0
matplotlib
xgboost    ← optional
lightgbm   ← optional
shap       ← optional
```
