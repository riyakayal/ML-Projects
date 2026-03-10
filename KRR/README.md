# Prediction of HOMO-LUMO Gaps with Kernel Ridge Regression (KRR)
We try a basic KRR model and a tuned KRR model.
The input file is "homolumo.csv" with 2 columns, smiles and HOMO-LUMO gap in eV.

## To run
```
python krr.py
# or,
python krr_tuned.py
```
The outputs will be in the **output_krr** directory. An example output is provided for krr.py.

---
| File | Description |
|-----|---|
|`plot0_data_distribution.jpg`|Gap histogram + box plot|
|`plot1_pred_vs_actual.jpg`|Predicted vs actual scatter (best model)|
|`plot2_residuals.jpg`|Residuals vs actual gap|
|`plot3_residual_hist.jpg`|Residual distribution|
|`plot4_baseline_vs_best.jpg`|Bar chart comparing baseline vs tuned KRR across all 4 metrics|
|`plot5_gridsearch_heatmap.jpg`|Alpha × gamma CV MAE heatmap — shows which region of hyperparameter space works best|
|`plot6_per_molecule_error.jpg`|Absolute error per test molecule (colour-coded)|
|`plot7_error_vs_gap_size.jpg`|Whether prediction error correlates with the gap value itself|
