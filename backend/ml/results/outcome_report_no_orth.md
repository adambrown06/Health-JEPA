# Outcome-prediction report — tag `no_orth`

- Checkpoint: `backend/ml/checkpoints/jepa_best_no_orth.pt`
- Device: `cuda`
- Splits: train=2989 · val=640 · test=640
- Best non-JEPA baseline (for bootstraps): **GBT**

## Observed test patient counts per outcome

| outcome | n |
|:---|---:|
| hba1c | 250 |
| ldl_chol | 83 |
| hdl_chol | 326 |
| total_chol | 331 |
| systolic_bp | 481 |
| diastolic_bp | 482 |
| bmi | 425 |
| glucose | 442 |

## MAE per model and outcome (real units)

| model | hba1c | ldl_chol | hdl_chol | total_chol | systolic_bp | diastolic_bp | bmi | glucose | macro MAE | macro R² |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| POP_MEAN | 1.098 | 25.640 | 11.456 | 31.880 | 10.579 | 6.856 | 6.978 | 35.866 | 16.294 | -0.003 |
| NO_CHANGE | 2.760 | 53.108 | 19.673 | 81.172 | 28.019 | 17.021 | 6.979 | 43.722 | 31.557 | -6.780 |
| RIDGE | 0.953 | 25.922 | 8.779 | 29.801 | 9.984 | 6.494 | 6.462 | 28.891 | 14.661 | 0.118 |
| GBT | 0.751 | 24.800 | 7.241 | 26.309 | 9.091 | 5.521 | 2.558 | 26.210 | 12.810 | 0.362 |
| MLP_RAW | 1.131 | 31.690 | 8.376 | 29.015 | 10.794 | 6.860 | 5.757 | 30.434 | 15.507 | -0.103 |
| JEPA_RIDGE | 0.929 | 29.716 | 9.146 | 29.462 | 9.092 | 5.813 | 4.687 | 29.377 | 14.778 | 0.192 |
| JEPA_RIDGE_PRED | 0.930 | 27.774 | 9.240 | 29.100 | 9.144 | 5.834 | 5.141 | 29.640 | 14.600 | 0.157 |
| JEPA_TWIN | 0.994 | 29.615 | 9.533 | 32.165 | 9.970 | 6.026 | 6.485 | 29.504 | 15.537 | 0.039 |
| JEPA_TWIN_NAIVE | 1.092 | 31.563 | 12.162 | 33.364 | 10.429 | 6.219 | 7.116 | 33.651 | 16.950 | -0.092 |

## R² per model and outcome

| model | hba1c | ldl_chol | hdl_chol | total_chol | systolic_bp | diastolic_bp | bmi | glucose | macro MAE | macro R² |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| POP_MEAN | -0.004 | -0.004 | -0.001 | -0.000 | -0.004 | -0.010 | -0.000 | -0.004 | 16.294 | -0.003 |
| NO_CHANGE | -6.859 | -3.233 | -2.792 | -6.667 | -13.748 | -12.194 | -7.472 | -1.276 | 31.557 | -6.780 |
| RIDGE | 0.216 | -0.175 | 0.329 | 0.096 | 0.082 | 0.093 | 0.095 | 0.205 | 14.661 | 0.118 |
| GBT | 0.429 | 0.082 | 0.540 | 0.284 | 0.237 | 0.254 | 0.756 | 0.315 | 12.810 | 0.362 |
| MLP_RAW | -0.056 | -0.888 | 0.400 | 0.100 | -0.317 | -0.461 | 0.240 | 0.159 | 15.507 | -0.103 |
| JEPA_RIDGE | 0.280 | -0.239 | 0.310 | 0.119 | 0.205 | 0.181 | 0.463 | 0.217 | 14.778 | 0.192 |
| JEPA_RIDGE_PRED | 0.275 | -0.233 | 0.282 | 0.132 | 0.208 | 0.118 | 0.270 | 0.201 | 14.600 | 0.157 |
| JEPA_TWIN | 0.147 | -0.432 | 0.256 | -0.080 | 0.102 | 0.144 | 0.030 | 0.147 | 15.537 | 0.039 |
| JEPA_TWIN_NAIVE | -0.106 | -0.382 | -0.103 | -0.155 | 0.054 | 0.118 | -0.095 | -0.067 | 16.950 | -0.092 |

## Paired bootstrap MAE: model − GBT (negative Δ ⇒ model better)

| model | outcome | Δ MAE | 95% CI | p (two-sided) | % test patients model-better |
|:---|:---|---:|:---:|---:|---:|
| POP_MEAN | hba1c | 0.346 | [0.259, 0.431] | 0.000 | 26.800% |
| POP_MEAN | ldl_chol | 0.802 | [-2.402, 3.978] | 0.646 | 49.398% |
| POP_MEAN | hdl_chol | 4.234 | [3.281, 5.260] | 0.000 | 33.742% |
| POP_MEAN | total_chol | 5.565 | [3.648, 7.364] | 0.000 | 40.785% |
| POP_MEAN | systolic_bp | 1.475 | [0.935, 2.005] | 0.000 | 41.996% |
| POP_MEAN | diastolic_bp | 1.346 | [0.975, 1.711] | 0.000 | 37.967% |
| POP_MEAN | bmi | 4.429 | [3.911, 4.947] | 0.000 | 17.647% |
| POP_MEAN | glucose | 9.694 | [7.218, 12.176] | 0.000 | 30.090% |
| NO_CHANGE | hba1c | 2.003 | [1.600, 2.390] | 0.000 | 39.600% |
| NO_CHANGE | ldl_chol | 28.252 | [19.466, 37.612] | 0.000 | 31.325% |
| NO_CHANGE | hdl_chol | 12.384 | [10.386, 14.557] | 0.000 | 33.436% |
| NO_CHANGE | total_chol | 54.782 | [47.439, 62.313] | 0.000 | 24.773% |
| NO_CHANGE | systolic_bp | 18.963 | [15.234, 22.828] | 0.000 | 34.304% |
| NO_CHANGE | diastolic_bp | 11.526 | [9.168, 13.928] | 0.000 | 33.402% |
| NO_CHANGE | bmi | 4.401 | [2.618, 7.156] | 0.000 | 50.353% |
| NO_CHANGE | glucose | 17.467 | [13.386, 21.970] | 0.000 | 44.118% |
| RIDGE | hba1c | 0.200 | [0.126, 0.273] | 0.000 | 34.800% |
| RIDGE | ldl_chol | 1.061 | [-2.393, 5.523] | 0.668 | 55.422% |
| RIDGE | hdl_chol | 1.530 | [0.971, 2.061] | 0.000 | 40.798% |
| RIDGE | total_chol | 3.496 | [1.739, 5.116] | 0.000 | 40.785% |
| RIDGE | systolic_bp | 0.886 | [0.415, 1.347] | 0.000 | 43.659% |
| RIDGE | diastolic_bp | 0.983 | [0.686, 1.294] | 0.000 | 38.174% |
| RIDGE | bmi | 3.908 | [3.504, 4.347] | 0.000 | 18.118% |
| RIDGE | glucose | 2.686 | [1.320, 4.147] | 0.000 | 42.986% |
| MLP_RAW | hba1c | 0.377 | [0.270, 0.492] | 0.000 | 32.000% |
| MLP_RAW | ldl_chol | 6.834 | [1.405, 14.617] | 0.004 | 40.964% |
| MLP_RAW | hdl_chol | 1.123 | [0.572, 1.657] | 0.000 | 45.092% |
| MLP_RAW | total_chol | 2.707 | [0.998, 4.539] | 0.002 | 43.807% |
| MLP_RAW | systolic_bp | 1.699 | [0.973, 2.573] | 0.000 | 41.164% |
| MLP_RAW | diastolic_bp | 1.333 | [0.854, 1.958] | 0.000 | 38.589% |
| MLP_RAW | bmi | 3.202 | [2.791, 3.655] | 0.000 | 20.235% |
| MLP_RAW | glucose | 4.202 | [2.552, 6.110] | 0.000 | 40.724% |
| JEPA_RIDGE | hba1c | 0.178 | [0.104, 0.256] | 0.000 | 35.200% |
| JEPA_RIDGE | ldl_chol | 4.881 | [0.408, 9.463] | 0.022 | 42.169% |
| JEPA_RIDGE | hdl_chol | 1.916 | [1.271, 2.594] | 0.000 | 42.331% |
| JEPA_RIDGE | total_chol | 3.149 | [1.466, 4.874] | 0.000 | 45.015% |
| JEPA_RIDGE | systolic_bp | -0.004 | [-0.304, 0.282] | 0.992 | 49.064% |
| JEPA_RIDGE | diastolic_bp | 0.297 | [0.082, 0.515] | 0.000 | 43.776% |
| JEPA_RIDGE | bmi | 2.130 | [1.809, 2.474] | 0.000 | 24.471% |
| JEPA_RIDGE | glucose | 3.167 | [1.523, 4.773] | 0.000 | 42.308% |
| JEPA_RIDGE_PRED | hba1c | 0.179 | [0.103, 0.256] | 0.000 | 35.200% |
| JEPA_RIDGE_PRED | ldl_chol | 2.947 | [-1.677, 7.833] | 0.200 | 49.398% |
| JEPA_RIDGE_PRED | hdl_chol | 2.014 | [1.311, 2.746] | 0.000 | 42.945% |
| JEPA_RIDGE_PRED | total_chol | 2.794 | [1.060, 4.690] | 0.000 | 44.713% |
| JEPA_RIDGE_PRED | systolic_bp | 0.051 | [-0.249, 0.358] | 0.720 | 47.609% |
| JEPA_RIDGE_PRED | diastolic_bp | 0.315 | [0.052, 0.589] | 0.010 | 43.568% |
| JEPA_RIDGE_PRED | bmi | 2.582 | [2.165, 3.041] | 0.000 | 23.529% |
| JEPA_RIDGE_PRED | glucose | 3.415 | [1.671, 5.094] | 0.000 | 42.081% |
| JEPA_TWIN | hba1c | 0.243 | [0.165, 0.327] | 0.000 | 34.000% |
| JEPA_TWIN | ldl_chol | 4.773 | [0.521, 9.537] | 0.016 | 44.578% |
| JEPA_TWIN | hdl_chol | 2.293 | [1.651, 2.986] | 0.000 | 36.196% |
| JEPA_TWIN | total_chol | 5.872 | [3.865, 8.162] | 0.000 | 40.785% |
| JEPA_TWIN | systolic_bp | 0.872 | [0.412, 1.298] | 0.000 | 42.620% |
| JEPA_TWIN | diastolic_bp | 0.510 | [0.202, 0.830] | 0.000 | 41.909% |
| JEPA_TWIN | bmi | 3.930 | [3.467, 4.414] | 0.000 | 17.882% |
| JEPA_TWIN | glucose | 3.301 | [1.836, 4.968] | 0.000 | 45.023% |
| JEPA_TWIN_NAIVE | hba1c | 0.339 | [0.238, 0.443] | 0.000 | 34.800% |
| JEPA_TWIN_NAIVE | ldl_chol | 6.655 | [2.085, 10.822] | 0.002 | 43.373% |
| JEPA_TWIN_NAIVE | hdl_chol | 4.936 | [3.929, 6.030] | 0.000 | 31.902% |
| JEPA_TWIN_NAIVE | total_chol | 7.044 | [4.746, 9.357] | 0.000 | 39.275% |
| JEPA_TWIN_NAIVE | systolic_bp | 1.320 | [0.831, 1.836] | 0.000 | 40.748% |
| JEPA_TWIN_NAIVE | diastolic_bp | 0.704 | [0.381, 1.051] | 0.000 | 42.324% |
| JEPA_TWIN_NAIVE | bmi | 4.569 | [4.015, 5.140] | 0.000 | 16.941% |
| JEPA_TWIN_NAIVE | glucose | 7.472 | [5.051, 10.021] | 0.000 | 38.009% |

## Verdicts

- POP_MEAN: macro MAE 16.294 (++3.484 vs GBT)
- NO_CHANGE: macro MAE 31.557 (++18.747 vs GBT)
- RIDGE: macro MAE 14.661 (++1.850 vs GBT)
- GBT: baseline (macro MAE 12.810)
- MLP_RAW: macro MAE 15.507 (++2.697 vs GBT)
- JEPA_RIDGE: macro MAE 14.778 (++1.968 vs GBT)
- JEPA_RIDGE_PRED: macro MAE 14.600 (++1.790 vs GBT)
- JEPA_TWIN: macro MAE 15.537 (++2.727 vs GBT)
- JEPA_TWIN_NAIVE: macro MAE 16.950 (++4.139 vs GBT)

## Propensity-matched subcohort (prescriber-confounding control)

Matched 1:1 on logit(P(z=A|x)) from multinomial LR on pooled pre-window features
(caliper = 0.2σ of logit).

### metformin_vs_lisinopril

- n_matched = **318**
- SMD(logit) pre -> post: 0.297 -> 0.027

| model | macro MAE (matched) |
|:---|---:|
| POP_MEAN | 17.436 |
| NO_CHANGE | 32.689 |
| RIDGE | 15.683 |
| GBT | 14.052 |
| MLP_RAW | 15.968 |
| JEPA_RIDGE | 15.799 |
| JEPA_RIDGE_PRED | 15.719 |
| JEPA_TWIN | 16.868 |
| JEPA_TWIN_NAIVE | 18.532 |

### metformin_vs_atorvastatin

- n_matched = **338**
- SMD(logit) pre -> post: 0.298 -> 0.029

| model | macro MAE (matched) |
|:---|---:|
| POP_MEAN | 16.284 |
| NO_CHANGE | 32.009 |
| RIDGE | 15.142 |
| GBT | 13.047 |
| MLP_RAW | 15.837 |
| JEPA_RIDGE | 14.853 |
| JEPA_RIDGE_PRED | 15.006 |
| JEPA_TWIN | 15.914 |
| JEPA_TWIN_NAIVE | 17.483 |

### atorvastatin_vs_lisinopril

- n_matched = **380**
- SMD(logit) pre -> post: 0.016 -> 0.002

| model | macro MAE (matched) |
|:---|---:|
| POP_MEAN | 16.577 |
| NO_CHANGE | 29.753 |
| RIDGE | 14.101 |
| GBT | 12.758 |
| MLP_RAW | 14.563 |
| JEPA_RIDGE | 14.994 |
| JEPA_RIDGE_PRED | 14.450 |
| JEPA_TWIN | 14.781 |
| JEPA_TWIN_NAIVE | 16.652 |
