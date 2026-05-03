# Outcome-prediction report — tag `small_z`

- Checkpoint: `ml/checkpoints/jepa_best_small_z.pt`
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
| JEPA_RIDGE | 0.938 | 29.116 | 9.062 | 28.610 | 9.067 | 5.768 | 4.499 | 28.813 | 14.484 | 0.210 |
| JEPA_RIDGE_PRED | 0.947 | 29.264 | 9.259 | 28.250 | 9.136 | 5.883 | 4.910 | 29.028 | 14.585 | 0.141 |
| JEPA_TWIN | 0.953 | 27.897 | 8.639 | 30.864 | 9.988 | 6.209 | 6.506 | 28.902 | 14.995 | 0.086 |
| JEPA_TWIN_NAIVE | 1.084 | 31.045 | 11.407 | 32.549 | 10.284 | 6.274 | 7.088 | 33.141 | 16.609 | -0.047 |

## R² per model and outcome

| model | hba1c | ldl_chol | hdl_chol | total_chol | systolic_bp | diastolic_bp | bmi | glucose | macro MAE | macro R² |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| POP_MEAN | -0.004 | -0.004 | -0.001 | -0.000 | -0.004 | -0.010 | -0.000 | -0.004 | 16.294 | -0.003 |
| NO_CHANGE | -6.859 | -3.233 | -2.792 | -6.667 | -13.748 | -12.194 | -7.472 | -1.276 | 31.557 | -6.780 |
| RIDGE | 0.216 | -0.175 | 0.329 | 0.096 | 0.082 | 0.093 | 0.095 | 0.205 | 14.661 | 0.118 |
| GBT | 0.429 | 0.082 | 0.540 | 0.284 | 0.237 | 0.254 | 0.756 | 0.315 | 12.810 | 0.362 |
| MLP_RAW | -0.056 | -0.888 | 0.400 | 0.100 | -0.317 | -0.461 | 0.240 | 0.159 | 15.507 | -0.103 |
| JEPA_RIDGE | 0.262 | -0.213 | 0.322 | 0.175 | 0.213 | 0.184 | 0.505 | 0.231 | 14.484 | 0.210 |
| JEPA_RIDGE_PRED | 0.246 | -0.530 | 0.283 | 0.197 | 0.206 | 0.121 | 0.398 | 0.204 | 14.585 | 0.141 |
| JEPA_TWIN | 0.209 | -0.286 | 0.348 | 0.030 | 0.080 | 0.102 | 0.039 | 0.161 | 14.995 | 0.086 |
| JEPA_TWIN_NAIVE | -0.075 | -0.303 | 0.005 | -0.050 | 0.039 | 0.103 | -0.087 | -0.007 | 16.609 | -0.047 |

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
| JEPA_RIDGE | hba1c | 0.186 | [0.112, 0.260] | 0.000 | 35.200% |
| JEPA_RIDGE | ldl_chol | 4.268 | [-0.367, 9.341] | 0.076 | 44.578% |
| JEPA_RIDGE | hdl_chol | 1.835 | [1.169, 2.511] | 0.000 | 43.252% |
| JEPA_RIDGE | total_chol | 2.324 | [0.624, 4.048] | 0.004 | 44.713% |
| JEPA_RIDGE | systolic_bp | -0.033 | [-0.316, 0.256] | 0.820 | 50.104% |
| JEPA_RIDGE | diastolic_bp | 0.249 | [0.055, 0.461] | 0.008 | 44.398% |
| JEPA_RIDGE | bmi | 1.941 | [1.632, 2.283] | 0.000 | 25.176% |
| JEPA_RIDGE | glucose | 2.591 | [0.923, 4.183] | 0.002 | 42.534% |
| JEPA_RIDGE_PRED | hba1c | 0.195 | [0.122, 0.271] | 0.000 | 36.400% |
| JEPA_RIDGE_PRED | ldl_chol | 4.449 | [-1.389, 11.655] | 0.148 | 49.398% |
| JEPA_RIDGE_PRED | hdl_chol | 2.028 | [1.337, 2.715] | 0.000 | 42.331% |
| JEPA_RIDGE_PRED | total_chol | 1.962 | [0.350, 3.623] | 0.008 | 45.619% |
| JEPA_RIDGE_PRED | systolic_bp | 0.039 | [-0.242, 0.317] | 0.802 | 48.649% |
| JEPA_RIDGE_PRED | diastolic_bp | 0.362 | [0.139, 0.618] | 0.000 | 42.324% |
| JEPA_RIDGE_PRED | bmi | 2.357 | [1.997, 2.774] | 0.000 | 24.235% |
| JEPA_RIDGE_PRED | glucose | 2.795 | [1.087, 4.442] | 0.000 | 44.570% |
| JEPA_TWIN | hba1c | 0.200 | [0.123, 0.277] | 0.000 | 38.400% |
| JEPA_TWIN | ldl_chol | 3.071 | [-1.826, 8.587] | 0.258 | 44.578% |
| JEPA_TWIN | hdl_chol | 1.392 | [0.811, 1.954] | 0.000 | 42.638% |
| JEPA_TWIN | total_chol | 4.544 | [2.774, 6.353] | 0.000 | 39.577% |
| JEPA_TWIN | systolic_bp | 0.888 | [0.457, 1.297] | 0.000 | 42.827% |
| JEPA_TWIN | diastolic_bp | 0.695 | [0.397, 1.001] | 0.000 | 42.116% |
| JEPA_TWIN | bmi | 3.949 | [3.498, 4.423] | 0.000 | 17.412% |
| JEPA_TWIN | glucose | 2.710 | [1.173, 4.382] | 0.002 | 44.118% |
| JEPA_TWIN_NAIVE | hba1c | 0.330 | [0.234, 0.433] | 0.000 | 31.600% |
| JEPA_TWIN_NAIVE | ldl_chol | 6.225 | [1.414, 11.246] | 0.014 | 36.145% |
| JEPA_TWIN_NAIVE | hdl_chol | 4.187 | [3.189, 5.239] | 0.000 | 31.595% |
| JEPA_TWIN_NAIVE | total_chol | 6.218 | [4.094, 8.238] | 0.000 | 38.369% |
| JEPA_TWIN_NAIVE | systolic_bp | 1.174 | [0.712, 1.675] | 0.000 | 41.164% |
| JEPA_TWIN_NAIVE | diastolic_bp | 0.757 | [0.472, 1.077] | 0.000 | 41.079% |
| JEPA_TWIN_NAIVE | bmi | 4.539 | [4.014, 5.108] | 0.000 | 17.882% |
| JEPA_TWIN_NAIVE | glucose | 6.984 | [4.405, 9.468] | 0.000 | 36.878% |

## Verdicts

- POP_MEAN: macro MAE 16.294 (++3.484 vs GBT)
- NO_CHANGE: macro MAE 31.557 (++18.747 vs GBT)
- RIDGE: macro MAE 14.661 (++1.850 vs GBT)
- GBT: baseline (macro MAE 12.810)
- MLP_RAW: macro MAE 15.507 (++2.697 vs GBT)
- JEPA_RIDGE: macro MAE 14.484 (++1.674 vs GBT)
- JEPA_RIDGE_PRED: macro MAE 14.585 (++1.774 vs GBT)
- JEPA_TWIN: macro MAE 14.995 (++2.185 vs GBT)
- JEPA_TWIN_NAIVE: macro MAE 16.609 (++3.799 vs GBT)

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
| JEPA_RIDGE | 15.554 |
| JEPA_RIDGE_PRED | 15.537 |
| JEPA_TWIN | 15.270 |
| JEPA_TWIN_NAIVE | 17.540 |

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
| JEPA_RIDGE | 14.360 |
| JEPA_RIDGE_PRED | 14.282 |
| JEPA_TWIN | 15.451 |
| JEPA_TWIN_NAIVE | 17.087 |

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
| JEPA_RIDGE | 14.685 |
| JEPA_RIDGE_PRED | 14.678 |
| JEPA_TWIN | 14.658 |
| JEPA_TWIN_NAIVE | 16.730 |
