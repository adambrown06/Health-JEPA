# Outcome-prediction report — tag `no_sigreg`

- Checkpoint: `ml/checkpoints/jepa_best_no_sigreg.pt`
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
| JEPA_RIDGE | 0.920 | 27.909 | 9.185 | 28.615 | 9.049 | 5.788 | 4.761 | 29.037 | 14.408 | 0.207 |
| JEPA_RIDGE_PRED | 0.905 | 27.472 | 9.242 | 28.239 | 9.116 | 5.809 | 5.167 | 28.907 | 14.357 | 0.180 |
| JEPA_TWIN | 1.024 | 31.405 | 9.124 | 30.638 | 10.242 | 6.251 | 6.449 | 30.129 | 15.658 | 0.049 |
| JEPA_TWIN_NAIVE | 1.137 | 29.043 | 11.907 | 32.335 | 10.696 | 6.424 | 6.930 | 34.373 | 16.606 | -0.062 |

## R² per model and outcome

| model | hba1c | ldl_chol | hdl_chol | total_chol | systolic_bp | diastolic_bp | bmi | glucose | macro MAE | macro R² |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| POP_MEAN | -0.004 | -0.004 | -0.001 | -0.000 | -0.004 | -0.010 | -0.000 | -0.004 | 16.294 | -0.003 |
| NO_CHANGE | -6.859 | -3.233 | -2.792 | -6.667 | -13.748 | -12.194 | -7.472 | -1.276 | 31.557 | -6.780 |
| RIDGE | 0.216 | -0.175 | 0.329 | 0.096 | 0.082 | 0.093 | 0.095 | 0.205 | 14.661 | 0.118 |
| GBT | 0.429 | 0.082 | 0.540 | 0.284 | 0.237 | 0.254 | 0.756 | 0.315 | 12.810 | 0.362 |
| MLP_RAW | -0.056 | -0.888 | 0.400 | 0.100 | -0.317 | -0.461 | 0.240 | 0.159 | 15.507 | -0.103 |
| JEPA_RIDGE | 0.291 | -0.189 | 0.306 | 0.170 | 0.212 | 0.187 | 0.452 | 0.230 | 14.408 | 0.207 |
| JEPA_RIDGE_PRED | 0.300 | -0.264 | 0.268 | 0.196 | 0.208 | 0.143 | 0.374 | 0.219 | 14.357 | 0.180 |
| JEPA_TWIN | 0.068 | -0.334 | 0.271 | 0.021 | 0.074 | 0.106 | 0.078 | 0.113 | 15.658 | 0.049 |
| JEPA_TWIN_NAIVE | -0.177 | -0.095 | -0.095 | -0.074 | 0.009 | 0.062 | -0.034 | -0.096 | 16.606 | -0.062 |

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
| JEPA_RIDGE | hba1c | 0.169 | [0.097, 0.243] | 0.000 | 34.800% |
| JEPA_RIDGE | ldl_chol | 3.077 | [-1.155, 7.778] | 0.176 | 49.398% |
| JEPA_RIDGE | hdl_chol | 1.955 | [1.296, 2.637] | 0.000 | 42.025% |
| JEPA_RIDGE | total_chol | 2.313 | [0.674, 3.908] | 0.004 | 46.224% |
| JEPA_RIDGE | systolic_bp | -0.045 | [-0.316, 0.248] | 0.720 | 49.896% |
| JEPA_RIDGE | diastolic_bp | 0.270 | [0.072, 0.488] | 0.006 | 43.361% |
| JEPA_RIDGE | bmi | 2.201 | [1.878, 2.545] | 0.000 | 24.000% |
| JEPA_RIDGE | glucose | 2.837 | [1.155, 4.347] | 0.002 | 41.403% |
| JEPA_RIDGE_PRED | hba1c | 0.153 | [0.075, 0.231] | 0.000 | 38.000% |
| JEPA_RIDGE_PRED | ldl_chol | 2.654 | [-1.920, 7.953] | 0.300 | 50.602% |
| JEPA_RIDGE_PRED | hdl_chol | 2.014 | [1.310, 2.722] | 0.000 | 42.638% |
| JEPA_RIDGE_PRED | total_chol | 1.944 | [0.439, 3.562] | 0.014 | 44.713% |
| JEPA_RIDGE_PRED | systolic_bp | 0.026 | [-0.245, 0.313] | 0.866 | 49.896% |
| JEPA_RIDGE_PRED | diastolic_bp | 0.288 | [0.055, 0.544] | 0.012 | 44.606% |
| JEPA_RIDGE_PRED | bmi | 2.618 | [2.265, 2.985] | 0.000 | 23.059% |
| JEPA_RIDGE_PRED | glucose | 2.682 | [0.919, 4.462] | 0.002 | 42.760% |
| JEPA_TWIN | hba1c | 0.270 | [0.180, 0.357] | 0.000 | 36.000% |
| JEPA_TWIN | ldl_chol | 6.618 | [2.109, 11.568] | 0.002 | 34.940% |
| JEPA_TWIN | hdl_chol | 1.873 | [1.248, 2.530] | 0.000 | 40.184% |
| JEPA_TWIN | total_chol | 4.295 | [2.541, 6.197] | 0.000 | 44.411% |
| JEPA_TWIN | systolic_bp | 1.152 | [0.721, 1.607] | 0.000 | 41.164% |
| JEPA_TWIN | diastolic_bp | 0.737 | [0.437, 1.047] | 0.000 | 45.021% |
| JEPA_TWIN | bmi | 3.889 | [3.455, 4.361] | 0.000 | 18.118% |
| JEPA_TWIN | glucose | 3.938 | [2.209, 5.643] | 0.000 | 42.760% |
| JEPA_TWIN_NAIVE | hba1c | 0.384 | [0.272, 0.485] | 0.000 | 29.600% |
| JEPA_TWIN_NAIVE | ldl_chol | 4.277 | [-0.114, 8.586] | 0.060 | 43.373% |
| JEPA_TWIN_NAIVE | hdl_chol | 4.674 | [3.645, 5.745] | 0.000 | 29.448% |
| JEPA_TWIN_NAIVE | total_chol | 6.007 | [3.815, 8.133] | 0.000 | 41.390% |
| JEPA_TWIN_NAIVE | systolic_bp | 1.596 | [1.062, 2.171] | 0.000 | 40.333% |
| JEPA_TWIN_NAIVE | diastolic_bp | 0.908 | [0.599, 1.235] | 0.000 | 41.079% |
| JEPA_TWIN_NAIVE | bmi | 4.379 | [3.882, 4.924] | 0.000 | 17.647% |
| JEPA_TWIN_NAIVE | glucose | 8.239 | [5.706, 10.978] | 0.000 | 35.973% |

## Verdicts

- POP_MEAN: macro MAE 16.294 (++3.484 vs GBT)
- NO_CHANGE: macro MAE 31.557 (++18.747 vs GBT)
- RIDGE: macro MAE 14.661 (++1.850 vs GBT)
- GBT: baseline (macro MAE 12.810)
- MLP_RAW: macro MAE 15.507 (++2.697 vs GBT)
- JEPA_RIDGE: macro MAE 14.408 (++1.598 vs GBT)
- JEPA_RIDGE_PRED: macro MAE 14.357 (++1.547 vs GBT)
- JEPA_TWIN: macro MAE 15.658 (++2.847 vs GBT)
- JEPA_TWIN_NAIVE: macro MAE 16.606 (++3.795 vs GBT)

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
| JEPA_RIDGE | 15.742 |
| JEPA_RIDGE_PRED | 15.687 |
| JEPA_TWIN | 16.670 |
| JEPA_TWIN_NAIVE | 17.864 |

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
| JEPA_RIDGE | 14.371 |
| JEPA_RIDGE_PRED | 14.518 |
| JEPA_TWIN | 16.296 |
| JEPA_TWIN_NAIVE | 17.072 |

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
| JEPA_RIDGE | 14.503 |
| JEPA_RIDGE_PRED | 14.174 |
| JEPA_TWIN | 14.882 |
| JEPA_TWIN_NAIVE | 16.807 |
