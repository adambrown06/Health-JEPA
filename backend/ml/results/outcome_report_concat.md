# Outcome-prediction report — tag `concat`

- Checkpoint: `backend/ml/checkpoints/jepa_best_concat.pt`
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
| JEPA_RIDGE | 0.936 | 28.717 | 9.176 | 29.103 | 9.295 | 5.853 | 4.726 | 29.068 | 14.609 | 0.191 |
| JEPA_RIDGE_PRED | 0.943 | 26.695 | 10.326 | 28.750 | 9.474 | 5.964 | 6.427 | 29.524 | 14.763 | 0.132 |
| JEPA_TWIN | 1.066 | 32.695 | 9.214 | 31.447 | 10.011 | 6.227 | 6.376 | 29.046 | 15.760 | 0.028 |
| JEPA_TWIN_NAIVE | 1.108 | 32.581 | 11.853 | 32.403 | 10.518 | 6.565 | 6.963 | 32.877 | 16.859 | -0.081 |

## R² per model and outcome

| model | hba1c | ldl_chol | hdl_chol | total_chol | systolic_bp | diastolic_bp | bmi | glucose | macro MAE | macro R² |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| POP_MEAN | -0.004 | -0.004 | -0.001 | -0.000 | -0.004 | -0.010 | -0.000 | -0.004 | 16.294 | -0.003 |
| NO_CHANGE | -6.859 | -3.233 | -2.792 | -6.667 | -13.748 | -12.194 | -7.472 | -1.276 | 31.557 | -6.780 |
| RIDGE | 0.216 | -0.175 | 0.329 | 0.096 | 0.082 | 0.093 | 0.095 | 0.205 | 14.661 | 0.118 |
| GBT | 0.429 | 0.082 | 0.540 | 0.284 | 0.237 | 0.254 | 0.756 | 0.315 | 12.810 | 0.362 |
| MLP_RAW | -0.056 | -0.888 | 0.400 | 0.100 | -0.317 | -0.461 | 0.240 | 0.159 | 15.507 | -0.103 |
| JEPA_RIDGE | 0.258 | -0.173 | 0.297 | 0.128 | 0.169 | 0.167 | 0.467 | 0.216 | 14.609 | 0.191 |
| JEPA_RIDGE_PRED | 0.246 | -0.058 | 0.132 | 0.150 | 0.163 | 0.136 | 0.096 | 0.194 | 14.763 | 0.132 |
| JEPA_TWIN | 0.107 | -0.553 | 0.286 | 0.004 | 0.076 | 0.103 | 0.063 | 0.139 | 15.760 | 0.028 |
| JEPA_TWIN_NAIVE | -0.086 | -0.368 | -0.103 | -0.053 | -0.001 | 0.046 | -0.068 | -0.018 | 16.859 | -0.081 |

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
| JEPA_RIDGE | hba1c | 0.183 | [0.110, 0.260] | 0.000 | 36.000% |
| JEPA_RIDGE | ldl_chol | 3.891 | [-0.235, 7.999] | 0.058 | 46.988% |
| JEPA_RIDGE | hdl_chol | 1.941 | [1.267, 2.668] | 0.000 | 39.877% |
| JEPA_RIDGE | total_chol | 2.806 | [1.016, 4.594] | 0.000 | 44.411% |
| JEPA_RIDGE | systolic_bp | 0.198 | [-0.096, 0.489] | 0.186 | 47.817% |
| JEPA_RIDGE | diastolic_bp | 0.339 | [0.135, 0.558] | 0.002 | 45.021% |
| JEPA_RIDGE | bmi | 2.172 | [1.823, 2.489] | 0.000 | 25.412% |
| JEPA_RIDGE | glucose | 2.848 | [1.250, 4.566] | 0.002 | 43.439% |
| JEPA_RIDGE_PRED | hba1c | 0.192 | [0.114, 0.274] | 0.000 | 39.200% |
| JEPA_RIDGE_PRED | ldl_chol | 1.933 | [-2.516, 6.627] | 0.420 | 48.193% |
| JEPA_RIDGE_PRED | hdl_chol | 3.081 | [2.284, 3.978] | 0.000 | 36.810% |
| JEPA_RIDGE_PRED | total_chol | 2.438 | [0.754, 4.253] | 0.004 | 45.619% |
| JEPA_RIDGE_PRED | systolic_bp | 0.380 | [0.006, 0.758] | 0.050 | 45.946% |
| JEPA_RIDGE_PRED | diastolic_bp | 0.450 | [0.215, 0.698] | 0.000 | 42.324% |
| JEPA_RIDGE_PRED | bmi | 3.876 | [3.410, 4.350] | 0.000 | 21.647% |
| JEPA_RIDGE_PRED | glucose | 3.320 | [1.446, 5.180] | 0.000 | 40.498% |
| JEPA_TWIN | hba1c | 0.313 | [0.225, 0.403] | 0.000 | 30.800% |
| JEPA_TWIN | ldl_chol | 7.934 | [2.417, 14.340] | 0.002 | 45.783% |
| JEPA_TWIN | hdl_chol | 1.972 | [1.363, 2.601] | 0.000 | 40.491% |
| JEPA_TWIN | total_chol | 5.134 | [3.229, 6.955] | 0.000 | 41.088% |
| JEPA_TWIN | systolic_bp | 0.920 | [0.441, 1.405] | 0.000 | 45.322% |
| JEPA_TWIN | diastolic_bp | 0.714 | [0.388, 1.043] | 0.000 | 41.494% |
| JEPA_TWIN | bmi | 3.821 | [3.358, 4.304] | 0.000 | 19.059% |
| JEPA_TWIN | glucose | 2.853 | [1.333, 4.501] | 0.000 | 45.249% |
| JEPA_TWIN_NAIVE | hba1c | 0.357 | [0.256, 0.454] | 0.000 | 34.000% |
| JEPA_TWIN_NAIVE | ldl_chol | 7.842 | [2.842, 13.264] | 0.002 | 44.578% |
| JEPA_TWIN_NAIVE | hdl_chol | 4.627 | [3.598, 5.721] | 0.000 | 30.368% |
| JEPA_TWIN_NAIVE | total_chol | 6.078 | [4.129, 8.096] | 0.000 | 38.369% |
| JEPA_TWIN_NAIVE | systolic_bp | 1.414 | [0.900, 1.972] | 0.000 | 44.283% |
| JEPA_TWIN_NAIVE | diastolic_bp | 1.050 | [0.711, 1.418] | 0.000 | 36.929% |
| JEPA_TWIN_NAIVE | bmi | 4.416 | [3.893, 4.971] | 0.000 | 17.176% |
| JEPA_TWIN_NAIVE | glucose | 6.725 | [4.254, 9.380] | 0.000 | 35.747% |

## Verdicts

- POP_MEAN: macro MAE 16.294 (++3.484 vs GBT)
- NO_CHANGE: macro MAE 31.557 (++18.747 vs GBT)
- RIDGE: macro MAE 14.661 (++1.850 vs GBT)
- GBT: baseline (macro MAE 12.810)
- MLP_RAW: macro MAE 15.507 (++2.697 vs GBT)
- JEPA_RIDGE: macro MAE 14.609 (++1.799 vs GBT)
- JEPA_RIDGE_PRED: macro MAE 14.763 (++1.953 vs GBT)
- JEPA_TWIN: macro MAE 15.760 (++2.950 vs GBT)
- JEPA_TWIN_NAIVE: macro MAE 16.859 (++4.048 vs GBT)

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
| JEPA_RIDGE | 15.807 |
| JEPA_RIDGE_PRED | 16.135 |
| JEPA_TWIN | 16.519 |
| JEPA_TWIN_NAIVE | 18.093 |

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
| JEPA_RIDGE | 14.685 |
| JEPA_RIDGE_PRED | 14.971 |
| JEPA_TWIN | 16.382 |
| JEPA_TWIN_NAIVE | 17.304 |

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
| JEPA_RIDGE | 14.661 |
| JEPA_RIDGE_PRED | 14.750 |
| JEPA_TWIN | 14.773 |
| JEPA_TWIN_NAIVE | 16.718 |
