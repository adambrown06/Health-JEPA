# Outcome-prediction report — tag `no_lewm`

- Checkpoint: `ml/checkpoints/jepa_best_no_lewm.pt`
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
| JEPA_RIDGE | 0.920 | 27.917 | 9.186 | 28.619 | 9.048 | 5.788 | 4.762 | 29.042 | 14.410 | 0.207 |
| JEPA_RIDGE_PRED | 0.903 | 27.452 | 9.231 | 28.231 | 9.108 | 5.820 | 5.167 | 28.866 | 14.347 | 0.180 |
| JEPA_TWIN | 1.018 | 31.407 | 9.158 | 30.524 | 10.215 | 6.221 | 6.494 | 30.130 | 15.646 | 0.051 |
| JEPA_TWIN_NAIVE | 1.138 | 29.371 | 12.014 | 32.125 | 10.692 | 6.385 | 6.985 | 34.489 | 16.650 | -0.064 |

## R² per model and outcome

| model | hba1c | ldl_chol | hdl_chol | total_chol | systolic_bp | diastolic_bp | bmi | glucose | macro MAE | macro R² |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| POP_MEAN | -0.004 | -0.004 | -0.001 | -0.000 | -0.004 | -0.010 | -0.000 | -0.004 | 16.294 | -0.003 |
| NO_CHANGE | -6.859 | -3.233 | -2.792 | -6.667 | -13.748 | -12.194 | -7.472 | -1.276 | 31.557 | -6.780 |
| RIDGE | 0.216 | -0.175 | 0.329 | 0.096 | 0.082 | 0.093 | 0.095 | 0.205 | 14.661 | 0.118 |
| GBT | 0.429 | 0.082 | 0.540 | 0.284 | 0.237 | 0.254 | 0.756 | 0.315 | 12.810 | 0.362 |
| MLP_RAW | -0.056 | -0.888 | 0.400 | 0.100 | -0.317 | -0.461 | 0.240 | 0.159 | 15.507 | -0.103 |
| JEPA_RIDGE | 0.291 | -0.190 | 0.306 | 0.169 | 0.212 | 0.187 | 0.452 | 0.230 | 14.410 | 0.207 |
| JEPA_RIDGE_PRED | 0.301 | -0.259 | 0.270 | 0.196 | 0.208 | 0.130 | 0.371 | 0.223 | 14.347 | 0.180 |
| JEPA_TWIN | 0.073 | -0.335 | 0.271 | 0.027 | 0.079 | 0.107 | 0.069 | 0.113 | 15.646 | 0.051 |
| JEPA_TWIN_NAIVE | -0.181 | -0.102 | -0.102 | -0.059 | 0.015 | 0.066 | -0.045 | -0.103 | 16.650 | -0.064 |

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
| JEPA_RIDGE | hba1c | 0.169 | [0.097, 0.243] | 0.000 | 34.400% |
| JEPA_RIDGE | ldl_chol | 3.086 | [-1.156, 7.777] | 0.178 | 49.398% |
| JEPA_RIDGE | hdl_chol | 1.956 | [1.296, 2.638] | 0.000 | 42.025% |
| JEPA_RIDGE | total_chol | 2.317 | [0.681, 3.910] | 0.004 | 46.224% |
| JEPA_RIDGE | systolic_bp | -0.046 | [-0.320, 0.246] | 0.712 | 49.896% |
| JEPA_RIDGE | diastolic_bp | 0.270 | [0.072, 0.488] | 0.006 | 43.361% |
| JEPA_RIDGE | bmi | 2.202 | [1.879, 2.545] | 0.000 | 24.000% |
| JEPA_RIDGE | glucose | 2.842 | [1.159, 4.357] | 0.002 | 41.176% |
| JEPA_RIDGE_PRED | hba1c | 0.151 | [0.074, 0.229] | 0.000 | 38.000% |
| JEPA_RIDGE_PRED | ldl_chol | 2.638 | [-1.933, 7.930] | 0.306 | 53.012% |
| JEPA_RIDGE_PRED | hdl_chol | 2.004 | [1.303, 2.705] | 0.000 | 42.331% |
| JEPA_RIDGE_PRED | total_chol | 1.935 | [0.412, 3.595] | 0.018 | 44.713% |
| JEPA_RIDGE_PRED | systolic_bp | 0.017 | [-0.257, 0.301] | 0.914 | 50.312% |
| JEPA_RIDGE_PRED | diastolic_bp | 0.298 | [0.059, 0.568] | 0.010 | 45.021% |
| JEPA_RIDGE_PRED | bmi | 2.619 | [2.260, 2.991] | 0.000 | 22.588% |
| JEPA_RIDGE_PRED | glucose | 2.643 | [0.896, 4.373] | 0.002 | 41.629% |
| JEPA_TWIN | hba1c | 0.265 | [0.175, 0.355] | 0.000 | 36.400% |
| JEPA_TWIN | ldl_chol | 6.622 | [2.147, 11.523] | 0.000 | 34.940% |
| JEPA_TWIN | hdl_chol | 1.905 | [1.267, 2.559] | 0.000 | 39.571% |
| JEPA_TWIN | total_chol | 4.179 | [2.419, 6.033] | 0.000 | 44.411% |
| JEPA_TWIN | systolic_bp | 1.126 | [0.675, 1.572] | 0.000 | 39.917% |
| JEPA_TWIN | diastolic_bp | 0.708 | [0.416, 1.018] | 0.000 | 44.813% |
| JEPA_TWIN | bmi | 3.933 | [3.497, 4.405] | 0.000 | 16.471% |
| JEPA_TWIN | glucose | 3.935 | [2.164, 5.631] | 0.000 | 42.534% |
| JEPA_TWIN_NAIVE | hba1c | 0.386 | [0.276, 0.484] | 0.000 | 28.800% |
| JEPA_TWIN_NAIVE | ldl_chol | 4.595 | [0.163, 9.003] | 0.040 | 42.169% |
| JEPA_TWIN_NAIVE | hdl_chol | 4.782 | [3.750, 5.868] | 0.000 | 28.528% |
| JEPA_TWIN_NAIVE | total_chol | 5.796 | [3.641, 7.900] | 0.000 | 41.994% |
| JEPA_TWIN_NAIVE | systolic_bp | 1.593 | [1.067, 2.152] | 0.000 | 39.293% |
| JEPA_TWIN_NAIVE | diastolic_bp | 0.870 | [0.555, 1.195] | 0.000 | 41.909% |
| JEPA_TWIN_NAIVE | bmi | 4.433 | [3.939, 4.980] | 0.000 | 16.235% |
| JEPA_TWIN_NAIVE | glucose | 8.352 | [5.808, 11.118] | 0.000 | 34.842% |

## Verdicts

- POP_MEAN: macro MAE 16.294 (++3.484 vs GBT)
- NO_CHANGE: macro MAE 31.557 (++18.747 vs GBT)
- RIDGE: macro MAE 14.661 (++1.850 vs GBT)
- GBT: baseline (macro MAE 12.810)
- MLP_RAW: macro MAE 15.507 (++2.697 vs GBT)
- JEPA_RIDGE: macro MAE 14.410 (++1.600 vs GBT)
- JEPA_RIDGE_PRED: macro MAE 14.347 (++1.537 vs GBT)
- JEPA_TWIN: macro MAE 15.646 (++2.836 vs GBT)
- JEPA_TWIN_NAIVE: macro MAE 16.650 (++3.840 vs GBT)

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
| JEPA_RIDGE | 15.745 |
| JEPA_RIDGE_PRED | 15.664 |
| JEPA_TWIN | 16.620 |
| JEPA_TWIN_NAIVE | 17.910 |

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
| JEPA_RIDGE | 14.370 |
| JEPA_RIDGE_PRED | 14.513 |
| JEPA_TWIN | 16.237 |
| JEPA_TWIN_NAIVE | 17.139 |

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
| JEPA_RIDGE | 14.507 |
| JEPA_RIDGE_PRED | 14.174 |
| JEPA_TWIN | 14.933 |
| JEPA_TWIN_NAIVE | 16.828 |
