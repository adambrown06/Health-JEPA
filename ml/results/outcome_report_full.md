# Outcome-prediction report — tag `full`

- Checkpoint: `ml/checkpoints/jepa_best_full.pt`
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
| GRU_E2E | 0.771 | 25.245 | 8.286 | 27.628 | 8.953 | 5.567 | 4.755 | 26.105 | 13.414 | 0.299 |
| TSENCODER_E2E | 0.832 | 23.722 | 7.958 | 27.600 | 9.205 | 5.641 | 4.982 | 28.129 | 13.509 | 0.287 |
| JEPA_RIDGE | 0.929 | 29.698 | 9.151 | 29.463 | 9.093 | 5.813 | 4.685 | 29.382 | 14.777 | 0.192 |
| JEPA_RIDGE_PRED | 0.933 | 27.765 | 9.235 | 29.208 | 9.152 | 5.838 | 5.151 | 29.651 | 14.617 | 0.155 |
| JEPA_TWIN | 1.008 | 29.517 | 9.578 | 32.087 | 10.017 | 6.049 | 6.496 | 29.594 | 15.543 | 0.036 |
| JEPA_TWIN_NAIVE | 1.105 | 31.629 | 12.189 | 33.318 | 10.447 | 6.215 | 7.114 | 33.687 | 16.963 | -0.095 |

## R² per model and outcome

| model | hba1c | ldl_chol | hdl_chol | total_chol | systolic_bp | diastolic_bp | bmi | glucose | macro MAE | macro R² |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| POP_MEAN | -0.004 | -0.004 | -0.001 | -0.000 | -0.004 | -0.010 | -0.000 | -0.004 | 16.294 | -0.003 |
| NO_CHANGE | -6.859 | -3.233 | -2.792 | -6.667 | -13.748 | -12.194 | -7.472 | -1.276 | 31.557 | -6.780 |
| RIDGE | 0.216 | -0.175 | 0.329 | 0.096 | 0.082 | 0.093 | 0.095 | 0.205 | 14.661 | 0.118 |
| GBT | 0.429 | 0.082 | 0.540 | 0.284 | 0.237 | 0.254 | 0.756 | 0.315 | 12.810 | 0.362 |
| MLP_RAW | -0.056 | -0.888 | 0.400 | 0.100 | -0.317 | -0.461 | 0.240 | 0.159 | 15.507 | -0.103 |
| GRU_E2E | 0.424 | 0.065 | 0.420 | 0.194 | 0.230 | 0.255 | 0.468 | 0.339 | 13.414 | 0.299 |
| TSENCODER_E2E | 0.338 | 0.141 | 0.462 | 0.216 | 0.217 | 0.239 | 0.410 | 0.274 | 13.509 | 0.287 |
| JEPA_RIDGE | 0.280 | -0.237 | 0.309 | 0.119 | 0.206 | 0.181 | 0.464 | 0.217 | 14.777 | 0.192 |
| JEPA_RIDGE_PRED | 0.271 | -0.230 | 0.283 | 0.128 | 0.204 | 0.109 | 0.280 | 0.197 | 14.617 | 0.155 |
| JEPA_TWIN | 0.125 | -0.421 | 0.249 | -0.073 | 0.094 | 0.139 | 0.029 | 0.148 | 15.543 | 0.036 |
| JEPA_TWIN_NAIVE | -0.124 | -0.387 | -0.111 | -0.144 | 0.051 | 0.115 | -0.094 | -0.068 | 16.963 | -0.095 |

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
| GRU_E2E | hba1c | 0.017 | [-0.046, 0.080] | 0.596 | 44.000% |
| GRU_E2E | ldl_chol | 0.372 | [-2.132, 3.058] | 0.836 | 49.398% |
| GRU_E2E | hdl_chol | 1.047 | [0.504, 1.574] | 0.000 | 46.012% |
| GRU_E2E | total_chol | 1.349 | [-0.217, 2.810] | 0.084 | 46.828% |
| GRU_E2E | systolic_bp | -0.138 | [-0.421, 0.133] | 0.348 | 53.222% |
| GRU_E2E | diastolic_bp | 0.050 | [-0.123, 0.227] | 0.592 | 50.207% |
| GRU_E2E | bmi | 2.199 | [1.879, 2.549] | 0.000 | 25.882% |
| GRU_E2E | glucose | -0.111 | [-1.466, 1.250] | 0.896 | 48.869% |
| TSENCODER_E2E | hba1c | 0.080 | [0.018, 0.147] | 0.014 | 39.600% |
| TSENCODER_E2E | ldl_chol | -1.112 | [-3.494, 1.409] | 0.372 | 49.398% |
| TSENCODER_E2E | hdl_chol | 0.719 | [0.254, 1.139] | 0.004 | 45.092% |
| TSENCODER_E2E | total_chol | 1.293 | [-0.112, 2.717] | 0.068 | 47.432% |
| TSENCODER_E2E | systolic_bp | 0.115 | [-0.217, 0.461] | 0.556 | 51.143% |
| TSENCODER_E2E | diastolic_bp | 0.125 | [-0.098, 0.356] | 0.302 | 46.058% |
| TSENCODER_E2E | bmi | 2.429 | [2.074, 2.796] | 0.000 | 27.059% |
| TSENCODER_E2E | glucose | 1.930 | [0.606, 3.307] | 0.002 | 42.308% |
| JEPA_RIDGE | hba1c | 0.178 | [0.104, 0.256] | 0.000 | 35.200% |
| JEPA_RIDGE | ldl_chol | 4.862 | [0.374, 9.454] | 0.022 | 42.169% |
| JEPA_RIDGE | hdl_chol | 1.921 | [1.275, 2.600] | 0.000 | 41.718% |
| JEPA_RIDGE | total_chol | 3.149 | [1.464, 4.873] | 0.000 | 44.713% |
| JEPA_RIDGE | systolic_bp | -0.003 | [-0.304, 0.283] | 0.986 | 49.272% |
| JEPA_RIDGE | diastolic_bp | 0.297 | [0.083, 0.516] | 0.000 | 43.983% |
| JEPA_RIDGE | bmi | 2.128 | [1.808, 2.473] | 0.000 | 24.471% |
| JEPA_RIDGE | glucose | 3.172 | [1.533, 4.786] | 0.000 | 42.308% |
| JEPA_RIDGE_PRED | hba1c | 0.181 | [0.107, 0.258] | 0.000 | 35.600% |
| JEPA_RIDGE_PRED | ldl_chol | 2.934 | [-1.658, 7.876] | 0.206 | 48.193% |
| JEPA_RIDGE_PRED | hdl_chol | 2.008 | [1.315, 2.732] | 0.000 | 43.252% |
| JEPA_RIDGE_PRED | total_chol | 2.903 | [1.175, 4.833] | 0.000 | 44.411% |
| JEPA_RIDGE_PRED | systolic_bp | 0.058 | [-0.235, 0.372] | 0.688 | 47.817% |
| JEPA_RIDGE_PRED | diastolic_bp | 0.319 | [0.055, 0.601] | 0.008 | 43.776% |
| JEPA_RIDGE_PRED | bmi | 2.592 | [2.178, 3.034] | 0.000 | 23.529% |
| JEPA_RIDGE_PRED | glucose | 3.426 | [1.682, 5.126] | 0.000 | 42.308% |
| JEPA_TWIN | hba1c | 0.257 | [0.176, 0.345] | 0.000 | 33.200% |
| JEPA_TWIN | ldl_chol | 4.654 | [0.290, 9.508] | 0.034 | 46.988% |
| JEPA_TWIN | hdl_chol | 2.336 | [1.696, 3.017] | 0.000 | 36.196% |
| JEPA_TWIN | total_chol | 5.793 | [3.738, 7.947] | 0.000 | 41.088% |
| JEPA_TWIN | systolic_bp | 0.919 | [0.452, 1.339] | 0.000 | 41.164% |
| JEPA_TWIN | diastolic_bp | 0.533 | [0.222, 0.861] | 0.000 | 42.116% |
| JEPA_TWIN | bmi | 3.941 | [3.491, 4.418] | 0.000 | 17.647% |
| JEPA_TWIN | glucose | 3.391 | [1.906, 5.052] | 0.000 | 44.118% |
| JEPA_TWIN_NAIVE | hba1c | 0.352 | [0.247, 0.453] | 0.000 | 33.600% |
| JEPA_TWIN_NAIVE | ldl_chol | 6.689 | [1.980, 11.406] | 0.002 | 43.373% |
| JEPA_TWIN_NAIVE | hdl_chol | 4.964 | [3.953, 6.111] | 0.000 | 31.902% |
| JEPA_TWIN_NAIVE | total_chol | 7.001 | [4.769, 9.196] | 0.000 | 38.369% |
| JEPA_TWIN_NAIVE | systolic_bp | 1.338 | [0.840, 1.866] | 0.000 | 40.125% |
| JEPA_TWIN_NAIVE | diastolic_bp | 0.701 | [0.372, 1.046] | 0.000 | 42.739% |
| JEPA_TWIN_NAIVE | bmi | 4.567 | [4.023, 5.149] | 0.000 | 16.471% |
| JEPA_TWIN_NAIVE | glucose | 7.513 | [5.092, 10.073] | 0.000 | 37.330% |

## Verdicts

- POP_MEAN: macro MAE 16.294 (++3.484 vs GBT)
- NO_CHANGE: macro MAE 31.557 (++18.747 vs GBT)
- RIDGE: macro MAE 14.661 (++1.850 vs GBT)
- GBT: baseline (macro MAE 12.810)
- MLP_RAW: macro MAE 15.507 (++2.697 vs GBT)
- GRU_E2E: macro MAE 13.414 (++0.603 vs GBT)
- TSENCODER_E2E: macro MAE 13.509 (++0.698 vs GBT)
- JEPA_RIDGE: macro MAE 14.777 (++1.967 vs GBT)
- JEPA_RIDGE_PRED: macro MAE 14.617 (++1.806 vs GBT)
- JEPA_TWIN: macro MAE 15.543 (++2.733 vs GBT)
- JEPA_TWIN_NAIVE: macro MAE 16.963 (++4.153 vs GBT)

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
| GRU_E2E | 14.779 |
| TSENCODER_E2E | 14.734 |
| JEPA_RIDGE | 15.798 |
| JEPA_RIDGE_PRED | 15.766 |
| JEPA_TWIN | 16.755 |
| JEPA_TWIN_NAIVE | 18.390 |

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
| GRU_E2E | 13.085 |
| TSENCODER_E2E | 13.611 |
| JEPA_RIDGE | 14.857 |
| JEPA_RIDGE_PRED | 15.017 |
| JEPA_TWIN | 16.019 |
| JEPA_TWIN_NAIVE | 17.570 |

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
| GRU_E2E | 13.130 |
| TSENCODER_E2E | 13.236 |
| JEPA_RIDGE | 14.989 |
| JEPA_RIDGE_PRED | 14.472 |
| JEPA_TWIN | 14.774 |
| JEPA_TWIN_NAIVE | 16.633 |
