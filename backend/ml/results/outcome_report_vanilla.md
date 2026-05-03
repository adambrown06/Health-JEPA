# Outcome-prediction report — tag `vanilla`

- Checkpoint: `backend/ml/checkpoints/jepa_best_vanilla.pt`
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
| JEPA_RIDGE | 0.923 | 28.098 | 9.108 | 28.844 | 9.202 | 5.827 | 4.666 | 29.194 | 14.483 | 0.189 |
| JEPA_RIDGE_PRED | 0.894 | 27.432 | 9.899 | 29.521 | 9.376 | 5.845 | 5.907 | 29.892 | 14.846 | 0.150 |
| JEPA_TWIN | 1.125 | 31.571 | 8.850 | 31.742 | 10.238 | 6.254 | 6.356 | 29.211 | 15.669 | 0.038 |
| JEPA_TWIN_NAIVE | 1.211 | 28.514 | 11.303 | 33.237 | 10.909 | 6.599 | 6.931 | 32.545 | 16.406 | -0.062 |

## R² per model and outcome

| model | hba1c | ldl_chol | hdl_chol | total_chol | systolic_bp | diastolic_bp | bmi | glucose | macro MAE | macro R² |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| POP_MEAN | -0.004 | -0.004 | -0.001 | -0.000 | -0.004 | -0.010 | -0.000 | -0.004 | 16.294 | -0.003 |
| NO_CHANGE | -6.859 | -3.233 | -2.792 | -6.667 | -13.748 | -12.194 | -7.472 | -1.276 | 31.557 | -6.780 |
| RIDGE | 0.216 | -0.175 | 0.329 | 0.096 | 0.082 | 0.093 | 0.095 | 0.205 | 14.661 | 0.118 |
| GBT | 0.429 | 0.082 | 0.540 | 0.284 | 0.237 | 0.254 | 0.756 | 0.315 | 12.810 | 0.362 |
| MLP_RAW | -0.056 | -0.888 | 0.400 | 0.100 | -0.317 | -0.461 | 0.240 | 0.159 | 15.507 | -0.103 |
| JEPA_RIDGE | 0.280 | -0.280 | 0.312 | 0.140 | 0.203 | 0.172 | 0.454 | 0.232 | 14.483 | 0.189 |
| JEPA_RIDGE_PRED | 0.302 | -0.084 | 0.223 | 0.104 | 0.167 | 0.156 | 0.128 | 0.205 | 14.846 | 0.150 |
| JEPA_TWIN | 0.023 | -0.391 | 0.315 | -0.022 | 0.032 | 0.130 | 0.073 | 0.140 | 15.669 | 0.038 |
| JEPA_TWIN_NAIVE | -0.146 | -0.131 | -0.055 | -0.110 | -0.063 | 0.062 | -0.056 | 0.006 | 16.406 | -0.062 |

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
| JEPA_RIDGE | hba1c | 0.172 | [0.101, 0.247] | 0.000 | 32.000% |
| JEPA_RIDGE | ldl_chol | 3.288 | [-1.376, 8.798] | 0.200 | 46.988% |
| JEPA_RIDGE | hdl_chol | 1.881 | [1.220, 2.615] | 0.000 | 42.638% |
| JEPA_RIDGE | total_chol | 2.527 | [0.902, 4.104] | 0.002 | 43.505% |
| JEPA_RIDGE | systolic_bp | 0.108 | [-0.199, 0.411] | 0.448 | 47.609% |
| JEPA_RIDGE | diastolic_bp | 0.310 | [0.108, 0.510] | 0.002 | 44.191% |
| JEPA_RIDGE | bmi | 2.104 | [1.761, 2.453] | 0.000 | 25.882% |
| JEPA_RIDGE | glucose | 2.993 | [1.396, 4.590] | 0.002 | 40.045% |
| JEPA_RIDGE_PRED | hba1c | 0.143 | [0.055, 0.225] | 0.002 | 41.200% |
| JEPA_RIDGE_PRED | ldl_chol | 2.654 | [-1.350, 7.077] | 0.204 | 49.398% |
| JEPA_RIDGE_PRED | hdl_chol | 2.668 | [1.913, 3.374] | 0.000 | 37.423% |
| JEPA_RIDGE_PRED | total_chol | 3.216 | [1.352, 5.030] | 0.002 | 43.202% |
| JEPA_RIDGE_PRED | systolic_bp | 0.281 | [-0.081, 0.644] | 0.122 | 45.946% |
| JEPA_RIDGE_PRED | diastolic_bp | 0.327 | [0.111, 0.564] | 0.002 | 44.606% |
| JEPA_RIDGE_PRED | bmi | 3.348 | [2.892, 3.854] | 0.000 | 20.706% |
| JEPA_RIDGE_PRED | glucose | 3.704 | [1.953, 5.527] | 0.002 | 37.557% |
| JEPA_TWIN | hba1c | 0.372 | [0.276, 0.469] | 0.000 | 28.800% |
| JEPA_TWIN | ldl_chol | 6.644 | [2.070, 11.982] | 0.000 | 40.964% |
| JEPA_TWIN | hdl_chol | 1.605 | [0.996, 2.195] | 0.000 | 37.730% |
| JEPA_TWIN | total_chol | 5.435 | [3.418, 7.255] | 0.000 | 40.785% |
| JEPA_TWIN | systolic_bp | 1.141 | [0.605, 1.645] | 0.000 | 43.035% |
| JEPA_TWIN | diastolic_bp | 0.743 | [0.421, 1.068] | 0.000 | 43.154% |
| JEPA_TWIN | bmi | 3.801 | [3.331, 4.264] | 0.000 | 18.824% |
| JEPA_TWIN | glucose | 3.014 | [1.424, 4.847] | 0.000 | 45.928% |
| JEPA_TWIN_NAIVE | hba1c | 0.460 | [0.358, 0.562] | 0.000 | 27.600% |
| JEPA_TWIN_NAIVE | ldl_chol | 3.608 | [-0.839, 7.709] | 0.120 | 46.988% |
| JEPA_TWIN_NAIVE | hdl_chol | 4.072 | [3.030, 5.211] | 0.000 | 33.742% |
| JEPA_TWIN_NAIVE | total_chol | 6.916 | [4.841, 9.067] | 0.000 | 38.369% |
| JEPA_TWIN_NAIVE | systolic_bp | 1.803 | [1.214, 2.407] | 0.000 | 43.451% |
| JEPA_TWIN_NAIVE | diastolic_bp | 1.086 | [0.723, 1.453] | 0.000 | 37.344% |
| JEPA_TWIN_NAIVE | bmi | 4.380 | [3.820, 4.960] | 0.000 | 19.529% |
| JEPA_TWIN_NAIVE | glucose | 6.397 | [3.946, 8.921] | 0.000 | 37.557% |

## Verdicts

- POP_MEAN: macro MAE 16.294 (++3.484 vs GBT)
- NO_CHANGE: macro MAE 31.557 (++18.747 vs GBT)
- RIDGE: macro MAE 14.661 (++1.850 vs GBT)
- GBT: baseline (macro MAE 12.810)
- MLP_RAW: macro MAE 15.507 (++2.697 vs GBT)
- JEPA_RIDGE: macro MAE 14.483 (++1.672 vs GBT)
- JEPA_RIDGE_PRED: macro MAE 14.846 (++2.035 vs GBT)
- JEPA_TWIN: macro MAE 15.669 (++2.858 vs GBT)
- JEPA_TWIN_NAIVE: macro MAE 16.406 (++3.596 vs GBT)

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
| JEPA_RIDGE | 15.774 |
| JEPA_RIDGE_PRED | 16.068 |
| JEPA_TWIN | 16.322 |
| JEPA_TWIN_NAIVE | 17.433 |

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
| JEPA_RIDGE | 14.204 |
| JEPA_RIDGE_PRED | 15.044 |
| JEPA_TWIN | 16.338 |
| JEPA_TWIN_NAIVE | 16.900 |

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
| JEPA_RIDGE | 14.504 |
| JEPA_RIDGE_PRED | 14.482 |
| JEPA_TWIN | 14.960 |
| JEPA_TWIN_NAIVE | 16.289 |
