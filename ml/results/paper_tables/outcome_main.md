| model / ablation | macro MAE | macro R² |
|:---|---:|---:|
| POP_MEAN (shared) | 16.294 | -0.003 |
| NO_CHANGE (shared) | 31.557 | -6.780 |
| RIDGE (shared) | 14.661 | 0.118 |
| GBT (shared) | 12.810 | 0.362 |
| MLP_RAW (shared) | 15.507 | -0.103 |
| GRU_E2E (shared) | 13.414 | 0.299 |
| TSENCODER_E2E (shared) | 13.509 | 0.287 |
| | | |
| JEPA_RIDGE · `full` | 14.777 | 0.192 |
| JEPA_RIDGE_PRED · `full` | 14.617 | 0.155 |
| JEPA_TWIN · `full` | 15.543 | 0.036 |
| JEPA_RIDGE · `small_z` | 14.484 | 0.210 |
| JEPA_RIDGE_PRED · `small_z` | 14.585 | 0.141 |
| JEPA_TWIN · `small_z` | 14.995 | 0.086 |
| JEPA_RIDGE · `no_sigreg` | 14.408 | 0.207 |
| JEPA_RIDGE_PRED · `no_sigreg` | 14.357 | 0.180 |
| JEPA_TWIN · `no_sigreg` | 15.658 | 0.049 |
| JEPA_RIDGE · `no_orth` | 14.778 | 0.192 |
| JEPA_RIDGE_PRED · `no_orth` | 14.600 | 0.157 |
| JEPA_TWIN · `no_orth` | 15.537 | 0.039 |
| JEPA_RIDGE · `no_lewm` | 14.410 | 0.207 |
| JEPA_RIDGE_PRED · `no_lewm` | 14.347 | 0.180 |
| JEPA_TWIN · `no_lewm` | 15.646 | 0.051 |
| JEPA_RIDGE · `concat` | 14.609 | 0.191 |
| JEPA_RIDGE_PRED · `concat` | 14.763 | 0.132 |
| JEPA_TWIN · `concat` | 15.760 | 0.028 |
| JEPA_RIDGE · `vanilla` | 14.483 | 0.189 |
| JEPA_RIDGE_PRED · `vanilla` | 14.846 | 0.150 |
| JEPA_TWIN · `vanilla` | 15.669 | 0.038 |