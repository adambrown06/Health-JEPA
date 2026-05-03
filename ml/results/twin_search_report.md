# Counterfactual Twin Search Report

- Checkpoint: `ml/checkpoints/jepa_best.pt` (val loss 0.0181 at epoch 13)
- Twin library: **3,053 training patients** indexed in Qdrant (dim=128)
- Top-K: **10**
- Test patients evaluated: **198**

## Query latency (ms)

| stat | ms |
|---|---:|
| mean | 8.27 |
| median | 7.55 |
| p95 | 11.74 |
| p99 | 13.61 |

## Treatment-match rate (top-K twins that actually received query z)

| Query z | JEPA | random | lift |
|---|---:|---:|---:|
| metformin | 25.81% | 32.53% | -6.72pp |
| atorvastatin | 60.71% | 36.36% | +24.34pp |
| lisinopril | 15.71% | 31.77% | -16.06pp |

## Outcome divergence across z queries

- HbA1c (%): mean |Δ across z queries| = **0.213** (n=198)
- LDL cholesterol: mean |Δ across z queries| = **6.344** (n=140)
- Systolic BP (mmHg): mean |Δ across z queries| = **1.905** (n=198)
- Diastolic BP: mean |Δ across z queries| = **1.111** (n=198)
- BMI: mean |Δ across z queries| = **1.093** (n=198)
- Glucose: mean |Δ across z queries| = **4.259** (n=198)
- HDL cholesterol: mean |Δ across z queries| = **2.615** (n=198)
- Total cholesterol: mean |Δ across z queries| = **7.009** (n=198)

## Verdicts

- **CONCERN** — Query z=metformin surfaces z-treated twins above random: JEPA 25.8% vs random 32.5% (lift -6.7pp)
- **GOOD** — Query z=atorvastatin surfaces z-treated twins above random: JEPA 60.7% vs random 36.4% (lift +24.3pp)
- **CONCERN** — Query z=lisinopril surfaces z-treated twins above random: JEPA 15.7% vs random 31.8% (lift -16.1pp)
- **GOOD** — Predictor output for different z yields different HbA1c (%): mean |Δ| across z = 0.21 (need > 0.05 to be clinically noticeable)
- **GOOD** — Predictor output for different z yields different LDL cholesterol: mean |Δ| across z = 6.34 (need > 2.0 to be clinically noticeable)
- **GOOD** — Predictor output for different z yields different Systolic BP (mmHg): mean |Δ| across z = 1.91 (need > 1.0 to be clinically noticeable)
- **GOOD** — Query latency interactive: p95 = 11.7 ms

## Demo — patient 1548162 (true intervention: metformin)

Actual post-window outcomes:
- HbA1c (%): 6.40
- LDL cholesterol: —
- Systolic BP (mmHg): 126.04
- Diastolic BP: 74.21
- BMI: 34.72
- Glucose: 181.00
- HDL cholesterol: —
- Total cholesterol: —

### Counterfactual: if patient takes metformin

Twin treatment mix: metformin=1, atorvastatin=6, lisinopril=3

Top-5 twins:

| rank | patient_id | actually took | cos | HbA1c | LDL | sBP | BMI |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 1771123 | atorvastatin x | +0.9976 | 5.50 | — | 142.2 | 32.8 |
| 2 | 9328948 | atorvastatin x | +0.9976 | 6.60 | — | 125.9 | 34.2 |
| 3 | 1465710 | atorvastatin x | +0.9975 | — | 87.5 | 116.9 | 25.2 |
| 4 | 8501255 | atorvastatin x | +0.9973 | — | — | 144.0 | 31.3 |
| 5 | 9402316 | lisinopril x | +0.9972 | 5.70 | — | 144.6 | 38.8 |

### Counterfactual: if patient takes atorvastatin

Twin treatment mix: metformin=2, atorvastatin=4, lisinopril=4

Top-5 twins:

| rank | patient_id | actually took | cos | HbA1c | LDL | sBP | BMI |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 1465710 | atorvastatin ✓ | +0.9977 | — | 87.5 | 116.9 | 25.2 |
| 2 | 8501255 | atorvastatin ✓ | +0.9977 | — | — | 144.0 | 31.3 |
| 3 | 1771123 | atorvastatin ✓ | +0.9974 | 5.50 | — | 142.2 | 32.8 |
| 4 | 9328948 | atorvastatin ✓ | +0.9974 | 6.60 | — | 125.9 | 34.2 |
| 5 | 9402316 | lisinopril x | +0.9972 | 5.70 | — | 144.6 | 38.8 |

### Counterfactual: if patient takes lisinopril

Twin treatment mix: metformin=2, atorvastatin=4, lisinopril=4

Top-5 twins:

| rank | patient_id | actually took | cos | HbA1c | LDL | sBP | BMI |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 8501255 | atorvastatin x | +0.9976 | — | — | 144.0 | 31.3 |
| 2 | 1465710 | atorvastatin x | +0.9975 | — | 87.5 | 116.9 | 25.2 |
| 3 | 1771123 | atorvastatin x | +0.9975 | 5.50 | — | 142.2 | 32.8 |
| 4 | 9328948 | atorvastatin x | +0.9972 | 6.60 | — | 125.9 | 34.2 |
| 5 | 9402316 | lisinopril ✓ | +0.9972 | 5.70 | — | 144.6 | 38.8 |

## Demo — patient 3455792 (true intervention: atorvastatin)

Actual post-window outcomes:
- HbA1c (%): 6.00
- LDL cholesterol: 118.00
- Systolic BP (mmHg): 146.33
- Diastolic BP: 84.75
- BMI: 41.66
- Glucose: 151.50
- HDL cholesterol: 39.00
- Total cholesterol: 183.00

### Counterfactual: if patient takes metformin

Twin treatment mix: metformin=3, atorvastatin=4, lisinopril=3

Top-5 twins:

| rank | patient_id | actually took | cos | HbA1c | LDL | sBP | BMI |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 2710157 | metformin ✓ | +0.9978 | 5.85 | — | 146.2 | 29.3 |
| 2 | 9328948 | atorvastatin x | +0.9976 | 6.60 | — | 125.9 | 34.2 |
| 3 | 8473180 | metformin ✓ | +0.9974 | 10.50 | 101.0 | 127.7 | 32.8 |
| 4 | 1309629 | atorvastatin x | +0.9973 | — | — | 130.9 | 32.4 |
| 5 | 1846213 | atorvastatin x | +0.9972 | — | — | 123.8 | 25.5 |

### Counterfactual: if patient takes atorvastatin

Twin treatment mix: metformin=5, atorvastatin=3, lisinopril=2

Top-5 twins:

| rank | patient_id | actually took | cos | HbA1c | LDL | sBP | BMI |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 2710157 | metformin x | +0.9980 | 5.85 | — | 146.2 | 29.3 |
| 2 | 9328948 | atorvastatin ✓ | +0.9975 | 6.60 | — | 125.9 | 34.2 |
| 3 | 3032507 | lisinopril x | +0.9975 | — | — | 141.2 | 37.5 |
| 4 | 8473180 | metformin x | +0.9974 | 10.50 | 101.0 | 127.7 | 32.8 |
| 5 | 1613346 | atorvastatin ✓ | +0.9972 | — | — | 143.3 | 24.4 |

### Counterfactual: if patient takes lisinopril

Twin treatment mix: metformin=5, atorvastatin=4, lisinopril=1

Top-5 twins:

| rank | patient_id | actually took | cos | HbA1c | LDL | sBP | BMI |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 2710157 | metformin x | +0.9981 | 5.85 | — | 146.2 | 29.3 |
| 2 | 3032507 | lisinopril ✓ | +0.9976 | — | — | 141.2 | 37.5 |
| 3 | 1613346 | atorvastatin x | +0.9974 | — | — | 143.3 | 24.4 |
| 4 | 9328948 | atorvastatin x | +0.9973 | 6.60 | — | 125.9 | 34.2 |
| 5 | 8473180 | metformin x | +0.9973 | 10.50 | 101.0 | 127.7 | 32.8 |

## Demo — patient 6705212 (true intervention: lisinopril)

Actual post-window outcomes:
- HbA1c (%): 6.25
- LDL cholesterol: —
- Systolic BP (mmHg): 135.00
- Diastolic BP: 65.00
- BMI: —
- Glucose: 100.50
- HDL cholesterol: 58.00
- Total cholesterol: 167.00

### Counterfactual: if patient takes metformin

Twin treatment mix: metformin=0, atorvastatin=6, lisinopril=4

Top-5 twins:

| rank | patient_id | actually took | cos | HbA1c | LDL | sBP | BMI |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 1771123 | atorvastatin x | +0.9972 | 5.50 | — | 142.2 | 32.8 |
| 2 | 9143954 | lisinopril x | +0.9969 | 7.10 | — | 124.5 | — |
| 3 | 1452192 | lisinopril x | +0.9969 | 8.43 | — | 118.5 | 40.6 |
| 4 | 8501255 | atorvastatin x | +0.9968 | — | — | 144.0 | 31.3 |
| 5 | 1845719 | lisinopril x | +0.9968 | — | — | 124.0 | 21.3 |

### Counterfactual: if patient takes atorvastatin

Twin treatment mix: metformin=0, atorvastatin=5, lisinopril=5

Top-5 twins:

| rank | patient_id | actually took | cos | HbA1c | LDL | sBP | BMI |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 8501255 | atorvastatin ✓ | +0.9971 | — | — | 144.0 | 31.3 |
| 2 | 1771123 | atorvastatin ✓ | +0.9970 | 5.50 | — | 142.2 | 32.8 |
| 3 | 1452192 | lisinopril x | +0.9969 | 8.43 | — | 118.5 | 40.6 |
| 4 | 9143954 | lisinopril x | +0.9969 | 7.10 | — | 124.5 | — |
| 5 | 1613346 | atorvastatin ✓ | +0.9967 | — | — | 143.3 | 24.4 |

### Counterfactual: if patient takes lisinopril

Twin treatment mix: metformin=0, atorvastatin=5, lisinopril=5

Top-5 twins:

| rank | patient_id | actually took | cos | HbA1c | LDL | sBP | BMI |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 1771123 | atorvastatin x | +0.9970 | 5.50 | — | 142.2 | 32.8 |
| 2 | 8501255 | atorvastatin x | +0.9969 | — | — | 144.0 | 31.3 |
| 3 | 1452192 | lisinopril ✓ | +0.9968 | 8.43 | — | 118.5 | 40.6 |
| 4 | 9143954 | lisinopril ✓ | +0.9967 | 7.10 | — | 124.5 | — |
| 5 | 1613346 | atorvastatin x | +0.9967 | — | — | 143.3 | 24.4 |