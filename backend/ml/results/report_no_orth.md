# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 95.4s
- Best val loss: **0.0183** at epoch 13

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.7302 | 0.7487 | 0.7327 |
| Cosine sim | 0.9907 | 0.9908 | 0.9902 |
| Explained var | 0.1889 | 0.1783 | 0.2034 |
| Target emb std | 0.3247 | 0.3197 | 0.3313 |
| Ctx emb std | 0.3986 | 0.4070 | 0.4202 |
| z-sensitivity | 0.5330 | 0.5332 | 0.5336 |
| z-sens / std(s_y) | 1.6417 | 1.6677 | 1.6106 |

## Linear probe (intervention ← s_x)

- Train accuracy: **48.41%**
- Test accuracy:  **45.00%**
- Majority baseline: 35.16%
  - metformin: 54.55%
  - atorvastatin: 45.33%
  - lisinopril: 34.95%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.331 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.990 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.203
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 1.611 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 45.0% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0185

## Splits

- train: 2989 · val: 640 · test: 640
- intervention labels: ['metformin', 'atorvastatin', 'lisinopril']

## Config

```json
{
  "npz_path": "backend/training_data/patient_tensors.npz",
  "manifest_path": "backend/training_data/manifest.json",
  "intervention_map_path": "backend/training_data/intervention_map.json",
  "max_seq_len": 64,
  "batch_size": 64,
  "val_fraction": 0.15,
  "test_fraction": 0.15,
  "seed": 42,
  "num_workers": 0,
  "d_model": 128,
  "n_heads": 4,
  "n_layers": 3,
  "d_ff": 256,
  "z_dim": 64,
  "predictor_hidden": 256,
  "predictor_layers": 3,
  "predictor_style": "adaln",
  "dropout": 0.1,
  "ema_momentum_start": 0.996,
  "ema_momentum_end": 0.9999,
  "epochs": 80,
  "lr": 0.0002,
  "weight_decay": 0.0005,
  "warmup_epochs": 5,
  "grad_clip": 1.0,
  "early_stop_patience": 15,
  "loss_type": "cosine",
  "sigreg_lambda": 0.1,
  "sigreg_projections": 1024,
  "sigreg_knots": 17,
  "sigreg_on_predictor": true,
  "orth_lambda": 0.0,
  "out_dir": "backend/ml/results",
  "ckpt_dir": "backend/ml/checkpoints"
}
```