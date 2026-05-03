# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 95.5s
- Best val loss: **0.0273** at epoch 13

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.4447 | 0.4571 | 0.4512 |
| Cosine sim | 0.9863 | 0.9864 | 0.9862 |
| Explained var | 0.2339 | 0.2263 | 0.2273 |
| Target emb std | 0.3550 | 0.3503 | 0.3537 |
| Ctx emb std | 0.4903 | 0.4936 | 0.5056 |
| z-sensitivity | 0.6131 | 0.6132 | 0.6126 |
| z-sens / std(s_y) | 1.7269 | 1.7502 | 1.7319 |

## Linear probe (intervention ← s_x)

- Train accuracy: **49.18%**
- Test accuracy:  **44.22%**
- Majority baseline: 35.16%
  - metformin: 50.72%
  - atorvastatin: 46.67%
  - lisinopril: 34.95%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.354 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.986 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.227
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 1.732 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 44.2% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0124

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
  "seed": 442,
  "split_seed": 42,
  "append_seed_to_artifacts": true,
  "num_workers": 0,
  "d_model": 128,
  "n_heads": 4,
  "n_layers": 3,
  "d_ff": 256,
  "z_dim": 16,
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
  "orth_lambda": 0.5,
  "out_dir": "backend/ml/results",
  "ckpt_dir": "backend/ml/checkpoints"
}
```