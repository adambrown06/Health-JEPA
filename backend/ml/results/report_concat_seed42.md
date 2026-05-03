# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 60.7s
- Best val loss: **0.1081** at epoch 5

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.6858 | 0.6955 | 0.6845 |
| Cosine sim | 0.9463 | 0.9459 | 0.9453 |
| Explained var | 0.0298 | 0.0276 | 0.0306 |
| Target emb std | 0.3362 | 0.3356 | 0.3374 |
| Ctx emb std | 0.4523 | 0.4528 | 0.4643 |
| z-sensitivity | 0.0489 | 0.0490 | 0.0490 |
| z-sens / std(s_y) | 0.1455 | 0.1459 | 0.1452 |

## Linear probe (intervention ← s_x)

- Train accuracy: **49.55%**
- Test accuracy:  **42.97%**
- Majority baseline: 35.16%
  - metformin: 52.63%
  - atorvastatin: 44.44%
  - lisinopril: 31.55%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.337 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.945 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.031
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.145 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 43.0% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0097

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
  "split_seed": 42,
  "append_seed_to_artifacts": true,
  "num_workers": 0,
  "d_model": 128,
  "n_heads": 4,
  "n_layers": 3,
  "d_ff": 256,
  "z_dim": 64,
  "predictor_hidden": 256,
  "predictor_layers": 3,
  "predictor_style": "concat",
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