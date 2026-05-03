# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 56.7s
- Best val loss: **0.1214** at epoch 4

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.7000 | 0.7046 | 0.7006 |
| Cosine sim | 0.9396 | 0.9393 | 0.9410 |
| Explained var | 0.0230 | 0.0220 | 0.0251 |
| Target emb std | 0.3480 | 0.3472 | 0.3454 |
| Ctx emb std | 0.4550 | 0.4557 | 0.4651 |
| z-sensitivity | 0.0720 | 0.0718 | 0.0726 |
| z-sens / std(s_y) | 0.2070 | 0.2067 | 0.2102 |

## Linear probe (intervention ← s_x)

- Train accuracy: **49.28%**
- Test accuracy:  **42.19%**
- Majority baseline: 35.16%
  - metformin: 50.24%
  - atorvastatin: 46.67%
  - lisinopril: 29.13%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.345 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.941 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.025
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.210 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 42.2% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0047

## Splits

- train: 2989 · val: 640 · test: 640
- intervention labels: ['metformin', 'atorvastatin', 'lisinopril']

## Config

```json
{
  "npz_path": "training_data/patient_tensors.npz",
  "manifest_path": "training_data/manifest.json",
  "intervention_map_path": "training_data/intervention_map.json",
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
  "sigreg_lambda": 0.0,
  "sigreg_projections": 1024,
  "sigreg_knots": 17,
  "sigreg_on_predictor": true,
  "orth_lambda": 0.0,
  "out_dir": "ml/results",
  "ckpt_dir": "ml/checkpoints"
}
```