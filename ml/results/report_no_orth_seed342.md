# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 115.4s
- Best val loss: **0.0086** at epoch 19

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 1.9626 | 1.9896 | 1.9682 |
| Cosine sim | 0.9956 | 0.9957 | 0.9956 |
| Explained var | 0.1086 | 0.1022 | 0.0971 |
| Target emb std | 0.2831 | 0.2780 | 0.2822 |
| Ctx emb std | 0.3215 | 0.3270 | 0.3351 |
| z-sensitivity | 0.2815 | 0.2815 | 0.2815 |
| z-sens / std(s_y) | 0.9942 | 1.0125 | 0.9973 |

## Linear probe (intervention ← s_x)

- Train accuracy: **49.62%**
- Test accuracy:  **40.78%**
- Majority baseline: 35.16%
  - metformin: 49.28%
  - atorvastatin: 37.33%
  - lisinopril: 35.92%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.282 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.996 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.097
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.997 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 40.8% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0270

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
  "seed": 342,
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
  "out_dir": "ml/results",
  "ckpt_dir": "ml/checkpoints"
}
```