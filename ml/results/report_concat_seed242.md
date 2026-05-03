# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 74.4s
- Best val loss: **0.1240** at epoch 5

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.5718 | 0.5766 | 0.5727 |
| Cosine sim | 0.9375 | 0.9380 | 0.9382 |
| Explained var | 0.0234 | 0.0221 | 0.0231 |
| Target emb std | 0.3213 | 0.3195 | 0.3182 |
| Ctx emb std | 0.4222 | 0.4228 | 0.4298 |
| z-sensitivity | 0.0498 | 0.0501 | 0.0502 |
| z-sens / std(s_y) | 0.1549 | 0.1567 | 0.1578 |

## Linear probe (intervention ← s_x)

- Train accuracy: **48.81%**
- Test accuracy:  **43.75%**
- Majority baseline: 35.16%
  - metformin: 55.02%
  - atorvastatin: 43.11%
  - lisinopril: 33.01%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.318 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.938 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.023
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.158 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 43.8% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0049

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
  "seed": 242,
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
  "out_dir": "ml/results",
  "ckpt_dir": "ml/checkpoints"
}
```