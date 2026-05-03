# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 50.5s
- Best val loss: **0.1126** at epoch 4

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.7065 | 0.7133 | 0.7076 |
| Cosine sim | 0.9444 | 0.9437 | 0.9441 |
| Explained var | 0.0201 | 0.0184 | 0.0220 |
| Target emb std | 0.3388 | 0.3393 | 0.3400 |
| Ctx emb std | 0.4275 | 0.4272 | 0.4407 |
| z-sensitivity | 0.0910 | 0.0904 | 0.0903 |
| z-sens / std(s_y) | 0.2685 | 0.2665 | 0.2656 |

## Linear probe (intervention ← s_x)

- Train accuracy: **50.42%**
- Test accuracy:  **44.53%**
- Majority baseline: 35.16%
  - metformin: 52.15%
  - atorvastatin: 43.11%
  - lisinopril: 38.35%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.340 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.944 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.022
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.266 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 44.5% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0069

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
  "seed": 42,
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