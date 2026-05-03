# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 105.6s
- Best val loss: **0.0124** at epoch 16

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 1.5577 | 1.5839 | 1.5572 |
| Cosine sim | 0.9938 | 0.9938 | 0.9935 |
| Explained var | 0.1657 | 0.1567 | 0.1741 |
| Target emb std | 0.3151 | 0.3101 | 0.3211 |
| Ctx emb std | 0.3697 | 0.3766 | 0.3917 |
| z-sensitivity | 0.3953 | 0.3953 | 0.3953 |
| z-sens / std(s_y) | 1.2545 | 1.2750 | 1.2313 |

## Linear probe (intervention ← s_x)

- Train accuracy: **48.95%**
- Test accuracy:  **43.59%**
- Majority baseline: 35.16%
  - metformin: 51.67%
  - atorvastatin: 41.33%
  - lisinopril: 37.86%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.321 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.994 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.174
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 1.231 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 43.6% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0262

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