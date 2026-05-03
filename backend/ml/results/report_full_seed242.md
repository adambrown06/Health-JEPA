# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 108.8s
- Best val loss: **0.0143** at epoch 15

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 1.0159 | 1.0339 | 1.0181 |
| Cosine sim | 0.9929 | 0.9929 | 0.9928 |
| Explained var | 0.1578 | 0.1434 | 0.1497 |
| Target emb std | 0.3035 | 0.2994 | 0.3026 |
| Ctx emb std | 0.3726 | 0.3763 | 0.3848 |
| z-sensitivity | 0.3976 | 0.3975 | 0.3980 |
| z-sens / std(s_y) | 1.3103 | 1.3277 | 1.3151 |

## Linear probe (intervention ← s_x)

- Train accuracy: **50.72%**
- Test accuracy:  **40.94%**
- Majority baseline: 35.16%
  - metformin: 47.85%
  - atorvastatin: 40.44%
  - lisinopril: 34.47%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.303 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.993 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.150
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 1.315 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 40.9% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0180

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