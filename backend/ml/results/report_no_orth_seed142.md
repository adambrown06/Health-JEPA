# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 92.4s
- Best val loss: **0.0197** at epoch 12

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.5318 | 0.5440 | 0.5308 |
| Cosine sim | 0.9900 | 0.9902 | 0.9900 |
| Explained var | 0.1979 | 0.1885 | 0.2064 |
| Target emb std | 0.3251 | 0.3218 | 0.3248 |
| Ctx emb std | 0.4100 | 0.4193 | 0.4313 |
| z-sensitivity | 0.5905 | 0.5910 | 0.5910 |
| z-sens / std(s_y) | 1.8166 | 1.8363 | 1.8197 |

## Linear probe (intervention ← s_x)

- Train accuracy: **49.25%**
- Test accuracy:  **42.03%**
- Majority baseline: 35.16%
  - metformin: 47.85%
  - atorvastatin: 44.89%
  - lisinopril: 33.01%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.325 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.990 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.206
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 1.820 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 42.0% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0122

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
  "seed": 142,
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
  "out_dir": "backend/ml/results",
  "ckpt_dir": "backend/ml/checkpoints"
}
```