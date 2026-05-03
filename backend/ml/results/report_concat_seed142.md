# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 73.5s
- Best val loss: **0.0903** at epoch 5

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.7221 | 0.7262 | 0.7221 |
| Cosine sim | 0.9551 | 0.9548 | 0.9559 |
| Explained var | 0.0226 | 0.0219 | 0.0237 |
| Target emb std | 0.3132 | 0.3132 | 0.3097 |
| Ctx emb std | 0.3868 | 0.3911 | 0.3961 |
| z-sensitivity | 0.0536 | 0.0543 | 0.0537 |
| z-sens / std(s_y) | 0.1710 | 0.1733 | 0.1735 |

## Linear probe (intervention ← s_x)

- Train accuracy: **49.01%**
- Test accuracy:  **45.31%**
- Majority baseline: 35.16%
  - metformin: 45.93%
  - atorvastatin: 51.56%
  - lisinopril: 37.86%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.310 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.956 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.024
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.174 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 45.3% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0041

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