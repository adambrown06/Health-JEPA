# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 81.3s
- Best val loss: **0.0259** at epoch 12

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.5805 | 0.6025 | 0.5807 |
| Cosine sim | 0.9870 | 0.9871 | 0.9869 |
| Explained var | 0.2507 | 0.2384 | 0.2614 |
| Target emb std | 0.4121 | 0.4086 | 0.4123 |
| Ctx emb std | 0.6576 | 0.6737 | 0.6905 |
| z-sensitivity | 0.4203 | 0.4209 | 0.4220 |
| z-sens / std(s_y) | 1.0200 | 1.0301 | 1.0237 |

## Linear probe (intervention ← s_x)

- Train accuracy: **48.98%**
- Test accuracy:  **44.22%**
- Majority baseline: 35.16%
  - metformin: 48.33%
  - atorvastatin: 50.22%
  - lisinopril: 33.50%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.412 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.987 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.261
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 1.024 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 44.2% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0220

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
  "sigreg_lambda": 0.0,
  "sigreg_projections": 1024,
  "sigreg_knots": 17,
  "sigreg_on_predictor": true,
  "orth_lambda": 0.0,
  "out_dir": "backend/ml/results",
  "ckpt_dir": "backend/ml/checkpoints"
}
```