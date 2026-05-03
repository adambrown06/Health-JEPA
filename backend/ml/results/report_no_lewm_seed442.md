# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 82.5s
- Best val loss: **0.0264** at epoch 12

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.5480 | 0.5693 | 0.5526 |
| Cosine sim | 0.9871 | 0.9868 | 0.9872 |
| Explained var | 0.1929 | 0.1786 | 0.1884 |
| Target emb std | 0.3875 | 0.3872 | 0.3839 |
| Ctx emb std | 0.6055 | 0.6140 | 0.6196 |
| z-sensitivity | 0.4190 | 0.4196 | 0.4161 |
| z-sens / std(s_y) | 1.0812 | 1.0837 | 1.0839 |

## Linear probe (intervention ← s_x)

- Train accuracy: **48.58%**
- Test accuracy:  **43.28%**
- Majority baseline: 35.16%
  - metformin: 50.72%
  - atorvastatin: 43.56%
  - lisinopril: 35.44%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.384 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.987 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.188
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 1.084 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 43.3% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0213

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
  "seed": 442,
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