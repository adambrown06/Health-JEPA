# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 96.6s
- Best val loss: **0.0181** at epoch 13

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.7621 | 0.7811 | 0.7645 |
| Cosine sim | 0.9908 | 0.9909 | 0.9903 |
| Explained var | 0.1866 | 0.1764 | 0.2012 |
| Target emb std | 0.3242 | 0.3193 | 0.3309 |
| Ctx emb std | 0.3976 | 0.4061 | 0.4193 |
| z-sensitivity | 0.5141 | 0.5144 | 0.5146 |
| z-sens / std(s_y) | 1.5858 | 1.6111 | 1.5553 |

## Linear probe (intervention ← s_x)

- Train accuracy: **48.61%**
- Test accuracy:  **45.00%**
- Majority baseline: 35.16%
  - metformin: 54.07%
  - atorvastatin: 45.78%
  - lisinopril: 34.95%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.331 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.990 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.201
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 1.555 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 45.0% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0189

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
  "out_dir": "ml/results",
  "ckpt_dir": "ml/checkpoints"
}
```