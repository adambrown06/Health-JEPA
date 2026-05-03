# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 83.8s
- Best val loss: **0.0242** at epoch 13

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.8669 | 0.8953 | 0.8741 |
| Cosine sim | 0.9876 | 0.9879 | 0.9869 |
| Explained var | 0.2189 | 0.2066 | 0.2428 |
| Target emb std | 0.4097 | 0.4032 | 0.4202 |
| Ctx emb std | 0.6014 | 0.6114 | 0.6354 |
| z-sensitivity | 0.3204 | 0.3208 | 0.3228 |
| z-sens / std(s_y) | 0.7821 | 0.7955 | 0.7681 |

## Linear probe (intervention ← s_x)

- Train accuracy: **48.08%**
- Test accuracy:  **46.25%**
- Majority baseline: 35.16%
  - metformin: 54.07%
  - atorvastatin: 47.56%
  - lisinopril: 36.89%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.420 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.987 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.243
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.768 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 46.2% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0284

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
  "sigreg_lambda": 0.0,
  "sigreg_projections": 1024,
  "sigreg_knots": 17,
  "sigreg_on_predictor": true,
  "orth_lambda": 0.5,
  "out_dir": "ml/results",
  "ckpt_dir": "ml/checkpoints"
}
```