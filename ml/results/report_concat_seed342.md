# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 73.9s
- Best val loss: **0.0830** at epoch 5

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.7717 | 0.7741 | 0.7722 |
| Cosine sim | 0.9584 | 0.9585 | 0.9591 |
| Explained var | 0.0187 | 0.0177 | 0.0179 |
| Target emb std | 0.3140 | 0.3129 | 0.3109 |
| Ctx emb std | 0.3878 | 0.3915 | 0.3949 |
| z-sensitivity | 0.0566 | 0.0567 | 0.0568 |
| z-sens / std(s_y) | 0.1801 | 0.1812 | 0.1828 |

## Linear probe (intervention ← s_x)

- Train accuracy: **49.58%**
- Test accuracy:  **40.62%**
- Majority baseline: 35.16%
  - metformin: 44.50%
  - atorvastatin: 43.11%
  - lisinopril: 33.98%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.311 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.959 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.018
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.183 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 40.6% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0024

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
  "seed": 342,
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