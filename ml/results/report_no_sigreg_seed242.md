# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 84.6s
- Best val loss: **0.0202** at epoch 13

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.9515 | 0.9789 | 0.9536 |
| Cosine sim | 0.9900 | 0.9899 | 0.9899 |
| Explained var | 0.1809 | 0.1639 | 0.1728 |
| Target emb std | 0.3689 | 0.3649 | 0.3686 |
| Ctx emb std | 0.5731 | 0.5780 | 0.5965 |
| z-sensitivity | 0.2489 | 0.2493 | 0.2490 |
| z-sens / std(s_y) | 0.6748 | 0.6833 | 0.6754 |

## Linear probe (intervention ← s_x)

- Train accuracy: **49.58%**
- Test accuracy:  **42.81%**
- Majority baseline: 35.16%
  - metformin: 47.37%
  - atorvastatin: 45.78%
  - lisinopril: 34.95%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.369 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.990 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.173
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.675 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 42.8% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0274

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
  "sigreg_lambda": 0.0,
  "sigreg_projections": 1024,
  "sigreg_knots": 17,
  "sigreg_on_predictor": true,
  "orth_lambda": 0.5,
  "out_dir": "ml/results",
  "ckpt_dir": "ml/checkpoints"
}
```