# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 102.6s
- Best val loss: **0.0144** at epoch 15

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.9767 | 0.9944 | 0.9789 |
| Cosine sim | 0.9928 | 0.9928 | 0.9928 |
| Explained var | 0.1606 | 0.1459 | 0.1520 |
| Target emb std | 0.3041 | 0.3000 | 0.3032 |
| Ctx emb std | 0.3743 | 0.3780 | 0.3865 |
| z-sensitivity | 0.4005 | 0.4004 | 0.4009 |
| z-sens / std(s_y) | 1.3172 | 1.3350 | 1.3220 |

## Linear probe (intervention ← s_x)

- Train accuracy: **50.75%**
- Test accuracy:  **40.78%**
- Majority baseline: 35.16%
  - metformin: 47.85%
  - atorvastatin: 40.44%
  - lisinopril: 33.98%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.303 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.993 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.152
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 1.322 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 40.8% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0176

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
  "sigreg_lambda": 0.1,
  "sigreg_projections": 1024,
  "sigreg_knots": 17,
  "sigreg_on_predictor": true,
  "orth_lambda": 0.0,
  "out_dir": "ml/results",
  "ckpt_dir": "ml/checkpoints"
}
```