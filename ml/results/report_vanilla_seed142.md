# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 61.6s
- Best val loss: **0.0828** at epoch 5

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.8318 | 0.8359 | 0.8311 |
| Cosine sim | 0.9588 | 0.9586 | 0.9597 |
| Explained var | 0.0312 | 0.0313 | 0.0345 |
| Target emb std | 0.3286 | 0.3290 | 0.3256 |
| Ctx emb std | 0.4539 | 0.4610 | 0.4713 |
| z-sensitivity | 0.0523 | 0.0523 | 0.0532 |
| z-sens / std(s_y) | 0.1592 | 0.1589 | 0.1634 |

## Linear probe (intervention ← s_x)

- Train accuracy: **48.88%**
- Test accuracy:  **43.44%**
- Majority baseline: 35.16%
  - metformin: 47.85%
  - atorvastatin: 46.22%
  - lisinopril: 35.92%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.326 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.960 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.034
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.163 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 43.4% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0040

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
  "seed": 142,
  "split_seed": 42,
  "append_seed_to_artifacts": true,
  "num_workers": 0,
  "d_model": 128,
  "n_heads": 4,
  "n_layers": 3,
  "d_ff": 256,
  "z_dim": 16,
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
  "sigreg_lambda": 0.0,
  "sigreg_projections": 1024,
  "sigreg_knots": 17,
  "sigreg_on_predictor": true,
  "orth_lambda": 0.0,
  "out_dir": "ml/results",
  "ckpt_dir": "ml/checkpoints"
}
```