# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 61.5s
- Best val loss: **0.1156** at epoch 5

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.6348 | 0.6387 | 0.6361 |
| Cosine sim | 0.9423 | 0.9422 | 0.9432 |
| Explained var | 0.0199 | 0.0190 | 0.0204 |
| Target emb std | 0.3287 | 0.3277 | 0.3261 |
| Ctx emb std | 0.4296 | 0.4339 | 0.4418 |
| z-sensitivity | 0.0649 | 0.0648 | 0.0652 |
| z-sens / std(s_y) | 0.1974 | 0.1978 | 0.1999 |

## Linear probe (intervention ← s_x)

- Train accuracy: **49.98%**
- Test accuracy:  **42.34%**
- Majority baseline: 35.16%
  - metformin: 52.63%
  - atorvastatin: 42.22%
  - lisinopril: 32.04%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.326 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.943 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.020
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.200 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 42.3% vs majority 35.2%
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
  "seed": 242,
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