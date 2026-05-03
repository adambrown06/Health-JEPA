# Causal-JEPA Training Report

- Cohort: **aou_cardiometabolic_weekly_v2**
- Device: `cuda`
- Wallclock: 84.2s
- Best val loss: **0.0244** at epoch 13

## Headline metrics

| Metric | Train | Val | Test |
|---|---:|---:|---:|
| Latent MSE | 0.8378 | 0.8656 | 0.8450 |
| Cosine sim | 0.9875 | 0.9878 | 0.9868 |
| Explained var | 0.2215 | 0.2093 | 0.2457 |
| Target emb std | 0.4100 | 0.4035 | 0.4205 |
| Ctx emb std | 0.6017 | 0.6117 | 0.6355 |
| z-sensitivity | 0.2761 | 0.2765 | 0.2789 |
| z-sens / std(s_y) | 0.6734 | 0.6852 | 0.6634 |

## Linear probe (intervention ← s_x)

- Train accuracy: **48.11%**
- Test accuracy:  **46.41%**
- Majority baseline: 35.16%
  - metformin: 54.07%
  - atorvastatin: 48.00%
  - lisinopril: 36.89%

## Quality verdicts

- **GOOD** — No representation collapse: target std = 0.420 (>0.05)
- **GOOD** — Predictor tracks target direction: test cosine = +0.987 (want > 0.5)
- **GOOD** — Positive explained variance: test EV = +0.246
- **GOOD** — Predictor conditions on intervention: Δ(ŝ_y across z)/std(s_y) = 0.663 (want > 0.10)
- **GOOD** — Encoder separates treatment cohorts: probe acc 46.4% vs majority 35.2%
- **GOOD** — No pathological train/val gap: val−train MSE = +0.0278

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
  "orth_lambda": 0.0,
  "out_dir": "ml/results",
  "ckpt_dir": "ml/checkpoints"
}
```