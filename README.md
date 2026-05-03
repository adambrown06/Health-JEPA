# Health-JEPA

Action-conditional **Joint-Embedding Predictive Architecture (JEPA)** for irregular clinical time series, trained on an **All of Us** cardiometabolic cohort. This repository is the research artifact: model code, evaluation scripts, logged experiment outputs, and the ML4H-style paper sources under [`docs/paper/`](docs/paper/).

---

## Archive: Cohort Compass platform

The prior **Cohort Compass** monorepo (Next.js frontend, FastAPI/Celery backend, and `backend/ml/` layout) is frozen on branch **`archive/cohort-compass-full`** at the last snapshot before this extraction. To inspect or revive that tree:

```bash
git fetch origin
git checkout archive/cohort-compass-full
```

---

## Abstract

Clinicians often ask how a patient’s labs might evolve under one drug versus another. A useful answer needs both a **unified patient representation** (for retrieval-style reasoning across heterogeneous biomarkers) and **strong action conditioning** (to forecast trajectories under alternative interventions). **Health-JEPA** is an action-conditional JEPA for irregular clinical time series: it encodes pre-intervention history into a latent state and predicts, under a discrete intervention, the post-intervention latent—one label-free embedding that supports linear read-out of routine biomarkers, counterfactual twin retrieval, and systematic ablations of design choices from the video world-model literature.

Using the All of Us cardiometabolic cohort (**N = 4,269** patients, three first-line drugs, **25** biomarker/wearable channels), we ablate **AdaLN-Zero** conditioning, the **SIGReg** isotropic-Gaussian penalty, and **action-embedding orthogonality**, and benchmark against trees, MLPs, a bidirectional GRU, and an architecture-matched Transformer regressor. Three findings stand out:

1. **AdaLN-Zero is critical:** removing it collapses explained variance (about **0.20 → ≤ 0.03**) and z-sensitivity by an order of magnitude.
2. **Video-scale regularizers (SIGReg, orthogonality)** do not improve downstream prediction at this scale; **AdaLN-only** is the Pareto-best JEPA configuration here.
3. **Residual twin retrieval** (twins contribute deviations from a ridge baseline, not raw means) reduces selection bias and narrows the gap between naive twin averaging and linear JEPA readouts.

Metrics are also reported on a **propensity-score-matched** subcohort for prescriber-confounding control. The takeaway: world-model recipes validated on massive video should be **revalidated**, not blindly imported, for clinical time series—and a single unified action-conditional latent is the main clinical value-add of JEPA in this setting.

Full mathematical setup and citations are in [`docs/paper/paper.tex`](docs/paper/paper.tex) (or [`docs/paper/paper.md`](docs/paper/paper.md) for a readable draft).

---

## Results at a glance

Precomputed tables and figures from the completed experiment grid:

| Artifact | Description |
|----------|-------------|
| [`ml/results/paper_tables/outcome_main.md`](ml/results/paper_tables/outcome_main.md) | Macro **MAE** / **R²**: baselines vs JEPA heads × ablations |
| [`ml/results/paper_tables/ablation_training.md`](ml/results/paper_tables/ablation_training.md) | Training-time metrics across ablations |
| [`ml/results/ablation_multiseed_summary.json`](ml/results/ablation_multiseed_summary.json) | Mean ± std over seeds (where applicable) |
| [`docs/paper/figures/`](docs/paper/figures/) | Paper figures (e.g. AdaLN sensitivity, downstream bars) |

**Quick numbers (macro MAE on shared test split, from `outcome_main`):** strong classical baselines land near **12.8** (GBT) and **13.7** (GRU); frozen **JEPA + ridge** variants sit near **14.3–14.8** depending on ablation—competitive for a **single** unified embedding across **eight** outcomes, with AdaLN-heavy configs favored. See the table file above for the full matrix (including twin heads and propensity-matched analyses in the JSON reports).

---

## Repository layout

```
ml/                 # PyTorch model, training, baselines, Qdrant/twin eval, aggregated results
notebooks/          # All of Us extraction notebook → training shard (paths described inside)
docs/paper/         # paper.tex / paper.md, figures, exported tables
training_data/      # Not in git — place NPZ + manifests here (see Data below)
```

Large generated artifacts (**checkpoints**, local **Qdrant** stores) are listed in `.gitignore`; reproduce them with the commands below.

---

## Setup

From the repository root (required so imports resolve as `python -m ml.…`):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Data:** Per the All of Us DUA, participant tensors are not redistributed. Approved researchers can rebuild the cohort via the Researcher Workbench using [`notebooks/aou_extraction_pipeline.ipynb`](notebooks/aou_extraction_pipeline.ipynb), then place:

- `training_data/patient_tensors.npz`
- `training_data/manifest.json`
- `training_data/intervention_map.json`
- `training_data/feature_vocab.json` (for some eval paths)

---

## Reproduce experiments

All paths assume **current working directory = repo root**.

| Step | Command |
|------|---------|
| Train main model + ablations | `python -m ml.train_jepa` (see CLI flags in `ml/train_jepa.py`) |
| Batch ablations (PowerShell) | `powershell -File ml/run_ablations.ps1` |
| Downstream outcome eval | `powershell -File ml/run_outcome_evals.ps1` or `python -m ml.outcome_eval --help` |
| Regenerate paper tables | `python -m ml.aggregate_results` |
| Regenerate figures | `python -m ml.plot_ablation_figures` |
| Multi-seed training sweep | `python -m ml.run_ablation_seeds` |

---

## GitHub and local folder name

1. **GitHub:** In **Settings → General → Repository name**, set **`Health-JEPA`**.  
   This clone’s `origin` URL is already set to `https://github.com/adambrown06/Health-JEPA.git`; if your GitHub repo is still named `Cohort-Compass`, rename it on GitHub first (or run `git remote set-url origin …` to match whatever URL you use).

2. **Local folder:** Close Cursor (the IDE locks the directory on Windows), then rename **`Cohort Compass`** → **`Health-JEPA`** in Explorer, or from PowerShell:
   ```powershell
   Rename-Item -LiteralPath "$env:USERPROFILE\Cohort Compass" -NewName "Health-JEPA"
   ```
   Alternatively, run **`scripts/rename-folder-after-close-ide.ps1`** after closing the IDE (it renames the repo root folder).
