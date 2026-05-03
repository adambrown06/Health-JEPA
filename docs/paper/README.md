# Paper package

This folder is the self-contained draft of the paper
**"Action-Conditional JEPA for Clinical Trajectory Modelling: AdaLN-Zero
Helps, World-Model Regularizers Don't."**

## Files

| path | role |
|---|---|
| `paper.md` | Readable Markdown draft (recommended first read). |
| `paper.tex` | **Single-file Overleaf-ready:** paste entire file into the editor; bib + tables + figures are inlined (no uploads). |
| `paper.bib` | BibTeX references (JEPA, LeWorldModel, DiT/AdaLN, clinical baselines). |
| `tables/ablation_training.tex` | Table 1 — training ablations (auto-generated). |
| `tables/outcome_main.tex` | Table 2 — macro MAE / R² across models × ablations. |
| `tables/outcome_per_outcome.tex` | Table 3 — per-biomarker MAE for the full model. |
| `figures/fig_adaln_matters.png` | Figure 1 — EV and z-sensitivity across ablations. |
| `figures/fig_downstream_bars.png` | Figure 2 — macro MAE bars for all competitors. |

## Regenerating everything

From the repo root:

```powershell
# 1. Train all ablations (~10 min total on an RTX 4060 laptop)
powershell -File ml/run_ablations.ps1

# 2. Run the real-unit outcome evaluation for every checkpoint
powershell -File ml/run_outcome_evals.ps1

# 3. Rebuild tables (LaTeX / Markdown / CSV)
python -m ml.aggregate_results

# 4. Rebuild the two paper figures
python -m ml.plot_ablation_figures
```

Training uses data in `training_data/patient_tensors.npz`
(N=4,269 patient-intervention examples, 25 channels, 3 interventions).

## Compiling the PDF

### Overleaf --- paste-only (simplest)

1. Create a blank project, open **main.tex**, select all, delete.
2. Copy the **entire** contents of `paper.tex` from this repo and paste.
3. **Recompile** with **pdfLaTeX**. Overleaf will write `paper.bib` from the
   embedded `filecontents` block on the first run and run **BibTeX**
   automatically.

### Overleaf --- zip upload (if you use split files instead)

1. Zip **the contents** of this folder so the archive root looks like:
   ```
   paper.tex
   paper.bib
   figures/
   tables/
   ```
   (Do **not** nest an extra `docs/paper/` level unless `figures/` and
   `tables/` live under the same folder as `paper.tex`.)
2. In Overleaf: **New Project → Upload project** and select the zip.
3. **Menu** → **Main document** → choose `paper.tex`.
4. **Menu** → **Compiler** → **pdfLaTeX** (default).
5. **Recompile**. Overleaf runs **BibTeX** for `\bibliography{paper}`.

The LaTeX source uses **natbib before hyperref**, **T1 + UTF-8**, ASCII
punctuation in `paper.tex`, and standard **PNG** figures.

### Local toolchains

- **Tectonic** (from this directory): `tectonic paper.tex`
- **MiKTeX / TeX Live**: `latexmk -pdf paper.tex` or
  `pdflatex` → `bibtex paper` → `pdflatex` ×2

## Headline claims the paper defends

1. **AdaLN-Zero is the single critical ingredient.** Removing it collapses
   test explained variance from 0.20 → ≤ 0.03 and z-sensitivity by an
   order of magnitude (Figure 1, Table 1).
2. **LeWorldModel regularizers do *not* transfer to this dataset.** SIGReg
   and action-orthogonality don't help; the AdaLN-only variant (`no_lewm`)
   has the best macro MAE among JEPA heads (14.35).
3. **Frozen JEPA embeddings are competitive with end-to-end neural
   baselines** (macro MAE 14.35 vs 13.67 for a bidirectional GRU) while
   covering *all eight outcomes with one label-free embedding*, and come
   within two MAE points of gradient-boosted trees (12.81).
4. **Naive twin retrieval is biased** and worse than the population-mean
   head — the paper names this and proposes residual retrieval as the
   obvious fix.

These are all negative-result-aware findings; the paper is honest about
the ~4,269-sample scale and the single-seed limitation.
