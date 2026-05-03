"""
Aggregate JEPA ablation + baseline results into paper-ready tables (Markdown + CSV + LaTeX).

Reads all ``report_<tag>.json`` (training) and ``outcome_report_<tag>.json`` (downstream),
plus one run with sequence baselines for the non-JEPA comparators, and writes:

  - ml/results/paper_tables/ablation_training.md + .tex + .csv
  - ml/results/paper_tables/outcome_main.md + .tex + .csv
  - ml/results/paper_tables/outcome_per_outcome.md + .tex
  - ml/results/paper_tables/summary.json

Usage
-----
    python -m ml.aggregate_results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

# Ablation tags in display order
ABLATIONS = [
    ("full",      "AdaLN + SIGReg + Orth, z=64"),
    ("small_z",   "AdaLN + SIGReg + Orth, z=16"),
    ("no_sigreg", "AdaLN + Orth, z=64 (no SIGReg)"),
    ("no_orth",   "AdaLN + SIGReg, z=64 (no Orth)"),
    ("no_lewm",   "AdaLN only, z=64 (no SIGReg, no Orth)"),
    ("concat",    "Concat predictor + SIGReg + Orth, z=64"),
    ("vanilla",   "Concat + z=16, no LeWM regs (pre-LeWM JEPA)"),
]

DOWNSTREAM_MODELS_ORDER = [
    "POP_MEAN", "NO_CHANGE", "RIDGE", "GBT", "MLP_RAW",
    "GRU_E2E", "TSENCODER_E2E",
    "JEPA_RIDGE", "JEPA_RIDGE_PRED", "JEPA_TWIN",
]


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_training_reports() -> dict:
    out = {}
    for tag, _desc in ABLATIONS:
        p = RESULTS / f"report_{tag}.json" if tag != "full" else RESULTS / "report_full.json"
        if not p.exists():
            # fallback to legacy name for the full model
            if tag == "full" and (RESULTS / "report.json").exists():
                p = RESULTS / "report.json"
            else:
                continue
        out[tag] = json.loads(p.read_text(encoding="utf-8-sig"))
    return out


def load_outcome_reports() -> dict:
    out = {}
    for tag, _desc in ABLATIONS:
        p = RESULTS / f"outcome_report_{tag}.json"
        if p.exists():
            out[tag] = json.loads(p.read_text(encoding="utf-8-sig"))
    return out


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def fmt(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        if np.isnan(x):
            return "—"
        return f"{x:.{digits}f}"
    return str(x)


def fmt_pct(x, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.{digits}f}\\%"


# ---------------------------------------------------------------------
# Table 1 — JEPA training metrics across ablations
# ---------------------------------------------------------------------

def build_training_table(tr: dict) -> tuple[str, str, str]:
    cols = [
        ("val_loss",        "val loss"),
        ("mse",              "test MSE"),
        ("cosine_sim",       "test cos"),
        ("explained_var",    "test EV"),
        ("target_embedding_std", "σ(s_y)"),
        ("intervention_sensitivity_relative", "z-sens/σ"),
    ]

    md = ["| ablation | description | " + " | ".join(c[1] for c in cols) + " | probe acc |",
          "|:---|:---" + "|---:" * (len(cols) + 1) + "|"]
    tex = []
    tex.append(r"\begin{tabular}{ll" + "r" * (len(cols) + 1) + "}")
    tex.append(r"\toprule")
    tex.append("ablation & description & " + " & ".join(c[1].replace("σ", r"$\sigma$") for c in cols) + r" & probe acc \\")
    tex.append(r"\midrule")
    csv = ["ablation,description," + ",".join(c[0] for c in cols) + ",probe_acc"]

    def _tex_safe(s: str) -> str:
        return s.replace("_", r"\_")

    for tag, desc in ABLATIONS:
        if tag not in tr:
            continue
        r = tr[tag]
        test_m = r["metrics"]["test"]
        vals = []
        for key, _ in cols:
            if key == "val_loss":
                vals.append(r.get("best_val_loss", float("nan")))
            else:
                vals.append(test_m.get(key, float("nan")))
        probe = r["linear_probe"]["eval_acc"]
        row_md = (f"| `{tag}` | {desc} | "
                  + " | ".join(fmt(v, 4 if key == 'val_loss' else 3) for (key, _), v in zip(cols, vals))
                  + f" | {fmt_pct(probe)} |")
        md.append(row_md)
        tex.append(f"\\texttt{{{_tex_safe(tag)}}} & {desc} & "
                   + " & ".join(fmt(v, 4 if key == 'val_loss' else 3) for (key, _), v in zip(cols, vals))
                   + f" & {probe*100:.1f}\\% \\\\")
        csv.append(f"{tag},{desc}," + ",".join(str(v) for v in vals) + f",{probe}")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return "\n".join(md), "\n".join(tex), "\n".join(csv)


# ---------------------------------------------------------------------
# Table 2 — Downstream outcome prediction (macro)
# ---------------------------------------------------------------------

def build_outcome_macro_table(out: dict) -> tuple[str, str, str]:
    """One row per ablation for each JEPA-based head, plus non-JEPA baselines
    from the ``full`` run (since they don't depend on the checkpoint)."""

    full = out.get("full")
    if full is None:
        return "", "", ""

    # Baseline models (identical across ablations) taken from the `full` run.
    baseline_models = [m for m in DOWNSTREAM_MODELS_ORDER
                       if not m.startswith("JEPA_") and m in full["results"]]

    md = ["| model / ablation | macro MAE | macro R² |", "|:---|---:|---:|"]
    tex = [r"\begin{tabular}{lrr}", r"\toprule",
           r"model / ablation & macro MAE & macro $R^2$ \\", r"\midrule"]
    csv = ["model_or_ablation,macro_mae,macro_r2"]

    def _tex_safe(s: str) -> str:
        return s.replace("_", r"\_")

    # Baselines
    for m in baseline_models:
        r = full["results"][m]
        md.append(f"| {m} (shared) | {fmt(r['macro_mae'])} | {fmt(r['macro_r2'])} |")
        tex.append(f"{_tex_safe(m)} (shared) & {fmt(r['macro_mae'])} & {fmt(r['macro_r2'])} \\\\")
        csv.append(f"{m},{r['macro_mae']},{r['macro_r2']}")

    # JEPA variants — one row per ablation × {RIDGE, RIDGE_PRED, TWIN}
    tex.append(r"\midrule")
    md.append("| | | |")
    for tag, desc in ABLATIONS:
        if tag not in out:
            continue
        for jm in ["JEPA_RIDGE", "JEPA_RIDGE_PRED", "JEPA_TWIN"]:
            if jm in out[tag]["results"]:
                r = out[tag]["results"][jm]
                md.append(f"| {jm} · `{tag}` | {fmt(r['macro_mae'])} | {fmt(r['macro_r2'])} |")
                tex.append(f"{_tex_safe(jm)} [{_tex_safe(tag)}] & {fmt(r['macro_mae'])} & {fmt(r['macro_r2'])} \\\\")
                csv.append(f"{jm}|{tag},{r['macro_mae']},{r['macro_r2']}")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return "\n".join(md), "\n".join(tex), "\n".join(csv)


# ---------------------------------------------------------------------
# Table 3 — Per-outcome MAE table for the FULL model
# ---------------------------------------------------------------------

def build_outcome_per_outcome_table(out: dict) -> tuple[str, str]:
    full = out.get("full")
    if full is None:
        return "", ""
    outcomes = full["outcomes"]
    models = [m for m in DOWNSTREAM_MODELS_ORDER if m in full["results"]]

    # Pretty labels
    pretty = {
        "hba1c": "HbA1c", "ldl_chol": "LDL", "hdl_chol": "HDL",
        "total_chol": "TC", "triglycerides": "TG", "glucose": "Glucose",
        "creatinine": "Cr", "systolic_bp": "sBP", "diastolic_bp": "dBP",
        "bmi": "BMI",
    }
    col_labels = [pretty.get(o, o) for o in outcomes]

    md = ["| model | " + " | ".join(col_labels) + " | macro |",
          "|:---" + "|---:" * (len(outcomes) + 1) + "|"]

    tex = [r"\begin{tabular}{l" + "r" * (len(outcomes) + 1) + "}",
           r"\toprule",
           "model & " + " & ".join(col_labels) + r" & macro \\", r"\midrule"]

    # Identify best (lowest) MAE per outcome for highlighting
    best_per_col = {}
    for o in outcomes:
        best = (None, float("inf"))
        for m in models:
            v = full["results"][m]["per_outcome"][o]["mae"]
            if v is None or np.isnan(v):
                continue
            if v < best[1]:
                best = (m, v)
        best_per_col[o] = best[0]

    for m in models:
        row_md = [m]
        row_tex = [m.replace("_", r"\_")]
        r = full["results"][m]
        for o in outcomes:
            v = r["per_outcome"][o]["mae"]
            s = fmt(v)
            if best_per_col.get(o) == m and s != "—":
                s_tex = r"\textbf{" + s + "}"
                s_md = f"**{s}**"
            else:
                s_tex = s
                s_md = s
            row_md.append(s_md)
            row_tex.append(s_tex)
        row_md.append(fmt(r["macro_mae"]))
        row_tex.append(fmt(r["macro_mae"]))
        md.append("| " + " | ".join(row_md) + " |")
        tex.append(" & ".join(row_tex) + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return "\n".join(md), "\n".join(tex)


# ---------------------------------------------------------------------
# Table 4 — Retrieval (from twin_search_report / paper_eval_report)
# ---------------------------------------------------------------------

def build_retrieval_table() -> tuple[str, str]:
    paper = RESULTS / "paper_eval_report.json"
    if not paper.exists():
        return "", ""
    d = json.loads(paper.read_text(encoding="utf-8-sig"))
    q = d.get("qdrant", {}) or d.get("qdrant_unfiltered", {})
    # paper_eval.py actually writes under `qdrant_twins_unfiltered`
    q = d.get("qdrant_twins_unfiltered") or d.get("qdrant") or {}
    if not q:
        return "", ""
    md = ["| query z | JEPA hit@10 | random | lift |",
          "|:---|---:|---:|---:|"]
    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"query $z$ & JEPA hit@10 & random & lift \\", r"\midrule"]
    for z_label, row in q.items():
        if not isinstance(row, dict):
            continue
        jepa = row.get("hit_at_10", row.get("jepa"))
        rnd = row.get("random_rate", row.get("random"))
        lift = row.get("lift_pp", row.get("lift"))
        if jepa is None:
            continue
        md.append(f"| {z_label} | {jepa*100:.2f}% | {rnd*100:.2f}% | {lift:+.2f}pp |")
        tex.append(f"{z_label} & {jepa*100:.2f}\\% & {rnd*100:.2f}\\% & {lift:+.2f}pp \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return "\n".join(md), "\n".join(tex)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    tr = load_training_reports()
    out = load_outcome_reports()

    print(f"Loaded training reports: {list(tr.keys())}")
    print(f"Loaded outcome reports: {list(out.keys())}")

    # Table 1
    md1, tex1, csv1 = build_training_table(tr)
    (OUT / "ablation_training.md").write_text(md1, encoding="utf-8")
    (OUT / "ablation_training.tex").write_text(tex1, encoding="utf-8")
    (OUT / "ablation_training.csv").write_text(csv1, encoding="utf-8")

    # Table 2
    md2, tex2, csv2 = build_outcome_macro_table(out)
    (OUT / "outcome_main.md").write_text(md2, encoding="utf-8")
    (OUT / "outcome_main.tex").write_text(tex2, encoding="utf-8")
    (OUT / "outcome_main.csv").write_text(csv2, encoding="utf-8")

    # Table 3
    md3, tex3 = build_outcome_per_outcome_table(out)
    (OUT / "outcome_per_outcome.md").write_text(md3, encoding="utf-8")
    (OUT / "outcome_per_outcome.tex").write_text(tex3, encoding="utf-8")

    # Retrieval
    md4, tex4 = build_retrieval_table()
    if md4:
        (OUT / "retrieval.md").write_text(md4, encoding="utf-8")
        (OUT / "retrieval.tex").write_text(tex4, encoding="utf-8")

    summary = {
        "trainings_loaded": list(tr.keys()),
        "outcomes_loaded": list(out.keys()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] wrote tables to {OUT}")


if __name__ == "__main__":
    main()
