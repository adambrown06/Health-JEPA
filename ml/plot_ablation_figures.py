"""
Generate paper-ready ablation figures.

- fig_adaln_matters.png: test EV and z-sensitivity across ablations
- fig_downstream_bars.png: macro MAE bars for all models
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
OUT = Path("docs/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

ABLATIONS = [
    ("full",      "full"),
    ("small_z",   "small_z"),
    ("no_sigreg", "no_sigreg"),
    ("no_orth",   "no_orth"),
    ("no_lewm",   "no_lewm"),
    ("concat",    "concat"),
    ("vanilla",   "vanilla"),
]


def _read(name: str) -> dict:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else {}


def plot_adaln_matters():
    tags, ev, zsens = [], [], []
    for tag, _ in ABLATIONS:
        r = _read(f"report_{tag}.json")
        if not r:
            continue
        tags.append(tag)
        ev.append(r["metrics"]["test"]["explained_var"])
        zsens.append(r["metrics"]["test"]["intervention_sensitivity_relative"])
    colors = ["#4daf4a" if t not in ("concat", "vanilla") else "#e41a1c" for t in tags]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(tags))
    axes[0].bar(x, ev, color=colors, edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
    axes[0].set_title("Test explained variance\n(higher = better; chance = 0)")
    axes[0].set_ylabel("EV"); axes[0].grid(alpha=0.3, axis="y")
    axes[0].axhline(0, color="black", lw=0.6)
    for i, v in enumerate(ev):
        axes[0].text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)

    axes[1].bar(x, zsens, color=colors, edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
    axes[1].set_title("Intervention sensitivity / $\\sigma(s_y)$\n(higher = predictor really uses $z$)")
    axes[1].set_ylabel("z-sens / $\\sigma$")
    axes[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(zsens):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle("AdaLN-Zero is the critical ingredient — removing it collapses both EV and z-sensitivity.",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "fig_adaln_matters.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


def plot_downstream_bars():
    full = _read("outcome_report_full.json")
    if not full:
        return
    order = [
        "POP_MEAN", "NO_CHANGE", "MLP_RAW", "RIDGE",
        "JEPA_TWIN", "JEPA_RIDGE", "JEPA_RIDGE_PRED",
        "TSENCODER_E2E", "GRU_E2E", "GBT",
    ]
    models = [m for m in order if m in full["results"]]
    maes = [full["results"][m]["macro_mae"] for m in models]
    colors = []
    for m in models:
        if m == "GBT":
            colors.append("#377eb8")
        elif m.startswith("JEPA_"):
            colors.append("#4daf4a")
        elif m.endswith("_E2E"):
            colors.append("#984ea3")
        else:
            colors.append("#a6a6a6")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(models))
    bars = ax.barh(x, maes, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(x); ax.set_yticklabels(models, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("macro MAE over 8 biomarkers (lower = better)")
    ax.set_title("Downstream outcome prediction — full cohort, test split")
    ax.grid(alpha=0.3, axis="x")
    for bar, v in zip(bars, maes):
        ax.text(v + 0.1, bar.get_y() + bar.get_height()/2, f"{v:.2f}",
                va="center", fontsize=8)
    plt.tight_layout()
    out = OUT / "fig_downstream_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    plot_adaln_matters()
    plot_downstream_bars()
