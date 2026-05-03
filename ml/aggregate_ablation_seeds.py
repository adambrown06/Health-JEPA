"""
Aggregate ``report_<tag>_seed*.json`` from multi-seed ablation runs.

Writes ``ml/results/ablation_multiseed_summary.json`` with mean ± std
for val loss, test EV, σ(s_y), z-sens/σ. Prints LaTeX table rows to stdout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
RESULTS = _ROOT / "ml" / "results"

TAGS = ["full", "small_z", "no_sigreg", "no_orth", "no_lewm", "concat", "vanilla"]

DESC = {
    "full": r"AdaLN + SIGReg + Orth, z=64",
    "small_z": r"AdaLN + SIGReg + Orth, z=16",
    "no_sigreg": r"AdaLN + Orth, z=64 (no SIGReg)",
    "no_orth": r"AdaLN + SIGReg, z=64 (no Orth)",
    "no_lewm": r"AdaLN only, z=64 (no SIGReg, no Orth)",
    "concat": r"Concat predictor + SIGReg + Orth, z=64",
    "vanilla": r"Concat + z=16, no LeWM regs (pre-LeWM JEPA)",
}


def _load_reports(tag: str) -> list[dict]:
    paths = sorted(RESULTS.glob(f"report_{tag}_seed*.json"))
    out = []
    for p in paths:
        m = re.match(rf"report_{re.escape(tag)}_seed(\d+)\.json$", p.name)
        if not m:
            continue
        out.append((int(m.group(1)), json.loads(p.read_text(encoding="utf-8"))))
    out.sort(key=lambda x: x[0])
    return [r for _, r in out]


def _tex_pm(mean: float, std: float, nd: int = 3) -> str:
    return f"{mean:.{nd}f} \\pm {std:.{nd}f}"


def main() -> None:
    summary: dict[str, dict] = {}
    for tag in TAGS:
        reps = _load_reports(tag)
        if len(reps) != 5:
            print(f"[warn] {tag}: expected 5 reports, got {len(reps)}", flush=True)
        if not reps:
            continue
        val_loss = [r["best_val_loss"] for r in reps]
        te = [r["metrics"]["test"] for r in reps]
        ev = [t["explained_var"] for t in te]
        sigy = [t["target_embedding_std"] for t in te]
        zs = [t["intervention_sensitivity_relative"] for t in te]
        # Reference MSE / cos / probe: seed 42 (same splits as main paper).
        ref_by_seed = {r["config"]["seed"]: r for r in reps}
        ref = ref_by_seed.get(42, reps[0])
        mse = ref["metrics"]["test"]["mse"]
        cos = ref["metrics"]["test"]["cosine_sim"]
        probe = ref["linear_probe"]["eval_acc"] * 100.0

        summary[tag] = {
            "n_seeds": len(reps),
            "val_loss_mean": float(np.mean(val_loss)),
            "val_loss_std": float(np.std(val_loss, ddof=1)) if len(val_loss) > 1 else 0.0,
            "test_ev_mean": float(np.mean(ev)),
            "test_ev_std": float(np.std(ev, ddof=1)) if len(ev) > 1 else 0.0,
            "sigma_sy_mean": float(np.mean(sigy)),
            "sigma_sy_std": float(np.std(sigy, ddof=1)) if len(sigy) > 1 else 0.0,
            "z_sens_rel_mean": float(np.mean(zs)),
            "z_sens_rel_std": float(np.std(zs, ddof=1)) if len(zs) > 1 else 0.0,
            "test_mse_ref": mse,
            "test_cos_ref": cos,
            "probe_acc_ref_pct": probe,
        }

    out_path = RESULTS / "ablation_multiseed_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_path}\n")

    # LaTeX rows (mean±std for 4 metrics; ref seed 42 for MSE, cos, probe)
    for tag in TAGS:
        if tag not in summary:
            continue
        s = summary[tag]
        vl = _tex_pm(s["val_loss_mean"], s["val_loss_std"])
        ev = _tex_pm(s["test_ev_mean"], s["test_ev_std"])
        sg = _tex_pm(s["sigma_sy_mean"], s["sigma_sy_std"])
        zz = _tex_pm(s["z_sens_rel_mean"], s["z_sens_rel_std"])
        mse = s["test_mse_ref"]
        cs = s["test_cos_ref"]
        pr = s["probe_acc_ref_pct"]
        tt = tag.replace("_", r"\_")
        print(
            rf"\texttt{{{tt}}} & {DESC[tag]} & ${vl}$ & {mse:.3f} & {cs:.3f} & ${ev}$ & ${sg}$ & ${zz}$ & {pr:.1f}\% \\"
        )


if __name__ == "__main__":
    main()
