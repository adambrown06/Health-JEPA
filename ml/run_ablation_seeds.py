"""
Run all JEPA ablation variants across multiple global RNG seeds.

Splits are fixed via ``--split-seed 42``; only initialization / shuffle /
dropout noise changes with ``--seed``.

Usage (from repo root)::

    python -m ml.run_ablation_seeds
    python -m ml.run_ablation_seeds --resume   # skip completed reports

Outputs one ``report_<tag>_seed<seed>.json`` per run under ``ml/results/``.
Aggregate with ``python -m ml.aggregate_ablation_seeds``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _ROOT / "ml" / "results"


def _report_path(tag: str, seed: int) -> Path:
    return _RESULTS / f"report_{tag}_seed{seed}.json"

# (tag, z_dim, predictor_style, sigreg_lambda, orth_lambda)
VARIANTS: list[tuple[str, int, str, float, float]] = [
    ("full", 64, "adaln", 0.1, 0.5),
    ("small_z", 16, "adaln", 0.1, 0.5),
    ("no_sigreg", 64, "adaln", 0.0, 0.5),
    ("no_orth", 64, "adaln", 0.1, 0.0),
    ("no_lewm", 64, "adaln", 0.0, 0.0),
    ("concat", 64, "concat", 0.1, 0.5),
    ("vanilla", 16, "concat", 0.0, 0.0),
]

SEEDS = [42, 142, 242, 342, 442]

COMMON = [
    "-m", "ml.train_jepa",
    "--epochs", "80",
    "--d-model", "128",
    "--n-heads", "4",
    "--n-layers", "3",
    "--d-ff", "256",
    "--predictor-hidden", "256",
    "--predictor-layers", "3",
    "--early-stop-patience", "15",
    "--split-seed", "42",
    "--append-seed-to-artifacts",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip (tag, seed) when report_<tag>_seed<seed>.json already exists.",
    )
    args = ap.parse_args()

    for tag, z, pstyle, sig, orth in VARIANTS:
        for seed in SEEDS:
            if args.resume and _report_path(tag, seed).is_file():
                print(f"[skip] {tag} seed={seed} (report exists)", flush=True)
                continue
            cmd = [
                sys.executable,
                *COMMON,
                "--tag", tag,
                "--z-dim", str(z),
                "--predictor-style", pstyle,
                "--sigreg-lambda", str(sig),
                "--orth-lambda", str(orth),
                "--seed", str(seed),
            ]
            print("\n>>>", " ".join(cmd), flush=True)
            r = subprocess.run(cmd, cwd=str(_ROOT))
            if r.returncode != 0:
                raise SystemExit(r.returncode)
    print("\n[ok] All ablation × seed runs finished.")


if __name__ == "__main__":
    main()
