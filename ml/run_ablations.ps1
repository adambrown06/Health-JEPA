# Train the JEPA ablation matrix.
# Each run produces {jepa_best_<tag>.pt, report_<tag>.md, training_log_<tag>.jsonl}.

$env:PYTHONIOENCODING = "utf-8"

$common = @(
    "--epochs", "80",
    "--d-model", "128",
    "--n-heads", "4",
    "--n-layers", "3",
    "--d-ff", "256",
    "--predictor-hidden", "256",
    "--predictor-layers", "3",
    "--early-stop-patience", "15",
    "--seed", "42"
)

$runs = @(
    @{ tag = "no_sigreg";  z = 64; pstyle = "adaln";  sig = 0.0; orth = 0.5 },
    @{ tag = "no_orth";    z = 64; pstyle = "adaln";  sig = 0.1; orth = 0.0 },
    @{ tag = "no_lewm";    z = 64; pstyle = "adaln";  sig = 0.0; orth = 0.0 },
    @{ tag = "concat";     z = 64; pstyle = "concat"; sig = 0.1; orth = 0.5 },
    @{ tag = "vanilla";    z = 16; pstyle = "concat"; sig = 0.0; orth = 0.0 },
    @{ tag = "small_z";    z = 16; pstyle = "adaln";  sig = 0.1; orth = 0.5 }
)

foreach ($r in $runs) {
    Write-Host ""
    Write-Host "================================================================"
    Write-Host "ABLATION: tag=$($r.tag) z_dim=$($r.z) predictor=$($r.pstyle) sig=$($r.sig) orth=$($r.orth)"
    Write-Host "================================================================"
    python -m ml.train_jepa `
        @common `
        --tag $r.tag `
        --z-dim $r.z `
        --predictor-style $r.pstyle `
        --sigreg-lambda $r.sig `
        --orth-lambda $r.orth 2>&1 | Tee-Object -FilePath "ml/results/ablation_$($r.tag).log"
}

Write-Host ""
Write-Host "[done] Ablation matrix complete."
Get-ChildItem ml/checkpoints/jepa_best_*.pt | Select-Object Name, Length
