# Run outcome evaluations for every JEPA ablation.
# Sequence baselines (GRU, TSEncoder end-to-end) are trained ONCE with the
# 'full' checkpoint since their predictions do not depend on the JEPA model.

$env:PYTHONIOENCODING = "utf-8"

$tags = @("full", "small_z", "no_sigreg", "no_orth", "no_lewm", "concat", "vanilla")

Write-Host ""
Write-Host "=== Outcome eval: full (with sequence baselines) ==="
python -m ml.outcome_eval `
    --checkpoint "ml/checkpoints/jepa_best_full.pt" `
    --tag "full" `
    --seq-epochs 30 2>&1 | Tee-Object -FilePath "ml/results/outcome_run_full.log"

foreach ($tag in $tags) {
    if ($tag -eq "full") { continue }
    Write-Host ""
    Write-Host "=== Outcome eval: $tag ==="
    python -m ml.outcome_eval `
        --checkpoint "ml/checkpoints/jepa_best_$tag.pt" `
        --tag $tag `
        --skip-sequence-baselines 2>&1 | Tee-Object -FilePath "ml/results/outcome_run_$tag.log"
}

Write-Host ""
Write-Host "[done] All outcome evaluations complete."
