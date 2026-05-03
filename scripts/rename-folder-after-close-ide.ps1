# Run AFTER closing Cursor/VS Code (the folder must not be in use).
# Moves repo root from e.g. "Cohort Compass" -> "Health-JEPA".
$repoRoot = Split-Path -Parent $PSScriptRoot
$userParent = Split-Path -Parent $repoRoot
$currentName = Split-Path -Leaf $repoRoot
if ($currentName -ne 'Health-JEPA') {
    Rename-Item -LiteralPath $repoRoot -NewName 'Health-JEPA'
    Write-Host "Renamed to: $(Join-Path $userParent 'Health-JEPA')"
} else {
    Write-Host "Already named Health-JEPA."
}
