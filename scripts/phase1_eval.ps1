param(
  [Parameter(Mandatory = $true)][int]$Seed
)

$common = Join-Path $PSScriptRoot "phase_eval.ps1"
& $common -Phase "1" -Seed $Seed
exit $LASTEXITCODE