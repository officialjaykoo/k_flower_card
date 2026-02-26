param(
  [Parameter(Mandatory = $true)][int]$Seed
)

$common = Join-Path $PSScriptRoot "phase_run.ps1"
& $common -Phase "2" -Seed $Seed
exit $LASTEXITCODE