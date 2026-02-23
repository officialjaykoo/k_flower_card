param(
  [Parameter(Mandatory = $true)][int]$Seed
)

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "python not found: $python"
}

$configFeedforward = "scripts/configs/neat_feedforward.ini"
$runtimeConfig = "scripts/configs/runtime_phase1.json"
$outputDir = "logs/neat_phase1_seed$Seed"

$cmd = @(
  "scripts/neat_train.py",
  "--config-feedforward", $configFeedforward,
  "--runtime-config", $runtimeConfig,
  "--output-dir", $outputDir,
  "--seed", "$Seed",
  "--profile-name", "phase1_seed$Seed"
)

& $python @cmd
exit $LASTEXITCODE
