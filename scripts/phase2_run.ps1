param(
  [Parameter(Mandatory = $true)][int]$Seed
)

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "python not found: $python"
}

$configFeedforward = "scripts/configs/neat_feedforward.ini"
$runtimeConfig = "scripts/configs/runtime_phase2.json"
$resumeCheckpoint = "logs/neat_phase1_seed$Seed/checkpoints/neat-checkpoint-20"
$outputDir = "logs/neat_phase2_seed$Seed"

if (-not (Test-Path $resumeCheckpoint)) {
  throw "phase1 checkpoint not found: $resumeCheckpoint"
}

$cmd = @(
  "scripts/neat_train.py",
  "--config-feedforward", $configFeedforward,
  "--runtime-config", $runtimeConfig,
  "--output-dir", $outputDir,
  "--resume", $resumeCheckpoint,
  "--seed", "$Seed",
  "--profile-name", "phase2_seed$Seed"
)

& $python @cmd
exit $LASTEXITCODE
