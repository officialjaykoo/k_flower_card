param(
  [string]$ConfigFeedforward = "configs/neat_feedforward.ini",
  [string]$RuntimeConfig = "configs/neat_runtime_i3_w6.json",
  [string]$OutputDir = "logs/neat_python",
  [int]$Generations = 0,
  [int]$Workers = 0,
  [int]$Seed = 0,
  [string]$ProfileName = "i3_w6",
  [switch]$DryRun,
  [switch]$NeatVerbose
)

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "python not found: $python"
}

$cmd = @(
  "scripts/neat_train.py",
  "--config-feedforward", $ConfigFeedforward,
  "--runtime-config", $RuntimeConfig,
  "--output-dir", $OutputDir,
  "--profile-name", $ProfileName
)

if ($Generations -gt 0) {
  $cmd += @("--generations", "$Generations")
}
if ($Workers -gt 0) {
  $cmd += @("--workers", "$Workers")
}
if ($Seed -gt 0) {
  $cmd += @("--seed", "$Seed")
}
if ($DryRun) {
  $cmd += "--dry-run"
}
if ($NeatVerbose) {
  $cmd += "--verbose"
}

& $python @cmd
exit $LASTEXITCODE
