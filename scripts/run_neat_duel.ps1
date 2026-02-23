param(
  [Parameter(Mandatory = $true)][string]$GenomeAPath,
  [Parameter(Mandatory = $true)][string]$GenomeBPath,
  [string]$Seed = "neat-duel-1000",
  [int]$MaxSteps = 600,
  [ValidateSet("alternate", "fixed")][string]$FirstTurnPolicy = "alternate",
  [ValidateSet("human", "ai")][string]$FixedFirstTurn = "human"
)

if (-not (Test-Path $GenomeAPath)) {
  throw "genome A not found: $GenomeAPath"
}
if (-not (Test-Path $GenomeBPath)) {
  throw "genome B not found: $GenomeBPath"
}

# Project rule: testing simulation count is fixed to 1000.
$games = 1000

$cmd = @(
  "scripts/neat_duel_worker.mjs",
  "--genome-a", $GenomeAPath,
  "--genome-b", $GenomeBPath,
  "--games", "$games",
  "--seed", $Seed,
  "--max-steps", "$MaxSteps",
  "--first-turn-policy", $FirstTurnPolicy
)

if ($FirstTurnPolicy -eq "fixed") {
  $cmd += @("--fixed-first-turn", $FixedFirstTurn)
}

node @cmd
exit $LASTEXITCODE
