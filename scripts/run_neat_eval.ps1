param(
  [Parameter(Mandatory = $true)][string]$GenomePath,
  [int]$Games = 1000,
  [string]$Seed = "neat-eval",
  [int]$MaxSteps = 600,
  [string]$OpponentPolicy = "heuristic_v4",
  [ValidateSet("alternate", "fixed")][string]$FirstTurnPolicy = "alternate",
  [ValidateSet("human", "ai")][string]$FixedFirstTurn = "human",
  [double]$FitnessGoldScale = 10000.0,
  [double]$FitnessWinWeight = 2.5,
  [double]$FitnessLossWeight = 1.5,
  [double]$FitnessDrawWeight = 0.1
)

if (-not (Test-Path $GenomePath)) {
  throw "genome not found: $GenomePath"
}

$cmd = @(
  "scripts/neat_eval_worker.mjs",
  "--genome", $GenomePath,
  "--games", "$Games",
  "--seed", $Seed,
  "--max-steps", "$MaxSteps",
  "--opponent-policy", $OpponentPolicy,
  "--first-turn-policy", $FirstTurnPolicy,
  "--fitness-gold-scale", "$FitnessGoldScale",
  "--fitness-win-weight", "$FitnessWinWeight",
  "--fitness-loss-weight", "$FitnessLossWeight",
  "--fitness-draw-weight", "$FitnessDrawWeight"
)

if ($FirstTurnPolicy -eq "fixed") {
  $cmd += @("--fixed-first-turn", $FixedFirstTurn)
}

node @cmd
exit $LASTEXITCODE
