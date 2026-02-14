param(
  [string]$Champion = "heuristic_v3",
  [string]$Challenger = "heuristic_v3",
  [int]$GamesPerSide = 100000,
  [double]$PromoteThreshold = 0.52,
  [string]$Tag = "",
  [int]$Rounds = 1,
  [string]$LogMode = "train",
  [int]$Workers = 4,
  [switch]$Resume
)

if ($GamesPerSide -le 0) {
  Write-Error "GamesPerSide must be > 0."
  exit 2
}
if ($GamesPerSide % 2 -ne 0) {
  Write-Error "GamesPerSide must be even."
  exit 2
}
if ($Rounds -le 0) {
  Write-Error "Rounds must be > 0."
  exit 2
}
if ($PromoteThreshold -lt 0 -or $PromoteThreshold -gt 1) {
  Write-Error "PromoteThreshold must be between 0 and 1."
  exit 2
}
if ($Workers -le 0) {
  Write-Error "Workers must be > 0."
  exit 2
}

function Ratio {
  param([double]$Numerator, [double]$Denominator)
  if ($Denominator -le 0) { return 0.0 }
  return $Numerator / $Denominator
}

function NumOrZero {
  param($Value)
  if ($null -eq $Value) { return 0.0 }
  try { return [double]$Value } catch { return 0.0 }
}

function Save-State {
  param([string]$Path, $Obj)
  $Obj | ConvertTo-Json -Depth 8 | Set-Content -Path $Path -Encoding utf8
}

function Read-JsonFile {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    throw "Missing file: $Path"
  }
  return (Get-Content $Path -Raw | ConvertFrom-Json)
}

function Run-ParallelSim {
  param(
    [int]$Games,
    [string]$OutPath,
    [string]$HumanPolicy,
    [string]$AiPolicy,
    [string]$Mode,
    [int]$WorkerCount
  )
  $args = @(
    "-3", "scripts/run_parallel_selfplay.py",
    "$Games",
    "--workers", "$WorkerCount",
    "--output", $OutPath,
    "--",
    "--log-mode=$Mode",
    "--policy-human=$HumanPolicy",
    "--policy-ai=$AiPolicy"
  )
  Write-Host ">" "py $($args -join ' ')"
  & py @args
  if ($LASTEXITCODE -ne 0) {
    throw "Simulation failed for $OutPath"
  }
}

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$baseTag = if ([string]::IsNullOrWhiteSpace($Tag)) {
  Get-Date -Format "yyyyMMdd-HHmmss"
} else {
  $Tag
}

$statePath = "logs/champ-cycle-$baseTag-state.json"
$scoreboardPath = "logs/champion-scoreboard.jsonl"

if ($Resume -and (Test-Path $statePath)) {
  $state = Read-JsonFile -Path $statePath
  Write-Host "Resuming from state: $statePath"
} else {
  $state = [ordered]@{
    base_tag = $baseTag
    round = 1
    rounds = $Rounds
    champion = $Champion
    challenger = $Challenger
    run_a_done = $false
    run_b_done = $false
    summary_done = $false
    round_tag = $null
    run_a = $null
    run_b = $null
  }
  Save-State -Path $statePath -Obj $state
}

while ([int]$state.round -le [int]$state.rounds) {
  $r = [int]$state.round
  if ([string]::IsNullOrWhiteSpace($state.round_tag)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $state.round_tag = "$($state.base_tag)-r$r-$stamp"
    $state.run_a = "logs/champ-cycle-$($state.round_tag)-A.jsonl"
    $state.run_b = "logs/champ-cycle-$($state.round_tag)-B.jsonl"
    Save-State -Path $statePath -Obj $state
  }

  if (-not $state.run_a_done) {
    Run-ParallelSim -Games $GamesPerSide -OutPath $state.run_a -HumanPolicy $state.challenger -AiPolicy $state.champion -Mode $LogMode -WorkerCount $Workers
    $state.run_a_done = $true
    Save-State -Path $statePath -Obj $state
  }

  if (-not $state.run_b_done) {
    Run-ParallelSim -Games $GamesPerSide -OutPath $state.run_b -HumanPolicy $state.champion -AiPolicy $state.challenger -Mode $LogMode -WorkerCount $Workers
    $state.run_b_done = $true
    Save-State -Path $statePath -Obj $state
  }

  if (-not $state.summary_done) {
    $rep1 = Read-JsonFile -Path ($state.run_a -replace "\.jsonl$", "-report.json")
    $rep2 = Read-JsonFile -Path ($state.run_b -replace "\.jsonl$", "-report.json")

    $challengerWins = [int]$rep1.winners.human + [int]$rep2.winners.ai
    $championWins = [int]$rep1.winners.ai + [int]$rep2.winners.human
    $draws = [int]$rep1.winners.draw + [int]$rep2.winners.draw
    $total = ($GamesPerSide * 2)
    $decisive = $challengerWins + $championWins

    $challengerWinRateAll = Ratio -Numerator $challengerWins -Denominator $total
    $challengerWinRateDecisive = Ratio -Numerator $challengerWins -Denominator $decisive

    # Gold-first metric (primary), win rate is secondary.
    # run_a: challenger=human, run_b: challenger=ai
    $rep1Eco = $rep1.economy
    $rep2Eco = $rep2.economy
    $challengerAvgGoldDelta =
      (NumOrZero($rep1Eco.averageGoldDeltaHuman) + (-(NumOrZero($rep2Eco.averageGoldDeltaHuman)))) / 2.0
    $challengerCumGold1000 =
      (NumOrZero($rep1Eco.cumulativeGoldDeltaOver1000) + (-(NumOrZero($rep2Eco.cumulativeGoldDeltaOver1000)))) / 2.0

    $championBefore = [string]$state.champion
    $challengerBefore = [string]$state.challenger
    if ($challengerAvgGoldDelta -gt 0) {
      $promoted = $true
    } elseif ($challengerAvgGoldDelta -lt 0) {
      $promoted = $false
    } else {
      $promoted = $challengerWinRateDecisive -ge $PromoteThreshold
    }
    if ($promoted) {
      $oldChampion = $state.champion
      $state.champion = $state.challenger
      $state.challenger = $oldChampion
    }

    $summary = [ordered]@{
      round = $r
      tag = [string]$state.round_tag
      games_per_side = $GamesPerSide
      total_games = $total
      champion_before = $championBefore
      challenger_before = $challengerBefore
      challenger_wins = $challengerWins
      champion_wins = $championWins
      draws = $draws
      challenger_avg_gold_delta = $challengerAvgGoldDelta
      challenger_cum_gold_1000 = $challengerCumGold1000
      challenger_win_rate_all = $challengerWinRateAll
      challenger_win_rate_decisive = $challengerWinRateDecisive
      promote_threshold = $PromoteThreshold
      promoted = $promoted
      champion_after = [string]$state.champion
      challenger_after = [string]$state.challenger
      run_a = [string]$state.run_a
      run_b = [string]$state.run_b
      report_a = [string]($state.run_a -replace "\.jsonl$", "-report.json")
      report_b = [string]($state.run_b -replace "\.jsonl$", "-report.json")
    }

    $summaryPath = "logs/champ-cycle-$($state.round_tag)-summary.json"
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding utf8
    ($summary | ConvertTo-Json -Depth 8 -Compress) | Add-Content -Path $scoreboardPath -Encoding utf8

    $state.summary_done = $true
    Save-State -Path $statePath -Obj $state

    Write-Host ""
    Write-Host "Round $r summary:"
    Write-Host "  challenger wins : $challengerWins"
    Write-Host "  champion wins   : $championWins"
    Write-Host "  draws           : $draws"
    Write-Host ("  challenger avg_gold_delta : {0:N4}" -f $challengerAvgGoldDelta)
    Write-Host ("  challenger cum_gold_1000  : {0:N2}" -f $challengerCumGold1000)
    Write-Host ("  challenger WR (decisive): {0:P2}" -f $challengerWinRateDecisive)
    Write-Host ("  threshold             : {0:P2}" -f $PromoteThreshold)
    Write-Host "  promoted         : $promoted"
    Write-Host "  champion after   : $($state.champion)"
    Write-Host "  challenger after : $($state.challenger)"
    Write-Host "  summary file     : $summaryPath"
    Write-Host "  scoreboard file  : $scoreboardPath"
    Write-Host ""
  }

  $state.round = $r + 1
  $state.run_a_done = $false
  $state.run_b_done = $false
  $state.summary_done = $false
  $state.round_tag = $null
  $state.run_a = $null
  $state.run_b = $null
  Save-State -Path $statePath -Obj $state
}

Write-Host "Cycle complete. Champion: $($state.champion) | Challenger: $($state.challenger)"
Write-Host "State file kept for audit/resume metadata: $statePath"

