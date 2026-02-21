param(
  [string]$Champion = "heuristic_v3",
  [string]$Challenger = "heuristic_v3",
  [int]$GamesPerSide = 100000,
  [double]$PromoteThreshold = 0.52,
  [double]$GoldDeltaEpsilon = 50.0,
  [double]$WinRateEpsilon = 0.005,
  [double]$ZScoreGate = 1.96,
  [double]$GoFailRateTolerance = 0.01,
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
if ($GoldDeltaEpsilon -lt 0) {
  Write-Error "GoldDeltaEpsilon must be >= 0."
  exit 2
}
if ($WinRateEpsilon -lt 0 -or $WinRateEpsilon -gt 1) {
  Write-Error "WinRateEpsilon must be between 0 and 1."
  exit 2
}
if ($ZScoreGate -le 0) {
  Write-Error "ZScoreGate must be > 0."
  exit 2
}
if ($GoFailRateTolerance -lt 0 -or $GoFailRateTolerance -gt 1) {
  Write-Error "GoFailRateTolerance must be in [0, 1]."
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

function TraceChoice {
  param($Trace)
  if ($null -eq $Trace) { return $null }
  $compact = $Trace.ch
  if ($null -ne $compact -and -not [string]::IsNullOrWhiteSpace([string]$compact)) { return [string]$compact }
  return $null
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

function Summarize-RunJsonl {
  param(
    [string]$Path,
    [string]$ChallengerLabel,
    [string]$ChampionLabel
  )
  if (-not (Test-Path $Path)) {
    throw "Missing file: $Path"
  }
  $s = [ordered]@{
    total = 0
    challengerWins = 0
    championWins = 0
    draws = 0
    challengerBankruptInflicted = 0
    challengerBankruptSuffered = 0
    challengerGoldDeltaSum = 0.0
    challengerWorstSingleLoss = 0.0
    championWorstSingleLoss = 0.0
    mappedGames = 0
    challengerGoldDeltaSqSum = 0.0
    challengerGoDeclared = 0
    challengerGoFailed = 0
    championGoDeclared = 0
    championGoFailed = 0
  }
  Get-Content -Path $Path | ForEach-Object {
    $line = $_
    if ([string]::IsNullOrWhiteSpace($line)) { return }
    $obj = $line | ConvertFrom-Json
    $s.total += 1

    $winner = [string]$obj.winner
    if ($winner -eq "draw") { $s.draws += 1 }

    $policy = $obj.policy
    if ($null -eq $policy) { return }

    $myLabel = [string]$policy.mySide
    $yourLabel = [string]$policy.yourSide
    $myGold = NumOrZero($obj.finalGoldMy)
    $yourGold = NumOrZero($obj.finalGoldYour)
    if ($myGold -eq 0 -and $yourGold -eq 0 -and $null -ne $obj.gold) {
      $myGold = NumOrZero($obj.gold.mySide)
      $yourGold = NumOrZero($obj.gold.yourSide)
    }
    $decisionTrace = @($obj.decision_trace)

    $challengerSide = $null
    if ($ChallengerLabel -ne $ChampionLabel) {
      if ($myLabel -eq $ChallengerLabel -and $yourLabel -eq $ChampionLabel) {
        $challengerSide = "mySide"
      } elseif ($yourLabel -eq $ChallengerLabel -and $myLabel -eq $ChampionLabel) {
        $challengerSide = "yourSide"
      } elseif ($myLabel -eq $ChallengerLabel -and $yourLabel -ne $ChallengerLabel) {
        $challengerSide = "mySide"
      } elseif ($yourLabel -eq $ChallengerLabel -and $myLabel -ne $ChallengerLabel) {
        $challengerSide = "yourSide"
      }
    }

    if ($challengerSide -eq "mySide") {
      $challengerDelta = $myGold - $yourGold
      $championDelta = $yourGold - $myGold
      $s.mappedGames += 1
      $s.challengerGoldDeltaSum += $challengerDelta
      $s.challengerGoldDeltaSqSum += ($challengerDelta * $challengerDelta)
      if ($challengerDelta -lt $s.challengerWorstSingleLoss) { $s.challengerWorstSingleLoss = $challengerDelta }
      if ($championDelta -lt $s.championWorstSingleLoss) { $s.championWorstSingleLoss = $championDelta }
      if ($yourGold -le 0) { $s.challengerBankruptInflicted += 1 }
      if ($myGold -le 0) { $s.challengerBankruptSuffered += 1 }
      if ($winner -eq "mySide") { $s.challengerWins += 1 }
      elseif ($winner -eq "yourSide") { $s.championWins += 1 }
      foreach ($tr in $decisionTrace) {
        if ($null -eq $tr) { continue }
        if ([string]$tr.dt -ne "option") { continue }
        $choice = TraceChoice -Trace $tr
        if ([string]$choice -ne "go") { continue }
        $actor = [string]$tr.a
        if ($actor -eq "mySide") {
          $s.challengerGoDeclared += 1
          if ($winner -eq "yourSide") { $s.challengerGoFailed += 1 }
        } elseif ($actor -eq "yourSide") {
          $s.championGoDeclared += 1
          if ($winner -eq "mySide") { $s.championGoFailed += 1 }
        }
      }
    } elseif ($challengerSide -eq "yourSide") {
      $challengerDelta = $yourGold - $myGold
      $championDelta = $myGold - $yourGold
      $s.mappedGames += 1
      $s.challengerGoldDeltaSum += $challengerDelta
      $s.challengerGoldDeltaSqSum += ($challengerDelta * $challengerDelta)
      if ($challengerDelta -lt $s.challengerWorstSingleLoss) { $s.challengerWorstSingleLoss = $challengerDelta }
      if ($championDelta -lt $s.championWorstSingleLoss) { $s.championWorstSingleLoss = $championDelta }
      if ($myGold -le 0) { $s.challengerBankruptInflicted += 1 }
      if ($yourGold -le 0) { $s.challengerBankruptSuffered += 1 }
      if ($winner -eq "yourSide") { $s.challengerWins += 1 }
      elseif ($winner -eq "mySide") { $s.championWins += 1 }
      foreach ($tr in $decisionTrace) {
        if ($null -eq $tr) { continue }
        if ([string]$tr.dt -ne "option") { continue }
        $choice = TraceChoice -Trace $tr
        if ([string]$choice -ne "go") { continue }
        $actor = [string]$tr.a
        if ($actor -eq "yourSide") {
          $s.challengerGoDeclared += 1
          if ($winner -eq "mySide") { $s.challengerGoFailed += 1 }
        } elseif ($actor -eq "mySide") {
          $s.championGoDeclared += 1
          if ($winner -eq "yourSide") { $s.championGoFailed += 1 }
        }
      }
    } elseif ($winner -eq "mySide" -or $winner -eq "yourSide") {
      # Unresolvable label mapping (for example, identical labels on both sides).
      $s.draws += 1
    }
  }
  return $s
}

function Run-ParallelSim {
  param(
    [int]$Games,
    [string]$OutPath,
    [string]$MySidePolicy,
    [string]$YourSidePolicy,
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
    "--policy-my-side=$MySidePolicy",
    "--policy-your-side=$YourSidePolicy"
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
    Run-ParallelSim -Games $GamesPerSide -OutPath $state.run_a -MySidePolicy $state.challenger -YourSidePolicy $state.champion -Mode $LogMode -WorkerCount $Workers
    $state.run_a_done = $true
    Save-State -Path $statePath -Obj $state
  }

  if (-not $state.run_b_done) {
    Run-ParallelSim -Games $GamesPerSide -OutPath $state.run_b -MySidePolicy $state.champion -YourSidePolicy $state.challenger -Mode $LogMode -WorkerCount $Workers
    $state.run_b_done = $true
    Save-State -Path $statePath -Obj $state
  }

  if (-not $state.summary_done) {
    $sum1 = Summarize-RunJsonl -Path $state.run_a -ChallengerLabel $state.challenger -ChampionLabel $state.champion
    $sum2 = Summarize-RunJsonl -Path $state.run_b -ChallengerLabel $state.challenger -ChampionLabel $state.champion

    $challengerWins = [int]$sum1.challengerWins + [int]$sum2.challengerWins
    $championWins = [int]$sum1.championWins + [int]$sum2.championWins
    $draws = [int]$sum1.draws + [int]$sum2.draws
    $challengerBankruptInflicted = [int]$sum1.challengerBankruptInflicted + [int]$sum2.challengerBankruptInflicted
    $challengerBankruptSuffered = [int]$sum1.challengerBankruptSuffered + [int]$sum2.challengerBankruptSuffered
    $challengerGoDeclared = [int]$sum1.challengerGoDeclared + [int]$sum2.challengerGoDeclared
    $challengerGoFailed = [int]$sum1.challengerGoFailed + [int]$sum2.challengerGoFailed
    $championGoDeclared = [int]$sum1.championGoDeclared + [int]$sum2.championGoDeclared
    $championGoFailed = [int]$sum1.championGoFailed + [int]$sum2.championGoFailed
    $challengerGoFailRate = Ratio -Numerator $challengerGoFailed -Denominator $challengerGoDeclared
    $championGoFailRate = Ratio -Numerator $championGoFailed -Denominator $championGoDeclared
    $goRiskWorse = $false
    if ($challengerGoDeclared -ge 50 -and $championGoDeclared -ge 50) {
      $goRiskWorse = ($challengerGoFailRate -gt ($championGoFailRate + $GoFailRateTolerance))
    }
    $bankruptDiff = $challengerBankruptInflicted - $challengerBankruptSuffered
    $total = [int]$sum1.total + [int]$sum2.total
    $mappedTotal = [int]$sum1.mappedGames + [int]$sum2.mappedGames
    $decisive = $challengerWins + $championWins

    $challengerWinRateAll = Ratio -Numerator $challengerWins -Denominator $total
    $challengerWinRateDecisive = Ratio -Numerator $challengerWins -Denominator $decisive

    # Gold-first metric (primary), win rate is secondary.
    $challengerGoldDeltaSum = NumOrZero($sum1.challengerGoldDeltaSum) + NumOrZero($sum2.challengerGoldDeltaSum)
    $challengerGoldDeltaSqSum = NumOrZero($sum1.challengerGoldDeltaSqSum) + NumOrZero($sum2.challengerGoldDeltaSqSum)
    if ($mappedTotal -gt 0) {
      $challengerAvgGoldDelta = $challengerGoldDeltaSum / [double]$mappedTotal
    } else {
      $challengerAvgGoldDelta = 0.0
    }
    if ($mappedTotal -gt 1) {
      $goldVarianceRaw = ($challengerGoldDeltaSqSum - (($challengerGoldDeltaSum * $challengerGoldDeltaSum) / [double]$mappedTotal)) / [double]($mappedTotal - 1)
      if ($goldVarianceRaw -lt 0) { $goldVarianceRaw = 0.0 }
      $goldStd = [Math]::Sqrt($goldVarianceRaw)
      $goldSe = $goldStd / [Math]::Sqrt([double]$mappedTotal)
    } else {
      $goldStd = 0.0
      $goldSe = 0.0
    }
    if ($goldSe -gt 0) {
      $goldZAbs = [Math]::Abs($challengerAvgGoldDelta / $goldSe)
    } else {
      $goldZAbs = 0.0
    }
    $challengerCumGold1000 = $challengerAvgGoldDelta * 1000.0
    $challengerWorstA = NumOrZero($sum1.challengerWorstSingleLoss)
    $challengerWorstB = NumOrZero($sum2.challengerWorstSingleLoss)
    $challengerWorstSingleLoss = [Math]::Min($challengerWorstA, $challengerWorstB)
    $championWorstA = NumOrZero($sum1.championWorstSingleLoss)
    $championWorstB = NumOrZero($sum2.championWorstSingleLoss)
    $championWorstSingleLoss = [Math]::Min($championWorstA, $championWorstB)
    if ($decisive -gt 0) {
      $winRateEdge = $challengerWinRateDecisive - 0.5
      $p = $challengerWinRateDecisive
      $winRateEdgeSe = [Math]::Sqrt([Math]::Max(0.0, ($p * (1.0 - $p)) / [double]$decisive))
    } else {
      $winRateEdge = 0.0
      $winRateEdgeSe = 0.0
    }
    if ($winRateEdgeSe -gt 0) {
      $winRateZAbs = [Math]::Abs($winRateEdge / $winRateEdgeSe)
    } else {
      $winRateZAbs = 0.0
    }

    $championBefore = [string]$state.champion
    $challengerBefore = [string]$state.challenger
    $decisionReason = ""
    if ($bankruptDiff -gt 0) {
      $promoted = $true
      $decisionReason = "bankrupt_diff_positive"
    } elseif ($bankruptDiff -lt 0) {
      $promoted = $false
      $decisionReason = "bankrupt_diff_negative"
    } elseif ($challengerAvgGoldDelta -ge $GoldDeltaEpsilon -and $challengerAvgGoldDelta -ge ($GoldDeltaEpsilon + ($ZScoreGate * $goldSe))) {
      $promoted = $true
      $decisionReason = "gold_delta_above_epsilon_with_z_gate"
    } elseif ($challengerAvgGoldDelta -le (-1.0 * $GoldDeltaEpsilon) -and $challengerAvgGoldDelta -le ((-1.0 * $GoldDeltaEpsilon) - ($ZScoreGate * $goldSe))) {
      $promoted = $false
      $decisionReason = "gold_delta_below_negative_epsilon_with_z_gate"
    } elseif ($challengerAvgGoldDelta -ge $GoldDeltaEpsilon) {
      $promoted = $false
      $decisionReason = "gold_delta_positive_but_not_significant"
    } elseif ($challengerAvgGoldDelta -le (-1.0 * $GoldDeltaEpsilon)) {
      $promoted = $false
      $decisionReason = "gold_delta_negative_but_not_significant"
    } elseif ($winRateEdge -ge $WinRateEpsilon -and $winRateEdge -ge ($WinRateEpsilon + ($ZScoreGate * $winRateEdgeSe))) {
      $promoted = $true
      $decisionReason = "win_rate_above_epsilon_with_z_gate"
    } elseif ($winRateEdge -le (-1.0 * $WinRateEpsilon) -and $winRateEdge -le ((-1.0 * $WinRateEpsilon) - ($ZScoreGate * $winRateEdgeSe))) {
      $promoted = $false
      $decisionReason = "win_rate_below_negative_epsilon_with_z_gate"
    } elseif ($winRateEdge -ge $WinRateEpsilon) {
      $promoted = $false
      $decisionReason = "win_rate_positive_but_not_significant"
    } elseif ($winRateEdge -le (-1.0 * $WinRateEpsilon)) {
      $promoted = $false
      $decisionReason = "win_rate_negative_but_not_significant"
    } elseif ($challengerWorstSingleLoss -gt $championWorstSingleLoss) {
      $promoted = $true
      $decisionReason = "tie_break_by_max_single_loss"
    } elseif ($challengerWorstSingleLoss -lt $championWorstSingleLoss) {
      $promoted = $false
      $decisionReason = "tie_break_by_max_single_loss"
    } else {
      $promoted = $challengerWinRateDecisive -ge $PromoteThreshold
      $decisionReason = "fallback_promote_threshold"
    }
    if ($promoted -and $goRiskWorse) {
      $promoted = $false
      $decisionReason = "blocked_by_go_fail_rate_risk"
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
      challenger_bankrupt_inflicted = $challengerBankruptInflicted
      challenger_bankrupt_suffered = $challengerBankruptSuffered
      challenger_go_declared = $challengerGoDeclared
      challenger_go_failed = $challengerGoFailed
      champion_go_declared = $championGoDeclared
      champion_go_failed = $championGoFailed
      challenger_go_fail_rate = $challengerGoFailRate
      champion_go_fail_rate = $championGoFailRate
      go_fail_rate_tolerance = $GoFailRateTolerance
      go_risk_worse = $goRiskWorse
      bankrupt_diff = $bankruptDiff
      challenger_avg_gold_delta = $challengerAvgGoldDelta
      challenger_gold_delta_std = $goldStd
      challenger_gold_delta_se = $goldSe
      challenger_gold_delta_z_abs = $goldZAbs
      challenger_cum_gold_1000 = $challengerCumGold1000
      challenger_win_rate_all = $challengerWinRateAll
      challenger_win_rate_decisive = $challengerWinRateDecisive
      decisive_win_rate_edge = $winRateEdge
      decisive_win_rate_edge_se = $winRateEdgeSe
      decisive_win_rate_z_abs = $winRateZAbs
      gold_delta_epsilon = $GoldDeltaEpsilon
      win_rate_epsilon = $WinRateEpsilon
      z_score_gate = $ZScoreGate
      challenger_worst_single_loss = $challengerWorstSingleLoss
      champion_worst_single_loss = $championWorstSingleLoss
      promote_threshold = $PromoteThreshold
      decision_reason = $decisionReason
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
    Write-Host "  bankrupt (inflicted/suffered/diff): $challengerBankruptInflicted / $challengerBankruptSuffered / $bankruptDiff"
    Write-Host ("  challenger go fail rate  : {0:P2} ({1}/{2})" -f $challengerGoFailRate, $challengerGoFailed, $challengerGoDeclared)
    Write-Host ("  champion go fail rate    : {0:P2} ({1}/{2})" -f $championGoFailRate, $championGoFailed, $championGoDeclared)
    Write-Host ("  go fail tolerance        : {0:P2}" -f $GoFailRateTolerance)
    Write-Host ("  go risk worse            : {0}" -f $goRiskWorse)
    Write-Host ("  challenger avg_gold_delta : {0:N4}" -f $challengerAvgGoldDelta)
    Write-Host ("  challenger gold std       : {0:N4}" -f $goldStd)
    Write-Host ("  challenger gold se        : {0:N4}" -f $goldSe)
    Write-Host ("  challenger gold z(abs)    : {0:N4}" -f $goldZAbs)
    Write-Host ("  gold epsilon            : {0:N4}" -f $GoldDeltaEpsilon)
    Write-Host ("  z-score gate            : {0:N4}" -f $ZScoreGate)
    Write-Host ("  challenger cum_gold_1000  : {0:N2}" -f $challengerCumGold1000)
    Write-Host ("  challenger WR (decisive): {0:P2}" -f $challengerWinRateDecisive)
    Write-Host ("  challenger WR edge      : {0:P2}" -f $winRateEdge)
    Write-Host ("  challenger WR edge se   : {0:P4}" -f $winRateEdgeSe)
    Write-Host ("  challenger WR z(abs)    : {0:N4}" -f $winRateZAbs)
    Write-Host ("  win-rate epsilon        : {0:P2}" -f $WinRateEpsilon)
    Write-Host ("  challenger worst single loss : {0:N0}" -f $challengerWorstSingleLoss)
    Write-Host ("  champion worst single loss   : {0:N0}" -f $championWorstSingleLoss)
    Write-Host ("  threshold             : {0:P2}" -f $PromoteThreshold)
    Write-Host "  decision reason   : $decisionReason"
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
