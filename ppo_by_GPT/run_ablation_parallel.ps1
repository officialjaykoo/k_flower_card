param(
  [Parameter(Mandatory = $false)][string]$Python = ".\.venv\Scripts\python",
  [Parameter(Mandatory = $false)][Nullable[int]]$TotalUpdates = $null,
  [Parameter(Mandatory = $false)][Nullable[int]]$SaveEveryUpdates = 50,
  [Parameter(Mandatory = $false)][Nullable[int]]$LogEveryUpdates = 20,
  [Parameter(Mandatory = $false)][bool]$ResumeFromLatest = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$arms = @(
  @{ name = "A_go_explore";   config = "ppo_by_GPT/configs/ablation_A_go_explore.json"   },
  @{ name = "B_reward_scale"; config = "ppo_by_GPT/configs/ablation_B_reward_scale.json"  },
  @{ name = "C_weak_fixed";   config = "ppo_by_GPT/configs/ablation_C_weak_fixed.json"    },
  @{ name = "D_all";          config = "ppo_by_GPT/configs/ablation_D_all.json"           },
  @{ name = "E_gpt";          config = "ppo_by_GPT/configs/ablation_E_gpt.json"           }
)

$root = $PWD.Path
$jobs = @()

foreach ($arm in $arms) {
  $name = $arm.name
  $cfg = $arm.config
  if (-not (Test-Path $cfg)) {
    throw "config not found: $cfg"
  }
  $cfgObj = Get-Content -Path $cfg -Raw -Encoding UTF8 | ConvertFrom-Json
  $outDir = [string]$cfgObj.output_dir
  if ([string]::IsNullOrWhiteSpace($outDir)) {
    throw "output_dir missing in config: $cfg"
  }
  $resumeCkpt = Join-Path $outDir "latest.pt"
  $logFile = "logs/PPO_GPT/ablation_$name.log"
  if ($ResumeFromLatest -and (Test-Path $resumeCkpt)) {
    Write-Host "Starting ARM-$name -> $cfg (resume: $resumeCkpt)"
  } else {
    Write-Host "Starting ARM-$name -> $cfg (fresh)"
  }

  $job = Start-Job -ScriptBlock {
    param(
      $root,
      $py,
      $cfg,
      $outDir,
      $resumeCkpt,
      $resumeFromLatest,
      $log,
      $totalUpdates,
      $saveEveryUpdates,
      $logEveryUpdates
    )
    Set-Location $root
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
    $env:TORCH_COMPILE_DISABLE = "1"
    $env:TORCHDYNAMO_DISABLE = "1"
    $args = @(
      "ppo_by_GPT/scripts/train_ppo.py",
      "--runtime-config", $cfg,
      "--output-dir", $outDir
    )
    if ($resumeFromLatest -and (Test-Path $resumeCkpt)) {
      $args += @("--resume-checkpoint", $resumeCkpt)
    }
    if ($null -ne $totalUpdates) {
      if ($totalUpdates -le 0) { throw "TotalUpdates must be > 0, got=$totalUpdates" }
      $args += @("--total-updates", "$totalUpdates")
    }
    if ($null -ne $saveEveryUpdates) {
      if ($saveEveryUpdates -le 0) { throw "SaveEveryUpdates must be > 0, got=$saveEveryUpdates" }
      $args += @("--save-every-updates", "$saveEveryUpdates")
    }
    if ($null -ne $logEveryUpdates) {
      if ($logEveryUpdates -le 0) { throw "LogEveryUpdates must be > 0, got=$logEveryUpdates" }
      $args += @("--log-every-updates", "$logEveryUpdates")
    }
    & $py @args 2>&1 | Tee-Object -FilePath $log
  } -ArgumentList $root, $Python, $cfg, $outDir, $resumeCkpt, $ResumeFromLatest, $logFile, $TotalUpdates, $SaveEveryUpdates, $LogEveryUpdates

  $jobs += [PSCustomObject]@{ job = $job; name = $name; config = $cfg }
  Start-Sleep -Milliseconds 700
}

Write-Host ""
Write-Host "All $($jobs.Count) arms started. Logs in logs/PPO_GPT/ablation_*.log"
Write-Host ""

foreach ($j in $jobs) {
  $r = Wait-Job $j.job
  Write-Host "ARM-$($j.name) -> $($r.State)"
  if ($r.State -eq "Failed") {
    Receive-Job $j.job
  }
  Remove-Job $j.job
}

Write-Host ""
Write-Host "=== Ablation summary ==="

$summary = @()
foreach ($arm in $arms) {
  $cfgObj = Get-Content -Path $arm.config -Raw -Encoding UTF8 | ConvertFrom-Json
  $outDir = [string]$cfgObj.output_dir
  if ([string]::IsNullOrWhiteSpace($outDir)) {
    continue
  }
  $bestStage1 = Join-Path $outDir "best_metrics_stage1.json"
  $bestFallback = Join-Path $outDir "best_metrics.json"
  $bestPath = if (Test-Path $bestStage1) { $bestStage1 } elseif (Test-Path $bestFallback) { $bestFallback } else { "" }
  if ([string]::IsNullOrWhiteSpace($bestPath)) {
    $summary += [PSCustomObject]@{
      arm = $arm.name
      best_update = -1
      win_rate_1000 = [double]::NaN
      mean_final_gold_diff_1000 = [double]::NaN
      catastrophic_loss_rate_1000 = [double]::NaN
      best_file = "(missing)"
    }
    continue
  }
  $m = Get-Content -Path $bestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $summary += [PSCustomObject]@{
    arm = $arm.name
    best_update = [int]$m.update
    win_rate_1000 = [double]$m.win_rate_1000
    mean_final_gold_diff_1000 = [double]$m.mean_final_gold_diff_1000
    catastrophic_loss_rate_1000 = [double]$m.catastrophic_loss_rate_1000
    best_file = $bestPath
  }
}

$summary |
  Sort-Object -Property @{Expression = "mean_final_gold_diff_1000"; Descending = $true} |
  Format-Table -AutoSize

Write-Host ""
Write-Host "Done. Pick top arm by: higher mean_final_gold_diff_1000, lower catastrophic_loss_rate_1000."
