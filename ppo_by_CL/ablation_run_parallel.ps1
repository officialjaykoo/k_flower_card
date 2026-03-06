# 4-arm ablation 병렬 실행
# 프로젝트 루트에서 실행: .\ppo\run_ablation_parallel.ps1

param(
  [Parameter(Mandatory = $false)][string]$Python = ".\\.venv\\Scripts\\python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$arms = @(
    @{ name="A_go_explore";   config="ppo/configs/ablation_A_go_explore.json"  },
    @{ name="B_reward_scale"; config="ppo/configs/ablation_B_reward_scale.json" },
    @{ name="C_weak_fixed";   config="ppo/configs/ablation_C_weak_fixed.json"  },
    @{ name="D_all";          config="ppo/configs/ablation_D_all.json"          }
)

$root = $PWD.Path
$jobs = @()

foreach ($arm in $arms) {
    $name   = $arm.name
    $config = $arm.config
    $logFile = "logs/PPO/ablation_$name.log"

    Write-Host "Starting ARM-$name -> $config"

    $job = Start-Job -ScriptBlock {
        param($root, $py, $cfg, $log)
        Set-Location $root
        New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
        & $py "ppo/scripts/train_ppo.py" --runtime-config $cfg 2>&1 |
            Tee-Object -FilePath $log
    } -ArgumentList $root, $Python, $config, $logFile

    $jobs += [PSCustomObject]@{ job=$job; name=$name }
    Start-Sleep -Milliseconds 500  # 동시 시작 간격
}

Write-Host ""
Write-Host "All 4 arms running. Waiting for completion..."
Write-Host "Logs: logs/PPO/ablation_*.log"
Write-Host ""

foreach ($j in $jobs) {
    $result = Wait-Job $j.job
    $state  = $result.State
    Write-Host "ARM-$($j.name) -> $state"
    if ($state -eq "Failed") {
        Receive-Job $j.job
    }
    Remove-Job $j.job
}

Write-Host ""
Write-Host "=== Ablation done. Check logs/PPO/ablation_*.log ==="
