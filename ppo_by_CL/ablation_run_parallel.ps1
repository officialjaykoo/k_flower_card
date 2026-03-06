# 5-arm ablation 병렬 실행 (A/B/C/D = 단독 검증, E = Claude 최선 조합)
# 프로젝트 루트에서 실행

param(
  [Parameter(Mandatory = $false)][string]$Python = ".\\.venv\\Scripts\\python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$arms = @(
    @{ name="A_go_explore";   config="ppo/configs/ablation_A_go_explore.json"   },
    @{ name="B_reward_scale"; config="ppo/configs/ablation_B_reward_scale.json"  },
    @{ name="C_weak_fixed";   config="ppo/configs/ablation_C_weak_fixed.json"   },
    @{ name="D_all";          config="ppo/configs/ablation_D_all.json"           },
    @{ name="E_claude";       config="ppo/configs/ablation_E_claude.json"        }
)

$root = $PWD.Path
$jobs = @()

foreach ($arm in $arms) {
    $name    = $arm.name
    $config  = $arm.config
    $logFile = "logs/PPO/ablation_$name.log"
    Write-Host "Starting ARM-$name  ->  $config"

    $job = Start-Job -ScriptBlock {
        param($root, $py, $cfg, $log)
        Set-Location $root
        New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
        $env:TORCH_COMPILE_DISABLE = "1"
        $env:TORCHDYNAMO_DISABLE   = "1"
        & $py "ppo/scripts/train_ppo.py" --runtime-config $cfg 2>&1 |
            Tee-Object -FilePath $log
    } -ArgumentList $root, $Python, $config, $logFile

    $jobs += [PSCustomObject]@{ job=$job; name=$name }
    Start-Sleep -Milliseconds 800
}

Write-Host ""
Write-Host "All $($jobs.Count) arms running. Waiting..."
Write-Host "Logs: logs/PPO/ablation_*.log"
Write-Host ""

foreach ($j in $jobs) {
    $result = Wait-Job $j.job
    Write-Host "ARM-$($j.name) -> $($result.State)"
    if ($result.State -eq "Failed") { Receive-Job $j.job }
    Remove-Job $j.job
}

Write-Host ""
Write-Host "=== Ablation done. ==="
Write-Host "Check win_rate_1000 in logs/PPO/ablation_*/best_metrics_stage1.json"
