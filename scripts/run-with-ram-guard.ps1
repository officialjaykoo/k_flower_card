param(
  [double]$Threshold = 92,
  [int]$IntervalSec = 30,
  [int]$WindowSec = 180,
  [string]$Exe = "py",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgsToRun
)

if (-not $ArgsToRun -or $ArgsToRun.Count -eq 0) {
  Write-Error "No command arguments provided. Example: .\scripts\run-with-ram-guard.ps1 -- -3 scripts/02_train_value.py --input logs\*.jsonl --output models\value.json"
  exit 2
}

if ($IntervalSec -le 0) {
  Write-Error "IntervalSec must be > 0."
  exit 2
}

if ($WindowSec -le 0) {
  Write-Error "WindowSec must be > 0."
  exit 2
}

$maxHighCount = [int][Math]::Ceiling($WindowSec / $IntervalSec)
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = "logs"
$outLog = Join-Path $logDir "ram-guard-$stamp.out.log"
$errLog = Join-Path $logDir "ram-guard-$stamp.err.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "Running: $Exe $($ArgsToRun -join ' ')"
Write-Host "Guard : RAM >= $Threshold% for $WindowSec sec (check every $IntervalSec sec) => auto stop"
Write-Host "Logs  : $outLog , $errLog"

$proc = Start-Process -FilePath $Exe -ArgumentList $ArgsToRun -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$highCount = 0
$killedByGuard = $false

while (-not $proc.HasExited) {
  Start-Sleep -Seconds $IntervalSec

  $mem = (Get-Counter '\Memory\% Committed Bytes In Use').CounterSamples[0].CookedValue
  $live = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
  if ($null -eq $live) { break }

  if ($mem -ge $Threshold) { $highCount++ } else { $highCount = 0 }

  "{0} | RAM {1:N1}% | PID {2} | WS {3:N0} MB | CPU {4:N1}s | HighCount {5}/{6}" -f `
    (Get-Date), $mem, $proc.Id, ($live.WorkingSet64 / 1MB), $live.CPU, $highCount, $maxHighCount

  if ($highCount -ge $maxHighCount) {
    Write-Warning "RAM threshold exceeded for the window. Stopping process..."
    Stop-Process -Id $proc.Id -Force
    $killedByGuard = $true
    break
  }
}

if (-not $killedByGuard) {
  $proc.WaitForExit()
}

if ($killedByGuard) {
  Write-Host "Result: stopped by RAM guard."
  exit 3
}

Write-Host "Result: process exited with code $($proc.ExitCode)"
exit $proc.ExitCode
