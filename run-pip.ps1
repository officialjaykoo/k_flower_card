$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonDir = Get-ChildItem -LiteralPath (Join-Path $root '.tools') -Directory |
  Where-Object { $_.Name -like 'python-*-*' } |
  Sort-Object Name -Descending |
  Select-Object -First 1 -ExpandProperty FullName

if (-not $pythonDir) {
  throw 'Portable Python not found. Expected it under .tools.'
}

$env:PATH = "$pythonDir;" + (Join-Path $pythonDir 'Scripts') + ";$env:PATH"
& (Join-Path $pythonDir 'python.exe') -m pip @args
