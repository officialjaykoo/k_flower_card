$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$nodeDir = Get-ChildItem -LiteralPath (Join-Path $root '.tools') -Directory |
  Where-Object { $_.Name -like 'node-v*-win-*' } |
  Sort-Object Name -Descending |
  Select-Object -First 1 -ExpandProperty FullName

if (-not $nodeDir) {
  throw 'Portable Node.js not found. Expected it under .tools.'
}

$env:PATH = "$nodeDir;$env:PATH"
& (Join-Path $nodeDir 'npm.cmd') run preview
