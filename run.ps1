<#
.SYNOPSIS  Start CardioSage (backend + frontend) with one command.
.EXAMPLE   powershell -ExecutionPolicy Bypass -File .\run.ps1
.PARAMETER Dev        Also run the Vite dev server with hot reload on http://localhost:5173 (needs Node).
.PARAMETER Port       Backend port (default 8000).
.PARAMETER NoBrowser  Do not open the browser.
.PARAMETER Check      Start, verify health + UI + a real prediction, then shut down. Exit code 0 means everything works.
.PARAMETER Retrain    Train a new model version first.
#>
[CmdletBinding()]
param([switch]$Dev, [int]$Port = 8000, [switch]$NoBrowser, [switch]$Check, [switch]$Retrain)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

# 1. prerequisites, dependencies, build, model (all idempotent and quick when already done)
$global:LASTEXITCODE = 0
& "$PSScriptRoot\setup.ps1" -Retrain:$Retrain
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path '.run' | Out-Null
New-Item -ItemType Directory -Force -Path 'storage' | Out-Null   # SQLite database is created on first start

function PortBusy([int]$p) { return [bool](Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue) }
if (PortBusy $Port) { Fail "Port $Port is already in use. Close the other program or run: .\run.ps1 -Port 8010" }
if ($Dev -and (PortBusy 5173)) { Fail 'Port 5173 (Vite dev server) is already in use.' }

$procs = @()
function Stop-All {
  foreach ($p in $script:procs) { if ($p -and -not $p.HasExited) { & taskkill.exe /PID $p.Id /T /F *> $null } }
}

try {
  # 2. backend
  Write-Host '==> Starting backend' -ForegroundColor Cyan
  $env:CARDIOSAGE_API = "http://127.0.0.1:$Port"
  $be = Start-Process -FilePath $venvPy -ArgumentList @('-m','uvicorn','cardiosage.main:app','--host','127.0.0.1','--port',"$Port") `
        -WorkingDirectory (Join-Path $PSScriptRoot 'backend') -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput '.run\backend.out.log' -RedirectStandardError '.run\backend.err.log'
  $procs += $be

  $healthy = $false
  for ($i = 0; $i -lt 90; $i++) {
    if ($be.HasExited) { break }
    try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 2; if ($h) { $healthy = $true; break } } catch { Start-Sleep -Milliseconds 700 }
  }
  if (-not $healthy) {
    Write-Host '--- last backend log lines ---' -ForegroundColor Yellow
    Get-Content '.run\backend.err.log' -Tail 25 -ErrorAction SilentlyContinue
    Fail 'The backend did not start. Details are in .run\backend.err.log'
  }
  if ($h.status -ne 'ok') { Write-Host "    Backend is running but degraded: dataset=$($h.dataset_available) model=$($h.model_available) db=$($h.database_ok)" -ForegroundColor Yellow }

  # 3. frontend (dev server only when requested; the production build is served by the backend)
  $ui = "http://127.0.0.1:$Port"
  if ($Dev) {
    Write-Host '==> Starting frontend dev server' -ForegroundColor Cyan
    $fe = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c','npm.cmd','run','dev') -WorkingDirectory (Join-Path $PSScriptRoot 'frontend') `
          -PassThru -WindowStyle Hidden -RedirectStandardOutput '.run\frontend.out.log' -RedirectStandardError '.run\frontend.err.log'
    $procs += $fe
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest 'http://127.0.0.1:5173' -UseBasicParsing -TimeoutSec 2 | Out-Null; $ok = $true; break } catch { Start-Sleep -Milliseconds 700 } }
    if (-not $ok) { Fail 'The frontend dev server did not start. See .run\frontend.err.log' }
    $ui = 'http://localhost:5173'
  }

  # 4. self-check
  if ($Check) {
    Write-Host '==> Verifying' -ForegroundColor Cyan
    $page = Invoke-WebRequest "http://127.0.0.1:$Port/" -UseBasicParsing
    if ($page.Content -notmatch 'CardioSage') { Fail 'The UI was not served.' }
    $body = '{"features":{"age":57,"sex":1,"cp":0,"trestbps":140,"chol":241,"fbs":0,"restecg":1,"thalach":123,"exang":1,"oldpeak":0.2,"slope":1,"ca":0,"thal":3},"save":false}'
    $p = Invoke-RestMethod "http://127.0.0.1:$Port/api/predict" -Method Post -ContentType 'application/json' -Body $body
    if ($null -eq $p.probability_class1) { Fail 'Prediction endpoint returned no probability.' }
    Write-Host ("    OK: health={0}, UI served, prediction P(target=1)={1:N3} from {2}" -f $h.status, $p.probability_class1, $p.algorithm) -ForegroundColor Green
    return
  }

  Write-Host ''
  Write-Host "  CardioSage is running" -ForegroundColor Green
  Write-Host "    App:      $ui"
  Write-Host "    API docs: http://127.0.0.1:$Port/docs"
  Write-Host "    Logs:     .run\backend.err.log"
  Write-Host '  Press Ctrl+C to stop.' -ForegroundColor DarkGray
  if (-not $NoBrowser) { Start-Process $ui }

  while ($true) {
    Start-Sleep -Seconds 2
    foreach ($p in $procs) { if ($p.HasExited) { Fail "A server process exited unexpectedly (exit code $($p.ExitCode)). See the logs in .run\" } }
  }
}
finally {
  Stop-All
  Write-Host "`nStopped." -ForegroundColor DarkGray
}
