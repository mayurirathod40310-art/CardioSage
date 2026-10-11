<#
.SYNOPSIS  One-time (and safe-to-rerun) setup for CardioSage on Windows.
.DESCRIPTION
  Checks Python and Node, creates .venv, installs Python and JavaScript dependencies only when they changed,
  builds the frontend and trains the models if no saved model exists. Nothing outside this folder is modified.
.PARAMETER Retrain   Train a new model version even if one exists.
.PARAMETER Rebuild   Force a fresh frontend build.
#>
[CmdletBinding()]
param([switch]$Retrain, [switch]$Rebuild)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Run([string]$exe, [string[]]$arguments, [string]$what) {
  & $exe @arguments
  if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit code $LASTEXITCODE). See the output above." }
}

# ---- 1. prerequisites ------------------------------------------------------------------------
Step 'Checking prerequisites'
$py = @{ Exe = "py"; Extra = @("-3.13"); Version = "3.13" }
Write-Host "    Python $($py.Version)"
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Fail 'Node.js was not found. Install the LTS version from https://nodejs.org/ and rerun.' }
$nodeMajor = [int]((& node -v).TrimStart('v').Split('.')[0])
if ($nodeMajor -lt 18) { Fail "Node.js 18 or newer is required (found $(& node -v))." }
Write-Host "    Node $(& node -v)"
if (-not (Test-Path 'data\heart.csv')) { Fail 'data\heart.csv is missing. Restore it from the project (see DATASET.md for the source).' }

# ---- 2. Python environment ---------------------------------------------------------------------
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
  Step 'Creating virtual environment (.venv)'
  Run $py.Exe (@($py.Extra) + @('-m','venv','.venv')) 'Creating the virtual environment'
}
$reqFile = 'backend\requirements.txt'
$reqHash = (Get-FileHash $reqFile -Algorithm SHA256).Hash
$stamp = '.venv\.requirements.sha256'
if ((-not (Test-Path $stamp)) -or ((Get-Content $stamp -Raw).Trim() -ne $reqHash)) {
  Step 'Installing Python dependencies (first run takes a few minutes)'
  Run $venvPy @('-m','pip','install','--disable-pip-version-check','-q','--upgrade','pip') 'Upgrading pip'
  Run $venvPy @('-m','pip','install','--disable-pip-version-check','-r',$reqFile) 'Installing Python dependencies'
  Set-Content -Path $stamp -Value $reqHash
} else { Write-Host '    Python dependencies up to date' }

# ---- 3. Frontend -------------------------------------------------------------------------------
Push-Location frontend
try {
  $lockStamp = 'node_modules\.package-hash'
  $pkgHash = (Get-FileHash 'package.json' -Algorithm SHA256).Hash
  if ((-not (Test-Path 'node_modules')) -or (-not (Test-Path $lockStamp)) -or ((Get-Content $lockStamp -Raw).Trim() -ne $pkgHash)) {
    Step 'Installing JavaScript dependencies'
    Run 'npm.cmd' @('install','--no-audit','--no-fund') 'npm install'
    Set-Content -Path $lockStamp -Value $pkgHash
  } else { Write-Host '    JavaScript dependencies up to date' }
  $needBuild = $Rebuild -or (-not (Test-Path 'dist\index.html'))
  if (-not $needBuild) {
    $newest = Get-ChildItem -Recurse -File src, index.html, package.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newest.LastWriteTime -gt (Get-Item 'dist\index.html').LastWriteTime) { $needBuild = $true }
  }
  if ($needBuild) { Step 'Building the frontend'; Run 'npm.cmd' @('run','build') 'Frontend build' } else { Write-Host '    Frontend build up to date' }
} finally { Pop-Location }

# ---- 4. Models ----------------------------------------------------------------------------------
$haveModel = Test-Path 'models\current.json'
if ($Retrain -or -not $haveModel) {
  Step 'Training models (about one minute)'
  Push-Location backend
  try { Run $venvPy @('-m','cardiosage.training') 'Model training' } finally { Pop-Location }
} else { Write-Host '    Trained model found' }

Write-Host "`nSetup complete. Start the app with:  .\run.ps1" -ForegroundColor Green
