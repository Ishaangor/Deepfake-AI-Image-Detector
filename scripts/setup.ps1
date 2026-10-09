<#
.SYNOPSIS
  One-time setup for the AI Image Detector on Windows: creates .venv, installs pinned dependencies, verifies the model.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1          # GPU wheel if an NVIDIA GPU is found, else CPU
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Cpu     # force the CPU-only PyTorch wheel
#>
param(
    [switch]$Cpu,
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Fail($msg, $fix) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    if ($fix) { Write-Host "  fix: $fix" -ForegroundColor Yellow }
    exit 1
}

# 1. Find a Python interpreter (3.11+; 3.13 tested)
function Get-PyVersion([string[]]$cmd) {
    try {
        $rest = @($cmd | Select-Object -Skip 1)
        $out = & $cmd[0] @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return [version]("$out".Trim()) }
    } catch { }
    return $null
}
$PyCmd = $null
$candidates = if ($Python) { ,@($Python.Split(" ")) } else { @(@("py", "-3.13"), @("py", "-3.12"), @("py", "-3.11"), @("python")) }
foreach ($cand in $candidates) {
    if (-not (Get-Command $cand[0] -ErrorAction SilentlyContinue)) { continue }
    $v = Get-PyVersion $cand
    if ($v -and $v -ge [version]"3.11") { $PyCmd = $cand; break }
}
if (-not $PyCmd) { Fail "Python 3.11 or newer was not found." "Install Python 3.13 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and re-run." }
Write-Host ("Using Python: " + ($PyCmd -join " "))

# 2. Virtual environment
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment .venv ..."
    $rest = @($PyCmd | Select-Object -Skip 1)
    & $PyCmd[0] @rest -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create .venv." "Delete a half-created .venv folder and retry." }
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed." "Check your internet connection / proxy and retry." }

# 3. PyTorch: CUDA 12.4 wheel when an NVIDIA GPU is present, otherwise CPU
$useGpu = (-not $Cpu) -and [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$index = if ($useGpu) { "https://download.pytorch.org/whl/cu124" } else { "https://download.pytorch.org/whl/cpu" }
Write-Host ("Installing PyTorch 2.6.0 (" + $(if ($useGpu) { "CUDA 12.4" } else { "CPU" }) + ") - this downloads up to ~2.5 GB ...")
& $Py -m pip install torch==2.6.0 torchvision==0.21.0 --index-url $index
if ($LASTEXITCODE -ne 0) { Fail "PyTorch installation failed." "Retry; on GPU problems run: scripts\setup.ps1 -Cpu" }

# 4. Remaining pinned dependencies
Write-Host "Installing requirements.txt ..."
& $Py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed." "Read the pip error above; a clean retry is: Remove-Item -Recurse -Force .venv; scripts\setup.ps1" }

# 5. Verify environment, model files, calibration and one real prediction
& $Py scripts\check_env.py --smoke
if ($LASTEXITCODE -ne 0) { Fail "Environment check failed (see the FAIL lines above)." "Follow the 'fix:' hints, then re-run scripts\setup.ps1" }
Write-Host ""
Write-Host "Setup complete. Start the app with:  scripts\run.ps1" -ForegroundColor Green
