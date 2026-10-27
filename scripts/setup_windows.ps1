# One-time setup on Windows.
# Requires: Python 3.11-3.13 (64-bit, from python.org). On Windows-on-ARM laptops install the
# regular x64 "Windows installer (64-bit)"; it runs under emulation and all packages are available.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$Commit   = "a254152c1e7cf4aa1d7391cb9260d54a4185118c"
$Sha256   = "68BD5D3FD040FA6E4C464D99D2C7DCACAEBD2A40388EC0902133D4E26BE0F791"
$Size     = 152040442
$Vendor   = Join-Path $Root "third_party\rsna_knee_kaggle_model"
$Weights  = Join-Path $Vendor "best_model.pth"
$Url      = "https://media.githubusercontent.com/media/SlockAhuja/rsna-knee-mri-model/$Commit/best_model.pth"

Write-Host "== 1/4 Python virtual environment"
if (-not (Test-Path ".venv")) {
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv } else { python -m venv .venv }
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -c "import platform,sys; print(sys.version, platform.machine())"

Write-Host "== 2/4 Packages"
& $Py -m pip install --upgrade pip
& $Py -m pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path outputs | Out-Null
& $Py -m pip freeze | Out-File -Encoding utf8 outputs\requirements-frozen.txt

Write-Host "== 3/4 Model weights"
if ((Test-Path $Weights) -and ((Get-Item $Weights).Length -eq $Size)) {
    Write-Host "Weights already present."
} else {
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $Url -OutFile $Weights
}
$h = (Get-FileHash -Algorithm SHA256 $Weights).Hash
if ($h -ne $Sha256) { throw "SHA-256 mismatch: got $h, expected $Sha256. Do not continue." }
Write-Host "SHA-256 OK: $h"

Write-Host "== 4/4 Checks without patient data"
& $Py -m kneeai env
& $Py -m kneeai verify-model
& $Py -m kneeai smoke-test
Write-Host "`nSetup complete."
