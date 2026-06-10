#Requires -Version 5
# Build NVIDIA AmgX on Windows + MSVC 2022 + CUDA 12.x for use by
# torch-amgx setup.py. Mirror of scripts/build_amgx.sh.
#
# Prerequisites:
#   * VS 2022 Build Tools with C++17 support
#   * CUDA Toolkit 12.x (nvcc on PATH OR in a conda env)
#   * CMake bundled in the VS BuildTools install
#
# Environment knobs:
#   AMGX_SRC   : path to NVIDIA/AMGX checkout (default: ../third_party/AMGX)
#   BUILD_DIR  : where to drop CMake output (default: ../build/amgx)
#   CUDA_ARCH  : semicolon list of compute capabilities (default: 70;80;89;90;120)
param(
    [string]$AmgxSrc  = "$PSScriptRoot\..\third_party\AMGX",
    [string]$BuildDir = "$PSScriptRoot\..\build\amgx",
    [string]$CudaArch = "70;80;89;90;120",
    [string]$VcVars   = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    [string]$CMakeDir = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin",
    [string]$CudaPath = "D:\Software\Miniforge3\envs\torch-cu128\Library"
)

$ErrorActionPreference = "Stop"

# Strip the persistent Windows-registry-level Clash proxy that haunts
# every subprocess on tb16.
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:NO_PROXY = "*"
[System.Net.WebRequest]::DefaultWebProxy = $null

if (-not (Test-Path $AmgxSrc)) {
    throw "AmgX source not found at $AmgxSrc. Run: git submodule update --init --recursive"
}
if (-not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir | Out-Null }
$Install = Join-Path $BuildDir "install"
if (-not (Test-Path $Install)) { New-Item -ItemType Directory -Path $Install | Out-Null }

Write-Host "=== Configuring AmgX (sm_$($CudaArch -replace ';',', sm_')) ===" -ForegroundColor Cyan
$configureBat = @"
@echo off
call "$VcVars"
set PATH=$CMakeDir;$CudaPath\bin;%PATH%
set CUDA_PATH=$CudaPath
cd /d "$BuildDir"
cmake "$AmgxSrc" ^
  -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX="$Install" ^
  -DCMAKE_CUDA_ARCHITECTURES="$CudaArch" ^
  -DCUDAToolkit_ROOT="$CudaPath" ^
  -DAMGX_NO_RPATH=ON ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
"@
$configureBatFile = "$BuildDir\_configure.bat"
$configureBat | Set-Content -Path $configureBatFile -Encoding ASCII
cmd /c "$configureBatFile"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

Write-Host "=== Building AmgX (~30 min) ===" -ForegroundColor Cyan
$buildBat = @"
@echo off
call "$VcVars"
set PATH=$CMakeDir;$CudaPath\bin;%PATH%
cd /d "$BuildDir"
cmake --build . --config Release --parallel
cmake --install . --config Release
"@
$buildBatFile = "$BuildDir\_build.bat"
$buildBat | Set-Content -Path $buildBatFile -Encoding ASCII
cmd /c "$buildBatFile"
if ($LASTEXITCODE -ne 0) { throw "AmgX build failed" }

Write-Host "=== Done. AMGX_DIR=$Install ===" -ForegroundColor Green
Get-ChildItem -Path "$Install" -Recurse -Filter "amgx*.{dll,lib}" -ErrorAction SilentlyContinue | Select-Object FullName
