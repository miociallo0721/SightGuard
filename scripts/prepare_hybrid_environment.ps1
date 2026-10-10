param([ValidateSet('laya', 'vision')][string]$Model = 'laya')
# Explicit runtime setup only. Does not download model weights or change global settings.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtimeRoot = Join-Path $projectRoot 'models/hybrid'
$temporaryRoot = Join-Path $runtimeRoot 'tmp'
$cacheRoot = Join-Path $runtimeRoot 'cache'
New-Item -ItemType Directory -Force $temporaryRoot,$cacheRoot | Out-Null
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$env:TMPDIR = $temporaryRoot
$env:UV_CACHE_DIR = Join-Path $cacheRoot 'uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $cacheRoot 'python'
$env:HF_HOME = Join-Path $cacheRoot 'hf'
$env:TORCH_HOME = Join-Path $cacheRoot 'torch'
$env:PYTHONDONTWRITEBYTECODE = '1'
$runtimeName = if ($Model -eq 'laya') { 'env-structured' } else { 'env-vision' }
$runtimePath = Join-Path $runtimeRoot $runtimeName
uv venv $runtimePath --python (Join-Path $projectRoot '.venv/Scripts/python.exe') --allow-existing
if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
$runtimePython = Join-Path $runtimePath 'Scripts/python.exe'
if ($Model -eq 'laya') {
    uv pip install --python $runtimePython --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match -r (Join-Path $projectRoot 'configs/hybrid-structured-requirements.txt')
} else {
    uv pip install --python $runtimePython -r (Join-Path $projectRoot 'configs/hybrid-vision-requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Vision dependencies failed' }
    $sourceRoot = Join-Path $runtimeRoot 'laya-vision-source'
    if (!(Test-Path -LiteralPath $sourceRoot)) {
        git clone --no-checkout https://github.com/r33drichards/laya-vision $sourceRoot
        if ($LASTEXITCODE -ne 0) { throw 'Vision source clone failed' }
        git -C $sourceRoot checkout 9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c
        if ($LASTEXITCODE -ne 0) { throw 'Vision source checkout failed' }
    }
    $codeRevision = git -C $sourceRoot rev-parse HEAD
    if ($codeRevision -ne '9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c') {
        throw 'Check out pinned Vision commit 9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c in the source clone before retrying; existing source is preserved.'
    }
    uv pip install --python $runtimePython --no-deps -e $sourceRoot
    if ($LASTEXITCODE -ne 0) { throw 'Vision source install failed' }
    $sharedSite = Join-Path $projectRoot '.venv/Lib/site-packages'
    [System.IO.File]::WriteAllText((Join-Path $runtimePath 'Lib/site-packages/sightguard_shared_cpu_torch.pth'), ($sharedSite + [Environment]::NewLine), [System.Text.UTF8Encoding]::new($false))
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
Write-Output "Runtime prepared: $runtimePython. Local weights must be supplied separately."
