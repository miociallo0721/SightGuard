# Independent CUDA feasibility runtimes; no model downloads or CPU environment changes.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$hybridRoot = Join-Path $projectRoot 'models/hybrid'
$temporaryRoot = Join-Path $hybridRoot 'tmp'
$cacheRoot = Join-Path $hybridRoot 'cache'
New-Item -ItemType Directory -Force $temporaryRoot,$cacheRoot | Out-Null
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$env:TMPDIR = $temporaryRoot
$env:UV_CACHE_DIR = Join-Path $cacheRoot 'uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $cacheRoot 'python'
$env:HF_HOME = Join-Path $cacheRoot 'hf'
$env:TORCH_HOME = Join-Path $cacheRoot 'torch'
$env:CUDA_CACHE_PATH = Join-Path $cacheRoot 'cuda'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $cacheRoot 'torchinductor'
$env:TRITON_CACHE_DIR = Join-Path $cacheRoot 'triton'
New-Item -ItemType Directory -Force $env:CUDA_CACHE_PATH,$env:TORCHINDUCTOR_CACHE_DIR,$env:TRITON_CACHE_DIR | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'

function Invoke-UvChecked {
    & uv @args
    if ($LASTEXITCODE -ne 0) { throw "uv command failed: $args" }
}

$basePython = Join-Path $projectRoot '.venv/Scripts/python.exe'
$runtimePath = Join-Path $hybridRoot 'env-gpu-runtime'
Invoke-UvChecked venv $runtimePath --python $basePython --allow-existing
$runtimePython = Join-Path $runtimePath 'Scripts/python.exe'
# The observed ORT 1.31.0 Windows wheel requires CUDA 13 DLLs (not CUDA 12).
# Official CUDA 13.0 build; model/device compatibility still requires real tests.
Invoke-UvChecked pip install --python $runtimePython 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu130
$sharedSite = Join-Path $runtimePath 'Lib/site-packages'

foreach ($modelName in @('vision', 'structured')) {
    $modelRuntime = Join-Path $hybridRoot "env-$modelName-gpu"
    Invoke-UvChecked venv $modelRuntime --python $basePython --allow-existing
    $modelPython = Join-Path $modelRuntime 'Scripts/python.exe'
    [IO.File]::WriteAllText((Join-Path $modelRuntime 'Lib/site-packages/sightguard_shared_cuda_torch.pth'), ($sharedSite + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
    $requirements = Join-Path $projectRoot "configs/hybrid-$modelName-requirements.txt"
    $filteredPath = Join-Path $temporaryRoot "$modelName-gpu-requirements.txt"
    Get-Content -LiteralPath $requirements | Where-Object { $_ -notmatch '^(torch|onnxruntime|laya)==' } | Set-Content -LiteralPath $filteredPath -Encoding utf8
    Invoke-UvChecked pip install --python $modelPython -r $filteredPath
    if ($modelName -eq 'vision') {
        $sourceRoot = Join-Path $hybridRoot 'laya-vision-source'
        $revision = git -C $sourceRoot rev-parse HEAD
        if ($LASTEXITCODE -ne 0 -or $revision -ne '9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c') { throw 'Prepare the pinned local Vision source first; existing checkout is preserved.' }
        Invoke-UvChecked pip install --python $modelPython --no-deps -e $sourceRoot
    } else {
        Invoke-UvChecked pip install --python $modelPython 'onnxruntime-gpu==1.31.0' 'pillow==12.3.0'
        Invoke-UvChecked pip install --python $modelPython --no-deps 'laya==0.4.2'
    }
    # Do not silently accept a shadowing CPU torch package in a reused GPU environment.
    & $modelPython -c "import torch; assert torch.version.cuda == '13.0', torch.__version__; assert torch.cuda.is_available(), 'CUDA device unavailable'; x=torch.ones((2,2),device='cuda'); y=x@x; torch.cuda.synchronize(); assert y[0,0].item()==2; print(torch.__version__, torch.cuda.get_device_name(0), torch.backends.cudnn.version())"
    if ($LASTEXITCODE -ne 0) { throw "Incorrect CUDA torch in $modelRuntime; inspect this isolated environment." }
}
Write-Output 'GPU study runtimes prepared. Main/CPU environments and weights are unchanged.'
