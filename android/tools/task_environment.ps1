# Task-local environment only. All writable caches and temporary files stay on D:.
$ErrorActionPreference = 'Stop'
$env:ANDROID_HOME = 'D:\Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$env:ANDROID_USER_HOME = 'D:\Android\.android'
$env:ANDROID_AVD_HOME = 'D:\AndroidAvd'
$env:GRADLE_USER_HOME = 'D:\Android\.gradle'
$env:UV_CACHE_DIR = 'D:\Android\uv-cache'
$env:TEMP = 'D:\Temp\SightGuard'
$env:TMP = $env:TEMP
$env:JAVA_TOOL_OPTIONS = '-Djava.io.tmpdir=D:/Temp/SightGuard'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
