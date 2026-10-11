. "$PSScriptRoot/task_environment.ps1"
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Start-Process -FilePath "$env:ANDROID_HOME/emulator/emulator.exe" -WindowStyle Hidden -ArgumentList @(
    '-avd', 'medium_phone', '-port', '5554', '-memory', '1536', '-cores', '2',
    '-gpu', 'host', '-no-window', '-no-snapshot', '-no-boot-anim',
    '-skin', '540x1200', '-dpi-device', '210', '-camera-back', 'virtualscene'
) -RedirectStandardOutput "$repoRoot/android/emulator-low-memory.stdout.log" -RedirectStandardError "$repoRoot/android/emulator-low-memory.stderr.log"
