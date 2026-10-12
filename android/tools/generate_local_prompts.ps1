[CmdletBinding()]
param(
    [switch]$OnlyMissing
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Speech

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rawDirectory = Join-Path $repositoryRoot 'app/src/main/res/raw'
$manifestDirectory = Join-Path $repositoryRoot 'validation'
New-Item -ItemType Directory -Force -Path $rawDirectory, $manifestDirectory | Out-Null

$prompts = [System.Collections.Generic.List[object]]::new()
$prompts.Add([pscustomobject]@{ Name = 'ar_wall_near'; Text = '前方墙面可能较近，请注意。' })
$prompts.Add([pscustomobject]@{ Name = 'ar_wall_approaching'; Text = '相机可能正在靠近前方墙面，请注意。' })
$prompts.Add([pscustomobject]@{ Name = 'ar_corridor_left'; Text = '画面左侧可见地面较连续，通行条件尚未确认。' })
$prompts.Add([pscustomobject]@{ Name = 'ar_corridor_center'; Text = '画面中央可见地面较连续，通行条件尚未确认。' })
$prompts.Add([pscustomobject]@{ Name = 'ar_corridor_right'; Text = '画面右侧可见地面较连续，通行条件尚未确认。' })
$prompts.Add([pscustomobject]@{ Name = 'perception_started'; Text = '感知已开启' })
$prompts.Add([pscustomobject]@{ Name = 'perception_stopped'; Text = '感知已停止' })
$prompts.Add([pscustomobject]@{ Name = 'perception_interrupted'; Text = '视觉感知暂时中断' })
$prompts.Add([pscustomobject]@{ Name = 'camera_connection_failed'; Text = '摄像头连接失败' })
$prompts.Add([pscustomobject]@{ Name = 'perception_limited'; Text = '感知暂时受限' })
$prompts.Add([pscustomobject]@{ Name = 'perception_recovered'; Text = '感知已恢复' })
$prompts.Add([pscustomobject]@{ Name = 'camera_stopped'; Text = '摄像头已停止' })

$prompts.Add([pscustomobject]@{ Name = 'environment_wall'; Text = '前方检测到墙面。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_stairs'; Text = '前方检测到楼梯。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_step'; Text = '前方检测到台阶。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_pole'; Text = '前方检测到杆柱。' })

$prompts.Add([pscustomobject]@{ Name = 'environment_wall_left'; Text = '左侧检测到墙面。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_wall_center'; Text = '前方检测到墙面。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_wall_right'; Text = '右侧检测到墙面。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_stairs_left'; Text = '左侧检测到楼梯。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_stairs_center'; Text = '前方检测到楼梯。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_stairs_right'; Text = '右侧检测到楼梯。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_step_left'; Text = '左侧检测到台阶，请注意。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_step_center'; Text = '前方检测到台阶，请注意。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_step_right'; Text = '右侧检测到台阶，请注意。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_pole_left'; Text = '左侧检测到杆状结构。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_pole_center'; Text = '前方检测到杆状结构。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_pole_right'; Text = '右侧检测到杆状结构。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_railing_left'; Text = '左侧检测到栏杆或扶手。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_railing_center'; Text = '前方检测到栏杆或扶手。' })
$prompts.Add([pscustomobject]@{ Name = 'environment_railing_right'; Text = '右侧检测到栏杆或扶手。' })

$categories = @(
    [pscustomobject]@{ Key = 'pedestrian'; Text = '行人' },
    [pscustomobject]@{ Key = 'bicycle'; Text = '自行车' },
    [pscustomobject]@{ Key = 'car'; Text = '汽车' },
    [pscustomobject]@{ Key = 'motorcycle'; Text = '摩托车' },
    [pscustomobject]@{ Key = 'bus'; Text = '公交车' },
    [pscustomobject]@{ Key = 'truck'; Text = '卡车' },
    [pscustomobject]@{ Key = 'dog'; Text = '狗' },
    [pscustomobject]@{ Key = 'chair'; Text = '椅子' },
    [pscustomobject]@{ Key = 'target'; Text = '目标' },
    [pscustomobject]@{ Key = 'traffic_light'; Text = '红绿灯' },
    [pscustomobject]@{ Key = 'stop_sign'; Text = '停止标志' },
    [pscustomobject]@{ Key = 'train'; Text = '火车' }
)
$directions = @(
    [pscustomobject]@{ Key = 'left'; Text = '左侧' },
    [pscustomobject]@{ Key = 'center'; Text = '前方' },
    [pscustomobject]@{ Key = 'right'; Text = '右侧' }
)
foreach ($direction in $directions) {
    $prompts.Add([pscustomobject]@{
        Name = "alert_$($direction.Key)_risk_elevated"
        Text = "$($direction.Text)目标风险升高，请注意"
    })
    foreach ($category in $categories) {
        $prompts.Add([pscustomobject]@{
            Name = "alert_$($direction.Key)_$($category.Key)"
            Text = "$($direction.Text)有$($category.Text)"
        })
    }
}

$importantCategories = @(
    [pscustomobject]@{ Key = 'pedestrian'; Text = '行人' },
    [pscustomobject]@{ Key = 'bicycle'; Text = '自行车' },
    [pscustomobject]@{ Key = 'car'; Text = '汽车' },
    [pscustomobject]@{ Key = 'motorcycle'; Text = '摩托车' },
    [pscustomobject]@{ Key = 'bus'; Text = '公交车' },
    [pscustomobject]@{ Key = 'truck'; Text = '卡车' },
    [pscustomobject]@{ Key = 'traffic_light'; Text = '红绿灯' },
    [pscustomobject]@{ Key = 'stop_sign'; Text = '停止标志' },
    [pscustomobject]@{ Key = 'target'; Text = '目标' }
)
foreach ($direction in $directions) {
    foreach ($category in $importantCategories) {
        $baseText = "$($direction.Text)有$($category.Text)"
        $prompts.Add([pscustomobject]@{
            Name = "alert_$($direction.Key)_$($category.Key)_important"
            Text = "$baseText，请注意"
        })
    }
    $prompts.Add([pscustomobject]@{
        Name = "alert_$($direction.Key)_multiple_pedestrians"
        Text = "$($direction.Text)有多名行人"
    })
    $prompts.Add([pscustomobject]@{
        Name = "alert_$($direction.Key)_multiple_targets"
        Text = "$($direction.Text)有多个目标"
    })
}

$synthesizer = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try {
    $voice = $synthesizer.GetInstalledVoices() |
        ForEach-Object { $_.VoiceInfo } |
        Where-Object { $_.Name -eq 'Microsoft Huihui Desktop' -and $_.Culture.Name -eq 'zh-CN' } |
        Select-Object -First 1
    if ($null -eq $voice) {
        throw 'Microsoft Huihui Desktop (zh-CN) is not installed. No prompt audio was generated.'
    }

    $synthesizer.SelectVoice($voice.Name)
    $synthesizer.Rate = 0
    $synthesizer.Volume = 100
    $generatedCount = 0
    $files = foreach ($prompt in $prompts) {
        $outputPath = Join-Path $rawDirectory "$($prompt.Name).wav"
        if (-not ($OnlyMissing -and (Test-Path -LiteralPath $outputPath))) {
            $synthesizer.SetOutputToWaveFile($outputPath)
            $synthesizer.Speak($prompt.Text)
            $synthesizer.SetOutputToNull()
            $generatedCount += 1
        }
        [pscustomobject]@{
            resource = $prompt.Name
            text = $prompt.Text
            file = "app/src/main/res/raw/$($prompt.Name).wav"
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash.ToLowerInvariant()
            bytes = (Get-Item -LiteralPath $outputPath).Length
        }
    }

    $manifest = [ordered]@{
        schema_version = 2
        generated_at_utc = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
        generator = 'android/tools/generate_local_prompts.ps1'
        generator_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
        synthesis = [ordered]@{
            api = 'System.Speech.Synthesis.SpeechSynthesizer'
            voice = $voice.Name
            culture = $voice.Culture.Name
            source = 'Installed Windows offline voice; generated locally without network access.'
            rate = 0
            volume = 100
        }
        host = [ordered]@{
            os = [Environment]::OSVersion.VersionString
            powershell = $PSVersionTable.PSVersion.ToString()
        }
        clips = @($files)
    }
    $manifestPath = Join-Path $manifestDirectory 'local_audio_manifest.json'
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    $manifestHashPath = Join-Path $manifestDirectory 'local_audio_manifest.sha256'
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
    Set-Content -LiteralPath $manifestHashPath -Value "$manifestHash  local_audio_manifest.json" -Encoding ascii
    Write-Output "Generated $generatedCount offline WAV clips and recorded $($files.Count - $generatedCount) existing clips with '$($voice.Name)' into $rawDirectory"
    Write-Output "Manifest: $manifestPath"
    Write-Output "Manifest SHA-256: $manifestHash"
}
finally {
    $synthesizer.Dispose()
}
