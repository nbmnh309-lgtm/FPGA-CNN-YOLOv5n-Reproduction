# Metadata only: no pip install, inference, training, or downloads.
param(
    [string]$Python = 'python',
    [string]$Upstream = '',
    [string]$Weights = '',
    [string]$ModelYaml = '',
    [string]$DataYaml = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Output) { $Output = Join-Path $projectRoot 'outputs/windows_manifest.json' }
$collectorArgs = @((Join-Path $PSScriptRoot 'collect_reproduction_manifest.py'), '--output', $Output, '--torch-info')
if ($Upstream) { $collectorArgs += @('--upstream', $Upstream) }
if ($Weights) { $collectorArgs += @('--weights', $Weights) }
if ($ModelYaml) { $collectorArgs += @('--model-yaml', $ModelYaml) }
if ($DataYaml) { $collectorArgs += @('--data-yaml', $DataYaml) }
& $Python @collectorArgs
if ($LASTEXITCODE -ne 0) { throw "Manifest collector failed (exit $LASTEXITCODE)" }
Write-Host "Gửi file $Output sau khi kiểm tra nội dung. Đây chưa phải kết quả benchmark."
