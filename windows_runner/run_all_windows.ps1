param([switch]$Force, [switch]$SkipGpu)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Project .venv Python missing' }
Set-Location -LiteralPath $projectRoot
$runnerArgs = @('-B', 'windows_runner/runner.py')
if ($Force) { $runnerArgs += '--force' }
if ($SkipGpu) { $runnerArgs += '--skip-gpu' }
& $pythonExe @runnerArgs
exit $LASTEXITCODE
