$ErrorActionPreference = 'Stop'

$repository = Split-Path -Parent $PSScriptRoot
$uv = 'C:\Users\arach\.local\bin\uv.exe'
$catalog = Join-Path (Split-Path -Parent $repository) 'archive\TOI_2026.06.25_21.21.19.csv'
$logDirectory = Join-Path $repository 'logs'
$logFile = Join-Path $logDirectory 'tesscut-worker.log'

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
Set-Location $repository
$env:PYTHONUNBUFFERED = '1'

& $uv run --extra tesscut transitlens-tesscut-trainer `
    --catalog $catalog `
    --interval-seconds 300 `
    --max-per-cycle 2 `
    --minimum-per-class 20 *>> $logFile

exit $LASTEXITCODE
