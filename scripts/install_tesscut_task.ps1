$ErrorActionPreference = 'Stop'

$taskName = 'TransitLens TESSCut Trainer'
$runner = Join-Path $PSScriptRoot 'run_tesscut_worker.ps1'
$powerShell = (Get-Command powershell.exe).Source
$arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runner`""

$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description 'Collect labeled TESSCut light curves every five minutes and train TransitLens.' `
    -Force | Out-Null

Start-ScheduledTask -TaskName $taskName
Write-Output "Installed and started '$taskName'."
