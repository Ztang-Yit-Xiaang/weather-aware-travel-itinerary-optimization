#Requires -RunAsAdministrator

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$targetNames = @(
    ".codex_tmp_conversations_02294c846c7140278777f7127f83e121",
    ".codex_tmp_conversations_a0023334ebf1442eb6b6c5b7c9ca5c68",
    ".w1_direct_file_reaudit_tmp_20260803"
)

foreach ($targetName in $targetNames) {
    $targetPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $targetName))
    $expectedParent = [System.IO.Path]::GetDirectoryName($targetPath)
    if ($expectedParent -ne $repositoryRoot) {
        throw "Refusing target outside the repository: $targetName"
    }
    if (-not (Test-Path -LiteralPath $targetPath)) {
        Write-Host "Already absent: $targetName"
        continue
    }

    & takeown.exe /F $targetPath /A /R /D Y
    if ($LASTEXITCODE -ne 0) {
        throw "takeown failed for $targetName with exit code $LASTEXITCODE"
    }

    & icacls.exe $targetPath /inheritance:e /grant:r '*S-1-5-32-544:(OI)(CI)F' /T /C /Q
    if ($LASTEXITCODE -ne 0) {
        throw "icacls failed for $targetName with exit code $LASTEXITCODE"
    }

    Remove-Item -LiteralPath $targetPath -Recurse -Force
    if (Test-Path -LiteralPath $targetPath) {
        throw "Directory still exists after removal: $targetName"
    }
    Write-Host "Removed: $targetName"
}

Write-Host ""
Write-Host "Final verification"
foreach ($targetName in $targetNames) {
    [pscustomobject]@{
        Path = $targetName
        StillExists = Test-Path -LiteralPath (Join-Path $repositoryRoot $targetName)
    }
}
