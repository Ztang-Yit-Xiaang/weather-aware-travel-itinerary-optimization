#Requires -RunAsAdministrator

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$currentUserSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$userGrant = "*$($currentUserSid):(OI)(CI)F"

# These are the exact repository subtrees found to deny traversal on 2026-08-08.
# The script changes ownership/ACLs only. It never deletes, moves, or rewrites content.
$relativeTargets = @(
    ".codex_backups\california-coast-product-demo-v1-sandbox-acl",
    ".codex_tmp_pytest",
    ".pytest_cache",
    "results\outputs\pytest_tmp",
    "runs\e3ux-weather-repair-demo-v1\dashboard_product",
    "runs\e3ux-weather-repair-demo-v2\dashboard_product",
    "runs\e3ux-weather-repair-demo-v3\dashboard_product",
    "runs\e3ux-weather-repair-demo-v4\dashboard_product",
    "runs\e3ux-weather-repair-demo-v5\dashboard_product",
    "runs\e3ux-weather-repair-demo-v6\dashboard_product",
    "tmp_pytest",
    "tmp_test"
)

foreach ($relativeTarget in $relativeTargets) {
    $targetPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $relativeTarget))
    if (-not $targetPath.StartsWith($repositoryRoot + [System.IO.Path]::DirectorySeparatorChar)) {
        throw "Refusing target outside the repository: $relativeTarget"
    }
    if (-not (Test-Path -LiteralPath $targetPath)) {
        Write-Host "Already absent: $relativeTarget"
        continue
    }

    Write-Host "Repairing access: $relativeTarget"
    & takeown.exe /F $targetPath /R /D Y | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "takeown failed for $relativeTarget with exit code $LASTEXITCODE"
    }

    & icacls.exe $targetPath /grant:r $userGrant /T /C /Q | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "icacls failed for $relativeTarget with exit code $LASTEXITCODE"
    }
}

$scanErrors = @()
Get-ChildItem -LiteralPath $repositoryRoot -Force -Recurse `
    -ErrorAction SilentlyContinue -ErrorVariable +scanErrors | Out-Null
$denied = @(
    $scanErrors |
        Where-Object { $_.CategoryInfo.Category -eq "PermissionDenied" } |
        ForEach-Object { $_.CategoryInfo.TargetName } |
        Sort-Object -Unique
)

Write-Host ""
Write-Host "Final verification"
if ($denied.Count -eq 0) {
    Write-Host "PermissionDenied paths: 0"
    exit 0
}

Write-Host "PermissionDenied paths: $($denied.Count)"
$denied | ForEach-Object { Write-Host $_ }
exit 1
