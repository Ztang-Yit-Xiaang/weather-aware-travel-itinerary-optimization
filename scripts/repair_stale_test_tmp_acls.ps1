#Requires -RunAsAdministrator

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$currentUserSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$userGrant = "*$($currentUserSid):(OI)(CI)F"
$relativeTargets = @(
    "tmp\role-backend-pytest-20260808-a",
    "tmp\role-backend-pytest-20260808-b",
    "tmp\role-backend-pytest-20260808-c",
    "tmp\role-backend-pytest-20260808-d",
    "tmp\role-backend-pytest-20260808-e",
    "tmp\role-cross-slice-pytest-20260808-a",
    "tmp\role-frontend-final-20260808-a",
    "tmp\role-frontend-full-20260808-a",
    "tmp\role-frontend-hmac-20260808-a",
    "tmp\role-frontend-hmac-20260808-b",
    "tmp\role-frontend-hmac-20260808-c",
    "tmp\role-frontend-hmac-full-20260808-a",
    "tmp\role-frontend-hmac-integration-20260808-a",
    "tmp\role-frontend-integration-20260808-a",
    "tmp\role-frontend-integration-20260808-b",
    "tmp\role-frontend-pytest-20260808-a",
    "tmp\role-frontend-pytest-20260808-b",
    "tmp\role-frontend-replay-20260808-a",
    "tmp\role-impacted-pytest-20260808-a",
    "tmp\role-impacted-pytest-20260808-b",
    "tmp\role-legacy-pytest-20260808-a",
    "tmp\role-product-pytest-20260808-a",
    "tmp\role-replay-valid-20260808-a",
    "tmp\role-replay-valid-20260808-b",
    "tmp\role-replay-valid-20260808-c",
    "tmp\role-replay-valid-20260808-d",
    "tmp\role-replay-valid-20260808-e",
    "tmp\role-replay-valid-20260808-f",
    "tmp\stability-ui-pytest-20260810-d",
    "tmp\time-window-backend-pytest-20260808-a",
    "tmp\time-window-backend-pytest-20260808-b",
    "tmp\time-window-backend-pytest-20260808-c",
    "tmp_test\project_checks_full_1786209546514189600"
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
    & icacls.exe $targetPath /inheritance:e /grant:r $userGrant /T /C /Q | Out-Null
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
Write-Host "PermissionDenied paths: $($denied.Count)"
$denied | ForEach-Object { Write-Host $_ }
if ($denied.Count -ne 0) {
    exit 1
}
