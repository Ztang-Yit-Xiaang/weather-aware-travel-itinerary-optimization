#Requires -RunAsAdministrator

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$currentUserSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$userGrant = "*$($currentUserSid):(OI)(CI)F"
$targets = @(
    "C:\Users\Ztang_Yit_Xiaang\.codex\visualizations\2026\07\29\019faf6d-904b-7202-982c-780fa0473095\w1-final-focused",
    "C:\Users\Ztang_Yit_Xiaang\.codex\visualizations\2026\07\29\019faf6d-904b-7202-982c-780fa0473095\w1-launcher-reuse",
    "C:\Users\Ztang_Yit_Xiaang\.codex\visualizations\2026\07\29\019faf6d-904b-7202-982c-780fa0473095\w1-pytest-cors",
    "C:\Users\Ztang_Yit_Xiaang\.codex\visualizations\2026\07\29\019faf6d-904b-7202-982c-780fa0473095\w1-pytest-final",
    "C:\Users\Ztang_Yit_Xiaang\.codex\visualizations\2026\07\29\019faf6d-904b-7202-982c-780fa0473095\w1-pytest-main",
    "C:\Users\Ztang_Yit_Xiaang\.codex\visualizations\2026\07\29\019faf6d-904b-7202-982c-780fa0473095\w1-pytest-remediation",
    "E:\UMN Researches\Ju Research\PyGRANSO\.pytest_cache",
    "E:\UMN Researches\Ju Research\Report\codex-nested-probe-c67rbl6a"
)

foreach ($target in $targets) {
    $fullTarget = [System.IO.Path]::GetFullPath($target)
    if (-not (Test-Path -LiteralPath $fullTarget)) {
        Write-Host "Already absent: $fullTarget"
        continue
    }

    Write-Host "Repairing access: $fullTarget"
    & takeown.exe /F $fullTarget /R /D Y | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "takeown failed for $fullTarget with exit code $LASTEXITCODE"
    }
    & icacls.exe $fullTarget /grant:r $userGrant /T /C /Q | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "icacls failed for $fullTarget with exit code $LASTEXITCODE"
    }
}

$roots = @(
    "C:\Users\Ztang_Yit_Xiaang\.codex\visualizations\2026\07\29\019faf6d-904b-7202-982c-780fa0473095",
    "E:\UMN Researches\Ju Research"
)
$denied = @()
foreach ($root in $roots) {
    $scanErrors = @()
    Get-ChildItem -LiteralPath $root -Force -Recurse `
        -ErrorAction SilentlyContinue -ErrorVariable +scanErrors | Out-Null
    $denied += @(
        $scanErrors |
            Where-Object { $_.CategoryInfo.Category -eq "PermissionDenied" } |
            ForEach-Object { $_.CategoryInfo.TargetName }
    )
}
$denied = @($denied | Sort-Object -Unique)

Write-Host ""
Write-Host "Final verification"
if ($denied.Count -eq 0) {
    Write-Host "PermissionDenied paths across other workspace roots: 0"
    exit 0
}

Write-Host "PermissionDenied paths across other workspace roots: $($denied.Count)"
$denied | ForEach-Object { Write-Host $_ }
exit 1
