#Requires -RunAsAdministrator

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$currentUserSid = $identity.User.Value
$userGrant = "*$($currentUserSid):(OI)(CI)F"
$roots = @(
    "F:\UMN Courses\IE 5533\Project\weather-aware-travel-itinerary-optimization",
    "C:\Users\Ztang_Yit_Xiaang\.codex\visualizations\2026\07\29\019faf6d-904b-7202-982c-780fa0473095",
    "E:\UMN Researches\Ju Research"
)

foreach ($root in $roots) {
    $fullRoot = [System.IO.Path]::GetFullPath($root)
    if (-not (Test-Path -LiteralPath $fullRoot -PathType Container)) {
        throw "Configured writable root is unavailable: $fullRoot"
    }

    Write-Host "Repairing root owner/access: $fullRoot"
    # Root only: child artifact ownership and bytes are not changed.
    & takeown.exe /F $fullRoot | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "takeown failed for $fullRoot with exit code $LASTEXITCODE"
    }
    & icacls.exe $fullRoot /grant:r $userGrant /Q | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "icacls failed for $fullRoot with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Final verification"
foreach ($root in $roots) {
    $acl = Get-Acl -LiteralPath $root
    $matching = @(
        $acl.Access | Where-Object {
            $_.IdentityReference.Value -in @($identity.Name, $currentUserSid) -and
            $_.AccessControlType -eq "Allow" -and
            ($_.FileSystemRights -band [System.Security.AccessControl.FileSystemRights]::FullControl)
        }
    )
    [pscustomobject]@{
        Root = $root
        Owner = $acl.Owner
        UserFullControl = $matching.Count -gt 0
    }
}
