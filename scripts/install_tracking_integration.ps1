<#
.SYNOPSIS
Install AceTree-Py's tracking-enabled checkout.

.DESCRIPTION
Verifies that a Git checkout is on the tracking-integration branch, then
installs that checkout in editable mode. The GUI dependency set is installed
by default.
#>
[CmdletBinding()]
param(
    [ValidateSet("core", "gui", "all")]
    [string]$Variant = "gui",

    [string]$Python = "python",

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "tracking-integration"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$GitMarker = Join-Path $RepoRoot ".git"

if (Test-Path -LiteralPath $GitMarker) {
    $CurrentBranch = (& git -C $RepoRoot branch --show-current 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not determine the Git branch for $RepoRoot."
    }

    $CurrentBranch = "$CurrentBranch".Trim()
    if ([string]::IsNullOrWhiteSpace($CurrentBranch)) {
        throw "Detached checkout: switch to branch '$ExpectedBranch' before installing."
    }
    elseif ($CurrentBranch -ne $ExpectedBranch) {
        throw "This installer requires branch '$ExpectedBranch'; current branch is '$CurrentBranch'. Run: git switch $ExpectedBranch"
    }
}

$InstallTarget = if ($Variant -eq "core") {
    $RepoRoot
}
else {
    "${RepoRoot}[$Variant]"
}

Write-Host "Installing AceTree-Py '$Variant' from branch '$ExpectedBranch'."
Write-Host "$Python -m pip install --editable `"$InstallTarget`""

if ($DryRun) {
    return
}

& $Python -m pip install --editable $InstallTarget
if ($LASTEXITCODE -ne 0) {
    throw "AceTree-Py installation failed with exit code $LASTEXITCODE."
}

$VersionOutput = (& $Python -m acetree_py --version)
if ($LASTEXITCODE -ne 0) {
    throw "AceTree-Py installed, but its version check failed."
}
if ("$VersionOutput" -notmatch "\(tracking integration\)") {
    throw "The installed build did not identify itself as the tracking integration."
}
Write-Host "$VersionOutput"
