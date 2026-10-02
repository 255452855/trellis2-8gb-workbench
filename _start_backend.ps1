# ASCII-named wrapper around the Chinese-named backend launcher.
#
# Why this exists: 一键文生3D.bat is ASCII-only (batch files must match the
# console codepage, and UTF-8 Chinese in a .bat corrupts cmd.exe's parsing).
# An ASCII-only .bat therefore cannot contain the literal path of
# "_启动WSL后端.ps1" -- it turns into "_??WSL??.ps1" and PowerShell rejects it
# with "Illegal characters in path".
#
# So we locate it with a wildcard instead of hardcoding the name.
# This file contains no non-ASCII characters on purpose.

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$launcher = Get-ChildItem -LiteralPath $root -Filter '_*WSL*.ps1' -File |
            Where-Object { $_.Name -notlike '*bak*' } |
            Select-Object -First 1

if (-not $launcher) {
    Write-Host '[ERROR] cannot find the backend launcher (_*WSL*.ps1) in' $root
    exit 1
}

Write-Host ('[backend] using launcher: ' + $launcher.Name)
& $launcher.FullName
exit $LASTEXITCODE
