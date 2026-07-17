#Requires -RunAsAdministrator
[CmdletBinding()]
param(
    [string]$Distribution = 'Ubuntu-20.04'
)

$ErrorActionPreference = 'Stop'

Write-Host 'Enabling WSL 2 and Virtual Machine Platform...'
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl.exe --set-default-version 2

$installed = wsl.exe --list --quiet 2>$null
if ($installed -notcontains $Distribution) {
    Write-Host "Installing $Distribution without launching it..."
    wsl.exe --install --distribution $Distribution --no-launch
} else {
    Write-Host "$Distribution is already installed."
}

Write-Host ''
Write-Host 'If Windows requests a restart, restart once and then launch the distro:'
Write-Host "  wsl.exe --distribution $Distribution"
Write-Host 'On first launch, create the Linux username/password requested by Ubuntu.'
