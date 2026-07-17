[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Downloads = Join-Path $ProjectRoot 'artifacts\downloads'
$Checkpoints = Join-Path $ProjectRoot 'artifacts\checkpoints'
$Data = Join-Path $ProjectRoot 'data'
$Deps = Join-Path $ProjectRoot '_deps'
New-Item -ItemType Directory -Force -Path $Downloads, $Checkpoints, $Data, $Deps | Out-Null

function Get-ResumableFile {
    param([string]$Url, [string]$Destination, [long]$ExpectedBytes)
    if ((Test-Path -LiteralPath $Destination) -and
        ((Get-Item -LiteralPath $Destination).Length -eq $ExpectedBytes)) {
        Write-Host "verified: $Destination"
        return
    }
    & curl.exe -L --fail --retry 5 --retry-delay 3 -C - -o $Destination $Url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $Url" }
    $actual = (Get-Item -LiteralPath $Destination).Length
    if ($actual -ne $ExpectedBytes) {
        throw "size mismatch for $Destination ($actual != $ExpectedBytes)"
    }
}

Get-ResumableFile 'https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-mini.tgz' (Join-Path $Downloads 'v1.0-mini.tgz') 4168148189
Get-ResumableFile 'https://motional-nuscenes.s3.amazonaws.com/public/v1.0/can_bus.zip' (Join-Path $Downloads 'can_bus.zip') 780974697
Get-ResumableFile 'https://motional-nuscenes.s3.amazonaws.com/public/v1.0/nuScenes-map-expansion-v1.3.zip' (Join-Path $Downloads 'nuScenes-map-expansion-v1.3.zip') 398535531

$weight = Join-Path $Checkpoints 'VAD_tiny.pth'
& (Join-Path $PSScriptRoot 'download_gdrive_resumable.ps1') `
    -FileId '1KgCC_wFqPH0CQqdr6Pp2smBX5ARPaqne' `
    -Destination $weight `
    -ExpectedBytes 484968871
if ($LASTEXITCODE -ne 0) { throw 'VAD-Tiny checkpoint download failed' }
if ((Get-Item -LiteralPath $weight).Length -ne 484968871) {
    throw 'VAD-Tiny checkpoint size mismatch'
}

Get-ResumableFile 'https://codeload.github.com/hustvl/VAD/zip/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8' (Join-Path $Downloads 'VAD-main.zip') 542442
Get-ResumableFile 'https://codeload.github.com/open-mmlab/mmdetection3d/zip/refs/tags/v0.17.1' (Join-Path $Downloads 'mmdetection3d-v0.17.1.zip') 9020447

$vadRoot = Join-Path $Deps 'VAD'
if (!(Test-Path -LiteralPath $vadRoot)) {
    Expand-Archive -LiteralPath (Join-Path $Downloads 'VAD-main.zip') -DestinationPath $Deps
    Move-Item -LiteralPath (Join-Path $Deps 'VAD-1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8') -Destination $vadRoot
}
$mmdet3dRoot = Join-Path $Deps 'mmdetection3d-0.17.1'
if (!(Test-Path -LiteralPath $mmdet3dRoot)) {
    Expand-Archive -LiteralPath (Join-Path $Downloads 'mmdetection3d-v0.17.1.zip') -DestinationPath $Deps
}

$nuscRoot = Join-Path $Data 'nuscenes'
New-Item -ItemType Directory -Force -Path $nuscRoot | Out-Null
if (!(Test-Path -LiteralPath (Join-Path $nuscRoot 'v1.0-mini'))) {
    & tar.exe -xzf (Join-Path $Downloads 'v1.0-mini.tgz') -C $nuscRoot
}
if (!(Test-Path -LiteralPath (Join-Path $Data 'can_bus'))) {
    & tar.exe -xf (Join-Path $Downloads 'can_bus.zip') -C $Data
}
if (!(Test-Path -LiteralPath (Join-Path $nuscRoot 'maps\expansion'))) {
    & tar.exe -xf (Join-Path $Downloads 'nuScenes-map-expansion-v1.3.zip') -C (Join-Path $nuscRoot 'maps')
}

python (Join-Path $PSScriptRoot 'verify_assets.py') --root $ProjectRoot
Write-Host 'All minimal assets are downloaded, extracted, and size-verified.'
