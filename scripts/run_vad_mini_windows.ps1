[CmdletBinding()]
param(
    [int]$Frames = 12,
    [string]$CondaEnv = 'vad-study',
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$VadRoot = Join-Path $ProjectRoot '_deps\VAD'
$Mmdet3dRoot = Join-Path $ProjectRoot '_deps\mmdetection3d-0.17.1'
$DataLink = Join-Path $VadRoot 'data'

conda run -n $CondaEnv python (Join-Path $PSScriptRoot 'verify_assets.py') --root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw 'Asset verification failed' }

conda run -n $CondaEnv python (Join-Path $PSScriptRoot 'make_mini_config.py') `
    --vad-root $VadRoot --project-root $ProjectRoot
conda run -n $CondaEnv python (Join-Path $PSScriptRoot 'make_standalone_converter.py') `
    --vad-root $VadRoot
conda run -n $CondaEnv python (Join-Path $PSScriptRoot 'make_mini_visualizer.py') `
    --vad-root $VadRoot
conda run -n $CondaEnv python (Join-Path $PSScriptRoot 'patch_camera_only_windows.py') `
    --mmdet3d-root $Mmdet3dRoot --vad-root $VadRoot

if (!(Test-Path -LiteralPath $DataLink)) {
    New-Item -ItemType Junction -Path $DataLink -Target (Join-Path $ProjectRoot 'data') | Out-Null
}

$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$Mmdet3dRoot;$VadRoot"
$env:CUDA_VISIBLE_DEVICES = '0'
try {
    Push-Location $VadRoot
    conda run -n $CondaEnv python tools/data_converter/vad_nuscenes_converter_standalone.py nuscenes `
        --root-path ./data/nuscenes --out-dir ./data/nuscenes `
        --extra-tag vad_nuscenes --version v1.0-mini --canbus ./data
    if ($LASTEXITCODE -ne 0) { throw 'nuScenes conversion failed' }

    conda run -n $CondaEnv python (Join-Path $ProjectRoot 'scripts\make_mini_subset.py') `
        --input ./data/nuscenes/vad_nuscenes_infos_temporal_val.pkl `
        --output ./data/nuscenes/vad_nuscenes_infos_temporal_val_subset.pkl `
        --frames $Frames
    if ($LASTEXITCODE -ne 0) { throw 'Subset generation failed' }

    $prediction = 'work_dirs/vad_tiny_mini/predictions.pkl'
    conda run -n $CondaEnv python tools/test.py `
        projects/configs/VAD/VAD_tiny_mini.py `
        (Join-Path $ProjectRoot 'artifacts\checkpoints\VAD_tiny.pth') `
        --launcher none --out $prediction --format-only
    if ($LASTEXITCODE -ne 0) { throw 'VAD inference failed' }

    $formatted = Get-ChildItem -LiteralPath (Join-Path $VadRoot 'test\VAD_tiny_mini') `
        -Recurse -Filter 'results_nusc.pkl' |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $formatted) { throw 'Formatted results_nusc.pkl not found' }

    conda run -n $CondaEnv python (Join-Path $ProjectRoot 'scripts\inspect_predictions.py') `
        --raw $prediction --formatted $formatted.FullName --expected-frames $Frames
    if ($LASTEXITCODE -ne 0) { throw 'Prediction structure validation failed' }

    conda run -n $CondaEnv python tools/analysis_tools/visualization_mini.py `
        --result-path $formatted.FullName `
        --save-path (Join-Path $ProjectRoot 'outputs\vad_tiny_mini') `
        --dataroot ./data/nuscenes --version v1.0-mini
    if ($LASTEXITCODE -ne 0) { throw 'Visualization failed' }
} finally {
    Pop-Location
    $env:PYTHONPATH = $oldPythonPath
}

Write-Host "Predictions: $VadRoot\work_dirs\vad_tiny_mini\predictions.pkl"
Write-Host "Preview: $ProjectRoot\outputs\vad_tiny_mini\frame_000.jpg"
