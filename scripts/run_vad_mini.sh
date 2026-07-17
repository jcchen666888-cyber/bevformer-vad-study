#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAD_ROOT="${PROJECT_ROOT}/_deps/VAD"
FRAMES="${1:-12}"

cd "${VAD_ROOT}"
ln -sfn "${PROJECT_ROOT}/data" data
mkdir -p ckpts work_dirs/vad_tiny_mini
ln -sfn "${PROJECT_ROOT}/artifacts/checkpoints/VAD_tiny.pth" ckpts/VAD_tiny.pth

python "${PROJECT_ROOT}/scripts/make_mini_config.py" \
  --vad-root "${VAD_ROOT}" --project-root "${PROJECT_ROOT}"
python "${PROJECT_ROOT}/scripts/make_standalone_converter.py" \
  --vad-root "${VAD_ROOT}"

python tools/data_converter/vad_nuscenes_converter_standalone.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag vad_nuscenes \
  --version v1.0-mini \
  --canbus ./data

python "${PROJECT_ROOT}/scripts/make_mini_subset.py" \
  --input ./data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
  --output ./data/nuscenes/vad_nuscenes_infos_temporal_val_subset.pkl \
  --frames "${FRAMES}"

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  projects/configs/VAD/VAD_tiny_mini.py \
  ckpts/VAD_tiny.pth \
  --launcher none \
  --out work_dirs/vad_tiny_mini/predictions.pkl \
  --format-only

# VAD's test helper overwrites jsonfile_prefix with a timestamped directory.
# Resolve the newest formatted result deterministically after this run.
FORMATTED_RESULT="$(ls -t test/VAD_tiny_mini/*/pts_bbox/results_nusc.pkl | head -n 1)"
test -n "${FORMATTED_RESULT}"
python "${PROJECT_ROOT}/scripts/inspect_predictions.py" \
  --raw work_dirs/vad_tiny_mini/predictions.pkl \
  --formatted "${FORMATTED_RESULT}" \
  --expected-frames "${FRAMES}"

python "${PROJECT_ROOT}/scripts/make_mini_visualizer.py" \
  --vad-root "${VAD_ROOT}"
python tools/analysis_tools/visualization_mini.py \
  --result-path "${FORMATTED_RESULT}" \
  --save-path "${PROJECT_ROOT}/outputs/vad_tiny_mini" \
  --dataroot ./data/nuscenes \
  --version v1.0-mini

echo "Predictions: ${VAD_ROOT}/work_dirs/vad_tiny_mini/predictions.pkl"
echo "Visualization: ${PROJECT_ROOT}/outputs/vad_tiny_mini/frame_000.jpg"
