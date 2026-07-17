#!/usr/bin/env python3
"""Create a guarded nuScenes-mini-compatible copy of VAD's visualizer."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad-root", type=Path, default=Path("_deps/VAD"))
    args = parser.parse_args()
    root = args.vad_root.resolve()
    source = root / "tools/analysis_tools/visualization.py"
    destination = root / "tools/analysis_tools/visualization_mini.py"
    text = source.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "    parser.add_argument('--save-path', help='the dir to save visualization results')\n",
        "    parser.add_argument('--save-path', help='the dir to save visualization results')\n"
        "    parser.add_argument('--dataroot', required=True)\n"
        "    parser.add_argument('--version', default='v1.0-mini')\n",
        "visualizer argument block",
    )
    text = replace_once(
        text,
        "    bevformer_results = mmcv.load(inference_result_path)\n",
        "    mmcv.mkdir_or_exist(out_path)\n"
        "    bevformer_results = mmcv.load(inference_result_path)\n",
        "result loading statement",
    )
    text = replace_once(
        text,
        "    nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)\n",
        "    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)\n",
        "hard-coded nuScenes constructor",
    )
    text = replace_once(
        text,
        "        video.write(vis_img)\n",
        "        video.write(vis_img)\n"
        "        cv2.imwrite(osp.join(out_path, f'frame_{id:03d}.jpg'), vis_img)\n",
        "video frame write",
    )
    text = replace_once(
        text,
        "            mode_idx = [0, 1, 2, 3, 4, 5]\n"
        "            box.render_fut_trajs_grad_color(axes, linewidth=1, mode_idx=mode_idx, fut_ts=6, cmap='autumn')\n",
        "            # None renders every mode without introducing an extra list dimension.\n"
        "            box.render_fut_trajs_grad_color(axes, linewidth=1, mode_idx=None, fut_ts=6, cmap='autumn')\n",
        "multi-modal trajectory rendering",
    )
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(destination)


if __name__ == "__main__":
    main()
