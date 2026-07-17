#!/usr/bin/env python3
"""Generate a mini converter that does not import compiled MMDet3D ops.

The official converter only needs MMDet3D's nuScenes category mapping and a
NumPy projection helper.  Importing the full package eagerly loads unrelated
3D CUDA extensions, which prevents data conversion before the build finishes.
This guarded generator substitutes exactly those two pure-Python definitions;
model inference still uses the real MMDet3D installation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENT = '''class NuScenesDataset:
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }


def points_cam2img(points_3d, proj_mat, with_depth=False):
    """NumPy projection copied from MMDetection3D 0.17.1 box_np_ops."""
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    if len(proj_mat.shape) != 2:
        raise ValueError("projection matrix must be 2-D")
    d1, d2 = proj_mat.shape[:2]
    if (d1, d2) not in ((3, 3), (3, 4), (4, 4)):
        raise ValueError(f"unsupported projection shape: {(d1, d2)}")
    if d1 == 3:
        expanded = np.eye(4, dtype=proj_mat.dtype)
        expanded[:d1, :d2] = proj_mat
        proj_mat = expanded
    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    projected = point_2d[..., :2] / point_2d[..., 2:3]
    if with_depth:
        return np.concatenate([projected, point_2d[..., 2:3]], axis=-1)
    return projected
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad-root", type=Path, default=Path("_deps/VAD"))
    args = parser.parse_args()
    root = args.vad_root.resolve()
    source = root / "tools/data_converter/vad_nuscenes_converter.py"
    destination = root / "tools/data_converter/vad_nuscenes_converter_standalone.py"
    text = source.read_text(encoding="utf-8")
    old = (
        "from mmdet3d.datasets import NuScenesDataset\n"
        "from nuscenes.utils.geometry_utils import view_points\n"
        "from mmdet3d.core.bbox.box_np_ops import points_cam2img\n"
    )
    new = "from nuscenes.utils.geometry_utils import view_points\n\n" + REPLACEMENT + "\n"
    if text.count(old) != 1:
        raise RuntimeError("Official converter import block changed; review required")
    text = text.replace(old, new, 1)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(destination)


if __name__ == "__main__":
    main()
