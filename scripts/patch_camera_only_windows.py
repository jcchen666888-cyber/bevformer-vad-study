#!/usr/bin/env python3
"""Make MMDetection3D 0.17.1 import-safe for camera-only VAD on Windows.

The VAD camera model uses MMCV's compiled deformable-attention operator but
does not instantiate MMDet3D's voxel, ball-query, IoU or point-cloud kernels.
MMDet3D 0.17.1 imports all of those kernels eagerly.  This guarded patch keeps
the real model/dataset/core code, exposes MMCV's compatible symbols where
needed, and makes genuinely unavailable point-cloud IoU calls fail loudly.
It is a host-tested convenience path; the WSL route builds the full ops set.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OPS_INIT = '''"""Camera-only import bridge for VAD on Windows."""

from mmcv.ops import (
    Voxelization,
    points_in_boxes_all as points_in_boxes_batch,
    points_in_boxes_cpu,
    points_in_boxes_part as points_in_boxes_gpu,
)

__all__ = [
    "Voxelization",
    "points_in_boxes_batch",
    "points_in_boxes_cpu",
    "points_in_boxes_gpu",
]
'''

IOU_INIT = '''"""Import-safe iou3d placeholder for camera-only VAD inference."""


class _UnavailableIou3D:
    def __getattr__(self, name):
        raise RuntimeError(
            f"mmdet3d iou3d CUDA op {name!r} is unavailable in the native "
            "camera-only environment; use the WSL full-ops route"
        )


iou3d_cuda = _UnavailableIou3D()


def _unavailable(*args, **kwargs):
    return getattr(iou3d_cuda, "requested")(*args, **kwargs)


boxes_iou_bev = _unavailable
nms_gpu = _unavailable
nms_normal_gpu = _unavailable

__all__ = ["iou3d_cuda", "boxes_iou_bev", "nms_gpu", "nms_normal_gpu"]
'''

ROIAWARE_INIT = '''"""MMCV-backed point-in-box imports for camera-only VAD."""

from mmcv.ops import (
    RoIAwarePool3d,
    points_in_boxes_all as points_in_boxes_batch,
    points_in_boxes_cpu,
    points_in_boxes_part as points_in_boxes_gpu,
)

__all__ = [
    "RoIAwarePool3d",
    "points_in_boxes_gpu",
    "points_in_boxes_cpu",
    "points_in_boxes_batch",
]
'''

MODELS_INIT = '''"""Minimal model registry imports for camera-only VAD inference."""

from .builder import (
    FUSION_LAYERS,
    MIDDLE_ENCODERS,
    VOXEL_ENCODERS,
    build_backbone,
    build_detector,
    build_fusion_layer,
    build_head,
    build_loss,
    build_middle_encoder,
    build_model,
    build_neck,
    build_roi_extractor,
    build_shared_head,
    build_voxel_encoder,
)
from .detectors.base import Base3DDetector
from .detectors.single_stage_mono3d import SingleStageMono3DDetector
from .segmentors.base import Base3DSegmentor

__all__ = [
    "VOXEL_ENCODERS", "MIDDLE_ENCODERS", "FUSION_LAYERS",
    "build_backbone", "build_neck", "build_roi_extractor",
    "build_shared_head", "build_head", "build_loss", "build_detector",
    "build_fusion_layer", "build_model", "build_middle_encoder",
    "build_voxel_encoder", "Base3DDetector", "Base3DSegmentor",
    "SingleStageMono3DDetector",
]
'''

DETECTORS_INIT = '''from .base import Base3DDetector
from .mvx_two_stage import MVXTwoStageDetector

__all__ = ["Base3DDetector", "MVXTwoStageDetector"]
'''


def guarded_rewrite(path: Path, desired: str, markers: tuple[str, ...]) -> None:
    current = path.read_text(encoding="utf-8")
    if current == desired:
        print(f"already patched: {path}")
        return
    is_prior_camera_patch = "camera-only" in current.lower()
    if not all(marker in current for marker in markers) and not is_prior_camera_patch:
        raise RuntimeError(f"Upstream file changed; refusing to rewrite {path}")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(desired)
    print(f"patched: {path}")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already patched: {path}")
        return
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one guarded match in {path}")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text.replace(old, new, 1))
    print(f"patched: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdet3d-root", type=Path, required=True)
    parser.add_argument("--vad-root", type=Path, required=True)
    args = parser.parse_args()
    m = args.mmdet3d_root.resolve() / "mmdet3d"
    v = args.vad_root.resolve()

    replace_once(
        m / "__init__.py",
        "mmcv_maximum_version = '1.4.0'",
        "mmcv_maximum_version = '1.7.2'  # vad-bevformer-study Ada compatibility",
    )
    guarded_rewrite(
        m / "ops/__init__.py", OPS_INIT,
        ("from .ball_query import ball_query", "from .voxel import"),
    )
    guarded_rewrite(
        m / "ops/iou3d/__init__.py", IOU_INIT,
        ("from .iou3d_utils import",),
    )
    guarded_rewrite(
        m / "ops/roiaware_pool3d/__init__.py", ROIAWARE_INIT,
        ("from .points_in_boxes import", "from .roiaware_pool3d import"),
    )
    guarded_rewrite(
        m / "models/__init__.py", MODELS_INIT,
        ("from .backbones import *", "from .voxel_encoders import *"),
    )
    guarded_rewrite(
        m / "models/detectors/__init__.py", DETECTORS_INIT,
        ("from .centerpoint import CenterPoint", "from .votenet import VoteNet"),
    )
    replace_once(
        m / "datasets/pipelines/data_augment_utils.py",
        "from numba.errors import NumbaPerformanceWarning",
        "from numba.core.errors import NumbaPerformanceWarning",
    )
    metric = v / "projects/mmdet3d_plugin/VAD/planner/metric_stp3.py"
    if "trajs = trajs.to(segmentation.device)" in metric.read_text(encoding="utf-8"):
        print(f"already patched: {metric}")
    else:
        replace_once(
            metric,
            "        B, n_future, _ = trajs.shape\n",
            "        B, n_future, _ = trajs.shape\n"
            "        # Metric occupancy is CPU; keep metric-only indexing together.\n"
            "        trajs = trajs.to(segmentation.device)\n"
            "        gt_trajs = gt_trajs.to(segmentation.device)\n",
        )


if __name__ == "__main__":
    main()
