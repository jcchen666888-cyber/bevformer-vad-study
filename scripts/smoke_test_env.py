#!/usr/bin/env python3
"""Fail-fast environment test covering versions, GPU, and compiled ops."""

from __future__ import annotations

import json

import mmcv
import mmdet
import mmdet3d
import mmseg
import torch


def main() -> None:
    from mmcv.ops import MultiScaleDeformableAttention
    from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu

    assert torch.cuda.is_available(), "CUDA is not visible inside WSL"
    device = torch.device("cuda:0")
    # A real kernel launch catches 'no kernel image' failures on Ada GPUs.
    lhs = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    rhs = torch.eye(4, dtype=torch.float32, device=device)
    out = lhs @ rhs
    torch.cuda.synchronize()
    assert torch.equal(lhs, out)
    assert MultiScaleDeformableAttention is not None
    assert points_in_boxes_gpu is not None

    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "mmcv": mmcv.__version__,
        "mmdet": mmdet.__version__,
        "mmseg": mmseg.__version__,
        "mmdet3d": mmdet3d.__version__,
        "cuda_kernel_smoke_test": "passed",
        "compiled_ops_import": "passed",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
