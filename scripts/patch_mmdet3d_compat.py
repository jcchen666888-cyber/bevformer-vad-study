#!/usr/bin/env python3
"""Apply the narrow MMDetection3D 0.17.1 compatibility guard used here.

MMDetection3D 0.17.1 hard-codes MMCV <= 1.4.0.  Ada GPUs are materially more
reliable with the PyTorch 1.13/CUDA 11.7 stack, whose published MMCV wheel is
1.7.2.  VAD uses the unchanged 0.x APIs; this script only widens that import
guard.  The compiled-op and end-to-end smoke tests are the real compatibility
gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mmdet3d_root", type=Path)
    args = parser.parse_args()
    target = args.mmdet3d_root / "mmdet3d/__init__.py"
    text = target.read_text(encoding="utf-8")
    old = "mmcv_maximum_version = '1.4.0'"
    new = "mmcv_maximum_version = '1.7.2'  # vad-bevformer-study Ada compatibility"
    if new in text:
        print(f"already patched: {target}")
        return
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one MMCV guard in {target}")
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text.replace(old, new, 1))
    print(f"patched: {target}")


if __name__ == "__main__":
    main()
