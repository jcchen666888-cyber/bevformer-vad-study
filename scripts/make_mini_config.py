#!/usr/bin/env python3
"""Create a checked VAD-Tiny config for mini-split inference.

The released checkpoint was trained with BGR mean subtraction and unit
standard deviation.  The current upstream config uses ImageNet RGB
normalization; using it produces numerically wrong predictions.  This script
performs exact-text guarded edits so an upstream change fails loudly instead
of silently creating a bad experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_NORM = """img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)"""
CHECKPOINT_NORM = """img_norm_cfg = dict(
    # Original normalization used to train the released checkpoint.
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


def build_config(vad_root: Path, project_root: Path) -> Path:
    source = vad_root / "projects/configs/VAD/VAD_tiny_stage_2.py"
    template = project_root / "configs/VAD_tiny_mini.template.py"
    destination = vad_root / "projects/configs/VAD/VAD_tiny_mini.py"

    text = source.read_text(encoding="utf-8")
    text = replace_once(text, OLD_NORM, CHECKPOINT_NORM, "normalization block")
    text += "\n\n# ---- vad-bevformer-study mini overrides ----\n"
    text += template.read_text(encoding="utf-8")
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad-root", type=Path, default=Path("_deps/VAD"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    output = build_config(args.vad_root.resolve(), args.project_root.resolve())
    print(output)


if __name__ == "__main__":
    main()
