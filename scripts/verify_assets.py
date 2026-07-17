#!/usr/bin/env python3
"""Verify the exact minimal-download contract without importing PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SIZES = {
    "artifacts/downloads/v1.0-mini.tgz": 4_168_148_189,
    "artifacts/downloads/can_bus.zip": 780_974_697,
    "artifacts/downloads/nuScenes-map-expansion-v1.3.zip": 398_535_531,
    "artifacts/checkpoints/VAD_tiny.pth": 484_968_871,
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--hash", action="store_true", help="also compute SHA-256")
    args = parser.parse_args()
    root = args.root.resolve()
    report = []
    failed = False
    for relative, expected in EXPECTED_SIZES.items():
        path = root / relative
        actual = path.stat().st_size if path.exists() else None
        ok = actual == expected
        failed |= not ok
        row = {"path": relative, "expected_bytes": expected, "actual_bytes": actual, "ok": ok}
        if ok and args.hash:
            row["sha256"] = sha256(path)
        report.append(row)
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
