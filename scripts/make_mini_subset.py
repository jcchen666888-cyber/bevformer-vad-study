#!/usr/bin/env python3
"""Keep a short contiguous nuScenes mini validation sequence for inference."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "infos" not in payload:
        raise ValueError(f"Unexpected VAD annotation format: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/nuscenes/vad_nuscenes_infos_temporal_val.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/nuscenes/vad_nuscenes_infos_temporal_val_subset.pkl"),
    )
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    if args.frames < 3:
        raise ValueError("Use at least 3 frames so temporal BEV behavior is visible")

    payload = load_pickle(args.input)
    infos = payload["infos"]
    if not infos:
        raise ValueError("Input annotation contains no samples")

    first_scene = infos[0].get("scene_token")
    if first_scene is None:
        raise KeyError("scene_token is missing from VAD infos")
    contiguous = [item for item in infos if item.get("scene_token") == first_scene]
    selected = contiguous[: args.frames]
    if len(selected) != args.frames:
        raise RuntimeError(
            f"Requested {args.frames} frames, but the first validation scene "
            f"contains only {len(contiguous)}"
        )
    for previous, current in zip(selected, selected[1:]):
        if previous.get("next") != current.get("token"):
            raise RuntimeError(
                "Selected records are not consecutive along sample.next: "
                f"{previous.get('token')} -> {current.get('token')}"
            )
        if current.get("prev") != previous.get("token"):
            raise RuntimeError("Selected records disagree on sample.prev linkage")
        if current.get("timestamp", 0) <= previous.get("timestamp", 0):
            raise RuntimeError("Selected timestamps are not strictly increasing")

    subset = dict(payload)
    subset["infos"] = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(subset, stream, protocol=pickle.HIGHEST_PROTOCOL)

    version = subset.get("metadata", {}).get("version")
    if version != "v1.0-mini":
        raise RuntimeError(f"Expected v1.0-mini metadata, got {version!r}")
    print(
        f"wrote {len(selected)} contiguous frames from scene {first_scene} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
