#!/usr/bin/env python3
"""Validate VAD raw and formatted outputs and print a compact JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mmcv
import numpy as np


REQUIRED_PTS_KEYS = {
    "boxes_3d",
    "labels_3d",
    "scores_3d",
    "trajs_3d",
    "map_labels_3d",
    "map_scores_3d",
    "map_pts_3d",
    "ego_fut_preds",
    "ego_fut_cmd",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--formatted", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=12)
    args = parser.parse_args()

    raw = mmcv.load(str(args.raw))
    if not isinstance(raw, list) or len(raw) != args.expected_frames:
        raise RuntimeError(f"Expected {args.expected_frames} raw frames, got {len(raw)}")
    first_pts = raw[0].get("pts_bbox", raw[0])
    missing = REQUIRED_PTS_KEYS - set(first_pts)
    if missing:
        raise KeyError(f"Missing VAD output keys: {sorted(missing)}")

    formatted = mmcv.load(str(args.formatted))
    required_top = {"meta", "results", "map_results", "plan_results"}
    if required_top - set(formatted):
        raise KeyError(f"Formatted result keys: {sorted(formatted)}")
    counts = {
        key: len(formatted[key])
        for key in ("results", "map_results", "plan_results")
    }
    if set(counts.values()) != {args.expected_frames}:
        raise RuntimeError(f"Formatted frame counts mismatch: {counts}")

    token = next(iter(formatted["results"]))
    detections = formatted["results"][token]
    vectors = formatted["map_results"][token]["vectors"]
    plans, command = formatted["plan_results"][token]
    command_index = int(np.argmax(command[0, 0, 0]))
    cumulative_plan = np.asarray(plans[command_index]).cumsum(axis=0)
    report = {
        "raw_frames": len(raw),
        "raw_first_keys": sorted(first_pts),
        "formatted_counts": counts,
        "first_sample_token": token,
        "first_detection_count": len(detections),
        "first_map_vector_count": len(vectors),
        "command_index": command_index,
        "ego_plan_shape": list(np.asarray(first_pts["ego_fut_preds"]).shape),
        "selected_plan_endpoint_m": np.round(cumulative_plan[-1], 4).tolist(),
        "ok": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
