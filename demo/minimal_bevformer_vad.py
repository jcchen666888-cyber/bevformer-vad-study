#!/usr/bin/env python3
"""A small, inspectable BEVFormer -> VAD teaching loop.

This is not a neural-network reimplementation.  It preserves the dataflow and
the mathematics that are easiest to test locally:

1. six surround views expose different BEV reference points;
2. visibility-weighted cross-view sampling forms a BEV feature;
3. ego-motion warps the previous BEV into the current frame;
4. local maxima become vectorized agent queries;
5. nearest temporal matches produce constant-velocity forecasts;
6. a differentiable-style cost ranks quintic ego trajectories.

The official VAD checkpoint inference remains the reproduction proof.  This
file is the transparent companion used by the derivations and self-tests.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray

    @property
    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y)

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.y), len(self.x)


@dataclass(frozen=True)
class Camera:
    name: str
    yaw: float
    fov: float = math.radians(100.0)


@dataclass
class Candidate:
    terminal_offset: float
    speed: float
    xy: np.ndarray
    collision: float
    lane: float
    comfort: float
    speed_error: float
    total: float


CAMERAS = tuple(
    Camera(name, math.radians(yaw))
    for name, yaw in (
        ("front", 0),
        ("front_left", 60),
        ("back_left", 120),
        ("back", 180),
        ("back_right", -120),
        ("front_right", -60),
    )
)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def softmax(values: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def gaussian_field(
    grid: Grid, centers: np.ndarray, sigma_x: float = 1.1, sigma_y: float = 0.8
) -> np.ndarray:
    xx, yy = grid.mesh
    field = np.zeros(grid.shape, dtype=np.float64)
    for cx, cy in np.asarray(centers):
        field += np.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
    return np.clip(field, 0.0, 1.0)


def camera_visibility(grid: Grid, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
    """Return a visibility mask and deformable-attention-style logit.

    A BEV reference point p is projected only if its bearing lies inside the
    camera frustum.  Points near the optical axis and nearer to the camera
    receive larger logits; softmax across cameras turns these into weights.
    """

    xx, yy = grid.mesh
    bearing = np.arctan2(yy, xx)
    delta = wrap_angle(bearing - camera.yaw)
    distance = np.sqrt(xx**2 + yy**2)
    visible = (np.abs(delta) <= camera.fov / 2) & (distance >= 1.0)
    logit = -2.0 * (delta / (camera.fov / 2)) ** 2 - 0.015 * distance
    return visible, np.where(visible, logit, -1e9)


def multiview_to_bev(
    truth: np.ndarray, grid: Grid, cameras: Sequence[Camera], seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse six noisy view features with normalized visibility attention."""

    rng = np.random.default_rng(seed)
    features, logits, masks = [], [], []
    for index, camera in enumerate(cameras):
        mask, logit = camera_visibility(grid, camera)
        # Each camera sees the same latent scene through a different bounded
        # perturbation.  The sinusoid makes the demo deterministic by view.
        noise = 0.025 * rng.standard_normal(grid.shape)
        bias = 0.012 * np.sin((index + 1) * truth * np.pi)
        features.append(np.where(mask, np.clip(truth + noise + bias, 0, 1), 0.0))
        logits.append(logit)
        masks.append(mask)
    logit_stack = np.stack(logits)
    weights = softmax(logit_stack, axis=0)
    visible_any = np.any(np.stack(masks), axis=0)
    fused = np.sum(weights * np.stack(features), axis=0)
    return np.where(visible_any, fused, 0.0), weights


def bilinear_sample(field: np.ndarray, grid: Grid, query_x: np.ndarray, query_y: np.ndarray) -> np.ndarray:
    """Sample a regular BEV tensor at metric coordinates."""

    ix = (query_x - grid.x[0]) / (grid.x[1] - grid.x[0])
    iy = (query_y - grid.y[0]) / (grid.y[1] - grid.y[0])
    x0 = np.floor(ix).astype(int)
    y0 = np.floor(iy).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < len(grid.x)) & (y1 < len(grid.y))
    x0c, x1c = np.clip(x0, 0, len(grid.x) - 1), np.clip(x1, 0, len(grid.x) - 1)
    y0c, y1c = np.clip(y0, 0, len(grid.y) - 1), np.clip(y1, 0, len(grid.y) - 1)
    wx, wy = ix - x0, iy - y0
    sampled = (
        (1 - wx) * (1 - wy) * field[y0c, x0c]
        + wx * (1 - wy) * field[y0c, x1c]
        + (1 - wx) * wy * field[y1c, x0c]
        + wx * wy * field[y1c, x1c]
    )
    return np.where(valid, sampled, 0.0)


def warp_previous_bev(previous: np.ndarray, grid: Grid, ego_dx: float, ego_dy: float = 0.0) -> np.ndarray:
    """SE(2) translation case: current q samples previous q + delta_ego."""

    xx, yy = grid.mesh
    return bilinear_sample(previous, grid, xx + ego_dx, yy + ego_dy)


def find_peaks(field: np.ndarray, grid: Grid, count: int = 3, radius_cells: int = 6) -> np.ndarray:
    """Greedy non-maximum suppression returning metric (x, y) queries."""

    work = field.copy()
    peaks: List[Tuple[float, float]] = []
    for _ in range(count):
        flat_index = int(np.argmax(work))
        iy, ix = np.unravel_index(flat_index, work.shape)
        if work[iy, ix] < 0.25:
            break
        peaks.append((float(grid.x[ix]), float(grid.y[iy])))
        y0, y1 = max(0, iy - radius_cells), min(work.shape[0], iy + radius_cells + 1)
        x0, x1 = max(0, ix - radius_cells), min(work.shape[1], ix + radius_cells + 1)
        work[y0:y1, x0:x1] = -np.inf
    return np.asarray(peaks, dtype=np.float64)


def match_velocities(previous: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
    velocities = []
    unused = set(range(len(previous)))
    for point in current:
        if not unused:
            velocities.append(np.zeros(2))
            continue
        index = min(unused, key=lambda item: float(np.linalg.norm(previous[item] - point)))
        unused.remove(index)
        velocities.append((point - previous[index]) / dt)
    return np.asarray(velocities)


def quintic_lateral(offset: float, tau: np.ndarray) -> np.ndarray:
    """Minimum-jerk boundary solution with zero end velocity/acceleration."""

    return offset * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def plan_ego(
    agents: np.ndarray,
    velocities: np.ndarray,
    horizon: float = 3.0,
    steps: int = 12,
    target_speed: float = 8.0,
) -> List[Candidate]:
    times = np.linspace(horizon / steps, horizon, steps)
    predicted = agents[None, :, :] + times[:, None, None] * velocities[None, :, :]
    candidates: List[Candidate] = []
    for offset in (-3.0, -1.5, 0.0, 1.5, 3.0):
        for speed in (5.0, 6.5, 8.0):
            tau = times / horizon
            xy = np.column_stack((speed * times, quintic_lateral(offset, tau)))
            delta = xy[:, None, :] - predicted
            collision = float(
                np.sum(np.exp(-0.5 * ((delta[..., 0] / 2.8) ** 2 + (delta[..., 1] / 1.1) ** 2)))
            )
            lane = float(np.mean((xy[:, 1] / 3.5) ** 2))
            comfort = float(np.mean(np.diff(xy[:, 1], n=2) ** 2))
            speed_error = (speed - target_speed) ** 2
            total = 28.0 * collision + 4.0 * lane + 16.0 * comfort + 0.7 * speed_error
            candidates.append(
                Candidate(offset, speed, xy, collision, lane, comfort, speed_error, total)
            )
    return sorted(candidates, key=lambda item: item.total)


def build_scene(grid: Grid):
    dt, ego_dx = 0.5, 2.0
    previous_agents = np.array([[13.5, 0.1], [21.5, 4.2], [16.5, -5.2]])
    world_velocities = np.array([[0.5, 0.0], [-0.7, -0.2], [0.2, 0.25]])
    current_agents = previous_agents + world_velocities * dt - np.array([ego_dx, 0.0])
    previous_truth = gaussian_field(grid, previous_agents)
    current_truth = gaussian_field(grid, current_agents)
    previous_bev, _ = multiview_to_bev(previous_truth, grid, CAMERAS, seed=7)
    current_bev, camera_weights = multiview_to_bev(current_truth, grid, CAMERAS, seed=11)
    aligned_previous = warp_previous_bev(previous_bev, grid, ego_dx)
    temporal_bev = 0.72 * current_bev + 0.28 * aligned_previous
    previous_queries = find_peaks(aligned_previous, grid)
    current_queries = find_peaks(temporal_bev, grid)
    velocities = match_velocities(previous_queries, current_queries, dt)
    candidates = plan_ego(current_queries, velocities)
    return {
        "previous_truth": previous_truth,
        "current_truth": current_truth,
        "previous_bev": previous_bev,
        "current_bev": current_bev,
        "aligned_previous": aligned_previous,
        "temporal_bev": temporal_bev,
        "camera_weights": camera_weights,
        "previous_queries": previous_queries,
        "current_queries": current_queries,
        "velocities": velocities,
        "candidates": candidates,
    }


def plot_result(grid: Grid, scene: dict, output: Path) -> None:
    best = scene["candidates"][0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax = axes[0]
    extent = [grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]]
    image = ax.imshow(scene["temporal_bev"], origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, label="temporal BEV occupancy")
    ax.axhline(3.5, color="white", ls="--", lw=1, alpha=0.6)
    ax.axhline(-3.5, color="white", ls="--", lw=1, alpha=0.6)
    ax.axhline(0, color="cyan", lw=1.2, alpha=0.8, label="vectorized lane center")
    queries = scene["current_queries"]
    velocities = scene["velocities"]
    ax.scatter(queries[:, 0], queries[:, 1], marker="x", s=80, color="lime", label="agent queries")
    ax.quiver(queries[:, 0], queries[:, 1], velocities[:, 0], velocities[:, 1], color="lime", scale=7)
    for candidate in scene["candidates"]:
        ax.plot(candidate.xy[:, 0], candidate.xy[:, 1], color="deepskyblue", alpha=0.12, lw=1)
    ax.plot(best.xy[:, 0], best.xy[:, 1], color="yellow", lw=3, label="selected ego plan")
    ax.scatter([0], [0], marker="^", s=100, color="white", edgecolor="black", label="ego")
    ax.set(xlim=(-5, 28), ylim=(-10, 10), xlabel="x forward (m)", ylabel="y left (m)")
    ax.set_title("Minimal closed loop: BEV → vectors → prediction → planning")
    ax.legend(loc="upper right", fontsize=8)

    top = scene["candidates"][:8]
    labels = [f"y={item.terminal_offset:+.1f}, v={item.speed:.1f}" for item in top]
    axes[1].barh(np.arange(len(top)), [item.total for item in top], color=["#f4d35e"] + ["#4f8bc9"] * 7)
    axes[1].set_yticks(np.arange(len(top)), labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("total planning cost (lower is better)")
    axes[1].set_title(
        "Candidate ranking\n"
        f"best: offset={best.terminal_offset:+.1f} m, speed={best.speed:.1f} m/s"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_gif(grid: Grid, scene: dict, output: Path, frames: int) -> None:
    best = scene["candidates"][0]
    agents = scene["current_queries"]
    velocities = scene["velocities"]
    images = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for frame in range(frames):
        progress = frame / max(frames - 1, 1)
        index = min(int(progress * (len(best.xy) - 1)), len(best.xy) - 1)
        t = progress * 3.0
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axhline(3.5, color="gray", ls="--")
        ax.axhline(-3.5, color="gray", ls="--")
        ax.axhline(0, color="teal", lw=1)
        predicted_agents = agents + t * velocities
        ax.scatter(predicted_agents[:, 0], predicted_agents[:, 1], s=90, color="tomato", label="predicted agents")
        ax.plot(best.xy[:, 0], best.xy[:, 1], color="goldenrod", lw=2)
        ax.scatter(best.xy[index, 0], best.xy[index, 1], marker="^", s=110, color="navy", label="ego plan")
        ax.set(xlim=(-2, 28), ylim=(-9, 9), xlabel="x (m)", ylabel="y (m)", title=f"Vector prediction and ego plan, t={t:.1f}s")
        ax.legend(loc="upper right")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        images.append(rgba[:, :, :3].copy())
        plt.close(fig)
    imageio.mimsave(output, images, duration=0.08, loop=0)


def nearest_error(points: np.ndarray, expected: np.ndarray) -> float:
    return max(float(np.min(np.linalg.norm(points - item, axis=1))) for item in expected)


def run_self_tests() -> None:
    weights = softmax(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
    assert np.allclose(np.sum(weights, axis=0), 1.0)

    tiny_grid = Grid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    center = bilinear_sample(np.array([[0.0, 1.0], [2.0, 3.0]]), tiny_grid, np.array(0.5), np.array(0.5))
    assert np.allclose(center, 1.5)

    grid = Grid(np.linspace(-10, 40, 126), np.linspace(-15, 15, 76))
    previous = gaussian_field(grid, np.array([[12.0, 0.0]]))
    current = gaussian_field(grid, np.array([[10.0, 0.0]]))
    aligned = warp_previous_bev(previous, grid, ego_dx=2.0)
    assert np.mean((aligned - current) ** 2) < np.mean((previous - current) ** 2) * 0.05

    expected = np.array([[8.0, -2.0], [17.0, 3.0], [25.0, -5.0]])
    peaks = find_peaks(gaussian_field(grid, expected), grid, count=3)
    assert len(peaks) == 3 and nearest_error(peaks, expected) < 0.6

    scene = build_scene(grid)
    costs = [candidate.total for candidate in scene["candidates"]]
    assert costs == sorted(costs)
    assert scene["candidates"][0].total < scene["candidates"][-1].total
    assert len(scene["current_queries"]) == 3
    print("PASS: 5/5 mathematical and closed-loop self-tests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("demo/outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return
    grid = Grid(np.linspace(-10, 40, 126), np.linspace(-15, 15, 76))
    scene = build_scene(grid)
    png = args.output_dir / "minimal_bevformer_vad.png"
    plot_result(grid, scene, png)
    print(f"wrote {png}")
    if args.save_gif:
        gif = args.output_dir / "minimal_bevformer_vad.gif"
        save_gif(grid, scene, gif, args.frames)
        print(f"wrote {gif}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
