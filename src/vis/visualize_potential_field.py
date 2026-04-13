#!/usr/bin/env python3
"""
Visualize potential field from CuriosityRewardManager logs.

Features:
- Load canonical point cloud + per-timestep heatmaps.
- Compute potential Φ and gradient ∇Φ analytically.
- Color by: Heatmap (counts) or Potential (Φ).
- Gradient arrows originate from each point in the canonical cloud.
- Full playback controls: play, pause, FPS, step size, next/prev.
"""

import argparse
import glob
import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh
import viser
from matplotlib import colormaps


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = np.cross(a, b)
    d = np.dot(a, b)
    if d < -0.999999:
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        rot_axis = np.cross(a, axis)
        rot_axis /= np.linalg.norm(rot_axis) + 1e-12
        return np.array([0.0, rot_axis[0], rot_axis[1], rot_axis[2]])
    w = np.sqrt((1.0 + d) * 2.0) * 0.5
    xyz = c / (2.0 * w + 1e-12)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def make_arrow_mesh(length=0.05, shaft_radius=0.0015, head_length=0.015, head_radius=0.004):
    shaft_len = max(1e-8, length - head_length)
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_len)
    shaft.apply_translation([0, 0, shaft_len / 2])
    head = trimesh.creation.cone(radius=head_radius, height=head_length)
    head.apply_translation([0, 0, shaft_len])
    return trimesh.util.concatenate([shaft, head])


def compute_potential_and_gradient_at_points(
    canonical_pc: np.ndarray,  # (M, 3)
    counts: np.ndarray,        # (M,)
    kernel_param: float = 0.05,
    decay_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Φ and ∇Φ **at each point in canonical_pc**.
    Returns:
        Phi: (M,) — potential at each point
        grad: (M, 3) — gradient at each point
    """
    M = canonical_pc.shape[0]
    diffs = canonical_pc[:, None, :] - canonical_pc[None, :, :]  # (M, M, 3)
    dists = np.linalg.norm(diffs, axis=2)                        # (M, M)
    K = np.exp(-dists / kernel_param)                            # (M, M)
    q = 1.0 / np.sqrt(1.0 + decay_rate * counts)                 # (M,)

    Phi = np.sum(q[None, :] * K, axis=1)  # (M,)

    dists_safe = np.maximum(dists, 1e-8)
    unit_vecs = diffs / dists_safe[:, :, None]  # (M, M, 3)
    grad_per_point = -(q[None, :, None] * K[:, :, None] / kernel_param) * unit_vecs
    grad = np.sum(grad_per_point, axis=1)  # (M, 3)

    return Phi, grad


def main(log_dir: str, port: int = 8080, share: bool = False):
    log_path = Path(log_dir)
    server = viser.ViserServer(port=port)
    if share:
        server.request_share_url()

    # --- Load data ---
    with open(log_path / "canonical_pointcloud.pkl", "rb") as f:
        canonical_pc = pickle.load(f)["canonical_pointcloud"]  # (M, 3)
    M = canonical_pc.shape[0]

    seg_files = sorted(glob.glob(str(log_path / "segment_*.pkl")))
    heatmaps = []
    for sf in seg_files:
        with open(sf, "rb") as f:
            data = pickle.load(f)
            heatmaps.extend(data["steps"]["heatmap"])
    H = np.stack(heatmaps, axis=0)  # (T, L, M)
    T, L, _ = H.shape

    kernel_param = 0.05
    decay_rate = 1.0

    # --- Scene ---
    pc_handle = server.scene.add_point_cloud(
        "/canonical_pc",
        points=canonical_pc,
        colors=np.ones_like(canonical_pc) * 0.7,
        point_size=0.005,
    )

    # Pre-create arrow handles (one per point, but we may hide some)
    arrow_mesh = make_arrow_mesh()
    arrow_handles = []
    for i in range(M):
        handle = server.scene.add_mesh_trimesh(
            f"/arrows/arrow_{i}",
            mesh=arrow_mesh,
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
            visible=False,
        )
        arrow_handles.append(handle)

    # --- UI ---
    with server.gui.add_folder("Visualization"):
        gui_timestep = server.gui.add_slider("Timestep", 0, T - 1, 1, 0)
        gui_kp = server.gui.add_slider("Keypoint", 0, L - 1, 1, 0)
        gui_color_by = server.gui.add_dropdown("Color By", ["Heatmap", "Potential"])
        gui_show_pc = server.gui.add_checkbox("Show Canonical PC", True)
        gui_show_arrows = server.gui.add_checkbox("Show Gradient Arrows", True)
        gui_arrow_scale = server.gui.add_slider("Arrow Scale", 0.1, 5.0, 0.1, 1.0)
        gui_point_size = server.gui.add_slider("Point Size", 0.001, 0.02, 0.001, 0.005)
        gui_arrow_skip = server.gui.add_slider("Arrow Skip", 1, 20, 1, 1)  # Show 1 in N arrows

    with server.gui.add_folder("Playback"):
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_fps = server.gui.add_slider("FPS", 0.1, 30.0, 0.1, 5.0)
        gui_step = server.gui.add_slider("Step Size", 1, 10, 1, 1)
        gui_next = server.gui.add_button("Next Frame")
        gui_prev = server.gui.add_button("Prev Frame")

    @gui_playing.on_update
    def _(_):
        gui_timestep.disabled = gui_playing.value

    @gui_next.on_click
    def _(_):
        gui_timestep.value = min(gui_timestep.value + gui_step.value, T - 1)

    @gui_prev.on_click
    def _(_):
        gui_timestep.value = max(gui_timestep.value - gui_step.value, 0)

    def update(_=None):
        t = gui_timestep.value
        kp = gui_kp.value
        counts = H[t, kp]  # (M,)

        # Compute potential at points
        Phi, grad = compute_potential_and_gradient_at_points(
            canonical_pc, counts, kernel_param, decay_rate
        )  # Phi: (M,), grad: (M, 3)

        # Choose coloring
        if gui_color_by.value == "Heatmap":
            vals = counts
        else:  # "Potential"
            vals = Phi

        if gui_show_pc.value:
            vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            colors = colormaps["jet"](vals_norm)[..., :3]
            pc_handle.colors = colors
            pc_handle.point_size = gui_point_size.value
            pc_handle.visible = True
        else:
            pc_handle.visible = False

        # Update arrows: one per point (with skip for performance)
        if gui_show_arrows.value:
            skip = gui_arrow_skip.value
            scale = gui_arrow_scale.value * 0.05
            for i in range(M):
                if i % skip != 0:
                    arrow_handles[i].visible = False
                    continue

                g = grad[i]
                if np.linalg.norm(g) < 1e-8:
                    arrow_handles[i].visible = False
                else:
                    direction = g / np.linalg.norm(g)
                    q = quat_from_two_vectors(np.array([0, 0, 1]), direction)
                    arrow_handles[i].position = canonical_pc[i]
                    arrow_handles[i].wxyz = q
                    arrow_handles[i].scale = scale
                    arrow_handles[i].visible = True
        else:
            for handle in arrow_handles:
                handle.visible = False

    # Register all update triggers
    for widget in [gui_timestep, gui_kp, gui_color_by, gui_show_pc, gui_show_arrows,
                   gui_arrow_scale, gui_point_size, gui_arrow_skip]:
        widget.on_update(update)

    update()

    print(f"Visualizer running at http://localhost:{port}")
    try:
        while True:
            if gui_playing.value:
                new_t = gui_timestep.value + gui_step.value
                if new_t >= T:
                    new_t = 0
                gui_timestep.value = new_t
            time.sleep(1.0 / gui_fps.value)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    main(args.log_dir, args.port, args.share)