#!/usr/bin/env python3
"""
Visualize per-face potential field on cube point cloud with gradient arrows and playback.

Features:
- Auto-detect 6 cube faces from point cloud.
- Compute potential & gradient ONLY on the face of the query point.
- Gradient arrows originate from canonical points.
- Playback: play, pause, FPS, step, next/prev.
- Color by: Heatmap (counts) or Potential (Φ).
- Interactive query point via sliders.
"""

import argparse
import glob
import pickle
import time
from pathlib import Path

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


def make_arrow_mesh(length=0.03, shaft_radius=0.0016, head_length=0.008, head_radius=0.006):
    shaft_len = max(1e-8, length - head_length)
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_len)
    shaft.apply_translation([0, 0, shaft_len / 2])
    head = trimesh.creation.cone(radius=head_radius, height=head_length)
    head.apply_translation([0, 0, shaft_len])
    return trimesh.util.concatenate([shaft, head])


def detect_cube_faces(points: np.ndarray, tol=5e-3):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    faces = {}
    for i, axis in enumerate(['x', 'y', 'z']):
        faces[f'{axis}_min'] = np.abs(points[:, i] - mins[i]) < tol
        faces[f'{axis}_max'] = np.abs(points[:, i] - maxs[i]) < tol
    return faces, mins, maxs


def find_face_of_point(p: np.ndarray, mins: np.ndarray, maxs: np.ndarray, tol=5e-3):
    for i, axis in enumerate(['x', 'y', 'z']):
        if abs(p[i] - mins[i]) < tol:
            return f'{axis}_min'
        if abs(p[i] - maxs[i]) < tol:
            return f'{axis}_max'
    candidates = []
    labels = []
    for i, axis in enumerate(['x', 'y', 'z']):
        candidates.extend([abs(p[i] - mins[i]), abs(p[i] - maxs[i])])
        labels.extend([f'{axis}_min', f'{axis}_max'])
    return labels[np.argmin(candidates)]

def per_point_face_normals(points: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    # normals in order: x_min, x_max, y_min, y_max, z_min, z_max
    normals = np.array([
        [-1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0],
        [ 0.0,  0.0,  1.0],
    ], dtype=np.float32)
    d = np.stack([
        np.abs(points[:, 0] - mins[0]),
        np.abs(points[:, 0] - maxs[0]),
        np.abs(points[:, 1] - mins[1]),
        np.abs(points[:, 1] - maxs[1]),
        np.abs(points[:, 2] - mins[2]),
        np.abs(points[:, 2] - maxs[2]),
    ], axis=1)  # (M, 6)
    idx = np.argmin(d, axis=1)  # (M,)
    return normals[idx]  # (M,3)

def project_to_tangent(vectors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    # v_tan = v - (v·n) n
    dot = np.sum(vectors * normals, axis=1, keepdims=True)
    return vectors - dot * normals

def face_normal_of_point(p: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    normals = np.array([
        [-1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0],
        [ 0.0,  0.0,  1.0],
    ], dtype=np.float32)
    d = np.array([
        abs(p[0] - mins[0]),
        abs(p[0] - maxs[0]),
        abs(p[1] - mins[1]),
        abs(p[1] - maxs[1]),
        abs(p[2] - mins[2]),
        abs(p[2] - maxs[2]),
    ], dtype=np.float32)
    return normals[int(np.argmin(d))]

def grad_at_point_full(
    q_point: np.ndarray,
    canonical_pc: np.ndarray,
    counts: np.ndarray,
    kernel_param: float = 0.05,
) -> np.ndarray:
    # q_j from sqrt decay
    q = 1.0 / np.sqrt(1.0 + counts)  # (M,)
    diffs = (q_point[None, :] - canonical_pc).astype(np.float32)  # (M,3)
    d = np.linalg.norm(diffs, axis=1) + 1e-8
    K = np.exp(-d / kernel_param)  # (M,)
    uv = diffs / d[:, None]
    contrib = -(q * K / kernel_param)[:, None] * uv
    return np.sum(contrib, axis=0)

def grad_at_point_on_face(
    q_point: np.ndarray,
    canonical_pc: np.ndarray,
    counts: np.ndarray,
    face_mask: np.ndarray,
    kernel_param: float = 0.05,
) -> np.ndarray:
    src = canonical_pc[face_mask]
    if src.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    q = 1.0 / np.sqrt(1.0 + counts[face_mask])  # (S,)
    diffs = (q_point[None, :] - src).astype(np.float32)  # (S,3)
    d = np.linalg.norm(diffs, axis=1) + 1e-8
    K = np.exp(-d / kernel_param)
    uv = diffs / d[:, None]
    contrib = -(q * K / kernel_param)[:, None] * uv
    return np.sum(contrib, axis=0)

from typing import Tuple
def compute_potential_and_gradient_on_face(
    canonical_pc: np.ndarray,  # (M, 3)
    counts: np.ndarray,        # (M,)
    face_mask: np.ndarray,     # (M,) bool
    kernel_param: float = 0.05,
    decay_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Φ and ∇Φ at each point in canonical_pc, but only using sources on the same face.
    Returns:
        Phi: (M,) — potential at each point
        grad: (M, 3) — gradient at each point
    """
    M = canonical_pc.shape[0]
    # Only use points on the face as sources
    source_pc = canonical_pc[face_mask]
    source_q = 1.0 / np.sqrt(1.0 + decay_rate * counts[face_mask])  # (S,)

    if len(source_pc) == 0:
        return np.zeros(M), np.zeros((M, 3))

    # Compute diffs from all points to sources
    diffs = canonical_pc[:, None, :] - source_pc[None, :, :]  # (M, S, 3)
    dists = np.linalg.norm(diffs, axis=2)                      # (M, S)
    K = np.exp(-dists / kernel_param)                          # (M, S)

    # Potential
    Phi = np.sum(source_q[None, :] * K, axis=1)  # (M,)

    # Gradient
    dists_safe = np.maximum(dists, 1e-8)
    unit_vecs = diffs / dists_safe[:, :, None]  # (M, S, 3)
    grad_per_source = -(source_q[None, :, None] * K[:, :, None] / kernel_param) * unit_vecs
    grad = np.sum(grad_per_source, axis=1)  # (M, 3)

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

    # --- Detect cube faces ---
    faces, mins, maxs = detect_cube_faces(canonical_pc, tol=5e-3)

    # --- Scene setup ---
    pc_handle = server.scene.add_point_cloud(
        "/canonical_pc",
        points=canonical_pc,
        colors=np.ones_like(canonical_pc) * 0.7,
        point_size=0.003,
        point_shape="rounded"
    )

    # Query point marker
    query_sphere = trimesh.creation.icosphere(radius=0.004, subdivisions=2)
    query_handle = server.scene.add_mesh_trimesh(
        "/query_point",
        mesh=query_sphere,
        position=(0, 0, 0),
        visible=True,
    )

    query_arrow_mesh = make_arrow_mesh()
    query_arrow_handle = server.scene.add_mesh_trimesh(
        "/query_grad",
        mesh=query_arrow_mesh,
        position=(0, 0, 0),
        wxyz=(1, 0, 0, 0),
        visible=False,
    )

    query_ctrl = server.scene.add_transform_controls(
        "/query_ctrl",
        scale=0.15,
        position=tuple(canonical_pc[0].tolist()),
        wxyz=(1, 0, 0, 0),
        active_axes=(True, True, True),
    )
    
    # Arrow handles (one per point)
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

    # --- UI Controls ---
    with server.gui.add_folder("Visualization"):
        gui_t = server.gui.add_slider("Timestep", 0, T - 1, 1, 0)
        gui_kp = server.gui.add_slider("Keypoint", 0, L - 1, 1, 0)
        gui_color_by = server.gui.add_dropdown("Color By", ["Heatmap", "Potential"])
        gui_show_arrows = server.gui.add_checkbox("Show Gradient Arrows", True)
        gui_arrow_scale = server.gui.add_slider("Arrow Scale", 1.0, 10.0, 1.0, 8.0)
        gui_arrow_skip = server.gui.add_slider("Arrow Skip", 1, 20, 1, 4)
        gui_point_size = server.gui.add_slider("Point Size", 0.001, 0.02, 0.001, 0.005)
        gui_use_face_only = server.gui.add_checkbox("Use Face-Only Potential", False)

    with server.gui.add_folder("Query Point"):
        gui_qx = server.gui.add_slider("X", float(mins[0] - 0.1), float(maxs[0] + 0.1), 0.001, float(canonical_pc[0, 0]))
        gui_qy = server.gui.add_slider("Y", float(mins[1] - 0.1), float(maxs[1] + 0.1), 0.001, float(canonical_pc[0, 1]))
        gui_qz = server.gui.add_slider("Z", float(mins[2] - 0.1), float(maxs[2] + 0.1), 0.001, float(canonical_pc[0, 2]))

    with server.gui.add_folder("Playback"):
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_fps = server.gui.add_slider("FPS", 1, 100.0, 1, 5.0)
        gui_step = server.gui.add_slider("Step Size", 1, 100, 1, 1)
        gui_next = server.gui.add_button("Next Frame")
        gui_prev = server.gui.add_button("Prev Frame")

    @gui_playing.on_update
    def _(_):
        gui_t.disabled = gui_playing.value

    @gui_next.on_click
    def _(_):
        gui_t.value = min(gui_t.value + gui_step.value, T - 1)

    @gui_prev.on_click
    def _(_):
        gui_t.value = max(gui_t.value - gui_step.value, 0)

    def update(_=None):
        t = gui_t.value
        kp = gui_kp.value
        counts = H[t, kp]  # (M,)

        # Update query point
        q_point = np.array([gui_qx.value, gui_qy.value, gui_qz.value], dtype=np.float32)
        query_handle.position = q_point
        query_ctrl.position = (float(q_point[0]), float(q_point[1]), float(q_point[2]))

        # Find face
        face_name = find_face_of_point(q_point, mins, maxs, tol=5e-3)
        face_mask = faces[face_name]

        # Compute potential and gradient
        if gui_use_face_only.value:
            Phi, grad = compute_potential_and_gradient_on_face(
                canonical_pc, counts, face_mask,
                kernel_param=0.05, decay_rate=1.0
            )
        else:
            # Full computation (original)
            diffs = canonical_pc[:, None, :] - canonical_pc[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            K = np.exp(-dists / 0.05)
            q = 1.0 / np.sqrt(1.0 + counts)
            Phi = np.sum(q[None, :] * K, axis=1)

            dists_safe = np.maximum(dists, 1e-8)
            unit_vecs = diffs / dists_safe[:, :, None]
            grad_per = -(q[None, :, None] * K[:, :, None] / 0.05) * unit_vecs
            grad = np.sum(grad_per, axis=1)

        normals_pts = per_point_face_normals(canonical_pc, mins, maxs)  # (M,3)
        grad_proj = project_to_tangent(grad, normals_pts)               # (M,3)

        # Choose values for coloring
        if gui_color_by.value == "Heatmap":
            vals = counts
        else:
            vals = Phi

        # Color: highlight active face, dim others
        vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        colors = colormaps["viridis"](vals_norm)[..., :3]
        # colors[~face_mask] *= 0.4  # Dim inactive faces

        pc_handle.colors = colors
        pc_handle.point_size = gui_point_size.value

        # Update arrows
        if gui_show_arrows.value:
            skip = gui_arrow_skip.value
            scale = gui_arrow_scale.value * 0.04
            for i in range(M):
                if i % skip != 0:
                    arrow_handles[i].visible = False
                    continue

                # g = grad[i]
                g = grad_proj[i]
                if np.linalg.norm(g) < 1e-8:
                    arrow_handles[i].visible = False
                else:
                    direction = g / np.linalg.norm(g)
                    q_rot = quat_from_two_vectors(np.array([0, 0, 1]), direction)
                    arrow_handles[i].position = canonical_pc[i]
                    arrow_handles[i].wxyz = q_rot
                    arrow_handles[i].scale = scale
                    arrow_handles[i].visible = True
        else:
            for handle in arrow_handles:
                handle.visible = False
                
        # Query gradient arrow (projected onto the query's face plane)
        n_q = face_normal_of_point(q_point, mins, maxs)
        if gui_use_face_only.value:
            gq = grad_at_point_on_face(q_point, canonical_pc, counts, face_mask, kernel_param=0.05)
        else:
            gq = grad_at_point_full(q_point, canonical_pc, counts, kernel_param=0.05)
        gq_tan = gq - np.dot(gq, n_q) * n_q

        if np.linalg.norm(gq_tan) > 1e-8:
            dir_q = gq_tan / (np.linalg.norm(gq_tan) + 1e-12)
            q_rot = quat_from_two_vectors(np.array([0, 0, 1]), dir_q)
            query_arrow_handle.position = q_point
            query_arrow_handle.wxyz = q_rot
            query_arrow_handle.scale = gui_arrow_scale.value * 0.06
            query_arrow_handle.visible = True
        else:
            query_arrow_handle.visible = False

    # Register all callbacks
    widgets = [
        gui_t, gui_kp, gui_color_by, gui_show_arrows, gui_arrow_scale,
        gui_arrow_skip, gui_point_size, gui_use_face_only,
        gui_qx, gui_qy, gui_qz
    ]
    for w in widgets:
        w.on_update(update)

    update()

    try:
        last_q = np.array([gui_qx.value, gui_qy.value, gui_qz.value], dtype=np.float32)
        while True:
            if gui_playing.value:
                new_t = gui_t.value + gui_step.value
                if new_t >= T:
                    new_t = 0
                gui_t.value = new_t

            qc = np.array(query_ctrl.position, dtype=np.float32)
            if np.linalg.norm(qc - last_q) > 1e-5:
                gui_qx.value = float(qc[0]); gui_qy.value = float(qc[1]); gui_qz.value = float(qc[2])
                last_q = qc

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