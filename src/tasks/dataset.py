import json
import os
import pickle
import random
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from pytorch3d.ops import sample_farthest_points


import numpy as np
import torch
import open3d as o3d


def build_axis_mask(points: np.ndarray,
                    rules: Dict[str, Tuple[str, float]]):
    """
    Create a mask for the point cloud based on the rules defined on the pcd axis (min, max)

    Args:
    ----
    points : np.ndarray, shape [N, 3]
        Point cloud coordinates, ordered as (x, y, z).
    rules : dict
        Like { "x": ("less", 0.3), "y": ("more", 0.8), "z": ("less", 0.3) }.
        - op: "less" or "more"
        - frac: float in [0, 1] (automatically clipped to this range)

    Returns:
    mask : Boolean array of the same length as points[0] (np.bool_ or torch.bool)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points should be an array/tensor of shape [N, 3].")

    # calculate min / max / span for each axis
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans_vals = maxs - mins

    axis2idx = {"x": 0, "y": 1, "z": 2}

    mask = np.ones((points.shape[0],), dtype=np.bool_)
    spans = {
        "x": (mins[0], maxs[0]),
        "y": (mins[1], maxs[1]),
        "z": (mins[2], maxs[2]),
    }

    for axis, (frac_min, frac_max) in rules.items():
        if axis not in axis2idx:
            raise ValueError(f"Unknown axis name {axis}, should be 'x'/'y'/'z'.")
        i = axis2idx[axis]

        frac_min = max(0.0, min(1.0, frac_min))
        frac_max = max(0.0, min(1.0, frac_max))
        thr_min = mins[i] + frac_min * spans_vals[i]
        thr_max = mins[i] + frac_max * spans_vals[i]
        print(f"Axis {axis}: thresholds {thr_min:.4f}, {thr_max:.4f} (min {spans[axis][0]:.4f}, max {spans[axis][1]:.4f})")

        coords = points[:, i]
        cur = (coords >= thr_min) & (coords <= thr_max)

        mask = mask & cur  # AND 叠加条件
        
    return mask


def debug_view_normals_o3d(pts: torch.Tensor, norms: torch.Tensor, mask: torch.Tensor = None, normal_length: float = 0.02):
    """
    交互式窗口：按住鼠标旋转缩放。可视化点与法向（LineSet）。
    """
    P = pts[0].detach().cpu().numpy()
    N = norms[0].detach().cpu().numpy()
    
    # mask = (P[:, 2] > 0.05).astype(np.bool_)   # (N,)
    if mask is None:
        mask = np.ones((P.shape[0],), dtype=np.bool_)
    else:
        mask = mask[0].detach().cpu().numpy()
    
    color_mask=(1.0, 0.0, 0.0),   # 红色：z > z_thresh
    color=(0.4, 0.4, 0.4),   # 灰色：z <= z_thresh

    # 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.normals = o3d.utility.Vector3dVector(N)
    
    colors = np.zeros_like(P)
    colors[mask]  = np.array(color_mask, dtype=np.float64)
    colors[~mask] = np.array(color, dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 用线段画法向（起点->终点）
    start = P
    end   = P + N * normal_length
    lines = np.stack([np.arange(P.shape[0]), np.arange(P.shape[0]) + P.shape[0]], axis=1)
    pts_lines = np.vstack([start, end]).astype(np.float64)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts_lines)
    line_set.lines  = o3d.utility.Vector2iVector(lines)
    
    line_colors = np.zeros((P.shape[0], 3), dtype=np.float64)
    line_colors[mask]  = np.array(color_mask, dtype=np.float64)
    line_colors[~mask] = np.array(color, dtype=np.float64)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, line_set, axes])


class FunctionalGraspingDataset(object):
    def __init__(self):
        pass

    def parse_pose(self, pose: Dict[str, Any]) -> torch.Tensor:
        position = torch.tensor([pose["position"][key] for key in ["x", "y", "z"]]).float()
        quaternion = torch.tensor([pose["quaternion"][key] for key in ["x", "y", "z", "w"]]).float()
        return torch.cat([position, quaternion])


class OakInkDataset(FunctionalGraspingDataset):
    dataset_name: str = "oakink"
    data: Dict[str, Any]
    # fmt: off
    dof_names: List[str] = [
        "rh_WRJ2", "rh_WRJ1",
        'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1',
        'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
        'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1',
        'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
        'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1',
    ]
    # fmt: on

    def __init__(
        self,
        dataset_dir: str,
        base_prim: str = "palm",
        device: Optional[Union[str, torch.device]] = None,
        pcl_num: int = 1024,
        num_object: Optional[int] = -1,
        num_object_per_category: Optional[int] = -1,
        queries: Optional[Dict[str, Any]] = None,
        *,
        original_categories_statistics_path: str = "data/ori.json",
        metainfo_path: str = "data/oakink_metainfo.csv",
        skipcode_path: str = "data/oakink_skipcode.csv",
        pretrained_embedding: bool = False,
        precomputed_sdf: bool = False,
        pose_level_sampling: bool = False,
    ):
        super().__init__()

        self.data = {}
        self.label_paths = []
        self.device = device
        self.pcl_num = pcl_num
        self.object_num = num_object
        self.max_per_cat = num_object_per_category
        self.object_geo_level = None  # TODO: remove those two lines or set from the queries
        self.object_scale = None
        self.pose_level_sampling = pose_level_sampling
        self.original_category_statistics = json.load(open(original_categories_statistics_path, "r"))

        category_dict = np.load("./data/category.npy", allow_pickle=True).tolist()
        num_full_categories = len(set(category_dict.values()))
        category_matrix = torch.eye(num_full_categories)

        metainfo: pd.DataFrame = pd.read_csv(metainfo_path)

        # filter invalid codes
        if os.path.exists(skipcode_path):
            skipcode: pd.DataFrame = pd.read_csv(skipcode_path)
            invalid_codes = skipcode.query("ret_code != 0")["code"].tolist()
            metainfo = metainfo[~metainfo["code"].isin(invalid_codes)]
        else:
            warnings.warn("Skipcode file not found, skipping invalid code filtering")

        # filter grasping poses
        for key, candidates in queries.items():
            if isinstance(candidates, list) or isinstance(candidates, ListConfig):
                metainfo = metainfo[metainfo[key].isin(candidates)]
            elif candidates is not None:
                metainfo = metainfo[metainfo[key] == candidates]

        if num_object_per_category != -1:
            instances = metainfo[["category", "code"]].drop_duplicates()
            instances = instances.groupby("category")["code"].head(num_object_per_category)
            metainfo = metainfo[metainfo["code"].isin(instances.values)]

        if num_object != -1:
            instances = metainfo["code"].drop_duplicates()
            instances = instances.sample(min(num_object, instances.shape[0]), random_state=42)
            metainfo = metainfo[metainfo["code"].isin(instances.values)]

        if "category" in queries:
            if isinstance(queries["category"], list) or isinstance(queries["category"], ListConfig):
                self.object_cat = "__".join(queries["category"])
            else:
                self.object_cat = queries["category"]
        else:
            self.object_cat = "all"

        for index, row in metainfo.iterrows():
            code, filepath = row["code"], row["filepath"]

            with open(filepath, "r") as f:
                data = json.load(f)

            if code not in self.data:
                self.data[code] = []
            self.data[code].append((data, row))
            self.label_paths.append(filepath)

        self.num_samples = sum([len(self.data[code]) for code in self.data])
        self.num_objects = len(self.data)
        self.categories = []
        self.object_codes = []
        self.grasp_names = []
        self.code_names = []
        self.clutser_ids = []
        self.object_categories = []
        self.category_object_codes = {}
        self.indices = torch.zeros(self.num_objects + 1, dtype=torch.long, device=self.device)
        self.inverse_indices = torch.zeros(self.num_samples, dtype=torch.long, device=self.device)
        self._joints = torch.zeros(self.num_samples, len(self.dof_names), device=self.device)
        self._object_poses = torch.zeros(self.num_samples, 7, device=self.device)
        self._pointclouds = torch.zeros(self.num_objects, 4096, 3, device=self.device)
        self._category_matrix = torch.zeros(self.num_objects, num_full_categories, device=self.device)
        self._raw_samples = []

        if precomputed_sdf:
            self._sdf_fields = torch.zeros(self.num_objects, 200, 200, 200, device="cpu")

        print(f"Total number of samples: {self.num_samples}")
        print(f"Total number of objects: {self.num_objects}")

        index = 0
        for cur, code in enumerate(self.data):
            self.indices[cur] = index
            for i, (sample, meta) in enumerate(self.data[code]):
                self._raw_samples.append(sample)
                self.code_names.append(meta["code"])
                self.grasp_names.append(meta["pose"])
                self.clutser_ids.append(int(meta["cluster"]))
                self._joints[index] = torch.tensor(
                    [sample["joints"][name] for name in self.dof_names], device=self.device
                )
                self._object_poses[index] = self.parse_pose(sample["object_pose_wrt_palm"])
                self.data[code][i] = index

                if meta["category"] not in self.categories:
                    self.categories.append(meta["category"])
                    self.category_object_codes[meta["category"]] = []
                if meta["code"] not in self.object_codes:
                    self.object_codes.append(meta["code"])
                    self.object_categories.append(meta["category"])
                    self.category_object_codes[meta["category"]].append(meta["code"])
                self.inverse_indices[index] = cur
                index += 1
        self.indices[-1] = index
        self.categories = list(self.categories)
        self.object_codes = list(self.object_codes)
        self.object_categories = list(self.object_categories)
        self.grasp_names = np.array(self.grasp_names)
        self.code_names = np.array(self.code_names)
        self.clutser_ids = np.array(self.clutser_ids)

        obj_pcl_buf_all_path = "data/pcl_buffer_4096_all.pkl"
        with open(obj_pcl_buf_all_path, "rb") as f:
            self.obj_pcl_buf_all = pickle.load(f)

        for i, code in enumerate(self.object_codes):
            self._pointclouds[i] = torch.from_numpy(self.obj_pcl_buf_all[code]).float().to(self.device)
            self._category_matrix[i] = category_matrix[category_dict[code]]
            if precomputed_sdf:
                self._sdf_fields[i] = torch.from_numpy(np.load("data/precomputed_sdf/" + code + ".npy")).float()

        if pretrained_embedding:
            self.embeddings = pd.read_csv("pointnet_pretrain_embeddings.csv")
            self.embeddings.set_index("code", inplace=True)
        else:
            self.embeddings = None

        # support pose-level sampling
        if self.pose_level_sampling:
            self.manipulated_codes = []
            for code in self.object_codes:
                self.manipulated_codes.extend([code] * len(self.data[code]))
            random.seed(42)
            random.shuffle(self.manipulated_codes)
            self._env_counter = {}
        else:
            self.manipulated_codes = deepcopy(self.object_codes)

        print(">>> Oakink Dataset Initialized")

    def resample(self, num_samples: int) -> List[str]:
        """Resample the current dataset to match the original distribution.

        Args:
            num_samples (int): Number of objects to sample

        Returns:
            List[str]: List of object codes
        """
        print("Resampling dataset...")
        print("Categories: ", self.categories)

        names = self.categories.copy()
        counts = [self.original_category_statistics[name] for name in names]
        probs = np.array(counts) / np.sum(counts)

        categories = np.random.choice(names, size=num_samples, p=probs)

        codes = []
        for category in categories:
            codes.append(random.choice(self.category_object_codes[category]))

        return codes

    def get_boundingbox(self, pointclouds: torch.Tensor) -> torch.Tensor:
        """Computes the bounding box of a point cloud.

        Args:
            pointclouds (torch.Tensor): Point cloud tensor of shape (..., N, 3)

        Returns:
            torch.Tensor: Bounding box tensor of shape (..., 6)
        """
        corner_max = torch.max(pointclouds, dim=-2)[0]
        corner_min = torch.min(pointclouds, dim=-2)[0]
        return torch.cat((corner_max, corner_min), dim=-1).to(self.device)

    def sample(self, object_indices: torch.LongTensor) -> Dict[str, Any]:
        if self.pose_level_sampling:
            indices = object_indices
            object_indices = self.inverse_indices[indices]

        else:
            assert object_indices.dim() == 1, "Object indices must be a 1D tensor"
            assert object_indices.dtype == torch.long, "Object indices must be a 1D tensor of longs"
            assert object_indices.max() < self.num_objects and object_indices.min() >= 0, "Object indices out of range"

            lower = self.indices[object_indices].float()
            upper = self.indices[object_indices + 1].float()
            indices = lower + (upper - lower) * torch.rand_like(object_indices, dtype=torch.float, device=self.device)
            indices = torch.floor(indices).long()
            indices = torch.min(indices, self.indices[-1] - 1)

        pointclouds = self._pointclouds[object_indices]
        pointclouds = sample_farthest_points(pointclouds, K=self.pcl_num)[0]
        boundingbox = self.get_boundingbox(pointclouds)
        category_onehot = self._category_matrix[object_indices]
        grasp = self.grasp_names[indices.detach().cpu().numpy()]
        code = self.code_names[indices.detach().cpu().numpy()]
        clutser_ids = self.clutser_ids[indices.detach().cpu().numpy()]

        return {
            "joints": self._joints[indices],
            "pose": self._object_poses[indices],
            "pointcloud": pointclouds,
            "index": indices,
            "object_index": object_indices,
            "bbox": boundingbox,
            "category_onehot": category_onehot,
            "grasp": grasp,
            "code": code,
            "cluster": clutser_ids,
        }

    def get_object_index(self, object_code: str) -> int:
        if self.pose_level_sampling:
            curr = self._env_counter.get(object_code, 0)
            self._env_counter[object_code] = curr + 1
            index = self.object_codes.index(object_code)
            count = self.indices[index + 1] - self.indices[index]
            return self.indices[index] + (curr % count)
        else:
            return self.object_codes.index(object_code)


def compute_implicit_sdf(
    vertices: torch.Tensor, faces: torch.Tensor, grid_size: int = 200, space: float = 0.5
) -> torch.Tensor:
    """Computes the Signed Distance Field (SDF) for a given 3D mesh represented by vertices and faces.

    The default grid size and space size are chosen to match this paper:
        https://arxiv.org/abs/2211.10957

    Args:
        vertices (torch.Tensor): Tensor of shape (N, 3) representing the 3D coordinates of the mesh vertices.
        faces (torch.Tensor): Tensor of shape (M, 3) representing the indices of the vertices that form each face of the mesh.
        grid_size (int, optional): The number of grid points along each axis. Defaults to 200.
        space (float, optional): The size of the space in which the mesh is defined. Defaults to 0.5.

    Returns:
        sdf_field (torch.Tensor): Tensor of shape (grid_size, grid_size, grid_size) representing the SDF field.
            - Positive values indicate that the point is outside the mesh.
            - Negative values indicate that the point is inside the mesh.
    """
    import kaolin as kal

    assert vertices.dim() == 2 and vertices.shape[1] == 3, "vertices must be of shape (N, 3)"
    assert faces.dim() == 2 and faces.shape[1] == 3, "faces must be of shape (M, 3)"
    assert vertices.device == faces.device, "vertices and faces must be on the same device"
    device = vertices.device

    unit = 2 * space / grid_size
    axis = torch.linspace(-space + unit / 2, space - unit / 2, grid_size, device=device)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid_points = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    grid_points = grid_points.unsqueeze(0)
    vertices = vertices.unsqueeze(0)

    face_vertices = kal.ops.mesh.index_vertices_by_faces(vertices, faces)

    squared_distance, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(grid_points, face_vertices)
    distance = torch.sqrt(squared_distance)

    sign = kal.ops.mesh.check_sign(vertices, faces, grid_points)

    sdf = distance * torch.where(sign, -1.0, 1.0)
    sdf_field = sdf.reshape(grid_size, grid_size, grid_size)
    return sdf_field


def point_to_mesh_distance(
    points: torch.Tensor, sdf: torch.Tensor, indices: torch.LongTensor, space: float = 0.5
) -> torch.Tensor:
    """Calculates the distance from each point to the mesh represented by the signed distance field (SDF).

    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) representing the coordinates of N points.
        sdf (torch.Tensor): Tensor of shape (M, D, D, D) representing the signed distance field.
        indices (torch.LongTensor): Tensor of shape (B,) representing the indices of the mesh elements corresponding to each point.
        space (float, optional): The spacing between grid cells in the SDF. Defaults to 0.5.

    Returns:
        torch.Tensor: Tensor of shape (B, N) representing the distance from each point to the mesh.
    """
    assert points.shape[-1] == 3
    assert points.dim() == 3 or points.dim() == 2, "points must be of shape (B, N, 3) or (N, 3)"
    assert sdf.dim() == 4 and sdf.shape[1] == sdf.shape[2] == sdf.shape[3], "sdf must be of shape (M, D, D, D)"
    assert indices.dim() == 1 and indices.dtype == torch.long, "indices must be a 1D tensor of longs"
    assert indices.shape[0] == points.shape[0], "indices must have the same batch size as points"
    assert indices.min() >= 0 and indices.max() < sdf.shape[0], "indices out of range"

    device, sdf_device = points.device, sdf.device
    points, indices = points.to(sdf_device), indices.to(sdf_device)

    ndim = points.dim()
    batch_size = points.shape[0]
    grid_size = sdf.shape[-1]

    points = points.view(batch_size, -1, 3)
    num_points = points.shape[1]

    coords = (points + space) / (2 * space) * grid_size
    coords = coords.floor().clamp(0, grid_size - 1).long()

    # Get the SDF values at the corresponding coordinates
    indices = indices.unsqueeze(1).expand(-1, num_points)
    coords_x, coords_y, coords_z = coords.unbind(dim=-1)
    sdf_values = sdf[indices, coords_x, coords_y, coords_z]

    return (sdf_values if ndim == 3 else sdf_values.squeeze(1)).to(device)


class ObjectDataset(FunctionalGraspingDataset):
    """Generic dataset for object-centric tasks."""
    object_meta = {
        "block": {
            "object_type": "mesh_file",
            "file_path": "./assets/urdf/objects/meshes/cube_multicolor.obj",
            # "scale": [0.05, 0.05, 0.05],   # per-axis scale
            "scale": [0.065, 0.065, 0.065],   # per-axis scale
        },
        "egg": {
            "object_type": "predefined_geometry",
            "geometry_type": "ellipsoid",
            "size": [0.03, 0.03, 0.04],    # axes a,b,c
        },
        "pen": {
            "object_type": "predefined_geometry",
            "geometry_type": "capsule",
            "size": [0.008, 0.2],          # [radius, length]  (length为中部圆柱段长度)
        },
    }

    def __init__(
        self,
        object: str,
        device: "Optional[Union[str, torch.device]]" = None,
        pcl_num: int = 1024,
    ):
        super().__init__()
        self.object = object
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.pcl_num = pcl_num
        
        self._pointclouds = self._create_object_pointclouds()

    # === 主函数：按对象类型创建点云 ===
    def _create_object_pointclouds(self) -> torch.Tensor:
        meta = self.object_meta[self.object]

        if meta["object_type"] == "mesh_file":
            import trimesh
            file_path = meta["file_path"]
            mesh = trimesh.load(file_path, force='mesh')
            scale = meta.get("scale", None)
            # 支持各向异性缩放
            if scale is not None:
                if isinstance(scale, (list, tuple, np.ndarray)) and len(scale) == 3:
                    S = np.eye(4, dtype=np.float64)
                    S[0, 0], S[1, 1], S[2, 2] = scale
                    mesh.apply_transform(S)
                else:
                    mesh.apply_scale(float(scale))

            # 采样 n 个点（单个对象），再按 num_objects 复制
            pts, _ = trimesh.sample.sample_surface(mesh, self.pcl_num)
            pts = torch.as_tensor(pts, dtype=torch.float32, device=self.device)  # (n,3)
            pts = pts.unsqueeze(0)  # (1,n,3)
            return pts

        elif meta["object_type"] == "predefined_geometry":
            gtype = meta["geometry_type"].lower()
            if gtype == "ellipsoid":
                a, b, c = meta["size"]  # 轴长
                pts = self._sample_ellipsoid_surface(self.pcl_num, a, b, c)                 # (n,3)
            elif gtype == "capsule":
                r, L = meta["size"]  # 半径r，中部圆柱长度L
                pts = self._sample_capsule_surface(self.pcl_num, r, L)                      # (n,3)
            else:
                raise ValueError(f"Unknown geometry_type: {gtype}")

            pts = torch.as_tensor(pts, dtype=torch.float32, device=self.device)
            pts = pts.unsqueeze(0)                                           # (1,n,3)
            return pts

        else:
            raise ValueError(f"Unknown object_type: {meta['object_type']}")

    # === 椭球表面采样（近似均匀）：从单位球面均匀采样后各向缩放 ===
    @staticmethod
    def _sample_ellipsoid_surface(n: int, a: float, b: float, c: float) -> np.ndarray:
        # 在单位球面均匀采样：z ~ U(-1,1), phi ~ U(0,2π)
        z = np.random.uniform(-1.0, 1.0, size=(n,))
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=(n,))
        r_xy = np.sqrt(1.0 - z**2)
        x = r_xy * np.cos(phi)
        y = r_xy * np.sin(phi)
        # 各向缩放到椭球
        pts = np.stack([a * x, b * y, c * z], axis=1)  # (n,3)
        return pts

    # === 胶囊体(z轴对齐)表面采样：按表面积比例分配到圆柱与两端半球 ===
    @staticmethod
    def _sample_capsule_surface(n: int, r: float, L: float) -> np.ndarray:
        # 总面积：圆柱侧面积 2π r L；两半球面积 4π r^2
        area_cyl = 2.0 * np.pi * r * L
        area_caps = 4.0 * np.pi * r * r
        area_total = area_cyl + area_caps
        n_cyl = max(0, int(round(n * area_cyl / area_total)))
        n_caps = n - n_cyl
        # 两个半球各一半
        n_cap_top = n_caps // 2
        n_cap_bot = n_caps - n_cap_top

        pts_list = []

        # 圆柱侧面：theta ~ U[0,2π], z ~ U[-L/2, L/2]
        if n_cyl > 0:
            theta = np.random.uniform(0.0, 2.0 * np.pi, size=(n_cyl,))
            z = np.random.uniform(-L / 2.0, L / 2.0, size=(n_cyl,))
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts_list.append(np.stack([x, y, z], axis=1))

        # 顶部半球 (z>=0)：在单位球面均匀采样，再筛选 z>=0，缩放到半径 r，并平移到 z=+L/2
        if n_cap_top > 0:
            pts_top = ObjectDataset._sample_hemisphere_surface(n_cap_top, r, upper=True)
            pts_top[:, 2] += L / 2.0
            pts_list.append(pts_top)

        # 底部半球 (z<=0)：同理，平移到 z=-L/2
        if n_cap_bot > 0:
            pts_bot = ObjectDataset._sample_hemisphere_surface(n_cap_bot, r, upper=False)
            pts_bot[:, 2] -= L / 2.0
            pts_list.append(pts_bot)

        if len(pts_list) == 0:
            return np.zeros((n, 3), dtype=np.float32)
        pts = np.concatenate(pts_list, axis=0)
        # 若因为四舍五入点数略偏，裁剪/补齐
        if pts.shape[0] > n:
            pts = pts[:n]
        elif pts.shape[0] < n:
            need = n - pts.shape[0]
            pts = np.concatenate([pts, pts[:need]], axis=0)
        return pts.astype(np.float32)

    @staticmethod
    def _sample_hemisphere_surface(n: int, r: float, upper: bool = True) -> np.ndarray:
        # 在单位球面均匀采样：z ~ U(-1,1), phi ~ U(0,2π)，然后筛选上/下半球
        # 为避免拒绝采样开销，直接用 z ~ U(0,1) 或 U(-1,0)
        if upper:
            z = np.random.uniform(0.0, 1.0, size=(n,))
        else:
            z = np.random.uniform(-1.0, 0.0, size=(n,))
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=(n,))
        r_xy = np.sqrt(1.0 - z**2)
        x = r_xy * np.cos(phi)
        y = r_xy * np.sin(phi)
        return (r * np.stack([x, y, z], axis=1)).astype(np.float32)

class BoxGridDataset(FunctionalGraspingDataset):
    """Simple dataset for box grid singulation task."""
    
    dof_names: List[str] = [
        "joint0.0", "joint1.0", "joint2.0", "joint3.0",
        "joint4.0", "joint5.0", "joint6.0", "joint7.0",
        "joint8.0", "joint9.0", "joint10.0", "joint11.0",
        "joint12.0", "joint13.0", "joint14.0", "joint15.0",
    ]
    
    def __init__(
        self,
        grid_rows: int = 1,
        grid_cols: int = 5,
        grid_layers: int = 1,
        box_width: float = 0.04,
        box_depth: float = 0.16,
        box_height: float = 0.24,
        device: Optional[Union[str, torch.device]] = None,
        pcl_num: int = 512,
    ):
        super().__init__()
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_layers = grid_layers
        self.box_width = box_width
        self.box_depth = box_depth
        self.box_height = box_height
        self.device = device
        self.pcl_num = pcl_num
        
        self.num_objects = grid_rows * grid_cols * grid_layers
        self.object_codes = [f"box_{i}" for i in range(self.num_objects)]
        self.manipulated_codes = deepcopy(self.object_codes)
        

        self._category_matrix = torch.ones(self.num_objects, 1, device=self.device)
        self._pointclouds = self._create_box_pointclouds()
        self._pointcloud_normals = self._create_box_normals()
        
        self.object_cat = "box"
        self.max_per_cat = -1
        self.object_geo_level = None
        self.object_scale = None
        self.label_paths = ["box_grid_singulation"]
        
        # self._sdf_fields = torch.zeros(self.num_objects, 200, 200, 200, device=device)
        
        print(f">>> BoxGridDataset initialized with {self.num_objects} boxes")
    
    # def _create_box_pointclouds(self) -> torch.Tensor:
    #     """Create pointclouds for box objects by sampling points on the 6 faces."""
    #     # Sample random points uniformly across all objects
    #     points = torch.rand(self.num_objects, self.pcl_num, 3, device=self.device)
        
    #     # Scale to box dimensions and center at origin
    #     points[:, :, 0] = points[:, :, 0] * self.box_width - self.box_width / 2    # x
    #     points[:, :, 1] = points[:, :, 1] * self.box_depth - self.box_depth / 2    # y  
    #     points[:, :, 2] = points[:, :, 2] * self.box_height - self.box_height / 2  # z
        
    #     n_per_face = self.pcl_num // 6
        
    #     # z boundaries
    #     points[:, :n_per_face, 2] = self.box_height / 2
    #     points[:, n_per_face:2*n_per_face, 2] = -self.box_height / 2
        
    #     # y boundaries
    #     points[:, 2*n_per_face:3*n_per_face, 1] = self.box_depth / 2
    #     points[:, 3*n_per_face:4*n_per_face, 1] = -self.box_depth / 2
        
    #     # x boundaries
    #     points[:, 4*n_per_face:5*n_per_face, 0] = self.box_width / 2
    #     points[:, 5*n_per_face:, 0] = -self.box_width / 2
        
    #     return points
    # def _create_box_normals(self) -> torch.Tensor:

    #     normals = torch.zeros(self.num_objects, self.pcl_num, 3, device=self.device)
    #     n_per_face = self.pcl_num // 6
    #     # +Z, -Z
    #     normals[:, :n_per_face, 2] = 1.0
    #     normals[:, n_per_face:2*n_per_face, 2] = -1.0
    #     # +Y, -Y
    #     normals[:, 2*n_per_face:3*n_per_face, 1] = 1.0
    #     normals[:, 3*n_per_face:4*n_per_face, 1] = -1.0
    #     # +X, -X
    #     normals[:, 4*n_per_face:5*n_per_face, 0] = 1.0
    #     normals[:, 5*n_per_face:, 0] = -1.0
    #     return normals

    def _compute_box_face_counts(self) -> list:
        """
        Returns counts for faces in order: +Z, -Z, +Y, -Y, +X, -X,
        proportional to face areas, summing to self.pcl_num.
        """
        import math
        W = float(self.box_width)
        D = float(self.box_depth)
        H = float(self.box_height)
        areas = torch.tensor([W*D, W*D, W*H, W*H, D*H, D*H], dtype=torch.float32)
        weights = areas / areas.sum()
        raw = weights * float(self.pcl_num)
        base = torch.floor(raw)
        remainder = int(self.pcl_num - int(base.sum().item()))
        counts = base.to(torch.long)
        if remainder > 0:
            frac = (raw - base)
            _, idx = torch.sort(frac, descending=True)
            counts[idx[:remainder]] += 1
        return counts.tolist()

    def _create_box_pointclouds(self) -> torch.Tensor:
        """Create pointclouds for box objects with face counts proportional to area."""
        counts = self._compute_box_face_counts()  # [n+Z, n-Z, n+Y, n-Y, n+X, n-X]
        # cache counts for normals
        self._box_face_counts = torch.as_tensor(counts, device=self.device, dtype=torch.long)

        N = self.num_objects
        P = self.pcl_num
        W = float(self.box_width)
        D = float(self.box_depth)
        H = float(self.box_height)

        pts = torch.empty(N, P, 3, device=self.device, dtype=torch.float32)
        s = 0

        # +Z: z = +H/2, (x,y) ∈ [-W/2,W/2]×[-D/2,D/2]
        n = int(counts[0])
        if n > 0:
            x = (torch.rand(N, n, device=self.device) - 0.5) * W
            y = (torch.rand(N, n, device=self.device) - 0.5) * D
            z = torch.full((N, n), H/2.0, device=self.device)
            pts[:, s:s+n, 0] = x; pts[:, s:s+n, 1] = y; pts[:, s:s+n, 2] = z
            s += n

        # -Z: z = -H/2
        n = int(counts[1])
        if n > 0:
            x = (torch.rand(N, n, device=self.device) - 0.5) * W
            y = (torch.rand(N, n, device=self.device) - 0.5) * D
            z = torch.full((N, n), -H/2.0, device=self.device)
            pts[:, s:s+n, 0] = x; pts[:, s:s+n, 1] = y; pts[:, s:s+n, 2] = z
            s += n

        # +Y: y = +D/2, (x,z) ∈ [-W/2,W/2]×[-H/2,H/2]
        n = int(counts[2])
        if n > 0:
            x = (torch.rand(N, n, device=self.device) - 0.5) * W
            z = (torch.rand(N, n, device=self.device) - 0.5) * H
            y = torch.full((N, n), D/2.0, device=self.device)
            pts[:, s:s+n, 0] = x; pts[:, s:s+n, 1] = y; pts[:, s:s+n, 2] = z
            s += n

        # -Y: y = -D/2
        n = int(counts[3])
        if n > 0:
            x = (torch.rand(N, n, device=self.device) - 0.5) * W
            z = (torch.rand(N, n, device=self.device) - 0.5) * H
            y = torch.full((N, n), -D/2.0, device=self.device)
            pts[:, s:s+n, 0] = x; pts[:, s:s+n, 1] = y; pts[:, s:s+n, 2] = z
            s += n

        # +X: x = +W/2, (y,z) ∈ [-D/2,D/2]×[-H/2,H/2]
        n = int(counts[4])
        if n > 0:
            y = (torch.rand(N, n, device=self.device) - 0.5) * D
            z = (torch.rand(N, n, device=self.device) - 0.5) * H
            x = torch.full((N, n), W/2.0, device=self.device)
            pts[:, s:s+n, 0] = x; pts[:, s:s+n, 1] = y; pts[:, s:s+n, 2] = z
            s += n

        # -X: x = -W/2
        n = int(counts[5])
        if n > 0:
            y = (torch.rand(N, n, device=self.device) - 0.5) * D
            z = (torch.rand(N, n, device=self.device) - 0.5) * H
            x = torch.full((N, n), -W/2.0, device=self.device)
            pts[:, s:s+n, 0] = x; pts[:, s:s+n, 1] = y; pts[:, s:s+n, 2] = z
            s += n

        return pts

    def _create_box_normals(self) -> torch.Tensor:
        """
        Create normals aligned with _create_box_pointclouds ordering.
        Uses cached self._box_face_counts and emits (N, P, 3).
        """
        assert hasattr(self, "_box_face_counts"), "Call _create_box_pointclouds before normals."
        counts = [int(c) for c in self._box_face_counts.tolist()]
        N = self.num_objects
        P = self.pcl_num
        normals = torch.zeros(N, P, 3, device=self.device, dtype=torch.float32)

        s = 0
        n = counts[0]
        if n > 0: normals[:, s:s+n, 2] =  1.0; s += n  # +Z
        n = counts[1]
        if n > 0: normals[:, s:s+n, 2] = -1.0; s += n  # -Z
        n = counts[2]
        if n > 0: normals[:, s:s+n, 1] =  1.0; s += n  # +Y
        n = counts[3]
        if n > 0: normals[:, s:s+n, 1] = -1.0; s += n  # -Y
        n = counts[4]
        if n > 0: normals[:, s:s+n, 0] =  1.0; s += n  # +X
        n = counts[5]
        if n > 0: normals[:, s:s+n, 0] = -1.0; s += n  # -X

        return normals
    def get_boundingbox(self, pointclouds: torch.Tensor) -> torch.Tensor:
        """Computes the bounding box of a point cloud.

        Args:
            pointclouds (torch.Tensor): Point cloud tensor of shape (..., N, 3)

        Returns:
            torch.Tensor: Bounding box tensor of shape (..., 6)
        """
        corner_max = torch.max(pointclouds, dim=-2)[0]
        corner_min = torch.min(pointclouds, dim=-2)[0]
        return torch.cat((corner_max, corner_min), dim=-1).to(self.device)
    
    def sample(self, object_indices: torch.LongTensor) -> Dict[str, Any]:
        """Sample data for given object indices."""
        assert object_indices.dim() == 1, "Object indices must be a 1D tensor"
        assert object_indices.dtype == torch.long, "Object indices must be a 1D tensor of longs"
        assert object_indices.max() < self.num_objects and object_indices.min() >= 0, "Object indices out of range"
        
        pointclouds = self._pointclouds[object_indices]
        if pointclouds.shape[1] > self.pcl_num:
            pointclouds = sample_farthest_points(pointclouds, K=self.pcl_num)[0]
        
        boundingbox = self.get_boundingbox(pointclouds)
        category_onehot = self._category_matrix[object_indices]
        
        batch_size = object_indices.shape[0]
        dummy_joints = torch.zeros(batch_size, 16, device=self.device) 
        dummy_poses = torch.zeros(batch_size, 7, device=self.device)
        dummy_poses[:, 6] = 1.0
        
        return {
            "joints": dummy_joints,
            "pose": dummy_poses,
            "pointcloud": pointclouds,
            "index": object_indices,
            "object_index": object_indices,
            "bbox": boundingbox,
            "category_onehot": category_onehot,
            "grasp": ["box_grasp"] * batch_size,
            "code": [self.object_codes[i] for i in object_indices.cpu().numpy()],
            "cluster": np.zeros(batch_size, dtype=np.int32),
        }
    
    def get_object_index(self, object_code: str) -> int:
        """Get the index of an object by its code."""
        return self.object_codes.index(object_code)
    
    def resample(self, num_samples: int) -> List[str]:
        """Resample objects for the singulation task."""
        # For singulation task, we just cycle through the available boxes
        codes = []
        for i in range(num_samples):
            codes.append(self.object_codes[i % self.num_objects])
        return codes


class TableTopDataset(FunctionalGraspingDataset):
    """Dataset for TableTop task supporting:
    - General URDF meshes: parse a URDF and sample a surface point cloud.
    - IsaacGym cube: generate a box surface point cloud given W,L,H.

    Conventions follow other datasets in this module for compatibility.
    """

    # Maintain a simple hand-DOF list for compatibility (unused for TableTop)
    dof_names: List[str] = [
        "joint0.0", "joint1.0", "joint2.0", "joint3.0",
        "joint4.0", "joint5.0", "joint6.0", "joint7.0",
        "joint8.0", "joint9.0", "joint10.0", "joint11.0",
        "joint12.0", "joint13.0", "joint14.0", "joint15.0",
    ]

    def __init__(
        self,
        mode: str,
        *,
        device: Optional[Union[torch.device, str]] = None,
        pcl_num: int = 1024,
        # URDF mode
        urdf_rel_path: Optional[str] = None,
        asset_root: Optional[str] = None,
        # Cube mode
        cube_width: Optional[float] = None,
        cube_length: Optional[float] = None,
        cube_height: Optional[float] = None,
        object_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.pcl_num = int(pcl_num)

        self.object_codes = ["edge_object"]
        self.manipulated_codes = deepcopy(self.object_codes)
        self._category_matrix = torch.ones(1, 1, device=self.device)
        self.object_cat = "edge"
        self.max_per_cat = -1
        self.object_geo_level = None
        self.object_scale = object_scale
        self.label_paths = ["table_top"]
        self._sdf_fields = torch.zeros(1, 200, 200, 200, device=self.device)

        if self.mode == "urdf":
            assert urdf_rel_path is not None and asset_root is not None, "URDF mode requires urdf_rel_path and asset_root"
            self._urdf_asset_root = asset_root
            self._urdf_rel_path = urdf_rel_path
            self._pointclouds = self._create_pointcloud_from_urdf(asset_root=asset_root, urdf_rel_path=urdf_rel_path)
            self._pointcloud_normals = self._create_urdf_normals()
            self._pointcloud_mask = self._create_pointcloud_mask()

            debug_view_normals_o3d(self._pointclouds, self._pointcloud_normals, self._pointcloud_mask)

        elif self.mode == "cube":
            assert cube_width is not None and cube_length is not None and cube_height is not None, "Cube mode requires cube_width, cube_length, cube_height"
            self._pointclouds = self._create_pointcloud_from_cube(cube_width, cube_length, cube_height)
            self._pointcloud_normals = self._create_cube_normals()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _create_pointcloud_from_cube(self, W: float, L: float, H: float) -> torch.Tensor:
        n = self.pcl_num
        points = torch.rand(1, n, 3, device=self.device)
        points[:, :, 0] = points[:, :, 0] * W - W / 2
        points[:, :, 1] = points[:, :, 1] * L - L / 2
        points[:, :, 2] = points[:, :, 2] * H - H / 2

        n_per_face = max(1, n // 6)
        points[:, :n_per_face, 2] = H / 2
        points[:, n_per_face:2 * n_per_face, 2] = -H / 2
        points[:, 2 * n_per_face:3 * n_per_face, 1] = L / 2
        points[:, 3 * n_per_face:4 * n_per_face, 1] = -L / 2
        points[:, 4 * n_per_face:5 * n_per_face, 0] = W / 2
        points[:, 5 * n_per_face:, 0] = -W / 2
        return points

    def _create_pointcloud_from_urdf(self, *, asset_root: str, urdf_rel_path: str) -> torch.Tensor:
        import trimesh
        import xml.etree.ElementTree as ET

        urdf_path = os.path.join(asset_root, urdf_rel_path)
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        base_dir = os.path.dirname(urdf_path)

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        meshes: List[trimesh.Trimesh] = []

        def _parse_origin(origin_elem) -> np.ndarray:
            # default identity
            T = np.eye(4, dtype=np.float64)
            if origin_elem is None:
                return T
            xyz = origin_elem.attrib.get("xyz", None)
            rpy = origin_elem.attrib.get("rpy", None)
            if xyz is not None:
                x, y, z = [float(v) for v in xyz.strip().split()]
                T[:3, 3] = [x, y, z]
            if rpy is not None:
                rr, pp, yy = [float(v) for v in rpy.strip().split()]
                # roll-pitch-yaw to rotation (XYZ extrinsic == RPY intrinsic)
                from math import cos, sin
                Rx = np.array([[1, 0, 0, 0], [0, cos(rr), -sin(rr), 0], [0, sin(rr), cos(rr), 0], [0, 0, 0, 1]], dtype=np.float64)
                Ry = np.array([[cos(pp), 0, sin(pp), 0], [0, 1, 0, 0], [-sin(pp), 0, cos(pp), 0], [0, 0, 0, 1]], dtype=np.float64)
                Rz = np.array([[cos(yy), -sin(yy), 0, 0], [sin(yy), cos(yy), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
                T[:4, :4] = (T @ (Rz @ (Ry @ Rx)))
            return T

        def _resolve_mesh_path(filename: str) -> str:
            if filename.startswith("package://"):
                # drop scheme and join to asset_root
                rel = filename[len("package://"):]
                return os.path.join(asset_root, rel)
            if filename.startswith("file://"):
                return filename[len("file://"):]
            if os.path.isabs(filename):
                return filename
            return os.path.join(base_dir, filename)

        # iterate visual meshes
        for link in root.findall("link"):
            for visual in link.findall("visual"):
                origin_elem = visual.find("origin")
                geom = visual.find("geometry")
                if geom is None:
                    continue
                mesh_elem = geom.find("mesh")
                if mesh_elem is None:
                    continue
                filename = mesh_elem.attrib.get("filename", None)
                if filename is None:
                    continue
                scale_attr = mesh_elem.attrib.get("scale", None)

                mesh_path = _resolve_mesh_path(filename)
                if not os.path.exists(mesh_path):
                    # try under asset_root/meshes
                    alt = os.path.join(asset_root, "meshes", os.path.basename(mesh_path))
                    mesh_path = alt if os.path.exists(alt) else mesh_path
                try:
                    m = trimesh.load(mesh_path, force='mesh')
                    if m.is_empty:
                        continue
                except Exception:
                    continue

                # apply scale if any
                if scale_attr is not None:
                    sv = [float(v) for v in scale_attr.strip().split()]
                    if len(sv) == 3:
                        S = np.eye(4, dtype=np.float64)
                        S[0, 0], S[1, 1], S[2, 2] = sv
                        m.apply_transform(S)
                    else:
                        m.apply_scale(float(sv[0]))

                if self.object_scale is not None:
                    s = float(self.object_scale)
                    S = np.eye(4, dtype=np.float64)
                    S[0, 0], S[1, 1], S[2, 2] = s, s, s
                    m.apply_transform(S)

                # apply origin transform
                T = _parse_origin(origin_elem)
                m.apply_transform(T)
                if isinstance(m, trimesh.Trimesh):
                    meshes.append(m)
                elif isinstance(m, trimesh.Scene):
                    for g in m.geometry.values():
                        if isinstance(g, trimesh.Trimesh):
                            meshes.append(g)

        if len(meshes) == 0:
            raise RuntimeError(f"No visual meshes found in URDF: {urdf_path}")

        combined = trimesh.util.concatenate(meshes)
        pts, _ = trimesh.sample.sample_surface(combined, self.pcl_num)
        pts = torch.as_tensor(pts, dtype=torch.float32, device=self.device).unsqueeze(0)
        return pts

    def _create_cube_normals(self) -> torch.Tensor:
        assert hasattr(self, "_pointclouds")
        n = int(self._pointclouds.shape[1])
        normals = torch.zeros(1, n, 3, device=self.device, dtype=torch.float32)

        n_per_face = max(1, n // 6)
        i0 = 0
        i1 = min(n_per_face, n);                 normals[:, i0:i1, :] = torch.tensor([0, 0, 1], device=self.device); i0 = i1
        i1 = min(i0 + n_per_face, n);            normals[:, i0:i1, :] = torch.tensor([0, 0, -1], device=self.device); i0 = i1
        i1 = min(i0 + n_per_face, n);            normals[:, i0:i1, :] = torch.tensor([0, 1, 0], device=self.device); i0 = i1
        i1 = min(i0 + n_per_face, n);            normals[:, i0:i1, :] = torch.tensor([0, -1, 0], device=self.device); i0 = i1
        i1 = min(i0 + n_per_face, n);            normals[:, i0:i1, :] = torch.tensor([1, 0, 0], device=self.device); i0 = i1
        if i0 < n:
            normals[:, i0:, :] = torch.tensor([-1, 0, 0], device=self.device)

        return normals


    def _create_urdf_normals(self) -> torch.Tensor:
        """
        Compute per-point normals for self._pointclouds[0] by rebuilding the
        combined visual mesh and using nearest face normals.
        Returns: (1, N, 3) float32 tensor on self.device
        """
        import os
        import numpy as np
        import trimesh
        import xml.etree.ElementTree as ET

        assert hasattr(self, "_urdf_asset_root") and hasattr(self, "_urdf_rel_path")
        urdf_path = os.path.join(self._urdf_asset_root, self._urdf_rel_path)
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        base_dir = os.path.dirname(urdf_path)

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        def _parse_origin(origin_elem) -> np.ndarray:
            T = np.eye(4, dtype=np.float64)
            if origin_elem is None:
                return T
            xyz = origin_elem.attrib.get("xyz", None)
            rpy = origin_elem.attrib.get("rpy", None)
            if xyz is not None:
                x, y, z = [float(v) for v in xyz.strip().split()]
                T[:3, 3] = [x, y, z]
            if rpy is not None:
                rr, pp, yy = [float(v) for v in rpy.strip().split()]
                from math import cos, sin
                Rx = np.array([[1, 0, 0, 0], [0, cos(rr), -sin(rr), 0], [0, sin(rr), cos(rr), 0], [0, 0, 0, 1]], dtype=np.float64)
                Ry = np.array([[cos(pp), 0, sin(pp), 0], [0, 1, 0, 0], [-sin(pp), 0, cos(pp), 0], [0, 0, 0, 1]], dtype=np.float64)
                Rz = np.array([[cos(yy), -sin(yy), 0, 0], [sin(yy), cos(yy), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
                T[:4, :4] = (T @ (Rz @ (Ry @ Rx)))
            return T

        def _resolve_mesh_path(filename: str) -> str:
            if filename.startswith("package://"):
                rel = filename[len("package://"):]
                return os.path.join(self._urdf_asset_root, rel)
            if filename.startswith("file://"):
                return filename[len("file://"):]
            if os.path.isabs(filename):
                return filename
            return os.path.join(base_dir, filename)

        meshes = []
        for link in root.findall("link"):
            for visual in link.findall("visual"):
                origin_elem = visual.find("origin")
                geom = visual.find("geometry")
                if geom is None:
                    continue
                mesh_elem = geom.find("mesh")
                if mesh_elem is None:
                    continue
                filename = mesh_elem.attrib.get("filename", None)
                if filename is None:
                    continue
                scale_attr = mesh_elem.attrib.get("scale", None)

                mesh_path = _resolve_mesh_path(filename)
                if not os.path.exists(mesh_path):
                    alt = os.path.join(self._urdf_asset_root, "meshes", os.path.basename(mesh_path))
                    mesh_path = alt if os.path.exists(alt) else mesh_path
                try:
                    m = trimesh.load(mesh_path, force='mesh')
                    if m.is_empty:
                        continue
                except Exception:
                    continue

                if scale_attr is not None:
                    sv = [float(v) for v in scale_attr.strip().split()]
                    if len(sv) == 3:
                        S = np.eye(4, dtype=np.float64)
                        S[0, 0], S[1, 1], S[2, 2] = sv
                        m.apply_transform(S)
                    else:
                        m.apply_scale(float(sv[0]))
                
                if self.object_scale is not None:
                    s = float(self.object_scale)
                    S = np.eye(4, dtype=np.float64)
                    S[0, 0], S[1, 1], S[2, 2] = s, s, s
                    m.apply_transform(S)

                T = _parse_origin(origin_elem)
                m.apply_transform(T)
                if isinstance(m, trimesh.Trimesh):
                    meshes.append(m)
                elif isinstance(m, trimesh.Scene):
                    for g in m.geometry.values():
                        if isinstance(g, trimesh.Trimesh):
                            meshes.append(g)

        if len(meshes) == 0:
            raise RuntimeError(f"No visual meshes found in URDF: {urdf_path}")

        combined = trimesh.util.concatenate(meshes)
        points = self._pointclouds[0].detach().cpu().numpy()  # (N,3)

        # closest triangle index for each point
        # returns (closest_points, distances, triangle_id)
        closest = trimesh.proximity.closest_point(combined, points)
        tri_idx = closest[2].astype(np.int64)

        face_normals = combined.face_normals  # (F,3)
        normals_np = face_normals[tri_idx]
        n = np.linalg.norm(normals_np, axis=1, keepdims=True)
        normals_np = normals_np / np.clip(n, 1e-12, None)

        normals = torch.as_tensor(normals_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # debug_view_normals_o3d(self._pointclouds, normals)
        return normals
    
    def _create_pointcloud_mask(self) -> torch.Tensor:
        
        assert hasattr(self, "_urdf_asset_root") and hasattr(self, "_urdf_rel_path")
        urdf_path = os.path.join(self._urdf_asset_root, self._urdf_rel_path)
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        base_dir = os.path.dirname(urdf_path)
        
        rules_json_path = os.path.join(base_dir, "mask_rules.json")
        if os.path.exists(rules_json_path):
            with open(rules_json_path, 'r') as f:
                rules = json.load(f)
        else:
            rules = {
                "x": (0.0, 1.0),
                "y": (0.0, 1.0),
                "z": (0.0, 1.0),
            }
            # save the rules for reference
            with open(rules_json_path, 'w') as f:
                json.dump(rules, f, indent=4)
                
        points = self._pointclouds[0].detach().cpu().numpy()  # (N,3)

        mask_np = build_axis_mask(points, rules)
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device).unsqueeze(0)
        
        return mask
        

    def get_boundingbox(self, pointclouds: torch.Tensor) -> torch.Tensor:
        corner_max = torch.max(pointclouds, dim=-2)[0]
        corner_min = torch.min(pointclouds, dim=-2)[0]
        return torch.cat((corner_max, corner_min), dim=-1).to(self.device)

    def sample(self, object_indices: torch.LongTensor) -> Dict[str, Any]:
        assert object_indices.dim() == 1 and object_indices.dtype == torch.long
        assert object_indices.max() < 1 and object_indices.min() >= 0
        pointclouds = self._pointclouds[object_indices]
        if pointclouds.shape[1] > self.pcl_num:
            pointclouds = sample_farthest_points(pointclouds, K=self.pcl_num)[0]
        bbox = self.get_boundingbox(pointclouds)
        onehot = self._category_matrix[object_indices]
        batch_size = object_indices.shape[0]
        dummy_joints = torch.zeros(batch_size, 16, device=self.device)
        dummy_poses = torch.zeros(batch_size, 7, device=self.device)
        dummy_poses[:, 6] = 1.0
        return {
            "joints": dummy_joints,
            "pose": dummy_poses,
            "pointcloud": pointclouds,
            "index": object_indices,
            "object_index": object_indices,
            "bbox": bbox,
            "category_onehot": onehot,
            "grasp": ["edge_object"] * batch_size,
            "code": [self.object_codes[i] for i in object_indices.cpu().numpy()],
            "cluster": np.zeros(batch_size, dtype=np.int32),
        }

    def get_object_index(self, object_code: str) -> int:
        return 0

    def resample(self, num_samples: int) -> List[str]:
        return [self.object_codes[0] for _ in range(num_samples)]
