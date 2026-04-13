import torch
from torch import nn, roll
from typing import Optional, Dict, Tuple
from .state_feature_bank import PushBox2DStateBank, LearnedHashStateBank
from .torch_utils import quat_conjugate, quat_apply, get_euler_xyz

@torch.no_grad()
def fps(points: torch.Tensor, k: int, start_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert points.dim() == 2, "points should be (N, D)"
    N, D = points.shape
    k = min(k, N)
    pts = points.to(dtype=torch.float32)
    device = pts.device

    dists = torch.full((N,), float('inf'), device=device)
    idx = torch.empty((k,), dtype=torch.long, device=device)

    farthest = torch.randint(0, N, (1,), device=device).item() if start_idx is None else int(start_idx) % N
    for i in range(k):
        idx[i] = farthest
        center = pts[farthest].view(1, D)
        dist2 = ((pts - center) ** 2).sum(dim=1)
        dists = torch.minimum(dists, dist2)
        farthest = torch.argmax(dists).item()

    centers = pts[idx]
    return idx, centers


@torch.no_grad()
def _assign_labels_by_nn(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    points:  (M, D)
    centers: (k, D)
    return:  labels (M,)
    """
    x2 = (points**2).sum(dim=1, keepdim=True)         # (M,1)
    c2 = (centers**2).sum(dim=1).unsqueeze(0)         # (1,k)
    dist2 = x2 - 2 * (points @ centers.T) + c2        # (M,k)
    labels = dist2.argmin(dim=1)
    return labels


def _infer_simhash_dim(num_key_states: int) -> int:
    S = int(num_key_states)
    if S < 2:
        raise ValueError(f"num_key_states must be >= 2 for hash state banks, got {S}")
    if (S & (S - 1)) != 0:
        raise ValueError(
            f"num_key_states must be a power of two for hash state banks, got {S}"
        )
    simhash_dim = S.bit_length() - 1
    return simhash_dim


class OcclusionMask(nn.Module):
    """
    Uses slab-method ray/AABB test. Assumes ray_dirs = (surface_point - keypoint),
    so the surface is at t = 1 along the ray.
    """
    def __init__(self, container_aabbs: torch.Tensor):
        """
        Args:
            container_aabbs: [#box, 2, 3]
                aabbs[i, 0] = min corner (x_min, y_min, z_min)
                aabbs[i, 1] = max corner (x_max, y_max, z_max)
        """
        super().__init__()
        self.register_buffer("container_aabbs", container_aabbs)

    @staticmethod
    def _ray_aabb_intersection(ray_origins: torch.Tensor, ray_dirs: torch.Tensor, aabbs: torch.Tensor):
        """
        Vectorized ray-AABB intersection using the slab method.
        Given ray_dirs = (surface_point - keypoint), the hit parameter for the
        target surface is t=1. We return t_entry if a valid hit exists in (eps, 1-eps),
        else +inf.

        Args:
            ray_origins: [B, K, N, 3]
            ray_dirs:    [B, K, N, 3]
            aabbs:       [M, 2, 3] (M=5)
        Returns:
            hit_t: [B, K, N, M] with first valid entry t or +inf
        """
        eps_dir = 1e-9
        eps_t = 1e-6


        if aabbs.dim() == 3:
            mins = aabbs[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            maxs = aabbs[:, 1].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif aabbs.dim() == 4:
            assert aabbs.shape[0] == ray_origins.shape[0]
            mins = aabbs[:, :, 0].unsqueeze(1).unsqueeze(1)
            maxs = aabbs[:, :, 1].unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(f"Unexpected aabbs shape: {aabbs.shape}")
    
        orig = ray_origins.unsqueeze(-2)  # [B,K,N,1,3]
        dirs = ray_dirs.unsqueeze(-2)     # [B,K,N,1,3]

        inv_dirs = torch.where(
            torch.abs(dirs) > eps_dir,
            1.0 / dirs,
            torch.full_like(dirs, float('inf'))
        )

        t1 = (mins - orig) * inv_dirs
        t2 = (maxs - orig) * inv_dirs

        t_min = torch.minimum(t1, t2)  # [B,K,N,M,3]
        t_max = torch.maximum(t1, t2)  # [B,K,N,M,3]

        t_min_overall = torch.max(t_min, dim=-1).values  # [B,K,N,M]
        t_max_overall = torch.min(t_max, dim=-1).values  # [B,K,N,M]

        # First positive intersection param:
        t_entry = torch.where(t_min_overall >= 0, t_min_overall, t_max_overall)

        valid = (t_min_overall <= t_max_overall) & (t_entry > eps_t) & (t_entry < 1.0 - eps_t)
        hit_t = torch.where(valid, t_entry, torch.full_like(t_entry, float('inf')))
        return hit_t

    def forward(self, keypoints: torch.Tensor, surface_points: torch.Tensor) -> torch.Tensor:
        """
        Compute occlusion mask for all (keypoint, surface_point) pairs.

        Args:
            keypoints:       [B, K, 3] world-space keypoints
            surface_points:  [B, M, 3] world-space points, or [M, 3] shared across batch
        Returns:
            mask: [B, K, M] float, 1 if visible, 0 if occluded
        """
        B, K, _ = keypoints.shape
        if surface_points.dim() == 2:
            # Shared across batch
            M = surface_points.shape[0]
            sp_exp = surface_points.unsqueeze(0).expand(B, M, 3)       # [B, M, 3]
        else:
            B2, M, _ = surface_points.shape
            assert B2 == B, "Batch size mismatch between keypoints and surface_points"
            sp_exp = surface_points

        # [B, K, M, 3]
        kp_exp = keypoints.unsqueeze(2).expand(B, K, M, 3)
        sp_exp = sp_exp.unsqueeze(1).expand(B, K, M, 3)

        ray_dirs = sp_exp - kp_exp
        ray_orig = kp_exp

        hit_t = self._ray_aabb_intersection(ray_orig, ray_dirs, self.container_aabbs)
        any_occluded = (hit_t < float('inf')).any(dim=-1)  # [B, K, M]
        mask = (~any_occluded).float()
        return mask

class OBBOcclusionMask(nn.Module):

    def __init__(self, obb_transform: torch.Tensor, obb_extents: torch.Tensor):
        # NOTE: obb_transform: mesh_local -> obb_local homogeneous transform
        super().__init__()
        self.register_buffer("obb_transform", obb_transform.to(torch.float32))
        self.register_buffer("obb_extents", obb_extents.to(torch.float32))
    def forward(
        self,
        keypoints_world: torch.Tensor,
        surface_points_world: torch.Tensor,
        object_positions: torch.Tensor,
        object_orientations: torch.Tensor,
    ) -> torch.Tensor:
        """
        True for visible, False for occluded
        """
        device = keypoints_world.device
        dtype = keypoints_world.dtype

        B, K, _ = keypoints_world.shape
        B_, M, _ = surface_points_world.shape
        assert B_ == B
        sp_world = surface_points_world.to(device=device, dtype=dtype)

        # [B,K,M,3]
        kp_exp = keypoints_world.unsqueeze(2).expand(B, K, M, 3)
        sp_exp = sp_world.unsqueeze(1).expand(B, K, M, 3)

        # make keypoint and mesh point to mesh local
        # p_mesh = R^T (p_world - t)
        q = object_orientations.to(device=device, dtype=dtype)
        t = object_positions.to(device=device, dtype=dtype)
        q_conj = quat_conjugate(q)

        q_conj_exp = q_conj.view(B, 1, 1, 4).expand(B, K, M, 4).reshape(-1, 4)

        p_rel_kp = (kp_exp - t.view(B, 1, 1, 3)).reshape(-1, 3)
        p_rel_sp = (sp_exp - t.view(B, 1, 1, 3)).reshape(-1, 3)

        kp_mesh = quat_apply(q_conj_exp, p_rel_kp).view(B, K, M, 3)
        sp_mesh = quat_apply(q_conj_exp, p_rel_sp).view(B, K, M, 3)

        # from mesh local to obb local
        # R_obb = self.obb_transform[:3, :3]    # (3,3)
        # t_obb = self.obb_transform[:3, 3]     # (3,)

        # # p_obb = R_obb @ p_mesh + t_obb
        # kp_obb = torch.matmul(kp_mesh, R_obb.t()) + t_obb.view(1, 1, 1, 3)
        # sp_obb = torch.matmul(sp_mesh, R_obb.t()) + t_obb.view(1, 1, 1, 3)
        kp_mesh_hom = torch.cat([kp_mesh, torch.ones_like(kp_mesh[..., :1])], dim=-1)  # (B,K,M,4)
        sp_mesh_hom = torch.cat([sp_mesh, torch.ones_like(sp_mesh[..., :1])], dim=-1)  # (B,K,M,4)

        to_origin = self.obb_transform
        kp_obb_hom = torch.einsum('ij,bkmj->bkmi', to_origin, kp_mesh_hom)  # (B,K,M,4)
        sp_obb_hom = torch.einsum('ij,bkmj->bkmi', to_origin, sp_mesh_hom)  # (B,K,M,4)
        
        kp_obb = kp_obb_hom[..., :3] / kp_obb_hom[..., 3:4]
        sp_obb = sp_obb_hom[..., :3] / sp_obb_hom[..., 3:4]

        # slab check
        ray_orig_local = kp_obb
        ray_dirs_local = sp_obb - kp_obb

        half = (self.obb_extents / 2.0).to(device=device, dtype=dtype)
        aabb = torch.stack([-half, half], dim=0).unsqueeze(0)

        hit_t = OcclusionMask._ray_aabb_intersection(
            ray_origins=ray_orig_local,
            ray_dirs=ray_dirs_local,
            aabbs=aabb,
        )

        any_occluded = (hit_t < float('inf')).any(dim=-1)
        mask = (~any_occluded).float()
        return mask

class CuriosityRewardManager:

    def __init__(
        self,
        num_keypoints: Optional[int] = None,
        num_object_points: Optional[int] = None,
        canonical_pointcloud: Optional[torch.Tensor] = None,
        *,
        device: Optional[torch.device] = None,
        cluster_k: int = 64,  # number of clusters for object point cloud
        max_clustering_iters: int = 10,  # max number of iterations for K-Means
        enable_predefined_clusters: bool = False,
        
        potential_kernel = "exponential",    # "inverse", "gaussian", "exponential"
        novelty_decay = "sqrt",       # "linear", "exponential", "sqrt", "logarithmic"
        kernel_param = 0.03,
        novelty_decay_rate = 2.0,            # exp/log/linear decay param
        use_potential_shaping = True,
        
        canonical_normals: Optional[torch.Tensor] = None, # (M, 3)
        mask_backface_points: bool = False,
        mask_palm_inward_points: bool = False,
        
        use_normal_in_clustering: bool = True,
        normal_weight: float = 0.5,

        num_envs: Optional[int] = None,

        state_feature_dim: Optional[int] = None,
        num_key_states: int = 32,
        state_counter_mode: str = "cluster",  # "cluster" | "point"
        state_num_points: int = 32,          # number of fixed canonical points to use (ordered)
        state_include_goal: bool = False,

        hash_code_dim: int = 256,
        hash_noise_scale: float = 0.3,
        hash_lambda_binary: float = 10.0,
        hash_ae_lr: float = 3e-4,
        hash_ae_steps: int = 5,
        hash_ae_num_minibatches: int = 8,
        hash_ae_update_freq: int = 16,
        hash_seed: int = 0,
        enable_predefined_state: bool = False,

        state_type: str = None,
        state_running_max_mode: Optional[str] = None,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canonical_pointcloud = canonical_pointcloud

        # runtime cache
        self.potential_per_kp_max: Optional[torch.Tensor] = None  # (N, L) previous potential per keypoint
        self.contact_coverage_per_kp_max: Optional[torch.Tensor] = None  # (N, L) previous contact bonus per keypoint

        self.L = num_keypoints
        self.M = num_object_points
        self.num_envs = num_envs
        
        self.cluster_k = cluster_k
        self.max_clustering_iters = max_clustering_iters
        self._point_to_cluster: Optional[torch.Tensor] = None  # (M,)
        
        self.use_normal_in_clustering = use_normal_in_clustering
        self.normal_weight = float(normal_weight)
        
        assert (canonical_normals is not None if mask_backface_points else True)
        self.canonical_normals = canonical_normals.to(self.device) if canonical_normals is not None else None
        self.mask_backface_points = bool(mask_backface_points)
        self.mask_palm_inward_points = bool(mask_palm_inward_points)
        
        # self._point_to_cluster = self._perform_clustering_w_fps(self.canonical_pointcloud) 
        # self._point_to_cluster_with_normals = self._perform_clustering_w_fps_with_normals(self.canonical_pointcloud, self.canonical_normals if self.use_normal_in_clustering else None)
        if not enable_predefined_clusters:
            self._point_to_cluster = self._perform_clustering_w_fps_with_normals(self.canonical_pointcloud, self.canonical_normals if self.use_normal_in_clustering else None)
        else:
            self._point_to_cluster = torch.arange(0, self.M, 1, device=self.device) // (self.M // self.cluster_k) # shape (M,)


        self.potential_kernel = potential_kernel    
        self.novelty_decay = novelty_decay
        self.kernel_param = kernel_param
        self.novelty_decay_rate = novelty_decay_rate
        self.use_potential_shaping = True

        self.occlusion_module = None  # type: Optional[OcclusionMask]

        self.state_num_points = int(state_num_points)
        self.state_feature_dim = int(state_feature_dim) if state_feature_dim is not None else None
        self.state_include_goal = bool(state_include_goal)
        self.num_key_states = int(num_key_states)
        self.state_counter_mode = str(state_counter_mode).lower()
        self._state_point_indices: Optional[torch.Tensor] = None  # (P,) long, fixed canonical indices
        self.enable_predefined_state = enable_predefined_state

        if self.state_feature_dim is not None:
            expected_F = int(self.state_num_points) * 3 * (2 if self.state_include_goal else 1)

            assert int(self.state_feature_dim) == expected_F, f"state_feature_dim must be 3*state_num_points*{(2 if self.state_include_goal else 1)} ({expected_F}) when state_include_goal={self.state_include_goal}, got {self.state_feature_dim}"

            pc_can = self.canonical_pointcloud.to(self.device, dtype=torch.float32)
            idx, _ = fps(pc_can, k=self.state_num_points, start_idx=0)
            self._state_point_indices = idx.to(self.device, dtype=torch.long)  # (P,)

            if self.state_counter_mode == "point":
                if self.canonical_pointcloud is None:
                    raise ValueError("canonical_pointcloud is required for state_counter_mode='point'")
                num_bins = int(self.state_num_points)
            elif self.state_counter_mode == "cluster":
                num_bins = int(self.cluster_k)
            else:
                raise ValueError(f"Unknown state_counter_mode: {self.state_counter_mode}")

            # state_type = "pcd"
            self.state_type = str(state_type).lower()
            assert self.state_type in ( "hash", "predefined"), f"Unknown state_type: {self.state_type}"

            if self.state_type == "hash":
                simhash_dim = _infer_simhash_dim(self.num_key_states)
                buffer_size = int(self.num_envs) * int(hash_ae_update_freq)
                self.state_bank = LearnedHashStateBank(
                    num_key_states=int(self.num_key_states),
                    feature_dim=int(self.state_feature_dim),
                    buffer_size=int(buffer_size),
                    num_hand_keypoints=int(self.L),
                    num_object_bins=int(num_bins),
                    device=self.device,
                    code_dim=hash_code_dim,
                    simhash_dim=simhash_dim,
                    noise_scale=hash_noise_scale,
                    lambda_binary=hash_lambda_binary,
                    ae_lr=hash_ae_lr,
                    ae_update_steps=hash_ae_steps,
                    ae_update_freq=hash_ae_update_freq,
                    ae_num_minibatches=hash_ae_num_minibatches,
                    seed=hash_seed,
                )            
            elif self.state_type == "predefined":
                self.state_bank = PushBox2DStateBank(
                    num_key_states=int(self.num_key_states),
                    num_hand_keypoints=int(self.L),
                    num_object_bins=int(num_bins),
                    device=self.device,
                )
            else:
                raise ValueError(f"Unknown state_type: {self.state_type}")

        # running-max buffers:
        # - global(legacy) mode: (N,L)
        # - state-feature mode: (N,S,L)
        self.state_running_max_mode = str(state_running_max_mode).lower()
        self.state_running_max_mode = str(state_running_max_mode).lower()
        assert self.state_running_max_mode in ("state", "global")

    def build_state_features_from_world_pc(
        self,
        pc_world: torch.Tensor,
        *,
        goal_positions: Optional[torch.Tensor] = None,
        goal_orientations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build state features from a world-space object point cloud.

        If state_include_goal is True, appends the goal point cloud features
        built from goal_positions / goal_orientations using the canonical cloud.
        """
        if self._state_point_indices is None:
            raise ValueError("state_feature_dim must be enabled to build state features")

        device = pc_world.device
        N = int(pc_world.shape[0])
        idx = self._state_point_indices
        feat_obj = pc_world.index_select(1, idx).reshape(N, -1).to(device=device, dtype=torch.float32)

        if not self.state_include_goal:
            return feat_obj

        if goal_positions is None or goal_orientations is None:
            raise ValueError("goal_positions and goal_orientations are required when state_include_goal=True")

        if goal_positions.dim() == 1:
            goal_positions = goal_positions.unsqueeze(0).expand(N, -1)
        if goal_orientations.dim() == 1:
            goal_orientations = goal_orientations.unsqueeze(0).expand(N, -1)

        pc_local = self.canonical_pointcloud.unsqueeze(0).expand(N, -1, -1)
        qg = goal_orientations.unsqueeze(1).expand(-1, pc_local.shape[1], -1)
        pg_world = quat_apply(qg, pc_local) + goal_positions.unsqueeze(1)
        feat_goal = pg_world.index_select(1, idx).reshape(N, -1).to(device=device, dtype=torch.float32)

        return torch.cat([feat_obj, feat_goal], dim=-1)

    def set_occlusion_module(self, container_aabbs: torch.Tensor) -> None:
        """
        Enable occlusion-aware potential computation by supplying static container AABBs.
        Args:
            container_aabbs: [5, 2, 3] tensor on the desired device
        """
        self.occlusion_module = OcclusionMask(container_aabbs)

    @torch.no_grad()
    def update_occlusion_aabbs(
        self,
        container_aabbs: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        if self.occlusion_module is None or isinstance(self.occlusion_module, OBBOcclusionMask):
            self.set_occlusion_module(container_aabbs)
            return

        mod = self.occlusion_module
        new_aabbs = container_aabbs.to(device=mod.container_aabbs.device, dtype=mod.container_aabbs.dtype)

        if mod.container_aabbs.shape != new_aabbs.shape:
            self.set_occlusion_module(new_aabbs)
            return

        if mod.container_aabbs.dim() == 3 or env_ids is None:
            mod.container_aabbs.copy_(new_aabbs)
        else:
            mod.container_aabbs[env_ids].copy_(new_aabbs[env_ids])
        
    def set_obb_occlusion_module(self, obb_transform: torch.Tensor, obb_extents: torch.Tensor) -> None:
        
        self.occlusion_module = OBBOcclusionMask(obb_transform, obb_extents)

    @torch.no_grad()
    def ensure_running_max_buffers(self, num_envs: int) -> None:
        N = int(num_envs)
        device = self.device

        if self.state_bank is None:
            return

        S = int(self.num_key_states)
        L = int(self.L)

        if self.state_running_max_mode == "state":
            # (N,S,L)
            if self.potential_per_kp_max is None or self.potential_per_kp_max.shape != (N, S, L):
                self.potential_per_kp_max = torch.zeros((N,S,L), device=device, dtype=torch.float32)
            if self.contact_coverage_per_kp_max is None or self.contact_coverage_per_kp_max.shape != (N, S, L):
                self.contact_coverage_per_kp_max = torch.zeros((N,S,L), device=device, dtype=torch.float32)
        else:
            # "global": (N,L)
            if self.potential_per_kp_max is None or self.potential_per_kp_max.shape != (N, L):
                self.potential_per_kp_max = torch.zeros((N,L), device=device, dtype=torch.float32)
            if self.contact_coverage_per_kp_max is None or self.contact_coverage_per_kp_max.shape != (N, L):
                self.contact_coverage_per_kp_max = torch.zeros((N,L), device=device, dtype=torch.float32)


    @torch.no_grad()
    def reset_running_max_buffers(self, env_ids: torch.Tensor) -> None:
        if env_ids is None:
            return
        if self.state_bank is None:
            # legacy path
            if self.potential_per_kp_max is not None:
                self.potential_per_kp_max[env_ids] = 0.0
            if self.contact_coverage_per_kp_max is not None:
                self.contact_coverage_per_kp_max[env_ids] = 0.0
            return

        self.potential_per_kp_max[env_ids] = 0.0
        if self.contact_coverage_per_kp_max is not None:
            self.contact_coverage_per_kp_max[env_ids] = 0.0


    @torch.no_grad()
    def _perform_clustering(self, pointcloud: torch.Tensor):
        M, _ = pointcloud.shape
        points = pointcloud.to(torch.float32)

        indices = torch.randperm(M, device=self.device)[:self.cluster_k]
        centers = points[indices].clone()  # (cluster_k, 3)

        for _ in range(self.max_clustering_iters):
            distances = torch.cdist(points.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)  # (M, cluster_k)
            labels = torch.argmin(distances, dim=1)  # (M,)
            new_centers = torch.zeros_like(centers)
            for i in range(self.cluster_k):
                mask = (labels == i)
                if mask.any():
                    new_centers[i] = points[mask].mean(dim=0)
                else:
                    new_centers[i] = centers[i]
            centers = new_centers

        return labels

    @torch.no_grad()
    def _check_clusters_by_normals(self, pts: torch.Tensor, normals: torch.Tensor, labels: torch.Tensor, k: int, max_angle_deg: float = 15.0):

        M = pts.shape[0]
        max_angle_rad = torch.deg2rad(torch.tensor(max_angle_deg, device=pts.device))
        new_labels = labels.clone()
        current_max_label = labels.max().item()

        for cluster_id in range(k):
            mask = (labels == cluster_id)
            if not mask.any():
                continue

            cluster_normals = normals[mask]
            mean_normal = cluster_normals.mean(dim=0)
            mean_normal = mean_normal / mean_normal.norm().clamp_min(1e-8)

            cos_angles = (cluster_normals * mean_normal).sum(dim=1).clamp(-1.0, 1.0)
            angles = torch.acos(cos_angles)

            outlier_mask = angles > max_angle_rad

        return new_labels

    @torch.no_grad()
    def _perform_clustering_w_fps_with_normals(self, pointcloud: torch.Tensor, normals: Optional[torch.Tensor] = None):
        
        M, D = pointcloud.shape
        k = min(self.cluster_k, M)
        pts = pointcloud.to(torch.float32)
        
        idx, centers = fps(pts, k)
        center_normals = None
        if normals is not None and self.use_normal_in_clustering:
            center_normals = normals[idx]
        
        for _ in range(self.max_clustering_iters):
            if normals is not None and self.use_normal_in_clustering:
                pos_dist = torch.cdist(pts, centers)  # (M, k)
                normal_sim = torch.mm(normals, center_normals.T)  # (M, k), cosθ
                normal_dist = 1.0 - normal_sim.clamp(-1, 1)
                combined_dist = (1 - self.normal_weight) * pos_dist + self.normal_weight * normal_dist
                labels = torch.argmin(combined_dist, dim=1)
            else:
                distances = torch.cdist(pts, centers)
                labels = torch.argmin(distances, dim=1)

            # 更新中心（仅位置）
            new_centers = torch.zeros_like(centers)
            for i in range(k):
                mask = (labels == i)
                if mask.any():
                    new_centers[i] = pts[mask].mean(dim=0)
                else:
                    new_centers[i] = centers[i]
            centers = new_centers

            if normals is not None and self.use_normal_in_clustering:
                new_center_normals = torch.zeros_like(center_normals)
                for i in range(k):
                    mask = (labels == i)
                    if mask.any():
                        # 法向平均后归一化
                        avg_normal = normals[mask].mean(dim=0)
                        new_center_normals[i] = avg_normal / avg_normal.norm().clamp_min(1e-8)
                    else:
                        new_center_normals[i] = center_normals[i]
                center_normals = new_center_normals

        if normals is not None and self.use_normal_in_clustering:
            _ = self._check_clusters_by_normals(pts, normals, labels, k, max_angle_deg=15.0)

        return labels

    
    @torch.no_grad()
    def compute_reward_from_canonical(
        self,
        *,
        object_positions: torch.Tensor,
        object_orientations: torch.Tensor,
        goal_positions: Optional[torch.Tensor] = None,
        goal_orientations: Optional[torch.Tensor] = None,
        keypoint_positions_world: torch.Tensor,
        contact_indices: Optional[torch.Tensor] = None,
        contact_mask: Optional[torch.Tensor] = None,
        task_contact_satisfied: Optional[torch.Tensor] = None,
        contact_forces_local: Optional[torch.Tensor] = None,
        keypoint_palm_dirs_world: Optional[torch.Tensor] = None,
        object_occlusion_positions: Optional[torch.Tensor] = None,
        object_occlusion_orientations: Optional[torch.Tensor] = None,
        state_features_world: Optional[torch.Tensor] = None,  # (N,F) optional state feature vector
        predefined_state_ids: Optional[torch.Tensor] = None,  # (N,) optional predefined state ids
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.compute_potential_field_reward(
            object_positions=object_positions,
            object_orientations=object_orientations,
            keypoint_positions_world=keypoint_positions_world,
            goal_positions=goal_positions,
            goal_orientations=goal_orientations,
            contact_indices=contact_indices,
            contact_mask=contact_mask,
            valid_task_contact=task_contact_satisfied,
            contact_forces_local=contact_forces_local,
            keypoint_palm_dirs_world=keypoint_palm_dirs_world,
            object_occlusion_positions=object_occlusion_positions,
            object_occlusion_orientations=object_occlusion_orientations,
            state_features_world=state_features_world,
            predefined_state_ids=predefined_state_ids,
        )


    @torch.no_grad()
    def compute_potential_field_reward(
        self,
        object_positions: torch.Tensor,         # (N, 3)
        object_orientations: torch.Tensor,      # (N, 4) [x,y,z,w]
        keypoint_positions_world: torch.Tensor, # (N, L, 3)
        goal_positions: Optional[torch.Tensor] = None,
        goal_orientations: Optional[torch.Tensor] = None,
        contact_indices: Optional[torch.Tensor] = None,
        contact_mask: Optional[torch.Tensor] = None,
        valid_task_contact: Optional[torch.Tensor] = None,
        contact_forces_local: Optional[torch.Tensor] = None,
        keypoint_palm_dirs_world: Optional[torch.Tensor] = None,
        object_occlusion_positions: Optional[torch.Tensor] = None,
        object_occlusion_orientations: Optional[torch.Tensor] = None,
        state_features_world: Optional[torch.Tensor] = None,  # (N,F)
        predefined_state_ids: Optional[torch.Tensor] = None,  # (N,)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = keypoint_positions_world.device
        N, L, _ = keypoint_positions_world.shape
        M = int(self.canonical_pointcloud.shape[0])

        valid_env = valid_task_contact.to(torch.bool)

        from .torch_utils import quat_apply
        q_expanded = object_orientations.unsqueeze(1).expand(-1, M, -1)           # (N, M, 4)
        pc_local = self.canonical_pointcloud.unsqueeze(0).expand(N, -1, -1)       # (N, M, 3)
        pc_world = quat_apply(q_expanded, pc_local) + object_positions.unsqueeze(1)  # (N, M, 3)

        if self.state_bank is not None and self.state_type in ("pcd", "hash"):
            if state_features_world is None:
                state_features_world = self.build_state_features_from_world_pc(
                    pc_world,
                    goal_positions=goal_positions,
                    goal_orientations=goal_orientations,
                )
            else:
                # shape should match state_feature_dim
                state_features_world = state_features_world.to(device=device, dtype=torch.float32)

        # bank update / assignment
        state_ids: Optional[torch.Tensor] = None
        if (self.state_bank is not None):
            if self.state_type == "hash":
                self.state_bank.push(state_features_world)
            else:
                pass

            if self.state_type in ("hash"):
                state_ids = self.state_bank.assign(state_features_world)  # (N,)
            elif self.state_type == "predefined":
                state_ids = predefined_state_ids if self.num_key_states > 1 else torch.zeros_like(predefined_state_ids)  # (N,) in [-1,S-1]
            self.ensure_running_max_buffers(num_envs=N)


        cm = contact_mask.to(torch.bool) if contact_mask is not None else torch.zeros((N, L), dtype=torch.bool, device=device)


        # bank.counter (S,L,Kbins)
        if (self.state_bank is not None) and (state_ids is not None) and cm.any():
            # contact_bins: (N,L) in [0..Kbins-1]
            if contact_indices is None:
                contact_indices = torch.zeros((N, L), dtype=torch.long, device=device)
            ci = contact_indices.to(device=device, dtype=torch.long).clamp(0, M - 1)
            if self.state_counter_mode == "point":
                raise NotImplementedError("State counter mode 'point' is not implemented")
            elif self.state_counter_mode == "cluster":
                contact_bins = self._point_to_cluster[ci]  # (N,L)
            else:
                raise ValueError(f"Unknown state_counter_mode: {self.state_counter_mode}")

            self.state_bank.add_contacts(
                state_ids=state_ids,
                contact_mask=(cm & valid_env.unsqueeze(1)),
                contact_bins=contact_bins,
            )

        
        if (self.state_bank is not None) and (state_ids is not None) and (self.state_bank.counts is not None):
            has_state = (state_ids >= 0)
            if has_state.any():
                counts_state = self.state_bank.counts.to(torch.float32)

                # map to (S,L,M)
                if counts_state.shape[-1] == M:
                    counts_SLM = counts_state
                elif counts_state.shape[-1] == self.cluster_k:
                    # per-point gather by cluster id
                    idx = self._point_to_cluster.view(1, 1, M).expand(self.num_key_states, L, M)
                    counts_SLM = counts_state.gather(2, idx)
                elif counts_state.shape[-1] == self.cluster_k * self.force_direction_bins:
                    counts_cluster = counts_state.view(self.num_key_states, L, self.cluster_k, self.force_direction_bins).sum(dim=-1)
                    idx = self._point_to_cluster.view(1, 1, M).expand(self.num_key_states, L, M)
                    counts_SLM = counts_cluster.gather(2, idx)
                else:
                    raise ValueError(f"Unexpected Kbins={counts_state.shape[-1]}")

                # decay -> q(S,L,M)
                if self.novelty_decay == "exponential":
                    q_SLM = torch.exp(-self.novelty_decay_rate * counts_SLM)
                elif self.novelty_decay == "linear":
                    q_SLM = 1.0 / (1.0 + self.novelty_decay_rate * counts_SLM)
                elif self.novelty_decay == "sqrt":
                    q_SLM = 1.0 / torch.sqrt(1.0 + self.novelty_decay_rate * counts_SLM)
                elif self.novelty_decay == "logarithmic":
                    q_SLM = 1.0 / torch.log(1.0 + counts_SLM + 1e-8)
                else:
                    raise ValueError(f"Unknown novelty_decay: {self.novelty_decay}")

                assert (state_ids >= 0).all()
                q_env = q_SLM[state_ids]

        # Distances d(N,L,M)
        dists = torch.norm(keypoint_positions_world.unsqueeze(2) - pc_world.unsqueeze(1), dim=-1)
        

        # Kernel K(N,L,M)
        if self.potential_kernel == "inverse":
            K = 1.0 / (dists + self.kernel_param)
        elif self.potential_kernel == "gaussian":
            K = torch.exp(-(dists ** 2) / (2.0 * (self.kernel_param ** 2)))
        elif self.potential_kernel == "exponential":
            K = torch.exp(-dists / self.kernel_param)
        else:
            raise ValueError(f"Unknown potential_kernel: {self.potential_kernel}")

        if self.occlusion_module is not None:
            if isinstance(self.occlusion_module, OBBOcclusionMask):
                assert object_occlusion_positions is not None and object_occlusion_orientations is not None
                occl_mask = self.occlusion_module(
                    keypoints_world=keypoint_positions_world,
                    surface_points_world=pc_world,
                    object_positions=object_occlusion_positions,
                    object_orientations=object_occlusion_orientations,
                )
            else:
                occl_mask = self.occlusion_module(keypoint_positions_world, pc_world)
            K = K * occl_mask
        
        normals_world = quat_apply(q_expanded, self.canonical_normals.unsqueeze(0).expand(N, -1, -1))
        
        if self.mask_backface_points:
            v = keypoint_positions_world.unsqueeze(2) - pc_world.unsqueeze(1)
            v_norm = torch.norm(v, dim=-1).clamp_min(1e-8)
            n = normals_world.unsqueeze(1)
            n_norm = torch.norm(n, dim=-1).clamp_min(1e-8)
            
            cos_theta = (v * n).sum(dim=-1) / (v_norm * n_norm)

            angle_weight = torch.clamp(cos_theta, min=0.0, max=1.0)

            K = K * angle_weight
            
        if self.mask_palm_inward_points:
            p = keypoint_palm_dirs_world.unsqueeze(2)       # (N, L, 1, 3)
            n = normals_world.unsqueeze(1)                  # (N, 1, M, 3)
            p_norm = torch.norm(p, dim=-1).clamp_min(1e-8)  # (N, L, 1)
            n_norm = torch.norm(n, dim=-1).clamp_min(1e-8)  # (N, 1, M)
            cos_phi = (p * n).sum(dim=-1) / (p_norm * n_norm)  # (N, L, M)

            palm_angle_weight = torch.clamp(-cos_phi, min=0.0, max=1.0)  # (N, L, M)

            K = K * palm_angle_weight
    
        Phi = (q_env * K).sum(dim=-1)  # (N,L)
        avg_Phi = Phi.mean(dim=1)               # (N,)
        max_Phi = Phi.amax(dim=1)


        assert state_ids is not None and self.state_bank is not None
        # Phi: (N,L)
        self.ensure_running_max_buffers(num_envs=N)

        has_state = (state_ids >= 0) & valid_env
        if self.state_running_max_mode == "state":
            env_idx = torch.arange(N, device=device, dtype=torch.long)
            sid = state_ids.clamp(min=0)
            prev_max = self.potential_per_kp_max[env_idx, sid]          # (N,L)
            delta_kp = torch.clamp(Phi - prev_max, min=0.0)             # (N,L)
            reward = torch.where(has_state, delta_kp.mean(dim=1), torch.zeros((N,), device=device, dtype=Phi.dtype))
            new_max = torch.maximum(prev_max, Phi)                      # (N,L)
            self.potential_per_kp_max[env_idx[has_state], sid[has_state]] = new_max[has_state]
        else:
            prev_max = self.potential_per_kp_max                          # (N,L)
            delta_kp = torch.clamp(Phi - prev_max, min=0.0)               # (N,L)
            reward = torch.where(valid_env, delta_kp.mean(dim=1), torch.zeros((N,), device=device, dtype=Phi.dtype))
            self.potential_per_kp_max[valid_env] = torch.maximum(prev_max[valid_env], Phi[valid_env])

            
        cm = contact_mask.to(torch.bool)
        
        contact_novelty_reward = torch.zeros((N, L), dtype=avg_Phi.dtype, device=device)

        
        if cm.any():
            ei, ki = torch.nonzero(cm, as_tuple=True)
            pj = contact_indices[ei, ki].clamp(0, M - 1)

            assert state_ids is not None and self.state_bank is not None

            si = state_ids[ei]  # (num_contacts,)

            cluster_ids = self._point_to_cluster[pj]  # (num_valid,)
            point_counts = self.state_bank.counts[si, ki, cluster_ids].to(torch.float32)
            # novelty from counts
            if self.novelty_decay == "exponential":
                point_novelty = torch.exp(-self.novelty_decay_rate * point_counts)
            elif self.novelty_decay == "linear":
                point_novelty = 1.0 / (1.0 + self.novelty_decay_rate * point_counts)
            elif self.novelty_decay == "sqrt":
                point_novelty = 1.0 / torch.sqrt(1.0 + self.novelty_decay_rate * point_counts)
            elif self.novelty_decay == "logarithmic":
                point_novelty = 1.0 / torch.log(1.0 + point_counts + 1e-8)
            else:
                raise ValueError(f"Unknown novelty_decay: {self.novelty_decay}")

            contact_novelty_reward[ei, ki] = point_novelty.to(contact_novelty_reward.dtype)

            

        assert state_ids is not None and self.state_bank is not None
        self.ensure_running_max_buffers(num_envs=N)
        has_state = (state_ids >= 0) & valid_env

        if self.state_running_max_mode == "state":
            env_idx = torch.arange(N, device=device, dtype=torch.long)
            sid = state_ids.clamp(min=0)
            prev_bonus = self.contact_coverage_per_kp_max[env_idx, sid]                  # (N,L)
            delta_bonus_kp = torch.clamp(contact_novelty_reward - prev_bonus, min=0)  # (N,L)
            bonus_term = torch.where(has_state, delta_bonus_kp.mean(dim=1), torch.zeros((N,), device=device, dtype=delta_bonus_kp.dtype))
            new_bonus = torch.maximum(prev_bonus, contact_novelty_reward)             # (N,L)
            self.contact_coverage_per_kp_max[env_idx[has_state], sid[has_state]] = new_bonus[has_state]
        else:
            # "global": ignore state_ids
            prev_bonus = self.contact_coverage_per_kp_max                                 # (N,L)
            delta_bonus_kp = torch.clamp(contact_novelty_reward - prev_bonus, min=0)   # (N,L)
            bonus_term = torch.where(valid_env, delta_bonus_kp.mean(dim=1), torch.zeros((N,), device=device, dtype=delta_bonus_kp.dtype))
            self.contact_coverage_per_kp_max[valid_env] = torch.maximum(prev_bonus[valid_env], contact_novelty_reward[valid_env])

        info = {
            "potential_field_reward": reward,
            "avg_potential": avg_Phi,
            "contact_count": cm.sum(dim=1),
            "cluster_novelty_reward": bonus_term,
        }

        if self.state_type == "hash":
            info.update({k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in self.state_bank.get_metrics().items()})
        
        return reward, info
