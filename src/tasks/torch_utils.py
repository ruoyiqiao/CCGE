from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion


@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, npoint: int):
    """
    Farthest Point Sampling (FPS)
    Args:
        xyz: (B, N, 3) 输入点云
        npoint: 目标点数
    Returns:
        centroids: (B, npoint) 采样到的点索引
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    # 随机初始化第一个采样点
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids



@torch.no_grad()
def build_group_lut(pc: torch.Tensor, fps_idx: torch.Tensor, *, use_cdist: bool = True):
    """
    一次性分组并构建查表 LUT
    Args:
        pc:      (N, M, 3) float  —— 原始点云
        fps_idx: (N, K)    long   —— FPS 选出的中心索引（相对 pc）
        use_cdist: 是否使用 torch.cdist（更简洁），False 时用广播算平方距离（省显存）

    Returns:
        assign_idx: (N, M) long     —— 每个原始点归属的最近中心 (0..K-1)
        LUT:        (N, M+1) long   —— 索引查表：0->0, j+1 -> assign_idx[j]+1
                                      用于把 contact state 从 1..M 映射到 1..K
    """
    assert pc.ndim == 3 and fps_idx.ndim == 2
    N, M, _ = pc.shape
    K = fps_idx.shape[1]
    device = pc.device

    # 中心坐标 (N, K, 3)
    centroids = torch.gather(pc, dim=1, index=fps_idx.unsqueeze(-1).expand(-1, -1, 3))

    # 分配最近中心
    if use_cdist:
        dists = torch.cdist(pc, centroids, p=2)         # (N, M, K)
    else:
        # 广播算平方距离，省显存：||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        x2 = (pc ** 2).sum(dim=2, keepdim=True)         # (N, M, 1)
        y2 = (centroids ** 2).sum(dim=2).unsqueeze(1)   # (N, 1, K)
        xy = pc @ centroids.transpose(1, 2)             # (N, M, K)
        dists = x2 + y2 - 2.0 * xy                      # (N, M, K)  # 非负但可能有微小负值
    assign_idx = dists.argmin(dim=2)                    # (N, M) in [0..K-1]

    # 构建 LUT： 0 -> 0,  j+1 -> assign_idx[j] + 1
    LUT = torch.zeros((N, M + 1), dtype=torch.long, device=device)
    LUT[:, 1:] = assign_idx + 1

    return assign_idx, LUT


@torch.no_grad()
def remap_contact_state(s_c: torch.Tensor, LUT: torch.Tensor):
    """
    多次调用，把 contact state 从 0..M 转成 0..K（0 保持 0）
    Args:
        s_c: (N, L) long     —— 0..M（0=无接触，1..M=原点索引+1）
        LUT: (N, M+1) long   —— 来自 build_group_lut

    Returns:
        s_c_grouped: (N, L) long —— 0..K（0=无接触，1..K=簇索引+1）
    """
    assert s_c.ndim == 2 and LUT.ndim == 2 and s_c.size(0) == LUT.size(0)
    N, L = s_c.shape
    M = LUT.size(1) - 1
    # 直接查表：按行 gather
    s_c_clamped = s_c.clamp_min(0).clamp_max(M)
    s_c_grouped = LUT.gather(dim=1, index=s_c_clamped)
    return s_c_grouped


@torch.jit.script
def reciprocal(x: torch.Tensor, eps: float = 0.0, scale: float = 1.0, negate: bool = False) -> torch.Tensor:
    """Computes the reciprocal of a tensor element-wise."""
    if negate:
        return -scale / (torch.abs(x) + eps)
    else:
        return scale / (torch.abs(x) + eps)


@torch.jit.script
def copysign(a: float, b: torch.Tensor) -> torch.Tensor:
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def get_euler_xyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_apply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def broadcast_quat_and_vec(q: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert q.size(-1) == 4 and v.size(-1) == 3
    assert q.device == v.device
    qshape, vshape = q.shape[:-1], v.shape[:-1]
    shape = torch.broadcast_shapes(qshape, vshape)
    q = q.broadcast_to(shape + (4,))
    v = v.broadcast_to(shape + (3,))
    return q, v


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q, v = broadcast_quat_and_vec(q, v)
    shape = v.shape
    return _quat_rotate(q.reshape(-1, 4), v.reshape(-1, 3)).reshape(shape)


@torch.jit.script
def quat_mul(a, b):
    a, b = torch.broadcast_tensors(a, b)
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def orientation_dis(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


@torch.jit.script
def quat_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    # 2 * torch.acos(torch.abs(mul[:, -1]))
    # return 2.0 * torch.asin(torch.clamp(torch.norm(torch.abs(mul[:, 0:3]), p=2, dim=-1), max=1.0))
    return 2.0 * torch.asin(torch.clamp(torch.norm(mul[:, 0:3], p=2, dim=-1), max=1.0))

@torch.jit.script
def quat_diff_rad_normalized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    # Normalize quaternions to ensure they're unit quaternions
    a_norm = a / torch.norm(a, p=2, dim=-1, keepdim=True)
    b_norm = b / torch.norm(b, p=2, dim=-1, keepdim=True)
    
    # Compute relative rotation quaternion
    b_conj = quat_conjugate(b_norm)
    mul = quat_mul(a_norm, b_conj)
    
    # Use scalar part to compute angle (handles negative scalar correctly)
    w = mul[..., 3]
    w_abs = torch.abs(w)
    w_clamped = torch.clamp(w_abs, 0.0, 1.0)  # Avoid numerical issues
    angle = 2 * torch.acos(w_clamped)
    
    return angle

@torch.jit.script
def quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to 6D representation.
    
    Args:
        q: Quaternion, shape (N, ..., 4) in [x, y, z, w] format
    Returns:
        6D representation, shape (N, ..., 6)
    """
    original_shape = q.shape[:-1]
    q_flat = q.view(-1, 4)
    
    x, y, z, w = q_flat[:, 0], q_flat[:, 1], q_flat[:, 2], q_flat[:, 3]
    
    # Compute first two columns of rotation matrix
    col1_x = 1.0 - 2.0 * (y * y + z * z)
    col1_y = 2.0 * (x * y + w * z)
    col1_z = 2.0 * (x * z - w * y)
    
    col2_x = 2.0 * (x * y - w * z)
    col2_y = 1.0 - 2.0 * (x * x + z * z)
    col2_z = 2.0 * (y * z + w * x)
    
    # Stack results and reshape to original dimensions + 6
    result_flat = torch.stack([col1_x, col1_y, col1_z, col2_x, col2_y, col2_z], dim=-1)
    return result_flat.view(original_shape + (6,))

@torch.jit.script
def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Normalizes a given input tensor to a range of [-1, 1].

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Normalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return 2 * (x - offset) / (upper - lower)


@torch.jit.script
def normalize(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Same as `scale_transform` but with a different name."""
    return scale_transform(x, lower, upper)


@torch.jit.script
def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Denormalizes a given input tensor from range of [-1, 1] to (lower, upper).

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Denormalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset


@torch.jit.script
def denormalize(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Same as `unscale_transform` but with a different name."""
    return unscale_transform(x, lower, upper)


@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Clamps a given input tensor to (lower, upper).

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Clamped transform of the tensor. Shape (N, dims)
    """
    return torch.max(torch.min(x, upper), lower)


@torch.jit.script
def random_xy_circle(
    num: int, max_com_distance_to_center: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns sampled uniform positions in circle (https://stackoverflow.com/a/50746409)"""
    # sample radius of circle
    radius = torch.sqrt(torch.rand(num, dtype=torch.float, device=device))
    radius *= max_com_distance_to_center
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)
    # x,y-position of the cube
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    return x, y


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def torch_rand_minmax(lower, upper, shape, device):
    return (upper - lower) * torch.rand(shape, device=device) + lower


@torch.jit.script
def random_z(num: int, min_height: float, max_height: float, device: torch.device) -> torch.Tensor:
    """Returns sampled height of the goal object."""
    z = torch.rand(num, dtype=torch.float, device=device)
    z = (max_height - min_height) * z + min_height

    return z


@torch.jit.script
def default_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns identity rotation transform."""
    quat = torch.zeros((num, 4), dtype=torch.float, device=device)
    quat[..., -1] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns sampled rotation in 3D as quaternion.

    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
    """
    # sample random orientation from normal distribution
    quat = torch.randn((num, 4), dtype=torch.float, device=device)
    # normalize the quaternion
    quat = F.normalize(quat, p=2.0, dim=-1, eps=1e-12)

    return quat


# @torch.jit.script
def random_position_within_dist(
    num: int, device: torch.device, base: torch.Tensor, max_dist: float, min_z: torch.Tensor
):
    """Generates random positions within max_dist of base."""
    rand = 2 * torch.rand((num, 3), dtype=torch.float, device=device) - 1
    base += rand * max_dist
    base[:, 2] = torch.max(base[:, 2], min_z)

    return base.clone()


@torch.jit.script
def random_orientation_within_angle(num: int, device: torch.device, base: torch.Tensor, max_angle: float):
    """Generates random quaternions within max_angle of base
    Ref: https://math.stackexchange.com/a/3448434
    """
    quat = torch.zeros((num, 4), dtype=torch.float, device=device)

    rand = torch.rand((num, 3), dtype=torch.float, device=device)

    c = torch.cos(rand[:, 0] * max_angle)
    n = torch.sqrt((1.0 - c) / 2.0)

    quat[:, 3] = torch.sqrt((1 + c) / 2.0)
    quat[:, 2] = (rand[:, 1] * 2.0 - 1.0) * n
    quat[:, 0] = (torch.sqrt(1 - quat[:, 2] ** 2.0) * torch.cos(2 * np.pi * rand[:, 2])) * n
    quat[:, 1] = (torch.sqrt(1 - quat[:, 2] ** 2.0) * torch.sin(2 * np.pi * rand[:, 2])) * n

    # floating point errors can cause it to  be slightly off, re-normalise
    quat = F.normalize(quat, p=2.0, dim=-1, eps=1e-12)

    return quat_mul(quat, base)


@torch.jit.script
def random_angular_vel(num: int, device: torch.device, magnitude_stdev: float) -> torch.Tensor:
    """Samples a random angular velocity with standard deviation `magnitude_stdev`"""

    axis = torch.randn((num, 3), dtype=torch.float, device=device)
    axis /= torch.norm(axis, p=2, dim=-1).view(-1, 1)
    magnitude = torch.randn((num, 1), dtype=torch.float, device=device)
    magnitude *= magnitude_stdev
    return magnitude * axis


@torch.jit.script
def random_yaw_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns sampled rotation around z-axis."""
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)


@torch.jit.script
def mirror_yaw_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns sampled rotation around z-axis."""
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = np.pi * torch.ones(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)


@torch.jit.script
def transformation_multiply(
    quat1: torch.Tensor, pos1: torch.Tensor, quat2: torch.Tensor, pos2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multiply two transformations.

    Args:
        quat1: Quaternion of the first transformation.
        pos1: Position of the first transformation.
        quat2: Quaternion of the second transformation.
        pos2: Position of the second transformation.

    Returns:
        The quaternion and position of the resulting transformation.
    """
    quat1, quat2 = torch.broadcast_tensors(quat1, quat2)
    pos1, pos2 = torch.broadcast_tensors(pos1, pos2)
    return quat_mul(quat1, quat2), quat_apply(quat1, pos2) + pos1


@torch.jit.script
def transformation_inverse(quat: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert a transformation.

    Args:
        quat: Quaternion of the transformation.
        pos: Position of the transformation.

    Returns:
        The quaternion and position of the inverted transformation.
    """
    quat = quat_conjugate(quat)
    return quat, -quat_apply(quat, pos)


@torch.jit.script
def transformation_apply(quat: torch.Tensor, pos: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a transformation to a vector.

    Args:
        quat: Quaternion of the transformation.
        pos: Position of the transformation.
        vec: Vector to transform.

    Returns:
        The transformed vector.
    """
    pos, vec = torch.broadcast_tensors(pos, vec)
    quaternion_shape = pos.shape[:-1] + (4,)
    quat = torch.broadcast_to(quat, quaternion_shape)
    return quat_apply(quat, vec) + pos


def quat_from_two_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion rotating vector a to vector b. a,b: (..., 3)/(3,). Return (..., 4)/(4,).
    """
    single = (a.dim() == 1)
    if single:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    a_n = a / (torch.norm(a, dim=-1, keepdim=True).clamp_min(1e-9))
    b_n = b / (torch.norm(b, dim=-1, keepdim=True).clamp_min(1e-9))
    v = torch.cross(a_n, b_n, dim=-1)
    c = (a_n * b_n).sum(dim=-1, keepdim=True)
    w = 1.0 + c

    opp = (w.squeeze(-1) < 1e-6)
    if opp.any(): # case: opposite vectors, pick 180deg about an arbitrary orthogonal axis
        axis = torch.zeros_like(a_n)
        # choose axis orthogonal to a_n
        idx = (torch.abs(a_n[:, 0]) < 0.9).float().unsqueeze(-1)
        ref = torch.where(idx > 0, torch.tensor([1.0, 0.0, 0.0], device=a.device).expand_as(a_n),
                            torch.tensor([0.0, 1.0, 0.0], device=a.device).expand_as(a_n))
        axis_opp = torch.cross(a_n, ref, dim=-1)
        axis_opp = axis_opp / (torch.norm(axis_opp, dim=-1, keepdim=True).clamp_min(1e-9))
        # 180deg rotation quaternion
        q_opp = torch.cat([axis_opp, torch.zeros(axis_opp.size(0), 1, device=a.device)], dim=-1)
        q = torch.cat([v, w], dim=-1)
        q[opp] = q_opp[opp]
    else:
        q = torch.cat([v, w], dim=-1)
    q = q / (torch.norm(q, dim=-1, keepdim=True).clamp_min(1e-9))
    if single:
        q = q.squeeze(0)
    return q