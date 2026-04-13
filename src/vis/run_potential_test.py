import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ----------------------------
# 1. 参数设置
# ----------------------------
# np.random.seed(42)
delta = 2e-2          # inverse kernal(/epsilon)
sigma = 2e-2         # exp kernel
alpha = 1.0           # scalar
gamma = 0.99          # RL discount (for shaping, not used in sim)
step_size = 2e-3    # agent 每步移动距离
contact_thresh = 0.005 # 接触阈值(not used for now)
max_steps = 20000
decay_mode = "exp"   # or "exp"

# ----------------------------
# 2. 创建立方体表面点云（均匀采样）
# ----------------------------

def create_cube_surface_points(n_per_face=200):
    points = []
    # 每个面：固定一个坐标为 ±0.5，其余两个在 [-0.5, 0.5]
    for axis in range(3):
        for sign in [-1, 1]:
            coords = np.random.uniform(-0.5, 0.5, size=(n_per_face, 3))
            coords[:, axis] = sign * 0.5
            points.append(coords)
    return np.vstack(points)

def create_nonuniform_cube_surface():
    points = []
    for axis in range(3):
        for sign in [-1, 1]:
            # 给 +Z 面（axis=2, sign=1）更多点
            if axis == 2 and sign == 1:
                n = 500  # 高密度
            else:
                n = 100  # 低密度
            coords = np.random.uniform(-0.5, 0.5, size=(n, 3))
            coords[:, axis] = sign * 0.5
            points.append(coords)
    return np.vstack(points)




# surface_points = create_cube_surface_points(n_per_face=300)
# surface_points = np.vstack([surface_points, np.random.uniform(-0.5,0.5,(500,3))])  
surface_points = create_nonuniform_cube_surface()

# surface_points = create_cube_surface_points(n_per_face=150)  # ~900 points
counts = np.zeros(len(surface_points)) 

# Build KDTree for fast nearest neighbor during contact
surface_tree = cKDTree(surface_points)

# ----------------------------
# 3. 势能场函数 Φ(x)
# ----------------------------
def compute_potential(x, surface_points, counts, delta=1e-3, simga=2e-2, alpha=1.0):
    """
    x: (3,) array
    returns scalar potential Φ(x)
    """
    diffs = surface_points - x  # (N, 3)
    dists = np.linalg.norm(diffs, axis=1)  # (N,)

    # K = 1.0 / (dists + delta)
    K = np.exp(-dists / simga)

    if decay_mode == "sqrt":
        q = alpha / np.sqrt(counts * 10 + 1.0)
    else:  # exp
        q = alpha * np.exp(-0.5 * counts)
    return np.sum(q * K)


def compute_gradient(x, surface_points, counts, eps=1e-5):
    """
    Numerical gradient of Φ at x via central difference.
    Returns (3,) gradient vector.
    """
    grad = np.zeros(3)
    for i in range(3):
        dx = np.zeros(3); dx[i] = eps
        phi_plus = compute_potential(x + dx, surface_points, counts, delta, sigma, alpha)
        phi_minus = compute_potential(x - dx, surface_points, counts, delta, sigma, alpha)
        grad[i] = (phi_plus - phi_minus) / (2 * eps)
    return grad

def compute_gradient_analytical(x, surface_points, counts, delta=1e-2, alpha=1.0, eps_norm=1e-8):
    """
    Analytical gradient of Φ(x) = sum_i q_i * exp(-||x - c_i|| / delta)
    
    Returns:
        grad: (3,) numpy array
    """
    diffs = x - surface_points  # (N, 3) — note: x - c_i
    dists = np.linalg.norm(diffs, axis=1)  # (N,)
    
    # Avoid division by zero
    dists_safe = np.maximum(dists, eps_norm)
    
    # Compute q_i
    if decay_mode == "sqrt":
        q = alpha / np.sqrt(counts * 10 + 1.0)
    else:  # exp
        q = alpha * np.exp(-0.5 * counts)
    
    # Kernel value
    K = np.exp(-dists / delta)  # (N,)
    
    # Gradient contribution per point: - (q_i * K_i / delta) * (x - c_i) / ||x - c_i||
    # Shape: (N, 3)
    grad_per_point = - (q * K / delta)[:, None] * (diffs / dists_safe[:, None])
    
    # Sum over all points
    grad = np.sum(grad_per_point, axis=0)
    return grad


# ----------------------------
# Helper: Cube collision handling
# ----------------------------
def is_inside_cube(pos, half_size=0.5):
    return np.all(np.abs(pos) < half_size)

def clamp_to_cube_surface(pos, half_size=0.5):
    if not is_inside_cube(pos, half_size):
        return pos.copy()
    # Project to nearest face
    abs_pos = np.abs(pos)
    axis = np.argmax(abs_pos)  # coordinate closest to surface
    clamped = pos.copy()
    clamped[axis] = np.sign(pos[axis]) * half_size
    return clamped

# simulate

agent_pos = np.array([0.6, 0.6, 0.6]) #TODO(ruoyi): further distance make zero gradient disappear(larger simga? distance clip?)
trajectory = [agent_pos.copy()]

for step in range(max_steps):
    grad = compute_gradient(agent_pos, surface_points, counts)
    if np.linalg.norm(grad) < 1e-6:
        break

    direction = grad / np.linalg.norm(grad)
    proposed = agent_pos + step_size * direction
    agent_pos = clamp_to_cube_surface(proposed, half_size=0.5)
    agent_pos = proposed
    trajectory.append(agent_pos.copy())

    # Contact check (only if near surface)
    dists, idxs = surface_tree.query(agent_pos, k=1)
    # if dists < contact_thresh: # FIX: add this make agent stuck
    counts[idxs] += 1

print(f"Simulation done. Final counts: min={counts.min():.1f}, max={counts.max():.1f}")


norm_counts = (counts - counts.min()) / (counts.max() - counts.min() + 1e-8)
colors = plt.cm.coolwarm(norm_counts)[:, :3]  # RGB

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(surface_points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 轨迹线
traj = np.array(trajectory)
lines = [[i, i+1] for i in range(len(traj)-1)]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(traj)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.paint_uniform_color([0, 1, 0])  # green


mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Exploration with Potential Field")
vis.add_geometry(pcd)
vis.add_geometry(line_set)
vis.add_geometry(mesh_frame)

opt = vis.get_render_option()
opt.point_size = 12.0
opt.line_width = 4.0
opt.background_color = np.asarray([1,1,1])
vis.run()
vis.destroy_window()



def plot_potential_slice(z_slice=0.0, xlim=(-1,1), ylim=(-1,1), resolution=150):
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    Phi = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            pt = np.array([X[i,j], Y[i,j], z_slice])
            Phi[i,j] = compute_potential(pt, surface_points, counts, delta, alpha)

    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, Phi, levels=50, cmap='viridis')
    plt.colorbar(label='Potential Φ')

    # 增大散点大小：s=20（默认通常是 5），按需调整
    plt.scatter(surface_points[:,0], surface_points[:,1], c=norm_counts, cmap='coolwarm', s=20, edgecolor='k', linewidth=0.2)

    traj_2d = np.array(trajectory)[:, :2]
    plt.plot(traj_2d[:,0], traj_2d[:,1], 'g-', linewidth=2, label='Agent Trajectory')
    plt.title(f'Potential Field Slice at z={z_slice}')
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend()
    plt.axis('equal')
    plt.show()


# plot_potential_slice(z_slice=0.0)


print(np.mean(counts != 0))