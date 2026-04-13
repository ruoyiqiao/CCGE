# Task Environments and CCGE Implementation

This directory contains Isaac Gym task environments and the core CCGE (Contact Coverage-Guided Exploration)
reward logic. It is the main place to understand how observations, rewards, and curiosity signals are
computed for each manipulation task.

## Entry Points and Task Registration

- `__init__.py` registers each task class into `isaacgym_task_map` and exposes `load_isaacgym_env()`.
- Tasks are created by `create_env_from_config()` with Hydra configs from `cfg/task/`.

## Tasks at a Glance

| Family | Files | Notes |
|---|---|---|
| XArm + Allegro (arm + hand) | `allegro_singulation.py`, `allegro_table_top.py`, `allegro_cube_in_box.py` | Arm + hand tasks (singulation, table-top, cube-in-box). |
| XArm + LEAP (arm + hand) | `leap_singulation.py`, `leap_table_top.py`, `leap_cube_in_box.py` | Arm + hand LEAP tasks analogous to the Allegro set. |
| In-hand Allegro | `inhand_manipulation_allegro.py` | Hand-only in-hand manipulation (no arm). |
| In-hand LEAP | `inhand_manipulation_leap.py` | Hand-only in-hand manipulation (no arm). |
| Bimanual LEAP | `leap_bimanual.py`, `leap_bimanual_board_lift.py` | Two-hand coordination tasks and board-lift task. |
| Planar Push | `push_2d_task.py` | 2D push task with simplified planar contact. |

## Core Concepts Implementation


**State Feature Discretization**

- `state_feature_bank.py` defines how state features are discretized for novelty and
  running-max tracking.
- `LearnedHashStateBank` uses an autoencoder + SimHash to map continuous state features
  into discrete state IDs.
- `PushBox2DStateBank` provides a predefined discretization for the planar push task.

**Datasets and Geometry**

- `dataset.py` loads object point clouds, normals, and metadata used by tasks.
- `torch_utils.py` and `isaacgym_utils.py` provide math utilities and Isaac Gym helpers
  for transforms, normals, and tensor ops.


## Integrating CCGE into a New Task

For the architecture overview and required-data summary, see the [main README](../../README.md#ccge-reward-architecture). Working examples: `allegro_singulation.py` and `leap_singulation.py` (search for `compute_curiosity_informed_reach_reward`).

### Step-by-step

**1. Define keypoints** — List URDF link names and optional per-link offsets:

```python
_keypoints_info_path = "assets/urdf/my_hand_keypoints.json"
```

```json
{"thumb_tip": [[0,0,0]], "index_tip": [[0,0,0]], "palm_link": [[0.02,0,0], [-0.02,0,0]]}
```

In `_define_robot_asset`, flatten into `keypoint_indices` (list of rigid-body indices) and `keypoint_offset` tensor `(1, L, 3)`.

**2. Update keypoints each frame** — In `_refresh_sim_tensors`:

```python
self.keypoint_positions = self.hand_rigid_body_positions[:, self.keypoint_indices, :]
self.keypoint_orientations = self.hand_rigid_body_orientations[:, self.keypoint_indices, :]
self.keypoint_positions_with_offset = self.keypoint_positions + \
    quat_apply(self.keypoint_orientations, self.keypoint_offset.expand(self.num_envs, -1, -1))
```

**3. Compute contact mask** — Combine distance and force thresholds:

```python
dists = torch.cdist(self.keypoint_positions_with_offset, pcl_world)  # (N, L, P)
near = dists.min(dim=2).values < self.contact_distance_threshold
force = self.keypoint_contact_forces.norm(dim=-1) > self.contact_force_threshold
self.keypoint_contact_mask = near & force  # (N, L)
```

**4. Create `CuriosityRewardManager`** — In `__init__`:

```python
from .curiosity_reward_manager import CuriosityRewardManager

self.reach_curiosity_mgr = CuriosityRewardManager(
    num_keypoints=L, num_object_points=M,
    canonical_pointcloud=canonical_pcd,       # (M, 3)
    canonical_normals=canonical_normals,       # (M, 3)
    device=self.device,
    # Clustering
    cluster_k=32, use_normal_in_clustering=True, normal_weight=0.5,
    # Potential field
    potential_kernel="exponential", kernel_param=0.03,
    novelty_decay="sqrt", novelty_decay_rate=2.0, use_potential_shaping=True,
    # Masking
    mask_backface_points=True, mask_palm_inward_points=True,
    # State bank
    num_envs=self.num_envs, state_type="hash",
    state_feature_dim=96, num_key_states=32,
    state_counter_mode="cluster", state_running_max_mode="state",
    # Hash AE (when state_type="hash")
    hash_code_dim=256, hash_noise_scale=0.3, hash_lambda_binary=10.0,
    hash_ae_lr=3e-4, hash_ae_steps=5, hash_ae_update_freq=16,
)
```

See `CuriosityRewardManager.__init__` for the full parameter list.

**5. Compute reward** — Each step during training:

```python
def compute_curiosity_informed_reach_reward(self):
    canonical = self.reach_curiosity_mgr.canonical_pointcloud  # (M, 3)
    pc_world = quat_apply(
        self.object_root_orientations[:, None].expand(-1, canonical.shape[0], -1),
        canonical[None].expand(self.num_envs, -1, -1),
    ) + self.object_root_positions[:, None]                    # (N, M, 3)

    state_feats = self.reach_curiosity_mgr.build_state_features_from_world_pc(pc_world)
    contact_indices = torch.cdist(
        self.keypoint_positions_with_offset, pc_world
    ).argmin(dim=-1)                                           # (N, L)

    reward, info = self.reach_curiosity_mgr.compute_reward_from_canonical(
        object_positions=self.object_root_positions,
        object_orientations=self.object_root_orientations,
        keypoint_positions_world=self.keypoint_positions_with_offset,
        contact_indices=contact_indices,
        contact_mask=self.keypoint_contact_mask,
        task_contact_satisfied=self.contact_satisfied,
        keypoint_palm_dirs_world=palm_dir_world,               # (N, L, 3)
        state_features_world=state_feats,
    )
    self.reach_curiosity_rew_scaled = reward
    self.contact_coverage_rew_scaled = info["cluster_novelty_reward"]
```

**6. Wire into `compute_reward`:**

```python
if self.training:
    self.compute_curiosity_informed_reach_reward()
if "energy_reach" in self.reward_type:
    self.rew_buf += self.reach_curiosity_rew_scaled
if "contact_coverage" in self.reward_type:
    self.rew_buf += self.contact_coverage_rew_scaled
```

**7. Handle resets** — In `reset_idx`:

```python
self.reach_curiosity_mgr.ensure_running_max_buffers(self.num_envs)
self.reach_curiosity_mgr.reset_running_max_buffers(env_ids)
```

### Config (YAML)

```yaml
env:
  rewardType: "target+energy_reach+contact_coverage"
  stateType: "hash"             # "hash" or "predefined"
  stateFeatureDim: 96           # 32 pts × 3; null to disable
  numKeyStates: 32
  stateRunningMaxMode: "state"  # "state" or "global"
  maskBackfacePoints: true
  maskPalmInwardPoints: true
  useNormalInClustering: true
```

### Key Files

| File | Role |
|------|------|
| `curiosity_reward_manager.py` | Potential field, contact bonus, running-max logic |
| `state_feature_bank.py` | `LearnedHashStateBank` / `PushBox2DStateBank` |
| `allegro_singulation.py` | Full CCGE integration example |
| `leap_singulation.py` | Full CCGE integration example |

## Common Configuration Pattern

Task behavior is controlled through `task.env.*` keys in `cfg/task/*.yaml` and
Hydra overrides from launch scripts. The typical flow is:

1. Task config loads parameters like observation space, reward terms, and curiosity options.
2. Task class constructs the simulation, assets, and buffers.
3. `compute_observations()` and `compute_reward()` use those configs each step.

## Adding a New Task

1. Create a new task file in `src/tasks/` and implement the task class.
2. Register it in `src/tasks/__init__.py` (add it to `isaacgym_task_map`).
3. Add a task config in `cfg/task/` and a training config in `cfg/train/`.
4. Optionally add a `train_*.sh` launcher script for reproducibility.
