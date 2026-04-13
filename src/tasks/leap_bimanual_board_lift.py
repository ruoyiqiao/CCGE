import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from isaacgym import gymapi, gymtorch
from torch import LongTensor, Tensor

from .leap_bimanual import LeapBimanualManipulationArti
from .dataset import BoxGridDataset
from .torch_utils import quat_apply, quat_conjugate, quat_mul, quat_rotate

from .curiosity_reward_manager import CuriosityRewardManager


class LeapBimanualBoardLift(LeapBimanualManipulationArti):

    _right_hand_links: List[str] = [
        "palm_lower",
        "mcp_joint", "pip", "dip", "fingertip",
        "mcp_joint_2", "pip_2", "dip_2", "fingertip_2",
        "mcp_joint_3", "pip_3", "dip_3", "fingertip_3",
        "thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip",
        
        "index_tip_head", "middle_tip_head", "ring_tip_head", "thumb_tip_head",
    ]
    _left_hand_links: List[str] = [
        "palm_lower_left",
        "mcp_joint", "pip", "dip", "fingertip",
        "mcp_joint_2", "pip_2", "dip_2", "fingertip_2",
        "mcp_joint_3", "pip_3", "dip_3", "fingertip_3",
        "thumb_left_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip",
        
        "index_tip_head", "middle_tip_head", "ring_tip_head", "thumb_tip_head",
    ]
    _left_xarm_init_dof_positions: Dict[str, float] = {
        "joint1": 0.0,
        "joint2": -1.0,
        "joint3": -0.5,
        "joint4": 0.0,
        "joint5": 1.57,
        "joint6": 0.0,
    }

    _right_xarm_init_dof_positions: Dict[str, float] = {
        "joint1": 0.0,
        "joint2": -1.0,
        "joint3": -0.5,
        "joint4": 0.0,
        "joint5": 1.57,
        "joint6": 0.0,
    }

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        # self._apply_hand_base_pose_overrides(cfg)
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        self.lowest_point_reward_scale = float(self.env_cfg.get("lowestPointRewardScale", 300.0))
        self.board_lift_target_height = float(self.env_cfg.get("boardLiftTargetHeight", 0.05))
        self.board_lift_success_height = float(self.env_cfg.get("boardLiftSuccessHeight", 0.50))
        self.board_lowest_point_max = torch.full((self.num_envs,), self.table_height, device=self.device, dtype=torch.float32)


        self._init_curiosity_managers()

    @staticmethod
    def _apply_hand_base_pose_overrides(cfg: Dict[str, Any]) -> None:
        env_cfg = cfg.get("env", {})
        overrides = env_cfg.get("handBasePoseOverrides", {})
        hands = env_cfg.get("hands", {})
        for side in ("right", "left"):
            if side not in overrides or side not in hands:
                continue
            side_override = overrides[side]
            init_pose = hands[side].setdefault("initPose", {})
            if "position" in side_override:
                init_pose["position"] = side_override["position"]
            if "orientation" in side_override:
                init_pose["orientation"] = side_override["orientation"]

    def _define_table(self) -> Dict[str, Any]:
        """Create a support table guaranteed to be strictly larger than the board in x/y."""
        obj_cfg = self.object_cfgs[0]
        board_size = obj_cfg.get("boxSize", [0.18, 0.12, 0.01])
        board_x, board_y, _ = float(board_size[0]), float(board_size[1]), float(board_size[2])

        table_cfg = self.env_cfg.get("table", {})
        self.table_x_length = float(table_cfg.get("size", [0.24, 0.20, 0.2])[0])
        self.table_y_length = float(table_cfg.get("size", [0.24, 0.20, 0.2])[1])
        self.table_thickness = float(table_cfg.get("size", [0.24, 0.20, 0.2])[2])
        self.table_height = float(table_cfg.get("height", 0.50))

        if self.table_x_length <= board_x or self.table_y_length <= board_y:
            raise ValueError(
                f"Table xy must be strictly larger than board xy; got table=({self.table_x_length}, {self.table_y_length}), "
                f"board=({board_x}, {board_y})"
            )

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        table_asset = self.gym.create_box(
            self.sim, self.table_x_length, self.table_y_length, self.table_thickness, asset_options
        )

        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for shape in rigid_shape_props:
            shape.friction = 1.0
            shape.restitution = 0.0
        self.gym.set_asset_rigid_shape_properties(table_asset, rigid_shape_props)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, self.table_height - self.table_thickness / 2.0)

        return {
            "asset": table_asset,
            "pose": pose,
            "name": "table",
            "num_rigid_bodies": self.gym.get_asset_rigid_body_count(table_asset),
            "num_rigid_shapes": self.gym.get_asset_rigid_shape_count(table_asset),
            "num_dofs": 0,
        }

    def _define_object(self) -> Dict[str, Any]:
        """Create rigid board objects with gym.create_box instead of articulated URDF loading."""
        config = {"warehouse": [], "poses": [], "count": len(self.object_cfgs), "num_rigid_bodies": 0, "num_rigid_shapes": 0}
        max_rigid_bodies, max_rigid_shapes = 0, 0

        for obj_cfg in self.object_cfgs:
            use_box = bool(obj_cfg.get("createBox", False))
            if not use_box:
                raise ValueError("LeapBimanualBoardLift expects object cfg with createBox=true.")

            box_size = obj_cfg.get("boxSize", [0.18, 0.12, 0.01])
            sx, sy, sz = float(box_size[0]), float(box_size[1]), float(box_size[2])

            asset_options = gymapi.AssetOptions()
            asset_options.density = float(obj_cfg.get("density", 500.0))
            asset_options.fix_base_link = False
            asset_options.disable_gravity = False
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.thickness = 0.001
            asset = self.gym.create_box(self.sim, sx, sy, sz, asset_options)

            rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
            for shape in rigid_shape_props:
                shape.friction = float(obj_cfg.get("friction", 1.0))
                shape.restitution = 0.0
            self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

            num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
            num_rigid_shapes = self.gym.get_asset_rigid_shape_count(asset)
            max_rigid_bodies = max(max_rigid_bodies, num_rigid_bodies)
            max_rigid_shapes = max(max_rigid_shapes, num_rigid_shapes)

            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(*obj_cfg["pose"][:3])
            print(obj_cfg["pose"][3:])
            obj_pose.r = gymapi.Quat(*obj_cfg["pose"][3:])
            obj_config = {
                "name": obj_cfg.get("name", "board"),
                "asset": asset,
                "asset_root": "",
                "asset_file": "",
                "pose": obj_pose,
                "color": obj_cfg.get("color", [0.6, 0.6, 0.6]),
                "track": obj_cfg.get("track", True),
                "num_dofs": 0,
                "num_rigid_bodies": num_rigid_bodies,
                "num_rigid_shapes": num_rigid_shapes,
                "box_size": [sx, sy, sz],
            }
            config["warehouse"].append(obj_config)
            config["poses"].append(obj_pose)

        config["num_rigid_bodies"] = max_rigid_bodies
        config["num_rigid_shapes"] = max_rigid_shapes
        return config

    def _create_envs(self, num_envs: int, spacing: float):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(num_envs))

        self.envs: List[gymapi.Env] = []
        self.hand_actor_handles: Dict[str, List[int]] = {"right": [], "left": []}
        self.hand_actor_handle_sim: Dict[str, torch.Tensor] = {
            "right": torch.zeros((self.num_envs,), dtype=torch.long, device=self.device),
            "left": torch.zeros((self.num_envs,), dtype=torch.long, device=self.device),
        }
        self.hand_actor_ids_env: Dict[str, int] = {"right": 0, "left": 0}
        self.hand_dof_starts: Dict[str, int] = {"right": 0, "left": 0}
        self.hand_wrist_rigid_body_id: Dict[str, int] = {"right": 0, "left": 0}
        self.forearm_rigid_body_id: Dict[str, int] = {"right": 0, "left": 0}
        self.object_actor_handles: List[List[int]] = []
        self.tracked_object_actor_ids_env: List[int] = []
        self.object_name_to_handles: List[Dict[str, int]] = [dict() for _ in range(num_envs)]

        self.gym_assets["current"]["hand_right"] = self._define_hand("right")
        self.gym_assets["current"]["hand_left"] = self._define_hand("left")
        self.gym_assets["current"]["table"] = self._define_table()
        self.gym_assets["current"]["objects"] = self._define_object()

        self.object_dofs_per_env = sum(cfg["num_dofs"] for cfg in self.gym_assets["current"]["objects"]["warehouse"])
        self.object_dof_per_env: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        self.object_handles_env: List[List[int]] = [[] for _ in range(num_envs)]
        self.object_handles_sim: List[List[int]] = [[] for _ in range(num_envs)]

        max_aggregate_bodies, max_aggregate_shapes = self.compute_maximum_aggregate_bodies_and_shapes()
        for env_id in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)
            self.aggregate_tracker.reset()

            if self.aggregate_mode != 0:
                self.gym.begin_aggregate(env, max_aggregate_bodies, max_aggregate_shapes, True)

            self._create_sim_actor(
                env,
                self.gym_assets["current"]["table"],
                env_id,
                name="table",
                pose=self.gym_assets["current"]["table"]["pose"],
                color=gymapi.Vec3(0.0, 0.0, 0.0),
            )

            self.hand_actor_handles["right"].append(self._spawn_hand(env, env_id, "right"))
            self.hand_actor_handles["left"].append(self._spawn_hand(env, env_id, "left"))

            obj_handles_env: List[int] = []
            for obj_cfg in self.gym_assets["current"]["objects"]["warehouse"]:
                index, handle = self._create_sim_actor(
                    env,
                    obj_cfg,
                    env_id,
                    name=obj_cfg["name"],
                    actor_handle=True,
                )
                obj_handles_env.append(handle)
                self.object_name_to_handles[env_id][obj_cfg["name"]] = handle
                self.object_handles_env[env_id].append(handle)
                self.object_handles_sim[env_id].append(index)

            if self.aggregate_mode != 0:
                self.gym.end_aggregate(env)

        self.object_handles_env = torch.tensor(self.object_handles_env, device=self.device, dtype=torch.int32)
        self.object_handles_sim = torch.tensor(self.object_handles_sim, device=self.device, dtype=torch.int32)

        self.left_hand_arm_dof_start = self.gym.get_actor_dof_index(self.envs[0], self.hand_actor_handles["left"][0], 0, gymapi.DOMAIN_ENV)
        self.right_hand_arm_dof_start = self.gym.get_actor_dof_index(self.envs[0], self.hand_actor_handles["right"][0], 0, gymapi.DOMAIN_ENV)
        self.left_hand_arm_dof_end = self.left_hand_arm_dof_start + self.gym_assets["current"]["hand_left"]["num_dofs"]
        self.right_hand_arm_dof_end = self.right_hand_arm_dof_start + self.gym_assets["current"]["hand_right"]["num_dofs"]
        self.left_hand_endeffector_index = self.gym.find_asset_rigid_body_index(self.gym_assets["current"]["hand_left"]["asset"], "link6")
        self.right_hand_endeffector_index = self.gym.find_asset_rigid_body_index(self.gym_assets["current"]["hand_right"]["asset"], "link6")
        self.left_arm_dof_slice = slice(self.left_hand_arm_dof_start, self.left_hand_arm_dof_start + 6)
        self.right_arm_dof_slice = slice(self.right_hand_arm_dof_start, self.right_hand_arm_dof_start + 6)
        self.left_hand_dof_slice = slice(self.left_hand_arm_dof_start + 6, self.left_hand_arm_dof_end)
        self.right_hand_dof_slice = slice(self.right_hand_arm_dof_start + 6, self.right_hand_arm_dof_end)
        self.left_hand_rigid_body_start = self.gym.get_actor_rigid_body_index(self.envs[0], self.hand_actor_handles["left"][0], 0, gymapi.DOMAIN_ENV)
        self.right_hand_rigid_body_start = self.gym.get_actor_rigid_body_index(self.envs[0], self.hand_actor_handles["right"][0], 0, gymapi.DOMAIN_ENV)
        self.left_hand_rigid_body_end = self.left_hand_rigid_body_start + self.gym_assets["current"]["hand_left"]["num_rigid_bodies"]
        self.right_hand_rigid_body_end = self.right_hand_rigid_body_start + self.gym_assets["current"]["hand_right"]["num_rigid_bodies"]

        if self.object_dofs_per_env > 0:
            self.object_dof_start = self.gym.get_actor_dof_index(self.envs[0], int(self.object_handles_env[0][0]), 0, gymapi.DOMAIN_ENV)
            self.object_dof_end = self.object_dof_start + self.object_dofs_per_env
        else:
            self.object_dof_start = self.left_hand_arm_dof_end
            self.object_dof_end = self.left_hand_arm_dof_end
        self.object_rigid_body_start = self.gym.get_actor_rigid_body_index(self.envs[0], int(self.object_handles_sim[0][0]), 0, gymapi.DOMAIN_ENV)
        self.object_rigid_body_end = self.object_rigid_body_start + self.gym_assets["current"]["objects"]["warehouse"][0]["num_rigid_bodies"]

    def _init_object_parts(self):
        obj_cfg = self.object_cfgs[0]
        size = obj_cfg.get("boxSize", [0.18, 0.12, 0.01])
        box_ds = BoxGridDataset(
            grid_rows=1,
            grid_cols=1,
            grid_layers=1,
            box_width=float(size[0]),
            box_depth=float(size[1]),
            box_height=float(size[2]),
            device=self.device,
            pcl_num=int(self.env_cfg.get("numObjectPoints", 1024)),
        )
        canonical_pc = box_ds._pointclouds[0].to(self.device)
        canonical_normals = box_ds._pointcloud_normals[0].to(self.device)
        self._object_parts = {
            "board": {
                "pointcloud": canonical_pc,
                "pointcloud_normals": canonical_normals,
                "rigid_body_indices": [0],
                "box_size": size,
            }
        }

    def _init_curiosity_managers(self):
        # One board curiosity manager per hand (both target the same rigid board)
        self._curiosity_managers = {"left": {}, "right": {}}
        canonical_pc = self._object_parts["board"]["pointcloud"]
        canonical_normals = self._object_parts["board"]["pointcloud_normals"]

        for hand_side in ("left", "right"):
            self._curiosity_managers[hand_side]["board"] = CuriosityRewardManager(
                num_keypoints=4,
                device=self.device,
                canonical_pointcloud=canonical_pc,
                k=32,
                cluster_k=32,
                max_clustering_iters=10,
                canonical_normals=canonical_normals,
                mask_backface_points=True,
                mask_palm_inward_points=True,
                use_normal_in_clustering=False,
                state_feature_dim=self.env_cfg.get("stateFeatureDim", None),
                num_key_states=int(self.env_cfg.get("numKeyStates", 32)),
                state_counter_mode=str(self.env_cfg.get("stateCounterMode", "cluster")),
                state_include_goal=bool(self.cfg["env"].get("stateIncludeGoal", False)),
                hash_code_dim=int(self.cfg["env"].get("hashCodeDim", 16)),
                hash_noise_scale=float(self.cfg["env"].get("hashNoiseScale", 0.3)),
                hash_lambda_binary=float(self.cfg["env"].get("hashLambdaBinary", 1.0)),
                hash_ae_lr=float(self.cfg["env"].get("hashAeLr", 3e-4)),
                hash_ae_steps=int(self.cfg["env"].get("hashAeSteps", 5)),
                hash_ae_update_freq=int(self.cfg["env"].get("hashAeUpdateFreq", 16)),
                hash_ae_num_minibatches=int(self.cfg["env"].get("hashAeNumMinibatches", 8)),
                hash_seed=int(self.cfg["env"].get("hashSeed", 0)),
                state_type=str(self.cfg["env"].get("stateType", "pcd")),
                state_running_max_mode=str(self.cfg["env"].get("stateRunningMaxMode", "state")),
                num_envs=self.num_envs,
            )

    def _update_parts_pointclouds(self):
        self._object_parts_world_pointclouds = {}
        board_local = self._object_parts["board"]["pointcloud"].unsqueeze(0).repeat(self.num_envs, 1, 1)
        board_q = self.object_root_orientations
        board_p = self.object_root_positions
        board_q_exp = board_q.unsqueeze(1).expand(-1, board_local.shape[1], -1)
        board_world = quat_apply(board_q_exp, board_local) + board_p.unsqueeze(1)
        self._object_parts_world_pointclouds["board"] = board_world
        self.board_lowest_point_world = board_world[..., 2].amin(dim=1)

    def _refresh_sim_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        left_forearm_rb_idx = self.forearm_rigid_body_id["right"]
        right_forearm_rb_idx = self.forearm_rigid_body_id["left"]
        self.right_eef_jacobian = self.jacobians_right[:, right_forearm_rb_idx - 1, :, :6]
        self.left_eef_jacobian = self.jacobians_left[:, left_forearm_rb_idx - 1, :, :6]

        root_states = self.root_states.view(self.num_envs, self.num_actors, 13)
        net_contact_forces = self.net_contact_forces.view(self.num_envs, self.num_rigid_bodies, 3)

        left_hand_slice = slice(self.left_hand_rigid_body_start, self.left_hand_rigid_body_end)
        right_hand_slice = slice(self.right_hand_rigid_body_start, self.right_hand_rigid_body_end)
        object_slice = slice(self.object_rigid_body_start, self.object_rigid_body_end)
        left_arm_links_slice = slice(self.left_hand_rigid_body_start, self.left_hand_rigid_body_start + 7)
        right_arm_links_slice = slice(self.right_hand_rigid_body_start, self.right_hand_rigid_body_start + 7)
        self.left_arm_contact_forces = net_contact_forces[:, left_arm_links_slice, :]
        self.right_arm_contact_forces = net_contact_forces[:, right_arm_links_slice, :]
        self.lh_contact_forces = net_contact_forces[:, left_hand_slice, :]
        self.rh_contact_forces = net_contact_forces[:, right_hand_slice, :]
        self.object_contact_forces = net_contact_forces[:, object_slice, :]
        self.left_fingertip_contact_forces = self.lh_contact_forces[:, self.left_fingertip_indices, :]
        self.right_fingertip_contact_forces = self.rh_contact_forces[:, self.right_fingertip_indices, :]
        self.left_keypoint_contact_forces = self.lh_contact_forces[:, self.left_keypoint_indices, :]
        self.right_keypoint_contact_forces = self.rh_contact_forces[:, self.right_keypoint_indices, :]
        self.table_contact_forces = net_contact_forces[:, self.table_rigid_body_indices[0], :]

        self.right_hand_root_positions = root_states[:, self.hand_actor_ids_env["right"], 0:3]
        self.right_hand_root_orientations = root_states[:, self.hand_actor_ids_env["right"], 3:7]
        self.right_hand_root_linear_velocities = root_states[:, self.hand_actor_ids_env["right"], 7:10]
        self.right_hand_root_angular_velocities = root_states[:, self.hand_actor_ids_env["right"], 10:13]
        self.left_hand_root_positions = root_states[:, self.hand_actor_ids_env["left"], 0:3]
        self.left_hand_root_orientations = root_states[:, self.hand_actor_ids_env["left"], 3:7]
        self.left_hand_root_linear_velocities = root_states[:, self.hand_actor_ids_env["left"], 7:10]
        self.left_hand_root_angular_velocities = root_states[:, self.hand_actor_ids_env["left"], 10:13]
        self.object_root_positions = root_states[:, self.object_handles_env[0], 0:3].view(self.num_envs, 3)
        self.object_root_orientations = root_states[:, self.object_handles_env[0], 3:7].view(self.num_envs, 4)
        self.object_root_linear_velocities = root_states[:, self.object_handles_env[0], 7:10].view(self.num_envs, 3)
        self.object_root_angular_velocities = root_states[:, self.object_handles_env[0], 10:13].view(self.num_envs, 3)

        self.rh_root_position = self.right_hand_rigid_body_positions[:, self.hand_wrist_rigid_body_id["right"]]
        self.rh_root_orientation = self.right_hand_rigid_body_orientations[:, self.hand_wrist_rigid_body_id["right"]]
        self.rh_root_linear_velocity = self.right_hand_rigid_body_linear_velocities[:, self.hand_wrist_rigid_body_id["right"]]
        self.rh_root_angular_velocity = self.right_hand_rigid_body_angular_velocities[:, self.hand_wrist_rigid_body_id["right"]]
        self.lh_root_position = self.left_hand_rigid_body_positions[:, self.hand_wrist_rigid_body_id["left"]]
        self.lh_root_orientation = self.left_hand_rigid_body_orientations[:, self.hand_wrist_rigid_body_id["left"]]
        self.lh_root_linear_velocity = self.left_hand_rigid_body_linear_velocities[:, self.hand_wrist_rigid_body_id["left"]]
        self.lh_root_angular_velocity = self.left_hand_rigid_body_angular_velocities[:, self.hand_wrist_rigid_body_id["left"]]

        self.left_keypoint_positions = self.left_hand_rigid_body_positions[:, self.left_keypoint_indices]
        self.left_keypoint_orientations = self.left_hand_rigid_body_orientations[:, self.left_keypoint_indices]
        self.right_keypoint_positions = self.right_hand_rigid_body_positions[:, self.right_keypoint_indices]
        self.right_keypoint_orientations = self.right_hand_rigid_body_orientations[:, self.right_keypoint_indices]
        self.left_keypoint_positions_with_offset = self.left_keypoint_positions + quat_apply(
            self.left_keypoint_orientations, self.left_keypoint_offsets.repeat(self.num_envs, 1, 1)
        )
        self.right_keypoint_positions_with_offset = self.right_keypoint_positions + quat_apply(
            self.right_keypoint_orientations, self.right_keypoint_offsets.repeat(self.num_envs, 1, 1)
        )

        self.left_fingertip_states = self.left_hand_rigid_body_states[:, self.left_virtual_tip_indices, :]
        self.left_fingertip_positions = self.left_fingertip_states[..., 0:3]
        self.left_fingertip_orientations = self.left_fingertip_states[..., 3:7]
        self.left_fingertip_linear_velocities = self.left_fingertip_states[..., 7:10]
        self.left_fingertip_angular_velocities = self.left_fingertip_states[..., 10:13]
        self.right_fingertip_states = self.right_hand_rigid_body_states[:, self.right_virtual_tip_indices, :]
        self.right_fingertip_positions = self.right_fingertip_states[..., 0:3]
        self.right_fingertip_orientations = self.right_fingertip_states[..., 3:7]
        self.right_fingertip_linear_velocities = self.right_fingertip_states[..., 7:10]
        self.right_fingertip_angular_velocities = self.right_fingertip_states[..., 10:13]

        rh_q_conj = quat_conjugate(self.right_hand_root_orientations)
        lh_q_conj = quat_conjugate(self.left_hand_root_orientations)
        self.object_positions_wrt_rh_palm = quat_rotate(rh_q_conj, self.object_root_positions - self.right_hand_root_positions)
        self.object_orientations_wrt_rh_palm = quat_mul(rh_q_conj, self.object_root_orientations)
        self.object_positions_wrt_lh_palm = quat_rotate(lh_q_conj, self.object_root_positions - self.left_hand_root_positions)
        self.object_orientations_wrt_lh_palm = quat_mul(lh_q_conj, self.object_root_orientations)
        self.lh_wrt_rh_position = quat_rotate(rh_q_conj, self.left_hand_root_positions - self.right_hand_root_positions)
        self.lh_wrt_rh_orientation = quat_mul(rh_q_conj, self.left_hand_root_orientations)

        self._update_parts_pointclouds()

    def compute_reach_reward_keypoints(self):
        # Reach reward: historical min distance from both hands' keypoints to board surface
        device = self.device
        N = self.num_envs
        board_pc = self._object_parts_world_pointclouds.get("board", None)
        if board_pc is None:
            self.reach_rew_keypoints = torch.zeros((N,), device=device, dtype=torch.float32)
            self.reach_rew_scaled_keypoints = self.reach_rew_keypoints
            return

        right_dists = torch.cdist(self.right_keypoint_positions_with_offset, board_pc)  # (N,Kr,P)
        right_cur_min = right_dists.min(dim=2)[0]  # (N,Kr)
        if (not hasattr(self, "right_keypoints_to_surface_dist_min")) or (
            self.right_keypoints_to_surface_dist_min.shape != right_cur_min.shape
        ):
            self.right_keypoints_to_surface_dist_min = torch.full_like(right_cur_min, 0.20)
        right_delta = (self.right_keypoints_to_surface_dist_min - right_cur_min).clamp_min(0.0)
        self.right_keypoints_to_surface_dist_min = torch.min(self.right_keypoints_to_surface_dist_min, right_cur_min)
        right_rew = right_delta.mean(dim=1)

        left_dists = torch.cdist(self.left_keypoint_positions_with_offset, board_pc)  # (N,Kl,P)
        left_cur_min = left_dists.min(dim=2)[0]  # (N,Kl)
        if (not hasattr(self, "left_keypoints_to_surface_dist_min")) or (
            self.left_keypoints_to_surface_dist_min.shape != left_cur_min.shape
        ):
            self.left_keypoints_to_surface_dist_min = torch.full_like(left_cur_min, 0.20)
        left_delta = (self.left_keypoints_to_surface_dist_min - left_cur_min).clamp_min(0.0)
        self.left_keypoints_to_surface_dist_min = torch.min(self.left_keypoints_to_surface_dist_min, left_cur_min)
        left_rew = left_delta.mean(dim=1)

        self.reach_rew_keypoints = 0.5 * (left_rew + right_rew)
        self.reach_rew_scaled_keypoints = self.reach_rew_keypoints * 5.0
        self.extras["reach_rew_keypoints"] = self.reach_rew_scaled_keypoints.clone()

    def _compute_curiosity_rewards_board(self):
        
        board_pc = self._object_parts_world_pointclouds["board"]  # (N,P,3)
        board_pos = self.object_root_positions
        board_ori = self.object_root_orientations
        task_contact_satisfied = self._check_bimanual_contact_criteria_board()

        for hand_side in ("right", "left"):
            if hand_side == "right":
                kp_pos_all = self.right_keypoint_positions_with_offset     # (N,K,3)
                kp_ori_all = self.right_keypoint_orientations             # (N,K,4)
                kp_force_all = self.right_keypoint_contact_forces         # (N,K,3)
                idx_lists = [
                    self.right_index_link_indices_among_keypoints,
                    self.right_middle_link_indices_among_keypoints,
                    self.right_ring_link_indices_among_keypoints,
                    self.right_thumb_link_indices_among_keypoints,
                ]
            else:
                kp_pos_all = self.left_keypoint_positions_with_offset
                kp_ori_all = self.left_keypoint_orientations
                kp_force_all = self.left_keypoint_contact_forces
                idx_lists = [
                    self.left_index_link_indices_among_keypoints,
                    self.left_middle_link_indices_among_keypoints,
                    self.left_ring_link_indices_among_keypoints,
                    self.left_thumb_link_indices_among_keypoints,
                ]

            rep_positions = []
            rep_orientations = []
            rep_forces = []
            rep_contact_indices = []
            rep_contact_masks = []

            for idx_list in idx_lists:
                if idx_list is None or len(idx_list) == 0:
                    assert False
                    # d_all = torch.cdist(kp_pos_all, board_pc)               # (N,K,P)
                    # N, K, P = d_all.shape
                    # d_flat = d_all.view(N, -1)
                    # min_flat = d_flat.argmin(dim=-1)                        # (N,)
                    # best_k = (min_flat // P)
                    # best_p = (min_flat % P)
                    # min_d = d_flat.gather(1, min_flat.unsqueeze(-1)).squeeze(-1)
                    # gather_pos = best_k.view(N, 1, 1).expand(-1, 1, 3)
                    # gather_ori = best_k.view(N, 1, 1).expand(-1, 1, 4)
                    # pos_rep = torch.gather(kp_pos_all, 1, gather_pos).squeeze(1)
                    # ori_rep = torch.gather(kp_ori_all, 1, gather_ori).squeeze(1)
                    # # use summed global force as fallback
                    # force_rep = kp_force_all.sum(dim=1)
                    # force_mag_any = ((kp_force_all.norm(dim=-1, p=2) > 0.01) & (kp_force_all.norm(dim=-1, p=2) < 50.0)).any(dim=-1)
                    # contact_mask_rep = (min_d < 0.012) & force_mag_any
                else:
                    idx = torch.as_tensor(idx_list, dtype=torch.long, device=self.device)
                    kp_pos_f = kp_pos_all[:, idx, :]                        # (N,Kf,3)
                    kp_ori_f = kp_ori_all[:, idx, :]                        # (N,Kf,4)
                    kp_force_f = kp_force_all[:, idx, :]                    # (N,Kf,3)

                    d = torch.cdist(kp_pos_f, board_pc)                     # (N,Kf,P)
                    N, Kf, P = d.shape
                    d_flat = d.view(N, -1)
                    min_flat = d_flat.argmin(dim=-1)                        # (N,)
                    min_d = d_flat.gather(1, min_flat.unsqueeze(-1)).squeeze(-1)
                    best_p = (min_flat % P)                                 # (N,)
                    best_k = (min_flat // P)                                # (N,)

                    gather_pos = best_k.view(N, 1, 1).expand(-1, 1, 3)
                    gather_ori = best_k.view(N, 1, 1).expand(-1, 1, 4)
                    pos_rep = torch.gather(kp_pos_f, 1, gather_pos).squeeze(1)   # (N,3)
                    ori_rep = torch.gather(kp_ori_f, 1, gather_ori).squeeze(1)   # (N,4)

                    # same strategy as bimanual_v2: finger force is sum of finger keypoint forces
                    force_rep = kp_force_f.sum(dim=1)                       # (N,3)
                    force_mag_any = ((kp_force_f.norm(dim=-1, p=2) > 0.01) & (kp_force_f.norm(dim=-1, p=2) < 50.0)).any(dim=-1)
                    contact_mask_rep = (min_d < 0.012) & force_mag_any

                rep_positions.append(pos_rep)
                rep_orientations.append(ori_rep)
                rep_forces.append(force_rep)
                rep_contact_indices.append(best_p)
                rep_contact_masks.append(contact_mask_rep)

            keypoints_pos = torch.stack(rep_positions, dim=1)               # (N,4,3)
            keypoints_ori = torch.stack(rep_orientations, dim=1)            # (N,4,4)
            keypoints_forces = torch.stack(rep_forces, dim=1)               # (N,4,3)
            contact_indices = torch.stack(rep_contact_indices, dim=1)       # (N,4)
            contact_mask = torch.stack(rep_contact_masks, dim=1)            # (N,4)

            axis_local = torch.tensor([-1.0, 0.0, 0.0], device=self.device, dtype=keypoints_pos.dtype).view(1, 1, 3)
            axis_local = axis_local.expand_as(keypoints_pos)
            dir_world = quat_apply(
                keypoints_ori.reshape(-1, 4),
                axis_local.reshape(-1, 3),
            ).view_as(keypoints_pos)

            mgr = self._curiosity_managers[hand_side]["board"]
            state_features_world = board_pc.index_select(1, mgr._state_point_indices).reshape(self.num_envs, -1).to(torch.float32)
            _, info = mgr.compute_reward_from_canonical(
                object_positions=board_pos,
                object_orientations=board_ori,
                keypoint_positions_world=keypoints_pos,           # (N,4,3)
                contact_indices=contact_indices,                  # (N,4)
                contact_mask=contact_mask,                        # (N,4)
                task_contact_satisfied=task_contact_satisfied,
                contact_coverage=1.0,
                contact_forces_local=dir_world,
                keypoint_palm_dirs_world=dir_world,
                state_features_world=state_features_world,
            )

            reach_curiosity_rew_scaled = info["potential_field_reward"] * 1.28
            contact_coverage_rew_scaled = info["cluster_novelty_reward"] * 200.0
            curiosity_rew = 0.5 * (reach_curiosity_rew_scaled + contact_coverage_rew_scaled)

            self.rew_buf += curiosity_rew
            self.extras[f"{hand_side}_reach_curiosity_rew"] = reach_curiosity_rew_scaled
            self.extras[f"{hand_side}_contact_coverage_rew"] = contact_coverage_rew_scaled
            self.extras[f"{hand_side}_curiosity_reward"] = curiosity_rew

    def _check_bimanual_contact_criteria_board(self) -> torch.Tensor:
        """Board contact gate: both hands must have at least one valid contact keypoint."""
        device = self.device
        n_envs = self.num_envs
        board_pc = self._object_parts_world_pointclouds.get("board", None)
        if board_pc is None:
            contact_satisfied = torch.zeros((n_envs,), dtype=torch.bool, device=device)
            self.extras["left_hand_contact_ok"] = contact_satisfied
            self.extras["right_hand_contact_ok"] = contact_satisfied
            self.extras["bimanual_contact_satisfied"] = contact_satisfied
            return contact_satisfied

        # Right hand
        right_kp_pos = self.right_keypoint_positions_with_offset
        right_kp_force = self.right_keypoint_contact_forces
        dists_r = torch.cdist(right_kp_pos, board_pc)
        min_dists_r, _ = dists_r.min(dim=-1)
        force_mag_r = right_kp_force.norm(dim=-1, p=2)
        near_surface_r = min_dists_r < 0.012
        has_force_r = (force_mag_r > 0.01) & (force_mag_r < 50.0)
        kp_contact_r = near_surface_r & has_force_r
        right_ok = kp_contact_r.any(dim=-1)

        # Left hand
        left_kp_pos = self.left_keypoint_positions_with_offset
        left_kp_force = self.left_keypoint_contact_forces
        dists_l = torch.cdist(left_kp_pos, board_pc)
        min_dists_l, _ = dists_l.min(dim=-1)
        force_mag_l = left_kp_force.norm(dim=-1, p=2)
        near_surface_l = min_dists_l < 0.012
        has_force_l = (force_mag_l > 0.01) & (force_mag_l < 50.0)
        kp_contact_l = near_surface_l & has_force_l
        left_ok = kp_contact_l.any(dim=-1)

        table_force_mag = self.table_contact_forces.norm(dim=-1)
        bottom_table_contact = (table_force_mag > 1e-2) & (table_force_mag < 50)
        bottom_table_contact &= (board_pc[:, :, 2].amin(dim=-1) - self.table_height) < 0.01

        contact_satisfied = (left_ok & right_ok) | bottom_table_contact
        self.extras["left_kp_contact_mask"] = kp_contact_l.sum(dim=-1)
        self.extras["right_kp_contact_mask"] = kp_contact_r.sum(dim=-1)
        self.extras["bottom_table_contact"] = bottom_table_contact
        self.extras["left_hand_contact_ok"] = left_ok
        self.extras["right_hand_contact_ok"] = right_ok
        self.extras["bimanual_contact_satisfied"] = contact_satisfied
        return contact_satisfied

    def compute_reward(self, actions: Tensor) -> None:
        self.rew_buf[:] = 0.0

        # 1) reach reward
        self.compute_reach_reward_keypoints()
        self.rew_buf += self.reach_rew_scaled_keypoints

        contact_satisfied = self._check_bimanual_contact_criteria_board()
        contact_gate = contact_satisfied.float()

        # 2) board lift target reward (historical max delta of lowest point)
        if not hasattr(self, "board_lowest_point_max"):
            self.board_lowest_point_max = self.board_lowest_point_world.clone()
        delta = (self.board_lowest_point_world - self.board_lowest_point_max).clamp_min(0.0)
        self.board_lowest_point_max = torch.where(
            contact_satisfied,
            torch.maximum(self.board_lowest_point_max, self.board_lowest_point_world),
            self.board_lowest_point_max,
        )
        self.targ_rew = delta * self.lowest_point_reward_scale * contact_gate
        self.rew_buf += self.targ_rew

        near_goal = (self.board_lowest_point_world >= self.board_lift_success_height) & contact_satisfied
        self.success_step_counter = torch.where(
            near_goal,
            self.success_step_counter + 1,
            torch.zeros_like(self.success_step_counter),
        )
        is_success = self.success_step_counter >= self.success_steps_required

        per_step_bonus = self.success_bonus / float(self.success_steps_required)
        maintain_bonus = near_goal.float() * per_step_bonus
        self.rew_buf += maintain_bonus
        self.rew_buf += is_success.float() * self.success_bonus * contact_gate

        # 3) ccge reward
        if "ccge" in self.reward_type and self.training:
            self._compute_curiosity_rewards_board()

        if self.terminate_on_success:
            self.reset_buf |= torch.where(is_success, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf |= torch.where(self.progress_buf >= self.max_episode_length - 1, 1, self.reset_buf)

        self.extras["board_lowest_point_world"] = self.board_lowest_point_world.clone()
        self.extras["board_lowest_point_max"] = self.board_lowest_point_max.clone()
        self.extras["board_lowest_point_delta"] = delta.clone()
        self.extras["targ_rew"] = self.targ_rew.clone()
        self.extras["contact_satisfied"] = contact_satisfied.clone()
        self.extras["near_goal"] = near_goal.clone()
        self.extras["is_success"] = is_success.clone()

        self.compute_done(is_success)

    def reset_idx(self, env_ids: LongTensor, first_time: bool = False) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(dtype=torch.long, device=self.device)

        hand_indices_list = [self.hand_actor_handle_sim[side][env_ids].to(torch.int32) for side in ("right", "left")]
        hand_actor_indices = torch.cat(hand_indices_list)
        hand_actor_indices = hand_actor_indices[hand_actor_indices.argsort()]
        object_indices = self.object_handles_sim[env_ids][:, 0]

        self.root_positions.view(self.num_envs, self.num_actors, 3)[env_ids, self.object_handles_env[0]] = torch.tensor(
            self.object_cfgs[0]["pose"][:3], device=self.device
        )
        self.root_orientations.view(self.num_envs, self.num_actors, 4)[env_ids, self.object_handles_env[0]] = torch.tensor(
            self.object_cfgs[0]["pose"][3:], device=self.device
        )
        self.root_linear_velocities.view(self.num_envs, self.num_actors, 3)[env_ids, self.object_handles_env[0]] = 0.0
        self.root_angular_velocities.view(self.num_envs, self.num_actors, 3)[env_ids, self.object_handles_env[0]] = 0.0

        self.left_hand_root_positions[env_ids] = torch.tensor(
            self.cfg["env"]["hands"]["left"]["initPose"]["position"], device=self.device
        )
        self.left_hand_root_orientations[env_ids] = torch.tensor(
            self.cfg["env"]["hands"]["left"]["initPose"]["orientation"], device=self.device
        )
        self.left_hand_root_linear_velocities[env_ids] = 0.0
        self.left_hand_root_angular_velocities[env_ids] = 0.0
        self.right_hand_root_positions[env_ids] = torch.tensor(
            self.cfg["env"]["hands"]["right"]["initPose"]["position"], device=self.device
        )
        self.right_hand_root_orientations[env_ids] = torch.tensor(
            self.cfg["env"]["hands"]["right"]["initPose"]["orientation"], device=self.device
        )
        self.right_hand_root_linear_velocities[env_ids] = 0.0
        self.right_hand_root_angular_velocities[env_ids] = 0.0

        self.left_hand_arm_dof_positions[env_ids, :] = self.gym_assets["current"]["hand_left"]["init"]["position"]
        self.left_hand_arm_dof_velocities[env_ids, :] = self.gym_assets["current"]["hand_left"]["init"]["velocity"]
        self.right_hand_arm_dof_positions[env_ids, :] = self.gym_assets["current"]["hand_right"]["init"]["position"]
        self.right_hand_arm_dof_velocities[env_ids, :] = self.gym_assets["current"]["hand_right"]["init"]["velocity"]

        left_hand_arm_slice = slice(self.left_hand_arm_dof_start, self.left_hand_arm_dof_end)
        right_hand_arm_slice = slice(self.right_hand_arm_dof_start, self.right_hand_arm_dof_end)
        self.prev_targets_buffer[env_ids, left_hand_arm_slice] = self.left_hand_arm_dof_positions[env_ids, :]
        self.curr_targets_buffer[env_ids, left_hand_arm_slice] = self.left_hand_arm_dof_positions[env_ids, :]
        self.prev_targets_buffer[env_ids, right_hand_arm_slice] = self.right_hand_arm_dof_positions[env_ids, :]
        self.curr_targets_buffer[env_ids, right_hand_arm_slice] = self.right_hand_arm_dof_positions[env_ids, :]

        hand_object_indices = torch.cat([hand_actor_indices, object_indices])
        hand_object_indices = hand_object_indices[hand_object_indices.argsort()]

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(hand_actor_indices),
            hand_actor_indices.shape[0],
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(hand_object_indices),
            hand_object_indices.shape[0],
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.curr_targets_buffer),
            gymtorch.unwrap_tensor(hand_actor_indices),
            hand_actor_indices.shape[0],
        )

        if hasattr(self, "rb_forces"):
            self.rb_forces[env_ids, :, :] = 0.0
        if hasattr(self, "rb_torques"):
            self.rb_torques[env_ids, :, :] = 0.0
        if self.use_pid_control:
            self._reset_pid_buffers(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_step_counter[env_ids] = 0
        self.successes[env_ids] = 0
        if hasattr(self, "board_lowest_point_max"):
            self.board_lowest_point_max[env_ids] = self.table_height
