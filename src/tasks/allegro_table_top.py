from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from isaacgym import gymapi
import numpy as np
import torch
import os
from dotenv import find_dotenv
from torch import LongTensor, Tensor

from .torch_utils import (
    random_yaw_orientation,
    random_orientation,
    random_orientation_within_angle,
    default_orientation,
)

from .allegro_singulation import XArmAllegroHandFunctionalManipulationUnderarm


class XArmAllegroHandTableTop(XArmAllegroHandFunctionalManipulationUnderarm):

    _table_x_length = 0.4
    _table_y_length = 0.4
    _table_thickness = 0.02
    _table_pose = [0.0, 0.0, 0.10]
    # _xarm_right_init_dof_positions: Dict[str, float] = {
    #     "joint1": 0.0,
    #     "joint2": 0.0,
    #     "joint3": -1.5,
    #     "joint4": 0.0,
    #     "joint5": 1.4,
    #     "joint6": 0.0,
    # }

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        env_cfg = cfg["env"]

        # Defaults
        env_cfg.setdefault("useUpperShelf", False)
        env_cfg.setdefault("numObjects", 1)
        env_cfg.setdefault("numObjectsPerEnv", 1)
        env_cfg.setdefault("renderTarget", False)

        # Object options
        env_cfg.setdefault("objectMode", "cube")  # "cube" or "urdf"
        env_cfg.setdefault("objectUrdfPath", "")  # relative to assets/urdf if objectMode == "urdf"
        env_cfg.setdefault("cubeWidth", 0.04)
        env_cfg.setdefault("cubeLength", 0.04)
        env_cfg.setdefault("cubeHeight", 0.12)

        # Placement options
        env_cfg.setdefault("placeSide", "back")  # "left" | "right" | "back" | "center"
        env_cfg.setdefault("edgeMargin", 1e-3)   # small safety offset from exact edge

        # Walls
        env_cfg.setdefault("wallLeft", False)
        env_cfg.setdefault("wallRight", False)
        env_cfg.setdefault("wallBack", False)
        env_cfg.setdefault("wallThickness", 0.01)
        env_cfg.setdefault("wallHeight", 0.15)
        env_cfg.setdefault("wallColor", (0.25, 0.25, 0.25))

        # Table collision toggle
        env_cfg.setdefault("tableNoCollision", False)

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self._configure_viewer()
        self._reset_action_indices()

    def _create_box_grid_dataset(self, device=None) -> None:
        """Override parent's dataset creation to support URDF or cube point clouds.
        """
        from .dataset import TableTopDataset
        object_mode = str(self.cfg["env"].get("objectMode", "cube")).lower()
        pcl_num = int(self.cfg["env"].get("numObjectPointCloudPoints", 1024))
        if object_mode == "urdf":
            urdf_rel = str(self.cfg["env"].get("objectUrdfPath", "")).strip()
            if urdf_rel == "":
                raise ValueError("env.objectUrdfPath must be set when objectMode == 'urdf'")
            object_asset_root = "./assets"
            self.grasping_dataset = TableTopDataset(
                mode="urdf",
                device=device,
                pcl_num=pcl_num,
                urdf_rel_path=urdf_rel,
                asset_root=object_asset_root,
            )
        else:
            W = float(self.cfg["env"].get("cubeWidth", 0.04))
            L = float(self.cfg["env"].get("cubeLength", 0.04))
            H = float(self.cfg["env"].get("cubeHeight", 0.12))
            self.grasping_dataset = TableTopDataset(
                mode="cube",
                device=device,
                pcl_num=pcl_num,
                cube_width=W,
                cube_length=L,
                cube_height=H,
            )
        self.num_categories = self.grasping_dataset._category_matrix.shape[1]

    def _define_table(self) -> Dict[str, Any]:
        # Same table definition as in XArmAllegroHandCubeInBox (copied for consistent friction)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        asset = self.gym.create_box(
            self.sim, self._table_x_length, self._table_y_length, self._table_thickness, asset_options
        )

        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        for s in rigid_shape_props:
            s.friction = 0.1
            s.restitution = 0.1
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

        num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
        num_rigid_shapes = self.gym.get_asset_rigid_shape_count(asset)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self._table_pose)

        return {
            "asset": asset,
            "pose": pose,
            "name": "table",
            "num_rigid_bodies": num_rigid_bodies,
            "num_rigid_shapes": num_rigid_shapes,
        }

    def _define_object(self) -> dict:
        """
        Build a single target object based on env config:
         - objectMode == "cube": create Isaac box with dimensions (W,L,H)
         - objectMode == "urdf": load from URDF path
        """
        config = {}
        config["warehouse"] = {"targ_obj": [], "surr_obj": []}

        target_asset_options = gymapi.AssetOptions()
        target_asset_options.density = 250.0
        target_asset_options.convex_decomposition_from_submeshes = True
        target_asset_options.override_com = True
        target_asset_options.override_inertia = True

        object_mode = self.cfg["env"]["objectMode"]
        if object_mode == "cube":
            W = float(self.cfg["env"]["cubeWidth"])
            L = float(self.cfg["env"]["cubeLength"])
            H = float(self.cfg["env"]["cubeHeight"])

            asset = self.gym.create_box(self.sim, W, L, H, target_asset_options)
            rprops = self.gym.get_asset_rigid_shape_properties(asset)
            for s in rprops:
                s.friction = 0.7
                s.restitution = 0.05
            self.gym.set_asset_rigid_shape_properties(asset, rprops)

            cfg = {"name": "target_cube", "asset": asset}
            cfg["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(asset)
            cfg["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(asset)
            config["warehouse"]["targ_obj"].append(cfg)
            config["count"] = 1

        elif object_mode == "urdf":
            urdf_rel = str(self.cfg["env"]["objectUrdfPath"]).strip()
            if urdf_rel == "":
                raise ValueError("env.objectUrdfPath must be set when objectMode == 'urdf'")
            # asset = self.gym.load_asset(self.sim, self._asset_root, urdf_rel, target_asset_options)
            asset = self.gym.load_asset(self.sim, "./assets", urdf_rel, target_asset_options)
            print(">>> Loaded URDF asset: ", urdf_rel)
            rprops = self.gym.get_asset_rigid_shape_properties(asset)
            for s in rprops:
                s.friction = 1.0
                s.restitution = 0.05
            self.gym.set_asset_rigid_shape_properties(asset, rprops)

            cfg = {"name": "target_urdf", "asset": asset}
            cfg["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(asset)
            cfg["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(asset)
            config["warehouse"]["targ_obj"].append(cfg)
            config["count"] = 1
        else:
            raise ValueError(f"Unsupported objectMode: {object_mode}")

        # One pose slot; actual placement handled in _create_envs
        config["poses"] = [gymapi.Transform()]
        return config

    def _define_walls_assets(self) -> Dict[str, Any]:
        """
        Create assets for vertical walls (left/right/back) with configured dimensions.
        """
        Wt = float(self._table_x_length)
        Dt = float(self._table_y_length)
        Ht = float(self._table_thickness)
        wall_h = float(self.cfg["env"]["wallHeight"])
        t = float(self.cfg["env"]["wallThickness"])

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001

        # Walls along Y extend in Z with thickness t along Y; along X extend in Z with thickness t along X.
        wall_y_asset = self.gym.create_box(self.sim, Wt + 2 * t, t, wall_h, asset_options)
        wall_x_asset = self.gym.create_box(self.sim, t, Dt + 2 * t, wall_h, asset_options)

        # Basic friction/restitution
        for a in [wall_y_asset, wall_x_asset]:
            rprops = self.gym.get_asset_rigid_shape_properties(a)
            for s in rprops:
                s.friction = 0.01
                s.restitution = 0.01
            self.gym.set_asset_rigid_shape_properties(a, rprops)

        return {
            "wall_y": wall_y_asset,
            "wall_x": wall_x_asset,
            "num_rigid_bodies": self.gym.get_asset_rigid_body_count(wall_y_asset) + self.gym.get_asset_rigid_body_count(wall_x_asset),
            "num_rigid_shapes": self.gym.get_asset_rigid_shape_count(wall_y_asset) + self.gym.get_asset_rigid_shape_count(wall_x_asset),
        }


    def compute_aggregate_bodies_and_shapes(self, env: int, gym_assets: Optional[Dict] = None) -> Tuple[int, int]:
        """
        Compute the number of rigid bodies and shapes spawned in one TableTop env,
        so begin_aggregate() reserves enough capacity (includes optional walls).
        """
        if gym_assets is None:
            gym_assets = self.gym_assets

        num_bodies, num_shapes = 0, 0

        # Robot
        num_bodies += gym_assets["current"]["robot"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["robot"]["num_rigid_shapes"]

        # Target object (single)
        obj_cfg = gym_assets["current"]["objects"]["warehouse"]["targ_obj"][0]
        num_bodies += obj_cfg["num_rigid_bodies"]
        num_shapes += obj_cfg["num_rigid_shapes"]

        # Table
        num_bodies += gym_assets["current"]["table"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["table"]["num_rigid_shapes"]

        # Visual target object
        num_bodies += gym_assets["current"]["visual_target_object"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["visual_target_object"]["num_rigid_shapes"]

        # Optional walls (left/right/back)
        wall_left = bool(self.cfg["env"].get("wallLeft", False))
        wall_right = bool(self.cfg["env"].get("wallRight", False))
        wall_back = bool(self.cfg["env"].get("wallBack", False))

        if "walls" in gym_assets["current"]:
            wx = gym_assets["current"]["walls"]["wall_x"]
            wy = gym_assets["current"]["walls"]["wall_y"]

            if wall_left:
                num_bodies += self.gym.get_asset_rigid_body_count(wx)
                num_shapes += self.gym.get_asset_rigid_shape_count(wx)
            if wall_right:
                num_bodies += self.gym.get_asset_rigid_body_count(wx)
                num_shapes += self.gym.get_asset_rigid_shape_count(wx)
            if wall_back:
                num_bodies += self.gym.get_asset_rigid_body_count(wy)
                num_shapes += self.gym.get_asset_rigid_shape_count(wy)

        return num_bodies, num_shapes
    
    def _create_envs(self, num_envs: int, spacing: float, num_objects_per_env: int = 1):
        print(">>> Setting up %d environments (TableTop)" % num_envs)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(num_envs))

        print(">>> Defining gym assets")
        self.gym_assets["current"]["robot"] = self._define_allegro_hand_with_arm()
        self.gym_assets["current"]["objects"] = self._define_object()
        self.gym_assets["current"]["table"] = self._define_table()
        self.gym_assets["current"]["visual_target_object"] = self._define_visual_target_object()
        # if self.use_upper_shelf:
        #     self.gym_assets["current"]["upper_shelf"] = self._define_upper_shelf()

        # Prepare walls assets (will spawn per-env based on toggles)
        walls_assets = self._define_walls_assets()
        self.gym_assets["current"]["walls"] = walls_assets

        self._define_camera()
        print(">>> Done defining gym assets")

        max_aggregate_bodies, max_aggregate_shapes = self.compute_maximum_aggregate_bodies_and_shapes()

        self.envs = []
        self.cameras_handle = []
        allegro_hand_indices = []
        table_indices = []
        visual_target_object_indices = []
        object_indices = [[] for _ in range(num_envs)]
        object_names = [[] for _ in range(num_envs)]
        object_encodings = [[] for _ in range(num_envs)]
        occupied_object_indices = []
        scene_object_indices = [[] for _ in range(num_envs)]

        print(">>> Creating environments")
        print("    - max_aggregate_bodies: ", max_aggregate_bodies)
        print("    - max_aggregate_shapes: ", max_aggregate_shapes)

        # Convenience parameters
        table_cfg = self.gym_assets["current"]["table"]
        Wt = float(self._table_x_length)
        Dt = float(self._table_y_length)
        Ht = float(self._table_thickness)
        table_center = self._table_pose  # [x, y, z]
        wall_h = float(self.cfg["env"]["wallHeight"])
        t = float(self.cfg["env"]["wallThickness"])
        wall_color = self.cfg["env"]["wallColor"]
        wall_left = bool(self.cfg["env"]["wallLeft"])
        wall_right = bool(self.cfg["env"]["wallRight"])
        wall_back = bool(self.cfg["env"]["wallBack"])
        table_no_collision = bool(self.cfg["env"]["tableNoCollision"])

        VISUAL_TARGET_COLLISION_FILTER = getattr(self, "VISUAL_TARGET_COLLISION_FILTER", 0x7FFFFFFF)
        table_filter = VISUAL_TARGET_COLLISION_FILTER if table_no_collision else 0

        obj_mode = self.cfg["env"]["objectMode"]
        W = float(self.cfg["env"]["cubeWidth"])
        L = float(self.cfg["env"]["cubeLength"])
        H = float(self.cfg["env"]["cubeHeight"])
        margin = float(self.cfg["env"]["edgeMargin"])
        place_side = str(self.cfg["env"]["placeSide"])
        

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.aggregate_tracker.reset()

            if self.aggregate_mode != 0:
                _nb, _ns = self.compute_aggregate_bodies_and_shapes(i)
                ok = self.gym.begin_aggregate(env, max_aggregate_bodies, max_aggregate_shapes, True)
                if not ok:
                    raise RuntimeError("begin_aggregate failed")

            # Add robot
            actor_index, actor_handle = self._create_sim_actor(
                env, self.gym_assets["current"]["robot"], i, actor_handle=True
            )
            allegro_hand_indices.append(actor_index)

            # Add target object
            poses = self.gym_assets["current"]["objects"]["poses"]
            obj_pose = gymapi.Transform()
            # Placement height: table top + half object height
            table_top_z = table_center[2] + Ht * 0.5

            if obj_mode == "cube":
                # edge placement (left/right/back); coordinates centered at table center (cx, cy)
                cx, cy = table_center[0], table_center[1]
                if place_side == "left":
                    x = cx - (Wt * 0.5 - W * 0.5 - margin)
                    y = cy
                elif place_side == "right":
                    x = cx + (Wt * 0.5 - W * 0.5 - margin)
                    y = cy
                elif place_side == "back":
                    # treat "back" as negative Y side
                    x = cx
                    y = cy - (Dt * 0.5 - L * 0.5 - margin)
                elif place_side == "center":
                    x, y = cx, cy
                else:
                    raise ValueError(f"Unsupported placeSide: {place_side}")

                z = table_top_z + H * 0.5 + 1e-3
                obj_pose.p = gymapi.Vec3(x, y, z)
                obj_pose.r = gymapi.Quat(0, 0, 0, 1)
            else:
                # URDF: center of table, slight lift
                cx, cy = table_center[0], table_center[1]
                z = table_top_z + 0.10
                obj_pose.p = gymapi.Vec3(cx, cy, z)
                # obj_pose.r = gymapi.Quat(0, 0, 0, 1)
                obj_pose.r = gymapi.Quat(0, 0, 0, 1)

            self.gym_assets["current"]["objects"]["poses"][0] = obj_pose

            cfg = self.gym_assets["current"]["objects"]["warehouse"]["targ_obj"][0]
            actor_index = self._create_sim_actor(env, cfg, i, "targ_obj", obj_pose, color=gymapi.Vec3(0.9, 0.9, 0.9))
            object_indices[i].append(actor_index)
            object_names[i].append(cfg["name"])
            object_encodings[i].append(0)
            occupied_object_indices.append(actor_index)
            scene_object_indices[i].append(0)

            # Add table
            actor_index, table_handle = self._create_sim_actor(
                env, table_cfg, -1, actor_handle=True, color=gymapi.Vec3(0.0, 0.0, 0.0), filter=table_filter
            )
            table_indices.append(table_handle)

            # side walls
            wx = self.gym_assets["current"]["walls"]["wall_x"]
            wy = self.gym_assets["current"]["walls"]["wall_y"]
            wall_col = gymapi.Vec3(*wall_color)
            cx, cy = table_center[0], table_center[1]
            wall_center_z = table_top_z + wall_h * 0.5

            if wall_left:
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(cx - (Wt * 0.5 + t * 0.5), cy, wall_center_z)
                pose.r = gymapi.Quat(0, 0, 0, 1)
                _ = self._create_sim_actor(env, {"asset": wx, "name": "wall_left", "num_rigid_bodies": 1, "num_rigid_shapes": 1, "pose": pose}, -1, actor_handle=False, color=wall_col)

            if wall_right:
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(cx + (Wt * 0.5 + t * 0.5), cy, wall_center_z)
                pose.r = gymapi.Quat(0, 0, 0, 1)
                _ = self._create_sim_actor(env, {"asset": wx, "name": "wall_right", "num_rigid_bodies": 1, "num_rigid_shapes": 1, "pose": pose}, -1, actor_handle=False, color=wall_col)

            if wall_back:
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(cx, cy - (Dt * 0.5 + t * 0.5), wall_center_z)
                pose.r = gymapi.Quat(0, 0, 0, 1)
                _ = self._create_sim_actor(env, {"asset": wy, "name": "wall_back", "num_rigid_bodies": 1, "num_rigid_shapes": 1, "pose": pose}, -1, actor_handle=False, color=wall_col)

            actor_index, actor_handle = self._create_sim_actor(
                env, self.gym_assets["current"]["visual_target_object"], i + self.num_envs,
                actor_handle=True, color=gymapi.Vec3(0.6, 0.72, 0.98)
            )
            visual_target_object_indices.append(actor_handle)

            if self.enable_rendered_pointcloud_observation or self.save_video:
                for k in range(self.num_cameras_per_env):
                    camera = self.gym.create_camera_sensor(env, self.camera_properties)
                    self.cameras_handle.append(camera)
                    self.gym.set_camera_location(
                        camera, env, self._camera_positions[k], self._camera_target_locations[k]
                    )

            if self.aggregate_mode != 0:
                ok = self.gym.end_aggregate(env)
                if not ok:
                    raise RuntimeError("end_aggregate failed")
                assert self.aggregate_tracker.aggregate_bodies <= max_aggregate_bodies
                assert self.aggregate_tracker.aggregate_shapes <= max_aggregate_shapes

            self.envs.append(env)

        env = self.envs[0]
        allegro_hand = self.gym.find_actor_handle(env, "allegro_hand")
        self.allegro_hand_index = self.gym.get_actor_index(env, allegro_hand, gymapi.DOMAIN_ENV)

        self.target_object_rigid_body_indices = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.surr_object_rigid_body_indices = torch.zeros((num_envs, 0), dtype=torch.long, device=self.device)
        for i in range(num_envs):
            env_i = self.envs[i]
            target_obj_handle = self.gym.find_actor_handle(env_i, "targ_obj")
            target_object_actor = self.gym.get_actor_index(env_i, target_obj_handle, gymapi.DOMAIN_ENV)
            target_rb_index = self.gym.get_actor_rigid_body_index(env_i, target_object_actor, 0, gymapi.DOMAIN_SIM)
            self.target_object_rigid_body_indices[i] = target_rb_index

        self.allegro_hand_dof_start = self.gym.get_actor_dof_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_dof_end = self.allegro_hand_dof_start + self.gym_assets["current"]["robot"]["num_dofs"]
        self.allegro_hand_indices = torch.tensor(allegro_hand_indices).long().to(self.device)
        self.allegro_hand_rigid_body_start = self.gym.get_actor_rigid_body_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_rigid_body_end = (
            self.allegro_hand_rigid_body_start + self.gym_assets["current"]["robot"]["num_rigid_bodies"]
        )

        self.table_indices = torch.tensor(table_indices).long().to(self.device)
        self.visual_target_object_indices = torch.tensor(visual_target_object_indices).long().to(self.device)
        self.object_indices = torch.tensor(object_indices).long().to(self.device)
        self.object_names = object_names
        self.object_encodings = torch.tensor(object_encodings).long().to(self.device)
        
        self.table_rigid_body_indices = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        for i in range(self.num_envs):
            env_i = self.envs[i]
            table_handle_i = int(self.table_indices[i].item())
            table_actor_index = self.gym.get_actor_index(env_i, table_handle_i, gymapi.DOMAIN_ENV)
            table_rb_index = self.gym.get_actor_rigid_body_index(env_i, table_actor_index, 0, gymapi.DOMAIN_SIM)
            self.table_rigid_body_indices[i] = table_rb_index
        
        self.occupied_object_indices = torch.tensor(occupied_object_indices).long().to(self.device)
        self.occupied_object_relative_indices = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.non_occupied_object_indices = torch.zeros((num_envs, 0), dtype=torch.long, device=self.device)
        self.surr_object_indices = torch.zeros((num_envs, 0), dtype=torch.long, device=self.device)
        self.scene_object_indices = torch.tensor(scene_object_indices).long().to(self.device)

        self.table_rigid_body_indices = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        for i in range(self.num_envs):
            env_i = self.envs[i]
            table_handle_i = int(self.table_indices[i].item())
            table_actor_index = self.gym.get_actor_index(env_i, table_handle_i, gymapi.DOMAIN_ENV)
            table_rb_index = self.gym.get_actor_rigid_body_index(env_i, table_actor_index, 0, gymapi.DOMAIN_SIM)
            self.table_rigid_body_indices[i] = table_rb_index

        print(f">>> Done creating {num_envs} environments (TableTop)")
            
    def check_contact_criteria(self):

        tip_contact = self.keypoint_contact_mask.any(dim=-1)

        table_contact = (self.table_contact_forces.norm(dim=-1) > 1e-2)

        hand_contact = (tip_contact) | (table_contact)


        self.extras["hand_contact"]        = hand_contact.clone()
        self.extras["tip_contact"]         = tip_contact.clone()
        self.extras["table_contact"]       = table_contact.clone()

        return hand_contact
        
    def compute_mask_success(self):
        pcl_world = self._get_target_surface_points_world()  # (N, P, 3)
        pcl_mask = self.grasping_dataset._pointcloud_mask[0]
        pcl_mask_region_center_world = pcl_world[:, pcl_mask].mean(dim=1)  # (N, 3)
        midpoint_to_object_mask_dist = torch.norm(self.midpoint_positions - pcl_mask_region_center_world, dim=-1)
        contact_mask_region_success = midpoint_to_object_mask_dist < 0.05
        # print(f"midpoint_to_object_mask_dist: {midpoint_to_object_mask_dist}, contact_mask_region_success: {contact_mask_region_success}")
        return contact_mask_region_success
        
    def compute_reward(self, actions: Tensor) -> None:

        reward_types = self.reward_type.strip().split("+")
        assert len(reward_types) > 0, f"No reward type specified, reward_type from cfg is {self.reward_type}"

        self.refresh_contact_mask() # self.contact_mask
        self.contact_satisfied = self.check_contact_criteria()


        self.compute_reach_reward_keypoints(); self.reach_rew_scaled = self.reach_rew_scaled_keypoints.clone()
        self.compute_curiosity_informed_reach_reward()
        
        if "target" in reward_types:
            self.compute_targ_reward()
        
        self.near_goal_steps += self.near_goal.to(torch.long)
        # print("add near_goal to self.near_goal_steps, self.near_goal_steps: ", self.near_goal_steps)
        self.near_goal_steps *= self.near_goal.to(torch.long) # avoid swing behavior
        # print("avoid swing behavior, self.near_goal_steps: ", self.near_goal_steps)

        
        is_success = self.near_goal_steps >= self.success_steps
        # print(f"is_success: {is_success}")
        goal_resets = is_success
        self.reset_goal_buf[:] = goal_resets
        
        self.extras["near_goal_steps"] = self.near_goal_steps.clone()
        self.extras["successes"] = self.successes.clone()

        # self.compute_action_reward(actions)
        # self.extras["action_penalty"] = self.action_penalty_scaled.clone()
        
        bonus_rew = self.near_goal * (self.reach_goal_bonus / self.success_steps)

        # initialize reward
        self.rew_buf[:] = 0.0
        if "target" in reward_types:
            self.rew_buf[:] += self.targ_rew_scaled
            reward_types.remove("target")
        if "bonus" in reward_types:
            self.rew_buf[:] += bonus_rew
            reward_types.remove("bonus")
        if "success" in reward_types:
            self.rew_buf[:] += is_success * 4000
            reward_types.remove("success")
        if "reach" in reward_types:
            self.rew_buf[:] += self.reach_rew_scaled
            reward_types.remove("reach")
        if "energy_reach" in reward_types:
            self.rew_buf[:] += self.reach_curiosity_rew_scaled
            reward_types.remove("energy_reach")
        if "contact_coverage" in reward_types:
            self.rew_buf[:] += self.contact_coverage_rew_scaled
            reward_types.remove("contact_coverage")

        assert len(reward_types) == 0, f"Unknown reward types {reward_types} specified."


        
        self.task_reward = self.rew_buf.clone()
        
        # self.compute_safety_penalty()
        
        self.rew_buf[:] = self.task_reward


        self.compute_done(is_success)