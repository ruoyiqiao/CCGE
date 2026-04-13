from typing import Any, Dict, Optional, Tuple

from isaacgym import gymapi
import torch
import numpy as np

from .leap_singulation import XArmLeapHandFunctionalManipulationUnderarm


class XArmLeapHandCubeInBox(XArmLeapHandFunctionalManipulationUnderarm):

    # _xarm_right_init_position = [0.00, 0.55, 0.00]
    _table_pose = [0.0, 0.0, 0.10]
    _xarm_right_init_position = [0.00, 0.40, 0.00]
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # Read/override environment options for Cube-In-Box
        env_cfg = cfg["env"]

        env_cfg["numObjects"] = 1
        env_cfg["numObjectsPerEnv"] = 1
        env_cfg.setdefault("observationSpecs", {}).setdefault("__dim__", {})["num_nearest_non_targets"] = 0

        self.container_inner_width = float(env_cfg.get("containerInnerWidth", 0.18))    # X (m)
        self.container_inner_depth = float(env_cfg.get("containerInnerDepth", 0.18))    # Y (m)
        self.container_height = float(env_cfg.get("containerHeight", 0.15))             # Z (m)
        self.container_wall_thickness = float(env_cfg.get("containerWallThickness", 0.01))
        self.container_floor_thickness = float(env_cfg.get("containerFloorThickness", 0.01))
        self.cube_height = float(env_cfg.get("cubeHeight", 0.12))
        self.cube_width = float(env_cfg.get("cubeWidth", 0.04))
        self.cube_length = float(env_cfg.get("cubeLength", 0.04))
        self.cube_init_jitter = float(env_cfg.get("cubeInitJitter", 0.0))              # random jitter inside container
        self.cube_out_clearance = float(env_cfg.get("cubeOutClearance", 0.02))         # success margin above rim

        # Book-keeping for container indices (optional)
        self._container_parts = ["floor", "wall_pos_y", "wall_neg_y", "wall_pos_x", "wall_neg_x"]
        self.container_indices = None

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

        # Precompute constant world Z of container rim for success check, per env tensor
        table_top_z = self._table_pose_tensor[2] + self._table_thickness * 0.5
        rim_z = table_top_z + self.container_floor_thickness + self.container_height
        self._container_rim_z = torch.full((self.num_envs,), float(rim_z), device=self.device, dtype=torch.float)

        if cfg["env"]["curiosity"]["enable_occlusion"]:
            self._occlusion_aabb_getter = self.get_container_aabbs
            self.reach_curiosity_mgr.set_occlusion_module(self._occlusion_aabb_getter())

    def _create_box_grid_dataset(self, device=None) -> None:
        # Create simple box grid dataset for singulation task
        from .dataset import BoxGridDataset

        self.grasping_dataset = BoxGridDataset(
            grid_rows=self._grid_rows,
            grid_cols=self._grid_cols,
            grid_layers=self._grid_layers,
            box_width=self.cube_width,
            box_depth=self.cube_length,
            box_height=self.cube_height,
            device=device,
        )

        self.num_categories = self.grasping_dataset._category_matrix.shape[1]

    def _define_container(self) -> dict:
        """
        Define static box container parts (floor + 4 walls).
        Assets are created once; poses are applied per-env when spawning.
        """
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001

        # Dimensions
        W = self.container_inner_width
        D = self.container_inner_depth
        H = self.container_height
        t = self.container_wall_thickness
        tf = self.container_floor_thickness

        # Create floor and 4 walls as separate assets
        floor_asset = self.gym.create_box(self.sim, W, D, tf, asset_options)

        # Walls along Y (front/back), extruded in Z
        wall_y_asset = self.gym.create_box(self.sim, W + 2 * t, t, H, asset_options)
        # Walls along X (left/right), extruded in Z
        wall_x_asset = self.gym.create_box(self.sim, t, D + 2 * t, H, asset_options)

        # Set some friction/restitution
        for a in [floor_asset, wall_y_asset, wall_x_asset]:
            rprops = self.gym.get_asset_rigid_shape_properties(a)
            for s in rprops:
                s.friction = 0.01
                s.restitution = 0.01
            self.gym.set_asset_rigid_shape_properties(a, rprops)

        # Store container assets (we'll place them per-env in _create_envs)
        return {
            "name": "container",
            "assets": {
                "floor": floor_asset,
                "wall_pos_y": wall_y_asset,
                "wall_neg_y": wall_y_asset,
                "wall_pos_x": wall_x_asset,
                "wall_neg_x": wall_x_asset,
            },
            "num_rigid_bodies": sum(self.gym.get_asset_rigid_body_count(a) for a in [floor_asset, wall_y_asset, wall_x_asset]),
            "num_rigid_shapes": sum(self.gym.get_asset_rigid_shape_count(a) for a in [floor_asset, wall_y_asset, wall_x_asset]),
        }

    def _define_table(self) -> Dict[str, Any]:
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

    def _define_object(self, dataset: str = "cube") -> dict:
        """
        Define one dynamic cube as the target object (no clutter).
        """
        config = {}
        config["warehouse"] = {"targ_obj": [], "surr_obj": []}

        target_asset_options = gymapi.AssetOptions()
        target_asset_options.density = 250.0
        target_asset_options.convex_decomposition_from_submeshes = True
        target_asset_options.override_com = True
        target_asset_options.override_inertia = True

        cube = self.gym.create_box(self.sim, self.cube_width, self.cube_length, self.cube_height, target_asset_options)

        rprops = self.gym.get_asset_rigid_shape_properties(cube)
        for s in rprops:
            s.friction = 0.9
            s.restitution = 0.05
        self.gym.set_asset_rigid_shape_properties(cube, rprops)

        config["warehouse"]["targ_obj"].append({
            "name": "target_cube",
            "asset": cube,
            "num_rigid_bodies": self.gym.get_asset_rigid_body_count(cube),
            "num_rigid_shapes": self.gym.get_asset_rigid_shape_count(cube),
        })
        config["count"] = 1

        # Single pose placeholder (filled in _create_envs with proper per-env transforms)
        config["poses"] = [gymapi.Transform()]
        return config

    def compute_aggregate_bodies_and_shapes(self, env: int, gym_assets: Optional[Dict] = None) -> Tuple[int, int]:
        if gym_assets is None:
            gym_assets = self.gym_assets

        num_bodies, num_shapes = 0, 0
        num_bodies += gym_assets["current"]["robot"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["robot"]["num_rigid_shapes"]

        # Only target (no clutter)
        num_current = gym_assets["current"]["objects"]["count"]
        num_bodies += gym_assets["current"]["objects"]["warehouse"]["targ_obj"][(env * self.num_objects_per_env) % num_current]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["objects"]["warehouse"]["targ_obj"][(env * self.num_objects_per_env) % num_current]["num_rigid_shapes"]

        # Table + visual target
        num_bodies += gym_assets["current"]["table"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["table"]["num_rigid_shapes"]
        num_bodies += gym_assets["current"]["visual_target_object"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["visual_target_object"]["num_rigid_shapes"]

        # Container parts (static)
        cont = gym_assets["current"]["container"]
        # 3 assets (floor, wall_xy reused) are used to spawn 5 actors; count per-actor rigid bodies/shapes = asset counts
        n_rb_floor = self.gym.get_asset_rigid_body_count(cont["assets"]["floor"])
        n_rs_floor = self.gym.get_asset_rigid_shape_count(cont["assets"]["floor"])
        n_rb_y = self.gym.get_asset_rigid_body_count(cont["assets"]["wall_pos_y"])  # same as wall_neg_y
        n_rs_y = self.gym.get_asset_rigid_shape_count(cont["assets"]["wall_pos_y"])  # same as wall_neg_y
        n_rb_x = self.gym.get_asset_rigid_body_count(cont["assets"]["wall_pos_x"])  # same as wall_neg_x
        n_rs_x = self.gym.get_asset_rigid_shape_count(cont["assets"]["wall_pos_x"])  # same as wall_neg_x
        num_bodies += (n_rb_floor + 2 * n_rb_y + 2 * n_rb_x)
        num_shapes += (n_rs_floor + 2 * n_rs_y + 2 * n_rs_x)

        if self.use_upper_shelf:
            num_bodies += gym_assets["current"]["upper_shelf"]["num_rigid_bodies"]
            num_shapes += gym_assets["current"]["upper_shelf"]["num_rigid_shapes"]

        return num_bodies, num_shapes

    def _create_envs(self, num_envs: int, spacing: float, num_objects_per_env: int = 1):
        print(">>> Setting up %d environments (Leap CubeInBox)" % num_envs)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(num_envs))

        print(">>> Defining gym assets")
        self.gym_assets["current"]["robot"] = self._define_allegro_hand_with_arm()
        self.gym_assets["current"]["objects"] = self._define_object()  # single cube
        self.gym_assets["current"]["table"] = self._define_table()
        self.gym_assets["current"]["visual_target_object"] = self._define_visual_target_object()
        if self.use_upper_shelf:
            self.gym_assets["current"]["upper_shelf"] = self._define_upper_shelf()

        # Container assets (static)
        self.gym_assets["current"]["container"] = self._define_container()

        self._define_camera()
        print(">>> Done defining gym assets")

        max_aggregate_bodies, max_aggregate_shapes = self.compute_maximum_aggregate_bodies_and_shapes()

        self.envs = []
        self.cameras_handle = []
        allegro_hand_indices = []
        table_indices = []
        upper_shelf_indices = [] if self.use_upper_shelf else None
        visual_target_object_indices = []
        container_indices = []  # list of lists (per env) of 5 actors

        object_indices = [[] for _ in range(num_envs)]
        object_encodings = [[] for _ in range(num_envs)]
        object_names = [[] for _ in range(num_envs)]
        occupied_object_indices = []
        # For CubeInBox we don't keep non-targets bookkeeping
        scene_object_indices = [[] for _ in range(num_envs)]
        # occupied_object_indices_per_env = [0 for _ in range(num_envs)]  # always the single cube (index 0)

        print(">>> Creating environments")
        print("    - max_aggregate_bodies: ", max_aggregate_bodies)
        print("    - max_aggregate_shapes: ", max_aggregate_shapes)

        # Precompute container base pose (same for all envs)
        table_top_z = self._table_pose[2] + self._table_thickness * 0.5
        base_z = table_top_z + self.container_floor_thickness * 0.5

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.aggregate_tracker.reset()

            if self.aggregate_mode != 0:
                _nb, _ns = self.compute_aggregate_bodies_and_shapes(i)
                ok = self.gym.begin_aggregate(env, max_aggregate_bodies, max_aggregate_shapes, True)
                if not ok:
                    raise RuntimeError("begin_aggregate failed")

            # Add robot
            actor_index, actor_handle = self._create_sim_actor(env, self.gym_assets["current"]["robot"], i, actor_handle=True)
            allegro_hand_indices.append(actor_index)

            # Place container parts (static) centered at table center
            cont = self.gym_assets["current"]["container"]["assets"]
            W, D, H, t, tf = (self.container_inner_width, self.container_inner_depth, self.container_height,
                              self.container_wall_thickness, self.container_floor_thickness)
            cx, cy = self._table_pose[0], self._table_pose[1]
            container_handles = []

            # Floor
            floor_pose = gymapi.Transform()
            floor_pose.p = gymapi.Vec3(cx, cy, base_z)
            floor_pose.r = gymapi.Quat(0, 0, 0, 1)
            f_actor_index, f_actor_handle = self._create_sim_actor(env, {"asset": cont["floor"], "name": "container_floor",
                                              "num_rigid_bodies": 1, "num_rigid_shapes": 1,
                                              "pose": floor_pose}, i, actor_handle=True, color=gymapi.Vec3(0.2, 0.2, 0.2))
            container_handles.append(f_actor_handle)

            # Walls: centered at rim height midpoint
            wall_center_z = table_top_z + tf + H * 0.5

            # +Y wall
            wall_posy_pose = gymapi.Transform()
            wall_posy_pose.p = gymapi.Vec3(cx, cy + (D * 0.5 + t * 0.5), wall_center_z)
            wall_posy_pose.r = gymapi.Quat(0, 0, 0, 1)
            y_plus_actor_index, y_plus_actor_handle = self._create_sim_actor(env, {"asset": cont["wall_pos_y"], "name": "container_wall_pos_y",
                                              "num_rigid_bodies": 1, "num_rigid_shapes": 1,
                                              "pose": wall_posy_pose}, i, actor_handle=True, color=gymapi.Vec3(0.25, 0.25, 0.25))
            container_handles.append(y_plus_actor_handle)

            # -Y wall
            wall_negy_pose = gymapi.Transform()
            wall_negy_pose.p = gymapi.Vec3(cx, cy - (D * 0.5 + t * 0.5), wall_center_z)
            wall_negy_pose.r = gymapi.Quat(0, 0, 0, 1)
            y_minus_actor_index, y_minus_actor_handle = self._create_sim_actor(env, {"asset": cont["wall_neg_y"], "name": "container_wall_neg_y",
                                              "num_rigid_bodies": 1, "num_rigid_shapes": 1,
                                              "pose": wall_negy_pose}, i, actor_handle=True, color=gymapi.Vec3(0.25, 0.25, 0.25))
            container_handles.append(y_minus_actor_handle)

            # +X wall
            wall_posx_pose = gymapi.Transform()
            wall_posx_pose.p = gymapi.Vec3(cx + (W * 0.5 + t * 0.5), cy, wall_center_z)
            wall_posx_pose.r = gymapi.Quat(0, 0, 0, 1)
            x_plus_actor_index, x_plus_actor_handle = self._create_sim_actor(env, {"asset": cont["wall_pos_x"], "name": "container_wall_pos_x",
                                              "num_rigid_bodies": 1, "num_rigid_shapes": 1,
                                              "pose": wall_posx_pose}, i, actor_handle=True, color=gymapi.Vec3(0.25, 0.25, 0.25))
            container_handles.append(x_plus_actor_handle)

            # -X wall
            wall_negx_pose = gymapi.Transform()
            wall_negx_pose.p = gymapi.Vec3(cx - (W * 0.5 + t * 0.5), cy, wall_center_z)
            wall_negx_pose.r = gymapi.Quat(0, 0, 0, 1)
            x_minus_actor_index, x_minus_actor_handle = self._create_sim_actor(env, {"asset": cont["wall_neg_x"], "name": "container_wall_neg_x",
                                              "num_rigid_bodies": 1, "num_rigid_shapes": 1,
                                              "pose": wall_negx_pose}, i, actor_handle=True, color=gymapi.Vec3(0.25, 0.25, 0.25))
            container_handles.append(x_minus_actor_handle)
            container_indices.append(container_handles)

            # Add cube (target) at center with optional jitter, resting on floor
            poses = self.gym_assets["current"]["objects"]["poses"]
            cube_pose = gymapi.Transform()
            jx = (np.random.rand() - 0.5) * self.cube_init_jitter
            jy = (np.random.rand() - 0.5) * self.cube_init_jitter
            cube_pose.p = gymapi.Vec3(
                cx + jx, cy + jy, table_top_z + self.container_floor_thickness + self.cube_height * 0.5 + 1e-3
            )
            cube_pose.r = gymapi.Quat(0, 0, 0, 1)
            poses[0] = cube_pose

            cfg = self.gym_assets["current"]["objects"]["warehouse"]["targ_obj"][0]
            actor_index = self._create_sim_actor(env, cfg, i, "targ_obj", cube_pose, color=gymapi.Vec3(0.9, 0.9, 0.9))
            object_indices[i].append(actor_index)
            object_names[i].append(cfg["name"])
            object_encodings[i].append(0)  # relative index in scene
            occupied_object_indices.append(actor_index)
            scene_object_indices[i].append(0)

            # Add table (static)
            actor_index, actor_handle = self._create_sim_actor(
                env, self.gym_assets["current"]["table"], i, actor_handle=True, color=gymapi.Vec3(0.0, 0.0, 0.0)
            )
            table_indices.append(actor_handle)

            # Visual goal marker (optional)
            actor_index, actor_handle = self._create_sim_actor(
                env, self.gym_assets["current"]["visual_target_object"], i + self.num_envs,
                actor_handle=True, color=gymapi.Vec3(0.6, 0.72, 0.98)
            )
            visual_target_object_indices.append(actor_handle)

            # Cameras if enabled
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

        # Indices and ranges consistent with base class
        env = self.envs[0]
        allegro_hand = self.gym.find_actor_handle(env, "allegro_hand")
        self.allegro_hand_index = self.gym.get_actor_index(env, allegro_hand, gymapi.DOMAIN_ENV)

        # Target object rigid body index (the cube) per env
        self.target_object_rigid_body_indices = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.surr_object_rigid_body_indices = torch.zeros((num_envs, 0), dtype=torch.long, device=self.device)  # no clutter

        for i in range(num_envs):
            env_i = self.envs[i]
            target_obj_handle = self.gym.find_actor_handle(env_i, "targ_obj")
            target_object_actor = self.gym.get_actor_index(env_i, target_obj_handle, gymapi.DOMAIN_ENV)
            target_rb_index = self.gym.get_actor_rigid_body_index(
                env_i, target_object_actor, 0, gymapi.DOMAIN_SIM
            )
            self.target_object_rigid_body_indices[i] = target_rb_index

        # DOF and rigid body slice ranges
        self.allegro_hand_dof_start = self.gym.get_actor_dof_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_dof_end = self.allegro_hand_dof_start + self.gym_assets["current"]["robot"]["num_dofs"]
        self.allegro_hand_indices = torch.tensor(allegro_hand_indices).long().to(self.device)
        self.allegro_hand_rigid_body_start = self.gym.get_actor_rigid_body_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_rigid_body_end = (
            self.allegro_hand_rigid_body_start + self.gym_assets["current"]["robot"]["num_rigid_bodies"]
        )

        self.table_indices = torch.tensor(table_indices).long().to(self.device)
        self.visual_target_object_indices = torch.tensor(visual_target_object_indices).long().to(self.device)
        self.upper_shelf_indices = torch.tensor(upper_shelf_indices).long().to(self.device) if self.use_upper_shelf else None
        self.object_indices = torch.tensor(object_indices).long().to(self.device)
        self.object_names = object_names
        self.object_encodings = torch.tensor(object_encodings).long().to(self.device)

        self.occupied_object_indices = torch.tensor(occupied_object_indices).long().to(self.device)
        self.occupied_object_relative_indices = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.non_occupied_object_indices = torch.zeros((num_envs, 0), dtype=torch.long, device=self.device)
        self.surr_object_indices = torch.zeros((num_envs, 0), dtype=torch.long, device=self.device)
        self.scene_object_indices = torch.tensor(scene_object_indices).long().to(self.device)

        self.container_indices = container_indices
        

        self.table_rigid_body_indices = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        for i in range(self.num_envs):
            env_i = self.envs[i]
            table_handle_i = int(self.table_indices[i].item())
            table_actor_index = self.gym.get_actor_index(env_i, table_handle_i, gymapi.DOMAIN_ENV)
            table_rb_index = self.gym.get_actor_rigid_body_index(env_i, table_actor_index, 0, gymapi.DOMAIN_SIM)
            self.table_rigid_body_indices[i] = table_rb_index


        self.container_floor_rigid_body_indices = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        for i in range(self.num_envs):
            env_i = self.envs[i]
            floor_handle_i = self.container_indices[i][0]  # [floor, +Y wall, -Y wall, +X wall, -X wall]
            floor_actor_index = self.gym.get_actor_index(env_i, floor_handle_i, gymapi.DOMAIN_ENV)
            floor_rb_index = self.gym.get_actor_rigid_body_index(env_i, floor_actor_index, 0, gymapi.DOMAIN_SIM)
            self.container_floor_rigid_body_indices[i] = floor_rb_index

        self.container_wall_rigid_body_indices = torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device)
        for i in range(self.num_envs):
            env_i = self.envs[i]
            for j in range(4):
                wall_handle_i = self.container_indices[i][j + 1]  # [floor, +Y wall, -Y wall, +X wall, -X wall]
                wall_actor_index = self.gym.get_actor_index(env_i, wall_handle_i, gymapi.DOMAIN_ENV)
                wall_rb_index = self.gym.get_actor_rigid_body_index(env_i, wall_actor_index, 0, gymapi.DOMAIN_SIM)
                self.container_wall_rigid_body_indices[i, j] = wall_rb_index

        print(f">>> Done creating {num_envs} environments (Leap CubeInBox)")

    def get_container_aabbs(self) -> torch.Tensor:
        """
        Returns the 5 AABBs of the container in world coordinates.
        Order: [floor, +Y wall, -Y wall, +X wall, -X wall]
        """
        device = self.device
        table_top_z = self._table_pose_tensor[2] + self._table_thickness * 0.5
        base_z = table_top_z + self.container_floor_thickness * 0.5
        wall_center_z = table_top_z + self.container_floor_thickness + self.container_height * 0.5

        cx = float(self._table_pose_tensor[0])
        cy = float(self._table_pose_tensor[1])
        W, D, H = self.container_inner_width, self.container_inner_depth, self.container_height
        t, tf = self.container_wall_thickness, self.container_floor_thickness

        aabbs = torch.zeros(5, 2, 3, device=device, dtype=torch.float)

        # Floor
        aabbs[0, 0] = torch.tensor([cx - W * 0.5, cy - D * 0.5, base_z - tf * 0.5], device=device)
        aabbs[0, 1] = torch.tensor([cx + W * 0.5, cy + D * 0.5, base_z + tf * 0.5], device=device)

        # +Y wall
        aabbs[1, 0] = torch.tensor([cx - (W * 0.5 + t), cy + D * 0.5, wall_center_z - H * 0.5], device=device)
        aabbs[1, 1] = torch.tensor([cx + (W * 0.5 + t), cy + D * 0.5 + t, wall_center_z + H * 0.5], device=device)

        # -Y wall
        aabbs[2, 0] = torch.tensor([cx - (W * 0.5 + t), cy - D * 0.5 - t, wall_center_z - H * 0.5], device=device)
        aabbs[2, 1] = torch.tensor([cx + (W * 0.5 + t), cy - D * 0.5, wall_center_z + H * 0.5], device=device)

        # +X wall
        aabbs[3, 0] = torch.tensor([cx + W * 0.5, cy - (D * 0.5 + t), wall_center_z - H * 0.5], device=device)
        aabbs[3, 1] = torch.tensor([cx + W * 0.5 + t, cy + (D * 0.5 + t), wall_center_z + H * 0.5], device=device)

        # -X wall
        aabbs[4, 0] = torch.tensor([cx - W * 0.5 - t, cy - (D * 0.5 + t), wall_center_z - H * 0.5], device=device)
        aabbs[4, 1] = torch.tensor([cx - W * 0.5, cy + (D * 0.5 + t), wall_center_z + H * 0.5], device=device)

        return aabbs



    def check_contact_criteria(self):

        tip_contact = self.keypoint_contact_mask.any(dim=-1)
        opposing_ok = torch.ones_like(tip_contact, dtype=torch.bool)

        y_plus_wall_indices = self.container_wall_rigid_body_indices[:, 0]
        y_plus_wall_contact = self.net_contact_forces[y_plus_wall_indices, :].norm(dim=-1) > 1e-2
        table_contact = self.net_contact_forces[self.container_floor_rigid_body_indices, :].norm(dim=-1) > 1e-2
        table_contact |= y_plus_wall_contact
        lift_thresh  = 0.08
        in_max_displacement   = (self.y_displacement < lift_thresh)

        env_contact = torch.zeros_like(table_contact)
        other_wall_contact = self.net_contact_forces[self.container_wall_rigid_body_indices[:, 1:], :].norm(dim=-1) > 1e-2
        env_contact = other_wall_contact.any(dim=-1)

        hand_contact = (tip_contact & opposing_ok) | (table_contact | y_plus_wall_contact)

        self.extras["hand_contact"]        = hand_contact.clone()
        self.extras["in_max_displacement"] = in_max_displacement.clone()
        self.extras["tip_contact"]         = tip_contact.clone()
        self.extras["opposing_normals_ok"] = opposing_ok.clone()
        self.extras["table_contact"]       = table_contact.clone()
        self.extras["env_detach"]          = (~env_contact).clone()
        self.extras["transition_contact"]  = (tip_contact & opposing_ok) & ~table_contact & ~env_contact

        return hand_contact & (~env_contact)



    def compute_targ_reward(self):
        """Compute target reward - reward for reaching the target state."""

        self.goal_position_dist = torch.norm(
            self.goal_position.unsqueeze(0) - self.object_root_positions, dim=1
        )
        # print(self.goal_position_dist[0])
        if not hasattr(self, "goal_position_dist_min"):
            self.goal_position_dist_min = torch.ones_like(self.goal_position_dist) * 0.60
        delta = self.goal_position_dist_min - self.goal_position_dist
        clipped_delta = torch.maximum(delta, torch.zeros_like(delta))
        # clipped_delta = clipped_delta * (self.y_displacement > 0.0).float() * self.contact_satisfied
        clipped_delta = clipped_delta * self.contact_satisfied

        
        self.targ_rew = clipped_delta * 60
        
        self.targ_rew_scaled = self.targ_rew * (1)
        self.near_goal = (self.goal_position_dist <= 0.075).float() * self.contact_satisfied

        self.goal_position_dist_min = torch.where(self.contact_satisfied, torch.min(self.goal_position_dist_min, self.goal_position_dist), self.goal_position_dist_min)
        self.extras["near_goal"] = self.near_goal.clone()
        self.extras["targ_rew"] = self.targ_rew_scaled.clone()
        self.extras["goal_position_dist_min"] = self.goal_position_dist_min.clone()




from .leap_singulation import XArm7LeapHandUnderarmDimensions
class XArm7LeapHandCubeInBox(XArmLeapHandCubeInBox):
    _dims = XArm7LeapHandUnderarmDimensions

    _xarm_allegro_hand_right_asset_file: str = "xarm7_leap_right.urdf"
    # _xarm_allegro_hand_left_asset_file: str = "xarm7_leap_right.urdf"

    _xarm_dof_names = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
    _arm_links = ["link_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7"]
    _xarm_eef_link: str = "link7"

    _xarm_right_init_dof_positions: Dict[str, float] = {
        "joint1": 0.0,
        "joint2": -1.0,
        "joint3": 0.0,
        "joint4": 0.5,
        "joint5": 0.0,
        "joint6": 0.0,
        "joint7": 0.0,
    }

    # good prepose
    # _xarm_right_init_dof_positions: Dict[str, float] = {
    #     "joint1": 0.0,
    #     "joint2": -0.4,
    #     "joint3": 0.0,
    #     "joint4": 0.9,
    #     "joint5": 0.0,
    #     "joint6": 1.3,
    #     "joint7": 0.0,
    # }