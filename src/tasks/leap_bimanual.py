from ast import Slice
import json
import math
import os
from collections import OrderedDict, deque
from re import T
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import omegaconf
import open3d as o3d
import pandas as pd
import pytorch3d
import trimesh
import torch
from dotenv import find_dotenv
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from scipy.spatial.transform import Rotation as R
from torch import LongTensor, Tensor
from .isaacgym_utils import (
    ActionSpec, ObservationSpec,
    draw_points,
    draw_axes,
    draw_boxes,
    get_action_indices,
    ik,
    orientation,
    position,
    print_action_space,
    print_asset_options,
    print_dof_properties,
    print_links_and_dofs,
    print_observation_space,
    random_orientation_within_angle,
    to_torch,
)
from .torch_utils import *
from .curiosity_reward_manager import CuriosityRewardManager, OBBOcclusionMask
from .state_feature_bank import LearnedHashStateBank
from .allegro_singulation import compute_relative_xarm_dof_positions


def rot6d_to_axis_angle(rot_6d: Tensor) -> Tensor:
    """Convert 6D rotation representation to axis-angle.
    Args:
        rot_6d (Tensor): shape (..., 6)
    """
    if rot_6d.numel() == 0:
        return rot_6d
    rot_mtx = rotation_6d_to_matrix(rot_6d.view(-1, 6)).view(*rot_6d.shape[:-1], 3, 3)
    aa = matrix_to_axis_angle(rot_mtx.reshape(-1, 3, 3)).view(*rot_6d.shape[:-1], 3)
    return aa


class AggregateTracker:
    aggregate_bodies: int
    aggregate_shapes: int
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.aggregate_bodies = 0
        self.aggregate_shapes = 0
    
    def update(self, bodies: int, shapes: int):
        self.aggregate_bodies += bodies
        self.aggregate_shapes += shapes


class LeapBimanualManipulationArti(VecTask):
    """Bimanual Leap-hand manipulation environment with floating root control."""
    _asset_root: os.PathLike = os.path.join(os.path.dirname(find_dotenv()), "assets/urdf")
    
    _tip_links: List[str] = ["index_tip_head", "middle_tip_head", "ring_tip_head", "thumb_tip_head"]
    _fingertips: List[str] = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
    
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
    
    _index_finger_links: List[str] = ["pip", "dip", "fingertip", "index_tip_head"]
    _middle_finger_links: List[str] = ["pip_2", "dip_2", "fingertip_2", "middle_tip_head"]
    _ring_finger_links: List[str] = ["pip_3", "dip_3", "fingertip_3", "ring_tip_head"]
    _thumb_links: List[str] = ["thumb_pip", "thumb_dip", "thumb_fingertip", "thumb_tip_head"]
    
    # _tip_links: List[str] = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]
    # _fingertips: List[str] = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]
    
    # _right_hand_links: List[str] = [
    #     "base_link",
    #     "link_0.0", "link_4.0", "link_8.0", "link_12.0",
    #     "link_1.0", "link_5.0", "link_9.0", "link_13.0",
    #     "link_2.0", "link_6.0", "link_10.0", "link_14.0",
    #     "link_3.0", "link_7.0", "link_11.0", "link_15.0",
        
    #     "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip",
    # ]
    # _left_hand_links: List[str] = [
    #     "base_link",
    #     "link_0.0", "link_4.0", "link_8.0", "link_12.0",
    #     "link_1.0", "link_5.0", "link_9.0", "link_13.0",
    #     "link_2.0", "link_6.0", "link_10.0", "link_14.0",
    #     "link_3.0", "link_7.0", "link_11.0", "link_15.0",
        
    #     "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip",
    # ]
    
    # _index_finger_links: List[str] = ["link_0.0", "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip"]
    # _middle_finger_links: List[str] = ["link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip"]
    # _ring_finger_links: List[str] = ["link_8.0", "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip"]
    # _thumb_links: List[str] = ["link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"]
    
    
    _arm_links: List[str] = ["link_base", "link1", "link2", "link3", "link4", "link5", "link6"]
    
    _xarm_dof_names: List[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    _hand_dof_names: List[str] = [
        "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
        "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
        "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
        "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
    ]
    
    _left_xarm_init_dof_positions: Dict[str, float] = {
        "joint1": 0.0,
        "joint2": -1.0,
        "joint3": -0.5,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": -1.57,
    }

    _right_xarm_init_dof_positions: Dict[str, float] = {
        "joint1": 0.0,
        "joint2": -1.0,
        "joint3": -0.5,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 1.57,
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
        self.cfg = cfg
        self.env_cfg = cfg["env"]
        self.sim_cfg = cfg["sim"]

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        
        self.sub_steps = self.cfg["sim"]["substeps"]
        # self.dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        # self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        # self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.translation_scale = self.env_cfg.get("translationScale", 1.0)
        self.orientation_scale = self.env_cfg.get("orientationScale", 1.0)
        self.finger_action_scale = self.env_cfg.get("fingerActionScale", 20.0)
        self.actions_moving_average = self.env_cfg.get("actionsMovingAverage", 0.8)
        self.root_control_dim = self.env_cfg.get("rootControlDim", 9)
        
        self.max_eef_translation_speed = self.env_cfg.get("maxEefTranslationSpeed", 0.15)
        self.max_eef_rotation_speed = self.env_cfg.get("maxEefRotationSpeed", 1.0)

        self.logging_cfg = cfg.get("logging", {})
        self.stack_frame_number = self.env_cfg.get("stackFrameNumber", 1)
        self.frames: deque = deque(maxlen=self.stack_frame_number)
        self.velocity_observation_scale = self.cfg["env"]["velocityObservationScale"]
        self.reward_type = self.cfg["env"]["rewardType"]
        self.use_precomputed_poses = False
        self.use_pre_poses_init = False
        
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.env_info_logging = self.cfg["logging"]["envInfo"]
        
        self.training = True
        
        # Hand configuration
        self.hand_cfg = self.env_cfg.get("hands", {})
        self.hand_base_links = {
            "right": self.hand_cfg["right"]["wristLink"],
            "left": self.hand_cfg["left"]["wristLink"]
        }
        
        self.task_cfg = self.env_cfg.get("taskConfig", {})
        self.target_specs = self.task_cfg.get("targets", {})
        self.distance_reward_scale = abs(
            self.task_cfg.get("distanceRewardScale", self.env_cfg.get("distanceRewardScale", 1.0))
        )
        self.success_steps_required = self.task_cfg.get("successSteps", self.env_cfg.get("successSteps", 40))
        self.success_bonus = self.task_cfg.get("successBonus", self.env_cfg.get("successBonus", 200.0))
        self.terminate_on_success = self.task_cfg.get(
            "terminateOnSuccess", self.env_cfg.get("terminateOnSuccess", True)
        )
        self.task_description = self.task_cfg.get("description", "")
        self.mode = self.env_cfg.get("mode", self.cfg.get("mode", "train"))
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.object_cfgs: List[Dict[str, Any]] = self.env_cfg.get("objects", [])
        self.tracked_object_mask: List[bool] = [bool(obj.get("track", False)) for obj in self.object_cfgs]
        tracked_objects = sum(self.tracked_object_mask)
        self.num_tracked_objects = max(1, tracked_objects) if self.object_cfgs else 1
        # self.num_hand_dofs = self.env_cfg.get("numHandDofs", 16) + 6 # for dummy dof
        self.num_hand_dofs = self.env_cfg.get("numHandDofs", 16)

        if cfg["env"].get("returnCuriosityInfo", False):
            self.curiosity_state_type = cfg["env"]["CuriosityInfo"].get("curiosityStateType", "policy_state")  # or "contact_force" or "contact_distance"
    
        self._root_states_view: Optional[Tensor] = None
        
        self.hand_keypoints: Dict[str, Dict[str, List[List[float]]]] = {}
        left_hand_keypoints_info_path = self.hand_cfg["left"]["keypoints"]
        right_hand_keypoints_info_path = self.hand_cfg["right"]["keypoints"]
        if os.path.exists(left_hand_keypoints_info_path):
            with open(left_hand_keypoints_info_path, "r") as f:
                self.hand_keypoints["left"] = json.load(f)
        if os.path.exists(right_hand_keypoints_info_path):
            with open(right_hand_keypoints_info_path, "r") as f:
                self.hand_keypoints["right"] = json.load(f)
        self._left_hand_flattened_keypoint_pairs: List[Tuple[str, int]] = []
        for link_name, offsets in self.hand_keypoints["left"].items():
            for k in range(len(offsets)):
                self._left_hand_flattened_keypoint_pairs.append((link_name, k))
        self._left_hand_keypoints = [ln for (ln, _) in self._left_hand_flattened_keypoint_pairs]
        
        self._right_hand_flattened_keypoint_pairs: List[Tuple[str, int]] = []
        for link_name, offsets in self.hand_keypoints["right"].items():
            for k in range(len(offsets)):
                self._right_hand_flattened_keypoint_pairs.append((link_name, k))
        self._right_hand_keypoints = [ln for (ln, _) in self._right_hand_flattened_keypoint_pairs]
        
        # Storage for specification metadata
        self._observation_specs: Sequence[ObservationSpec] = []
        self._action_specs: Sequence[ActionSpec] = []
        self._observation_space: Sequence[ObservationSpec] = []
        self._observation_space_extra: Sequence[ObservationSpec] = []
        self._action_space: Sequence[ActionSpec] = []
        self.observation_info: Dict[str, int] = {}
        self.gym_assets = {}
        self.gym_assets["current"] = {}
        self.gym_assets["target"] = {}
        
        # Aggregate tracker
        self.aggregate_tracker = AggregateTracker()
        
        # Configure MDP spaces
        self._configure_mdp_spaces()
        
        # Initialize VecTask
        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        
        # Initialize tensor buffers
        self.hand_dof_indices = {"right": [], "left": []}
        self.object_dof_indices = []
        # self._identify_dof_indices()
        
        self.use_relative_control = self.env_cfg.get("useRelativeControl", False)
        self.dof_speed_scale = self.env_cfg.get("dofSpeedScale", 1.0)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        
        # Hand joint limits
        self.hand_dof_lower_limits = {}
        self.hand_dof_upper_limits = {}
        for side in ["right", "left"]:
            asset = self.gym_assets["current"][f"hand_{side}"]["asset"]
            dof_props = self.gym.get_asset_dof_properties(asset)
            self.hand_dof_lower_limits[side] = torch.tensor(dof_props["lower"]).to(self.device)
            self.hand_dof_upper_limits[side] = torch.tensor(dof_props["upper"]).to(self.device)
        
        # Configure viewer
        self._configure_viewer()
        
        # PID control buffers
        self.use_pid_control = self.env_cfg.get("usePIDControl", False)
        if self.use_pid_control:
            self.kp_rot = self.env_cfg.get("kp_pos", 0.300)
            self.ki_rot = self.env_cfg.get("ki_pos", 0.001)
            self.kd_rot = self.env_cfg.get("kd_pos", 0.005)
            self.kp_pos = self.env_cfg.get("kp_rot", 20.0)
            self.ki_pos = self.env_cfg.get("ki_rot", 0.05)
            self.kd_pos = self.env_cfg.get("kd_rot", 0.20)
            self._allocate_pid_buffers()
            
        self.enable_exploration_logging = self.cfg["env"].get("enableExplorationLogging", False)
            
        # 扩展物体配置，支持多部件
        self._object_parts = {}  # 存储物体各部分信息
        self._curiosity_managers = {}  # 为每只手单独的curiosity管理器
        self._task_joint_indices = {}  # 任务相关关节的索引
        
        # 初始化物体部件
        self._init_object_parts()
        
        # 初始化curiosity managers
        self._init_curiosity_managers()
        
        self._finalize_env_setup()

        if self.cfg["env"].get("returnCuriosityInfo", False):
            self.curiosity_state_dim = self.extras["curiosity_states"].shape[1:] # remove env dim
        

    def _init_object_parts(self):
        """初始化物体部件，包括加载点云和设置索引"""
        obj_cfg = self.object_cfgs[0]  # 假设只有一个物体
        parts = self._parse_bottle_parts(obj_cfg["assetRoot"], obj_cfg["assetFile"])
        self._object_parts = parts
        
        # 设置关节索引
        task_joint_name = obj_cfg["taskJoint"]  # 从配置获取任务关节
        asset = self.gym_assets["current"]["objects"]["warehouse"][0]["asset"]
        joint_index = self.gym.find_asset_dof_index(asset, task_joint_name)
        if joint_index != -1:
            self._task_joint_indices[obj_cfg["taskType"]] = torch.tensor(joint_index, dtype=torch.long, device=self.device)
        
        # 设置刚体索引
        for part_name, part_info in parts.items():
            part_info["rigid_body_indices"] = []
            for link_name in part_info["link_names"]:
                rb_index = self.gym.find_asset_rigid_body_index(
                    self.gym_assets["current"]["objects"]["warehouse"][0]["asset"], 
                    link_name
                )
                if rb_index != -1:
                    part_info["rigid_body_indices"].append(rb_index)
                    
        for part_name, part_info in self._object_parts.items():
            if not part_info.get("visual_meshes"):
                continue
                
            mesh_paths = [m["path"] for m in part_info["visual_meshes"]]
            scales = [m["scale"] for m in part_info["visual_meshes"]]
            
            to_origin, extents, combined_mesh = self._compute_part_obb(mesh_paths, scales)

            part_info["obb_transform"] = torch.tensor(to_origin, dtype=torch.float32, device=self.device)
            part_info["obb_extents"] = torch.tensor(extents, dtype=torch.float32, device=self.device)
            
            # points = combined_mesh.sample(self.num_object_points)
            # part_info["pointcloud"] = torch.tensor(points, dtype=torch.float32, device=self.device)

    def _init_curiosity_managers(self):
        
        self._curiosity_managers["left"] = {}
        self._curiosity_managers["right"] = {}
        
        for part_name in ["top", "bottom"]:
            for hand_side in ["left", "right"]:
                self._curiosity_managers[hand_side][part_name] = (
                    CuriosityRewardManager(
                        num_keypoints=4,
                        device=self.device,
                        canonical_pointcloud=self._object_parts[part_name]["pointcloud"],
                        cluster_k=32,
                        max_clustering_iters=10,
                        canonical_normals=self._object_parts[part_name]["pointcloud_normals"],
                        mask_backface_points=True,
                        mask_palm_inward_points=True,
                        use_normal_in_clustering=False,

                        # state-feature bank
                        state_feature_dim=self.env_cfg.get("stateFeatureDim", None),
                        num_key_states=int(self.env_cfg.get("numKeyStates", 32)),
                        state_counter_mode=str(self.env_cfg.get("stateCounterMode", "cluster")),
                        state_include_goal=bool(self.cfg["env"].get("stateIncludeGoal", False)), # hash state exclusive config

                        hash_code_dim=int(self.cfg["env"].get("hashCodeDim", 16)),
                        hash_noise_scale=float(self.cfg["env"].get("hashNoiseScale", 0.3)),
                        hash_lambda_binary=float(self.cfg["env"].get("hashLambdaBinary", 1.0)),
                        hash_ae_lr=float(self.cfg["env"].get("hashAeLr", 3e-4)),
                        hash_ae_steps=int(self.cfg["env"].get("hashAeSteps", 5)),
                        hash_ae_update_freq=int(self.cfg["env"].get("hashAeUpdateFreq", 16)),
                        hash_ae_num_minibatches=int(self.cfg["env"].get("hashAeNumMinibatches", 8)),
                        hash_seed=int(self.cfg["env"].get("hashSeed", 0)),

                        state_type=str(self.cfg["env"].get("stateType", "pcd")), # so3 (#key_states: 12, 24, 60) / pcd /  hash
                        state_running_max_mode=str(self.cfg["env"].get("stateRunningMaxMode", "state")),

                        num_envs=self.num_envs,
                    )
                )
                occ_name = "top" if part_name == "bottom" else "bottom"
                occ_info = self._object_parts[occ_name]
                self._curiosity_managers[hand_side][part_name].set_obb_occlusion_module(
                    obb_transform=occ_info["obb_transform"],
                    obb_extents=occ_info["obb_extents"],
                )

        state_type = str(self.cfg["env"].get("stateType", "pcd")).lower()
        if state_type == "hash":
            for hand_side in ["left", "right"]:
                top_bank = self._curiosity_managers[hand_side]["top"].state_bank
                bottom_bank = self._curiosity_managers[hand_side]["bottom"].state_bank
                if isinstance(top_bank, LearnedHashStateBank) and isinstance(bottom_bank, LearnedHashStateBank):
                    bottom_bank.share_codebook_from(top_bank, update_enabled=False)

        # for hand_side, occ_part_name in [("right", "bottom"), ("left", "top")]:
        #     occ_info = self._object_parts[occ_part_name]
        #     self._curiosity_managers[hand_side].set_obb_occlusion_module(
        #         obb_transform=occ_info["obb_transform"],
        #         obb_extents=occ_info["obb_extents"],
        #     )

    # -------------------------------------------------------------------------
    # Isaac Gym setup
    # -------------------------------------------------------------------------
    def create_sim(self):
        self.dt = self.sim_cfg["dt"]
        self.up_axis_idx = 2  # z axis
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_cfg["envSpacing"])
        if self.env_cfg.get("randomize", False):
            self.apply_randomizations(self.cfg["task"]["randomization_params"])

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def train(self):
        self.training = True

    def eval(self, vis=False):
        self.training = False

    def _configure_viewer(self):
        """Viewer setup."""
        if self.viewer is not None:
            cam_pos = gymapi.Vec3(0.5, -0.3, 0.8)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.4)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _define_table(self) -> Dict[str, Any]:
        """Define a table asset for supporting objects."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        
        # Table dimensions
        self.table_x_length = 0.15
        self.table_y_length = 0.15
        self.table_thickness = 0.2
        self.table_height = 0.5
        
        # Create table asset
        table_asset = self.gym.create_box(
            self.sim, self.table_x_length, self.table_y_length, self.table_thickness, asset_options
        )
        
        # Configure rigid body properties
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for shape in rigid_shape_props:
            shape.friction = 1.0
            shape.restitution = 0.0
        self.gym.set_asset_rigid_shape_properties(table_asset, rigid_shape_props)
        
        num_rigid_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_rigid_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        
        # Table pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, self.table_height - self.table_thickness/2.0)
        
        return {
            "asset": table_asset,
            "pose": pose,
            "name": "table",
            "num_rigid_bodies": num_rigid_bodies,
            "num_rigid_shapes": num_rigid_shapes,
        }

    # -------------------------------------------------------------------------
    # Environment creation
    # -------------------------------------------------------------------------
    def compute_maximum_aggregate_bodies_and_shapes(self) -> Tuple[int, int]:
        """Compute the maximum number of rigid bodies and shapes in the environment."""
        max_aggregate_bodies, max_aggregate_shapes = 0, 0
        for _ in range(self.num_envs):
            num_bodies, num_shapes = self.compute_aggregate_bodies_and_shapes()
            max_aggregate_bodies = max(max_aggregate_bodies, num_bodies)
            max_aggregate_shapes = max(max_aggregate_shapes, num_shapes)
        return max_aggregate_bodies, max_aggregate_shapes

    def compute_aggregate_bodies_and_shapes(self) -> Tuple[int, int]:
        """Compute the number of rigid bodies and shapes in a single environment."""
        num_bodies, num_shapes = 0, 0
        
        # Add hand bodies and shapes
        for side in ["right", "left"]:
            num_bodies += self.gym_assets["current"][f"hand_{side}"]["num_rigid_bodies"]
            num_shapes += self.gym_assets["current"][f"hand_{side}"]["num_rigid_shapes"]
        
        # Add table bodies and shapes
        if "table" in self.gym_assets["current"]:
            num_bodies += self.gym_assets["current"]["table"]["num_rigid_bodies"]
            num_shapes += self.gym_assets["current"]["table"]["num_rigid_shapes"]
        
        # Add object bodies and shapes
        if "objects" in self.gym_assets["current"]:
            # Get the warehouse list containing all object configurations
            warehouse = self.gym_assets["current"]["objects"]["warehouse"]
            for obj_cfg in warehouse:
                num_bodies += obj_cfg["num_rigid_bodies"]
                num_shapes += obj_cfg["num_rigid_shapes"]
        
        return num_bodies, num_shapes


    def _identify_dof_indices(self):
        """Identify and record DOF indices for hands and objects."""
        # Initialize lists
        for side in ["right", "left"]:
            self.hand_dof_indices[side] = []
        self.object_dof_indices = []
        
        # Collect all hand DOF indices
        for env_id in range(self.num_envs):
            for side in ["right", "left"]:
                dof_start = self.hand_dof_starts[side]
                num_dofs = self.gym_assets["current"][f"hand_{side}"]["num_dofs"]
                # For the first environment, collect DOF indices (assuming all environments have the same DOF arrangement)
                if env_id == 0:
                    for i in range(num_dofs):
                        dof_idx = self.gym.get_actor_dof_index(
                            self.envs[env_id], 
                            self.hand_actor_handles[side][env_id], 
                            i, 
                            gymapi.DOMAIN_SIM
                        )
                        self.hand_dof_indices[side].append(dof_idx)
        
        # Collect all object DOF indices
        if self.object_dofs_per_env > 0:
            for env_id in range(self.num_envs):
                for obj_idx, obj_cfg in enumerate(self.gym_assets["current"]["objects"]["warehouse"]):
                    num_obj_dofs = obj_cfg.get("num_dofs", 0)
                    if num_obj_dofs > 0 and env_id == 0:  # Only collect for the first environment
                        obj_handle = self.object_actor_handles[env_id][obj_idx]
                        for dof_id in range(num_obj_dofs):
                            dof_idx = self.gym.get_actor_dof_index(
                                self.envs[env_id], 
                                obj_handle, 
                                dof_id, 
                                gymapi.DOMAIN_SIM
                            )
                            self.object_dof_indices.append(dof_idx)
        
        # Convert to tensors
        self.hand_dof_indices_t = {
            "right": torch.tensor(self.hand_dof_indices["right"], dtype=torch.long, device=self.device),
            "left": torch.tensor(self.hand_dof_indices["left"], dtype=torch.long, device=self.device)
        }
        self.object_dof_indices_t = torch.tensor(self.object_dof_indices, dtype=torch.long, device=self.device)
        
        # Get all hand DOF indices
        all_hand_dofs = torch.cat([self.hand_dof_indices_t["right"], self.hand_dof_indices_t["left"]])
        self.all_hand_dof_indices = all_hand_dofs.sort()[0]
        
    def _compute_part_obb(self, mesh_paths, scales):
        meshes = []
        for path, scale in zip(mesh_paths, scales):
            mesh = trimesh.load_mesh(path, process=False)
            S = np.eye(4)
            S[:3, :3] *= scale
            mesh.apply_transform(S)
            meshes.append(mesh)
        
        combined_mesh = trimesh.util.concatenate(meshes)
        
        to_origin, extents = trimesh.bounds.oriented_bounds(combined_mesh)
        
        return to_origin, extents, combined_mesh
        
    def _parse_bottle_parts(self, asset_root: str, asset_file: str) -> Dict[str, Dict]:
        """解析物体各部分信息，加载对应的网格并生成点云"""
        parts = {
            "bottom": {
                "link_names": ["bottom"],
                "visual_meshes": [],
                "pointcloud": None,
                "pointcloud_normals": None,
                "rigid_body_indices": []
            },
            "top": {
                "link_names": ["top"],
                "visual_meshes": [],
                "pointcloud": None,
                "pointcloud_normals": None,
                "rigid_body_indices": []
            }
        }
        
        # 解析URDF获取网格信息
        import xml.etree.ElementTree as ET
        urdf_path = os.path.join(asset_root, asset_file)
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # 查找link定义
        for part_name, part_info in parts.items():
            for link_name in part_info["link_names"]:
                link_elem = root.find(f".//link[@name='{link_name}']")
                if link_elem is not None:
                    # 提取视觉几何网格
                    visuals = link_elem.findall(".//visual/geometry/mesh")
                    for visual in visuals:
                        filename = visual.get('filename')
                        scale_str = visual.get('scale', "1 1 1")
                        scale = [float(s) for s in scale_str.split()]
                        
                        if filename:
                            mesh_path = os.path.join(asset_root, os.path.dirname(asset_file), filename)
                            if os.path.exists(mesh_path):
                                part_info["visual_meshes"].append({
                                    "path": mesh_path,
                                    "scale": scale
                                })
        
        # 加载网格并生成点云 - 使用trimesh处理缩放和采样
        for part_name, part_info in parts.items():
            if part_info["visual_meshes"]:
                combined_points = []
                combined_normals = []
                for mesh_info in part_info["visual_meshes"]:

                    mesh = trimesh.load_mesh(mesh_info["path"], process=False)
                    
                    scale_matrix = np.eye(4)
                    scale_matrix[:3, :3] *= mesh_info["scale"]  # 应用完整缩放向量
                    mesh.apply_transform(scale_matrix)
                    
                    # points = mesh.sample(512)
                    # points, face_idx = mesh.sample(512, return_index=True)
                    points, face_idx = trimesh.sample.sample_surface_even(mesh, 512, seed=42)
                    normals = mesh.face_normals[face_idx]
                    combined_points.append(points)
                    combined_normals.append(normals)
                # 合并点云
                all_points = np.concatenate(combined_points, axis=0)
                all_normals = np.concatenate(combined_normals, axis=0)
                
                # 随机采样到固定大小
                if len(all_points) > 1024:
                    indices = np.random.choice(len(all_points), 1024, replace=False)
                    all_points = all_points[indices]
                    
                norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
                all_normals = all_normals / (norms + 1e-8)
                    
                parts[part_name]["pointcloud"] = torch.tensor(all_points, dtype=torch.float32, device=self.device)
                parts[part_name]["pointcloud_normals"] = torch.tensor(all_normals, dtype=torch.float32, device=self.device)
                
        return parts
    
    def __configure_robot_dof_indices(self, hand_arm_asset: gymapi.Asset, side: str) -> None:
        
        dof_dict = self.gym.get_asset_dof_dict(hand_arm_asset)

        actuated_dof_indices = []
        arm_actuated_dof_indices = []
        hand_actuated_dof_indices = []
        hand_digits_actuated_dof_indices = []
        # hand_fingers_actuated_dof_indices = []
        # hand_thumb_actuated_dof_indices = []

        for name, index in dof_dict.items():
            if any([dof in name for dof in self._xarm_dof_names]):
                arm_actuated_dof_indices.append(index)
            elif any([dof in name for dof in self._hand_dof_names]):
                hand_actuated_dof_indices.append(index)
                hand_digits_actuated_dof_indices.append(index)

                # if any([dof in name for dof in self._hand_fingers_dof_names]):
                #     hand_fingers_actuated_dof_indices.append(index)
                # elif any([dof in name for dof in self._hand_thumb_dof_names]):
                #     hand_thumb_actuated_dof_indices.append(index)

            actuated_dof_indices.append(index)

        def _torchify(indices: List[int]) -> torch.LongTensor:
            return torch.tensor(sorted(indices)).long().to(self.device)

        if side == "right":
            self.right_actuated_dof_indices = _torchify(actuated_dof_indices)
            self.right_arm_actuated_dof_indices = _torchify(arm_actuated_dof_indices)
            self.right_hand_actuated_dof_indices = _torchify(hand_actuated_dof_indices)
            self.right_hand_digits_actuated_dof_indices = _torchify(hand_digits_actuated_dof_indices)
            # self.right_hand_fingers_actuated_dof_indices = _torchify(hand_fingers_actuated_dof_indices)
            # self.right_hand_thumb_actuated_dof_indices = _torchify(hand_thumb_actuated_dof_indices)
        else:
            self.left_actuated_dof_indices = _torchify(actuated_dof_indices)
            self.left_arm_actuated_dof_indices = _torchify(arm_actuated_dof_indices)
            self.left_hand_actuated_dof_indices = _torchify(hand_actuated_dof_indices)
            self.left_hand_digits_actuated_dof_indices = _torchify(hand_digits_actuated_dof_indices)
            # self.left_hand_fingers_actuated_dof_indices = _torchify(hand_fingers_actuated_dof_indices)
            # self.left_hand_thumb_actuated_dof_indices = _torchify(hand_thumb_actuated_dof_indices)
        
    def _define_hand(self, side: str) -> Dict[str, Any]:
        """Define a hand asset for the simulation."""
        cfg = self.hand_cfg[side]
        asset_root = cfg.get("assetRoot", str(self._asset_root))
        asset_file = cfg.get("assetFile")
        assert asset_file is not None, f"{side} hand asset file missing"
        
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True

        if self.env_info_logging:
            print_asset_options(asset_options, f"{side}_hand")
        
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        # Get counts
        num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
        num_rigid_shapes = self.gym.get_asset_rigid_shape_count(asset)
        num_dofs = self.gym.get_asset_dof_count(asset)
        
        
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        for shape in rigid_shape_props:
            shape.friction = 1.0
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)
        
        
        # Configure DOF properties
        dof_props = self.gym.get_asset_dof_properties(asset)
        pos_dof = 3
        rot_dof = 3
        root_dof = 6
        finger_dofs = num_dofs - root_dof
        
        stiffness = self.env_cfg.get("fingerStiffness", 4000)
        damping = self.env_cfg.get("fingerDamping", 80)
        root_stiffness = self.env_cfg.get("rootStiffness", 50.0)
        root_damping = self.env_cfg.get("rootDamping", 10.0)
        root_effort = self.env_cfg.get("rootEffort", 50.0)
        
        for i in range(num_dofs):
            name = self.gym.get_asset_dof_name(asset, i)
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if name in self._xarm_dof_names or name.startswith("joint"):
                # Arm DOFs
                dof_props["stiffness"][i] = 1000
                dof_props["damping"][i] = 100
                dof_props["velocity"][i] = 4.0
                dof_props["effort"][i] = 2000.0
            else:
                # hand DOFs
                dof_props["stiffness"][i] = 3.0
                dof_props["damping"][i] = 0.1
                dof_props["effort"][i] = 0.5
                dof_props["friction"][i] = 0.01
                dof_props["armature"][i] = 0.001
                
        if self.env_info_logging:
            print_dof_properties(self.gym, asset, dof_props, f"{side}_hand")
            
        dof_lower_limits = [dof_props["lower"][i] for i in range(num_dofs)]
        dof_upper_limits = [dof_props["upper"][i] for i in range(num_dofs)]
        dof_init_positions = [0.0 for _ in range(num_dofs)]
        dof_init_velocities = [0.0 for _ in range(num_dofs)]

        # reset xarm initial dof positions
        if side == "left":
            for name, value in self._left_xarm_init_dof_positions.items():
                dof_init_positions[self.gym.find_asset_dof_index(asset, name)] = value
        else:
            for name, value in self._right_xarm_init_dof_positions.items():
                dof_init_positions[self.gym.find_asset_dof_index(asset, name)] = value
            
        limits = {}
        limits["lower"] = torch.tensor(dof_lower_limits).float().to(self.device)
        limits["upper"] = torch.tensor(dof_upper_limits).float().to(self.device)

        init = {}
        init["position"] = torch.tensor(dof_init_positions).float().to(self.device)
        init["velocity"] = torch.tensor(dof_init_velocities).float().to(self.device)
        
        self.__configure_robot_dof_indices(asset, side)
        
        # Initial pose
        init_pose_cfg = cfg["initPose"]
        init_position = init_pose_cfg["position"]
        init_orientation = init_pose_cfg["orientation"]
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*init_position)
        pose.r = gymapi.Quat(*init_orientation)
        
        if side == "right":
            self.right_fingertip_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._fingertips]
            self.right_virtual_tip_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._tip_links]
            self.right_hand_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._right_hand_links]
            self.right_index_finger_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._index_finger_links]
            self.right_middle_finger_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._middle_finger_links]
            self.right_ring_finger_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._ring_finger_links]
            self.right_thumb_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._thumb_links]
            self.right_arm_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._arm_links]
        else:
            self.left_fingertip_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._fingertips]
            self.left_virtual_tip_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._tip_links]
            self.left_hand_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._left_hand_links]
            self.left_index_finger_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._index_finger_links]
            self.left_middle_finger_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._middle_finger_links]
            self.left_ring_finger_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._ring_finger_links]
            self.left_thumb_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._thumb_links]
            self.left_arm_links = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._arm_links]
        
        keypoint_indices: List[int] = []
        keypoint_offsets: List[List[float]] = []
        link_name_to_index: Dict[str, int] = {}
        _hand_links = self._left_hand_links if side == "left" else self._right_hand_links
        for link in _hand_links:
            li = self.gym.find_asset_rigid_body_index(asset, link)
            assert li != -1
            link_name_to_index[link] = li
        _flattened_keypoint_pairs = self._left_hand_flattened_keypoint_pairs if side == "left" else self._right_hand_flattened_keypoint_pairs
        for link_name, sub_k in _flattened_keypoint_pairs:
            if link_name not in link_name_to_index:
                continue
            keypoint_indices.append(link_name_to_index[link_name])
            keypoint_offsets.append(self.hand_keypoints[side][link_name][sub_k])
            
        def indices_among_keypoints_for_side(links: List[str], side: str) -> List[int]:
            """Return indices (among flattened keypoints) whose link_name is in `links`."""
            if side == "left":
                flat_pairs = self._left_hand_flattened_keypoint_pairs
            else:
                flat_pairs = self._right_hand_flattened_keypoint_pairs
            idxs: List[int] = []
            for j, (ln, _) in enumerate(flat_pairs):
                if ln in links:
                    idxs.append(j)
            return idxs

        if side == "left":
            self.left_index_link_indices_among_keypoints  = indices_among_keypoints_for_side(self._index_finger_links,  "left")
            self.left_middle_link_indices_among_keypoints = indices_among_keypoints_for_side(self._middle_finger_links, "left")
            self.left_ring_link_indices_among_keypoints   = indices_among_keypoints_for_side(self._ring_finger_links,   "left")
            self.left_thumb_link_indices_among_keypoints  = indices_among_keypoints_for_side(self._thumb_links,        "left")
        else:
            self.right_index_link_indices_among_keypoints  = indices_among_keypoints_for_side(self._index_finger_links,  "right")
            self.right_middle_link_indices_among_keypoints = indices_among_keypoints_for_side(self._middle_finger_links, "right")
            self.right_ring_link_indices_among_keypoints   = indices_among_keypoints_for_side(self._ring_finger_links,   "right")
            self.right_thumb_link_indices_among_keypoints  = indices_among_keypoints_for_side(self._thumb_links,        "right")
        # actor domain
        if side == "right":
            self.right_keypoint_indices = torch.tensor(keypoint_indices, dtype=torch.long, device=self.device)
            self.right_keypoint_offsets = torch.tensor(keypoint_offsets, dtype=torch.float, device=self.device)
        else:
            self.left_keypoint_indices = torch.tensor(keypoint_indices, dtype=torch.long, device=self.device)
            self.left_keypoint_offsets = torch.tensor(keypoint_offsets, dtype=torch.float, device=self.device)
        
        return {
            "asset": asset,
            "pose": pose,
            "name": f"{side}_hand",
            "finger_dofs": finger_dofs,
            "dof_props": dof_props,
            "limits": limits,
            "init": init,
            "num_dofs": num_dofs,
            "root_dofs": root_dof,
            "num_rigid_bodies": num_rigid_bodies,
            "num_rigid_shapes": num_rigid_shapes,
        }

    def _define_object(self) -> Dict[str, Any]:
        """Define and load object assets for the environment."""
        config = {
            "warehouse": [],
            "poses": [],
            "count": len(self.object_cfgs),
            "num_rigid_bodies": 0,
            "num_rigid_shapes": 0
        }
        
        # Determine maximum rigid bodies and shapes needed across all objects
        max_rigid_bodies = 0
        max_rigid_shapes = 0
        
        for obj_idx, obj_cfg in enumerate(self.object_cfgs):
            asset_root = obj_cfg.get("assetRoot", os.path.join(os.path.dirname(find_dotenv()), "assets"))
            asset_file = obj_cfg.get("assetFile")
            assert asset_file is not None, "Object assetFile missing"
            
            asset_options = gymapi.AssetOptions()
            asset_options.density = 500
            asset_options.fix_base_link = False
            asset_options.flip_visual_attachments = False
            asset_options.collapse_fixed_joints = True
            asset_options.disable_gravity = False
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.use_mesh_materials = True
            asset_options.thickness = 0.001
            
            asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            # asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, asset_options)
            num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
            num_rigid_shapes = self.gym.get_asset_rigid_shape_count(asset)
            
            # Track max counts for aggregate calculation
            max_rigid_bodies = max(max_rigid_bodies, num_rigid_bodies)
            max_rigid_shapes = max(max_rigid_shapes, num_rigid_shapes)
            
            # Handle DOFs if present
            num_dofs = self.gym.get_asset_dof_count(asset)
            if num_dofs > 0:
                dof_props = self.gym.get_asset_dof_properties(asset)
                for i in range(num_dofs):
                    # dof_props["driveMode"][i] = gymapi.DOF_MODE_NONE
                    dof_props["velocity"][i] = 100.0
                    dof_props["effort"][i] = 50.0
                    dof_props["stiffness"][i] = 0.1
                    dof_props["damping"][i] = 0.02
                    # dof_props["stiffness"][i] = 1
                    # dof_props["damping"][i] = 0.2
                    # dof_props["stiffness"][i] = 5
                    # dof_props["damping"][i] = 1.0
                    # dof_props["friction"][i] = 3.0
                    # dof_props["armature"][i] = 0.001
            else:
                dof_props = None
                
            # if obj_cfg.get("taskType") == "bottle_cap":
            #     parts = self._parse_bottle_parts(asset_root, asset_file)
            #     config["parts"][obj_idx] = parts
                
            if self.env_info_logging and dof_props is not None:
                print_dof_properties(self.gym, asset, dof_props, f"{obj_cfg['name']}")
            
            # Create object configuration
            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(*obj_cfg["pose"][:3])
            obj_pose.r = gymapi.Quat(*obj_cfg["pose"][3:])
            obj_config = {
                "name": obj_cfg.get("name", os.path.splitext(os.path.basename(asset_file))[0]),
                "asset": asset,
                "asset_root": asset_root,
                "asset_file": asset_file,
                "pose": obj_pose,
                "color": obj_cfg.get("color", [0.7, 0.7, 0.7]),
                "track": obj_cfg.get("track", False),
                "num_dofs": num_dofs,
                "num_rigid_bodies": num_rigid_bodies,
                "num_rigid_shapes": num_rigid_shapes,
            }
            
            if dof_props is not None:
                obj_config["dof_props"] = dof_props
            
            config["warehouse"].append(obj_config)
            config["poses"].append(obj_pose)
        
        # Set total counts based on the maximum needed for any single environment
        config["num_rigid_bodies"] = max_rigid_bodies
        config["num_rigid_shapes"] = max_rigid_shapes
        
        return config

    def _create_envs(self, num_envs: int, spacing: float):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(num_envs))
        
        print(">>> Defining gym assets")
        self.envs: List[gymapi.Env] = []
        self.hand_actor_handles: Dict[str, List[int]] = {"right": [], "left": []}
        self.hand_actor_handle_sim: Dict[str, torch.Tensor] = {"right": torch.zeros((self.num_envs,), dtype=torch.long, device=self.device), "left": torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)}
        self.hand_actor_ids_env: Dict[str, int] = {"right": 0, "left": 0}
        self.hand_dof_starts: Dict[str, int] = {"right": 0, "left": 0}
        self.hand_wrist_rigid_body_id: Dict[str, int] = {"right": 0, "left": 0}
        self.forearm_rigid_body_id: Dict[str, int] = {"right": 0, "left": 0}
        self.object_actor_handles: List[List[int]] = []
        self.tracked_object_actor_ids_env: List[int] = []
        self.object_name_to_handles: List[Dict[str, int]] = [dict() for _ in range(num_envs)]
        
        # Load assets
        self.gym_assets["current"]["hand_right"] = self._define_hand("right")
        self.gym_assets["current"]["hand_left"] = self._define_hand("left")
        self.gym_assets["current"]["table"] = self._define_table()
        self.gym_assets["current"]["objects"] = self._define_object()
        
        # Count DOFs per environment
        self.object_dofs_per_env = sum(cfg["num_dofs"] for cfg in self.gym_assets["current"]["objects"]["warehouse"] )
            
            
        self.object_dof_per_env: List[List[Slice]] = [[] for _ in range(num_envs)]
        self.object_handles_env: List[List[int]] = [[] for _ in range(num_envs)]
        self.object_handles_sim: List[List[int]] = [[] for _ in range(num_envs)]
        
        # Compute maximum aggregate bodies and shapes
        max_aggregate_bodies, max_aggregate_shapes = self.compute_maximum_aggregate_bodies_and_shapes()
        
        # Create environments
        for env_id in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)
            self.aggregate_tracker.reset()
            
            if self.aggregate_mode != 0:
                num_bodies, num_shapes = self.compute_aggregate_bodies_and_shapes()
                agg_success = self.gym.begin_aggregate(env, max_aggregate_bodies, max_aggregate_shapes, True)
                if not agg_success:
                    raise RuntimeError("begin_aggregate failed")
            
            # Create table
            table_handle = self._create_sim_actor(
                env,
                self.gym_assets["current"]["table"],
                env_id,  # group ID
                name="table",
                pose=self.gym_assets["current"]["table"]["pose"],
                color=gymapi.Vec3(0.0, 0.0, 0.0),
                # filter=0  # Add collision filter parameter
            )
            self.table_handle = table_handle
            
            # Create right hand
            rh_handle = self._spawn_hand(env, env_id, "right")
            self.hand_actor_handles["right"].append(rh_handle)
            
            # Create left hand
            lh_handle = self._spawn_hand(env, env_id, "left")
            self.hand_actor_handles["left"].append(lh_handle)
            
            # Create scene objects
            obj_handles_env: List[int] = []
            object_dof_offset = 0
            for obj_idx, obj_cfg in enumerate(self.gym_assets["current"]["objects"]["warehouse"]):
                index, handle = self._create_sim_actor(
                    env,
                    obj_cfg,
                    (
                        env_id
                        # if not obj_cfg.get("color_idx_offset", False)
                        # else env_id + obj_idx * num_envs
                    ),
                    name=obj_cfg["name"],
                    # filter=0,
                    actor_handle=True
                )
                obj_handles_env.append(handle)
                self.object_name_to_handles[env_id][obj_cfg["name"]] = handle
                
                # Store object DOF indices
                num_obj_dofs = obj_cfg.get("num_dofs", 0)
                if num_obj_dofs > 0:
                    indices = torch.zeros(num_obj_dofs, dtype=torch.long, device=self.device)
                    for dof_id in range(num_obj_dofs):
                        indices[dof_id] = self.gym.get_actor_dof_index(env, handle, dof_id, gymapi.DOMAIN_ENV)
                    start = object_dof_offset
                    end = object_dof_offset + num_obj_dofs
                    object_dof_offset = end
                    self.object_dof_per_env[env_id].append(indices)
                self.object_handles_env[env_id].append(handle)
                self.object_handles_sim[env_id].append(index)
            
            if self.aggregate_mode != 0:
                self.gym.end_aggregate(env)
                assert self.aggregate_tracker.aggregate_bodies == num_bodies
                assert self.aggregate_tracker.aggregate_shapes == num_shapes
                
                
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
        self.object_dof_start = self.gym.get_actor_dof_index(self.envs[0], self.object_handles_env[0][0], 0, gymapi.DOMAIN_ENV)
        self.object_dof_end = self.object_dof_start + self.object_dofs_per_env
        self.object_rigid_body_start = self.gym.get_actor_rigid_body_index(self.envs[0], self.object_handles_sim[0][0], 0, gymapi.DOMAIN_ENV)
        self.object_rigid_body_end = self.object_rigid_body_start + self.gym_assets["current"]["objects"]["warehouse"][0]["num_rigid_bodies"]
        


    def _spawn_hand(self, env: gymapi.Env, env_id: int, side: str) -> int:
        hand_cfg = self.gym_assets["current"][f"hand_{side}"]
        pose = hand_cfg["pose"]
        
        index, handle = self._create_sim_actor(
            env, 
            hand_cfg, 
            env_id, 
            name=f"{side}_hand", 
            pose=pose,
            actor_handle=True
        )
        
        # Get actor index in simulation domain
        actor_sim_idx = self.gym.get_actor_index(env, handle, gymapi.DOMAIN_SIM)
        self.hand_actor_handle_sim[side][env_id] = actor_sim_idx
        
        # Get actor index in environment domain
        actor_env_idx = self.gym.get_actor_index(env, handle, gymapi.DOMAIN_ENV)
        self.hand_actor_ids_env[side] = actor_env_idx
        
        # Get DOF start index
        dof_start = self.gym.get_actor_dof_index(env, handle, 0, gymapi.DOMAIN_ENV)
        self.hand_dof_starts[side] = dof_start
        
        # Get wrist rigid body ID
        rb_index = self.gym.find_actor_rigid_body_index(
            env, handle, self.hand_base_links[side], gymapi.DOMAIN_ACTOR
        )
        self.hand_wrist_rigid_body_id[side] = rb_index
        
        # Get forearm rigid body ID
        rb_index = self.gym.find_actor_rigid_body_index(
            env, handle, "link6", gymapi.DOMAIN_ACTOR
        )
        self.forearm_rigid_body_id[side] = rb_index
        
        return handle

    def _create_sim_actor(
        self,
        env: gymapi.Env,
        config: Dict[str, Any],
        group: int,
        name: Optional[str] = None,
        pose: Optional[gymapi.Transform] = None,
        color: Optional[gymapi.Vec3] = None,
        actor_handle: Optional[bool] = False,
        filter: int = 0
    ) -> Union[int, Tuple[int, int]]:

        asset = config.get("asset", None)
        name = name if name is not None else config["name"]
        pose = pose if pose is not None else config["pose"]
        assert asset is not None and name is not None and pose is not None, f"asset: {asset}, name: {name}, pose: {pose}"

        self.aggregate_tracker.update(config["num_rigid_bodies"], config["num_rigid_shapes"])

        # create the actor
        actor = self.gym.create_actor(env, asset, pose, name, group, filter, 0)
        # actor = self.gym.create_actor(env, asset, pose, name, 0, 1, 0) # ik solving use

        # set the dof properties if `dof_props` exists in the config
        dof_props = config.get("dof_props", None)
        if dof_props is not None:
            self.gym.set_actor_dof_properties(env, actor, dof_props)

        # set the color
        if color is not None:
            self.gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        else:
            # set the color of the contact sensors (blue by default)
            for name, index in self.gym.get_actor_rigid_body_dict(env, actor).items():
                if not name.startswith("sensor_"):
                    continue
                self.gym.set_rigid_body_color(
                    env, actor, index, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 0.8)
                )

        if actor_handle:
            return self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM), actor
        else:
            return self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)

    def _finalize_env_setup(self):
        
        # Cache DOF indices per hand/actor (simulation domain)
        self.hand_dof_indices_sim: Dict[str, torch.Tensor] = {
            "right": torch.zeros(
                (self.num_envs, self.gym_assets["current"]["hand_right"]["num_dofs"]), dtype=torch.long, device=self.device
            ),
            "left": torch.zeros(
                (self.num_envs, self.gym_assets["current"]["hand_left"]["num_dofs"]), dtype=torch.long, device=self.device
            ),
        }
        
        for env_id, env in enumerate(self.envs):
            rh_handle = self.hand_actor_handles["right"][env_id]
            lh_handle = self.hand_actor_handles["left"][env_id]
            
            for i in range(self.gym_assets["current"]["hand_right"]["num_dofs"]):
                self.hand_dof_indices_sim["right"][env_id, i] = self.gym.get_actor_dof_index(
                    env, rh_handle, i, gymapi.DOMAIN_SIM
                )
            
            for i in range(self.gym_assets["current"]["hand_left"]["num_dofs"]):
                self.hand_dof_indices_sim["left"][env_id, i] = self.gym.get_actor_dof_index(
                    env, lh_handle, i, gymapi.DOMAIN_SIM
                )
        
        self.tracked_object_actor_ids_env = []
        if self.object_actor_handles:
            env0 = self.envs[0]
            for idx, obj_cfg in enumerate(self.object_cfgs):
                if obj_cfg.get("track", False):
                    self.tracked_object_actor_ids_env.append(
                        self.gym.get_actor_index(env0, self.object_actor_handles[0][idx], gymapi.DOMAIN_ENV)
                    )
        
        # retrieve generic tensor descriptors for the simulation
        # - root_states: [num_envs * num_actors, 13]
        _root_states: torch.Tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # - dof_states: [num_envs * num_dofs, 2]
        _dof_states: torch.Tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # - dof_forces: [num_envs * num_dofs]
        _dof_forces: torch.Tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # - rigid_body_states: [num_envs * num_rigid_bodies, 13]
        _rigid_body_states: torch.Tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # - net_contact_forces: [num_envs * num_rigid_bodies, 3]
        _net_contact_forces: torch.Tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        # - force_sensor_states: [num_envs * num_force_sensors, 6]
        _force_sensor_states: torch.Tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # - jacobians: [num_envs, num_prims - 1, 6, num_dofs]
        _jacobians_right: torch.Tensor = self.gym.acquire_jacobian_tensor(self.sim, "right_hand")
        _jacobians_left: torch.Tensor = self.gym.acquire_jacobian_tensor(self.sim, "left_hand")
        
        if self.env_info_logging:
            print("root_states.shape: ", _root_states.shape)
            print("dof_states.shape: ", _dof_states.shape)
            print("rigid_body_states.shape: ", _rigid_body_states.shape)
            print("net_contact_forces.shape: ", _net_contact_forces.shape)
            print("force_sensor_states.shape: ", _force_sensor_states.shape)
            print("dof_forces.shape: ", _dof_forces.shape)
            # print("jacobians.shape: ", _jacobians.shape)
            
        self.num_actors: int = self.gym.get_sim_actor_count(self.sim) // self.num_envs
        self.num_dofs: int = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.num_force_sensors: int = self.gym.get_sim_force_sensor_count(self.sim) // self.num_envs
        self.num_rigid_bodies: int = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs

        if self.env_info_logging:
            print("num_actors: ", self.num_actors)
            print("num_dofs: ", self.num_dofs)
            print("num_force_sensors: ", self.num_force_sensors)
            print("num_rigid_bodies: ", self.num_rigid_bodies)
        
        self.root_states: torch.Tensor = gymtorch.wrap_tensor(_root_states)
        self.dof_states: torch.Tensor = gymtorch.wrap_tensor(_dof_states)
        self.dof_forces: torch.Tensor = gymtorch.wrap_tensor(_dof_forces)
        self.rigid_body_states: torch.Tensor = gymtorch.wrap_tensor(_rigid_body_states)
        self.net_contact_forces: torch.Tensor = gymtorch.wrap_tensor(_net_contact_forces)
        
        self.jacobians_right: torch.Tensor = gymtorch.wrap_tensor(_jacobians_right)
        self.jacobians_left: torch.Tensor = gymtorch.wrap_tensor(_jacobians_left)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        
        # create tensors to hold observations, actions, and rewards for each environment
        # only contiguous slices can be defined here
        # non-contiguous slices will be defined in `_refresh_sim_tensors`
        self.root_positions = self.root_states[:, 0:3]
        self.root_orientations = self.root_states[:, 3:7]
        self.root_linear_velocities = self.root_states[:, 7:10]
        self.root_angular_velocities = self.root_states[:, 10:13]
        
        root_states = self.root_states.view(self.num_envs, self.num_actors, 13)
        
        self.left_hand_root_states = root_states[:, self.hand_actor_ids_env["left"], :]
        self.left_hand_root_positions = self.left_hand_root_states[:, 0:3]
        self.left_hand_root_orientations = self.left_hand_root_states[:, 3:7]
        self.left_hand_root_linear_velocities = self.left_hand_root_states[:, 7:10]
        self.left_hand_root_angular_velocities = self.left_hand_root_states[:, 10:13]
        
        self.right_hand_root_states = root_states[:, self.hand_actor_ids_env["right"], :]
        self.right_hand_root_positions = self.right_hand_root_states[:, 0:3]
        self.right_hand_root_orientations = self.right_hand_root_states[:, 3:7]
        self.right_hand_root_linear_velocities = self.right_hand_root_states[:, 7:10]
        self.right_hand_root_angular_velocities = self.right_hand_root_states[:, 10:13]
        
        self.object_root_states = root_states[:, self.object_handles_env[0], :].view(self.num_envs, 13) # HACK: when one object
        self.object_root_positions = self.object_root_states[:, 0:3]
        self.object_root_orientations = self.object_root_states[:, 3:7]
        self.object_root_linear_velocities = self.object_root_states[:, 7:10]
        self.object_root_angular_velocities = self.object_root_states[:, 10:13]
        
        # self.init_object_root_positions = self.object_root_positions.clone()
        # self.init_object_root_orientations = self.object_root_orientations.clone()
        
        dof_states = self.dof_states.view(self.num_envs, self.num_dofs, 2)
        
        self.left_hand_dof_positions = dof_states[:, self.left_hand_arm_dof_start + 6 : self.left_hand_arm_dof_end, 0]
        self.left_hand_dof_velocities = dof_states[:, self.left_hand_arm_dof_start + 6 : self.left_hand_arm_dof_end, 1]
        self.right_hand_dof_positions = dof_states[:, self.right_hand_arm_dof_start + 6 : self.right_hand_arm_dof_end, 0]
        self.right_hand_dof_velocities = dof_states[:, self.right_hand_arm_dof_start + 6 : self.right_hand_arm_dof_end, 1]
        self.left_hand_arm_dof_positions = dof_states[:, self.left_hand_arm_dof_start : self.left_hand_arm_dof_end, 0]
        self.left_hand_arm_dof_velocities = dof_states[:, self.left_hand_arm_dof_start : self.left_hand_arm_dof_end, 1]
        self.right_hand_arm_dof_positions = dof_states[:, self.right_hand_arm_dof_start : self.right_hand_arm_dof_end, 0]
        self.right_hand_arm_dof_velocities = dof_states[:, self.right_hand_arm_dof_start : self.right_hand_arm_dof_end, 1]
        
        self.object_dof_positions = dof_states[:, self.object_dof_start : self.object_dof_end, 0]
        self.object_dof_velocities = dof_states[:, self.object_dof_start : self.object_dof_end, 1]
        
        
        self.init_left_hand_dof_positions = self.left_hand_dof_positions.clone()
        self.init_left_hand_dof_velocities = self.left_hand_dof_velocities.clone()
        self.init_right_hand_dof_positions = self.right_hand_dof_positions.clone()
        self.init_right_hand_dof_velocities = self.right_hand_dof_velocities.clone()
        
        self.init_object_dof_positions = self.object_dof_positions.clone()
        self.init_object_dof_velocities = self.object_dof_velocities.clone()
        
        rigid_body_states = self.rigid_body_states.view(self.num_envs, self.num_rigid_bodies, 13)
        self.left_hand_rigid_body_states = rigid_body_states[:, self.left_hand_rigid_body_start : self.left_hand_rigid_body_end, :]
        self.left_hand_rigid_body_positions = self.left_hand_rigid_body_states[:, :, 0:3]
        self.left_hand_rigid_body_orientations = self.left_hand_rigid_body_states[:, :, 3:7]
        self.left_hand_rigid_body_linear_velocities = self.left_hand_rigid_body_states[:, :, 7:10]
        self.left_hand_rigid_body_angular_velocities = self.left_hand_rigid_body_states[:, :, 10:13]
        self.right_hand_rigid_body_states = rigid_body_states[:, self.right_hand_rigid_body_start : self.right_hand_rigid_body_end, :]
        self.right_hand_rigid_body_positions = self.right_hand_rigid_body_states[:, :, 0:3]
        self.right_hand_rigid_body_orientations = self.right_hand_rigid_body_states[:, :, 3:7]
        self.right_hand_rigid_body_linear_velocities = self.right_hand_rigid_body_states[:, :, 7:10]
        self.right_hand_rigid_body_angular_velocities = self.right_hand_rigid_body_states[:, :, 10:13]
        self.object_rigid_body_states = rigid_body_states[:, self.object_rigid_body_start : self.object_rigid_body_end, :]
        self.object_rigid_body_positions = self.object_rigid_body_states[:, :, 0:3]
        self.object_rigid_body_orientations = self.object_rigid_body_states[:, :, 3:7]
        self.object_rigid_body_linear_velocities = self.object_rigid_body_states[:, :, 7:10]
        self.object_rigid_body_angular_velocities = self.object_rigid_body_states[:, :, 10:13]
        
        self.left_endeffector_states = self.left_hand_rigid_body_states[:, self.left_hand_endeffector_index, :]
        self.left_endeffector_positions = self.left_endeffector_states[:, 0:3]
        self.left_endeffector_orientations = self.left_endeffector_states[:, 3:7]
        self.left_endeffector_linear_velocities = self.left_endeffector_states[:, 7:10]
        self.left_endeffector_angular_velocities = self.left_endeffector_states[:, 10:13]
        self.right_endeffector_states = self.right_hand_rigid_body_states[:, self.right_hand_endeffector_index, :]
        self.right_endeffector_positions = self.right_endeffector_states[:, 0:3]
        self.right_endeffector_orientations = self.right_endeffector_states[:, 3:7]
        self.right_endeffector_linear_velocities = self.right_endeffector_states[:, 7:10]
        self.right_endeffector_angular_velocities = self.right_endeffector_states[:, 10:13]
        
        
        
        # Create buffer tensors
        kwargs = {"device": self.device, "dtype": torch.float}
        self.curr_targets_buffer = torch.zeros((self.num_envs, self.num_dofs), **kwargs)
        self.prev_targets_buffer = torch.zeros((self.num_envs, self.num_dofs), **kwargs)
        
        # Observation tensors
        self.rh_root_pose = torch.zeros((self.num_envs, 7), **kwargs)
        self.rh_root_velocity = torch.zeros((self.num_envs, 6), **kwargs)
        self.rh_finger_positions_obs = torch.zeros((self.num_envs, self.num_hand_dofs), **kwargs)
        self.rh_finger_velocities_obs = torch.zeros((self.num_envs, self.num_hand_dofs), **kwargs)
        self.lh_root_pose = torch.zeros_like(self.rh_root_pose)
        self.lh_root_velocity = torch.zeros_like(self.rh_root_velocity)
        self.lh_finger_positions_obs = torch.zeros_like(self.rh_finger_positions_obs)
        self.lh_finger_velocities_obs = torch.zeros_like(self.rh_finger_velocities_obs)
        self.object_root_pose = torch.zeros((self.num_envs, self.num_tracked_objects * 7), **kwargs)
        self.object_root_velocity = torch.zeros((self.num_envs, self.num_tracked_objects * 6), **kwargs)
        
        # Force/torque buffers for PD control
        self.rb_forces = torch.zeros((self.num_envs, self.num_rigid_bodies, 3), **kwargs)
        self.rb_torques = torch.zeros((self.num_envs, self.num_rigid_bodies, 3), **kwargs)
        
        # Action buffer
        self.actions = torch.zeros((self.num_envs, self.num_actions), **kwargs)
        
        # Target tracking
        self.hand_target_object_actor_ids: Dict[str, torch.Tensor] = {
            "right": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
            "left": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
        }
        self.hand_target_body_indices: Dict[str, torch.Tensor] = {
            "right": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
            "left": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
        }
        
        # Success tracking
        self.success_step_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.successes = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        
        # Reshape tensors for easier access
        self._root_states_view = self.root_states.view(self.num_envs, self.num_actors, 13)
        self.rb_states_view = self.rigid_body_states.view(self.num_envs, self.num_rigid_bodies, 13)
        
        # Get table rigid body index
        self.table_rigid_body_indices = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        for i in range(self.num_envs):
            env = self.envs[i]
            table_handle = self.gym.find_actor_handle(env, "table")
            table_actor_index = self.gym.get_actor_index(env, table_handle, gymapi.DOMAIN_ENV)
            table_rigid_body_index = self.gym.get_actor_rigid_body_index(
                env, table_actor_index, 0, gymapi.DOMAIN_ENV
            )
            self.table_rigid_body_indices[i] = table_rigid_body_index
        
        self.initial_root_states = torch.zeros_like(self.root_states)
        self.initial_dof_states = torch.zeros_like(self.dof_states)
        
        # 显式设置初始手位置
        for side in ["right", "left"]:
            hand_cfg = self.gym_assets["current"][f"hand_{side}"]
            pose = hand_cfg["pose"]
            actor_id = self.hand_actor_ids_env[side]
            # 为所有环境设置初始位置和方向
            self.initial_root_states.view(self.num_envs, self.num_actors, 13)[:, actor_id, 0:3] = torch.tensor(
                [pose.p.x, pose.p.y, pose.p.z], device=self.device)
            self.initial_root_states.view(self.num_envs, self.num_actors, 13)[:, actor_id, 3:7] = torch.tensor(
                [pose.r.x, pose.r.y, pose.r.z, pose.r.w], device=self.device)
            # 零速度
            self.initial_root_states.view(self.num_envs, self.num_actors, 13)[:, actor_id, 7:13] = 0.0
            
            # 设置手的初始DOF状态
            dof_start = self.hand_dof_starts[side]
            num_dofs = self.gym_assets["current"][f"hand_{side}"]["num_dofs"]
            # 从配置获取或设置默认的初始DOF位置
            initial_positions = torch.zeros(num_dofs, device=self.device)
            # 对于根部DOF（前6个），使用配置中的初始姿态（如果有的话）
            # 对于手指DOF，可以设置为半闭合或其他默认姿势
            # 这里简单地全部设为0，您可以根据需要调整
            self.initial_dof_states.view(self.num_envs, self.num_dofs, 2)[:, dof_start:dof_start+num_dofs, 0] = initial_positions.repeat(self.num_envs, 1)
            # DOF速度设为0
            self.initial_dof_states.view(self.num_envs, self.num_dofs, 2)[:, dof_start:dof_start+num_dofs, 1] = 0.0

        # 显式设置初始对象位置
        if self.object_actor_handles:
            for env_id in range(self.num_envs):
                for obj_idx, obj_cfg in enumerate(self.gym_assets["current"]["objects"]["warehouse"]):
                    if obj_idx >= len(self.object_actor_handles[env_id]):
                        continue
                        
                    handle = self.object_actor_handles[env_id][obj_idx]
                    obj_actor_index = self.gym.get_actor_index(self.envs[env_id], handle, gymapi.DOMAIN_SIM)
                    obj_pose = obj_cfg["pose"]
                    # 设置初始位置和方向
                    self.initial_root_states[obj_actor_index, 0:3] = torch.tensor(
                        [obj_pose.p.x, obj_pose.p.y, obj_pose.p.z], device=self.device)
                    self.initial_root_states[obj_actor_index, 3:7] = torch.tensor(
                        [obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w], device=self.device)
                    # 零速度
                    self.initial_root_states[obj_actor_index, 7:13] = 0.0
                    
                    # 设置对象的初始DOF状态（如果有的话）
                    num_obj_dofs = obj_cfg.get("num_dofs", 0)
                    if num_obj_dofs > 0:
                        obj_dof_props = obj_cfg["dof_props"]
                        # 获取对象在环境中的DOF起始索引
                        obj_dof_start = 0  # 需要实现正确的计算
                        # 可以从配置中获取初始位置或设为0
                        initial_obj_positions = torch.zeros(num_obj_dofs, device=self.device)
                        self.initial_dof_states.view(self.num_envs, self.num_dofs, 2)[env_id, obj_dof_start:obj_dof_start+num_obj_dofs, 0] = initial_obj_positions
                        # 速度设为0
                        self.initial_dof_states.view(self.num_envs, self.num_dofs, 2)[env_id, obj_dof_start:obj_dof_start+num_obj_dofs, 1] = 0.0

        self.reset_arm(first_time=True)

    def reset_arm(self, first_time=False):
        self.reset(first_time=first_time)
        for _ in range(10):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            self.compute_observations()

    def step_simulation(self, step_time=1):
        for _ in range(step_time):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            self.compute_observations()

    def destroy(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
                
            
    def _configure_specifications(self, specs: Dict, mdp_type: str) -> None:
        assert "__dim__" in specs, "spec must contain `__dim__`"
        assert mdp_type in ["observation", "action"], "mdp_type must be either `observation` or `action`"

        Spec = ObservationSpec if mdp_type == "observation" else ActionSpec

        dims: Dict[str, Union[str, int]] = specs.pop("__dim__")
        for name, value in dims.items():
            assert isinstance(value, int) or isinstance(value, str), "dim must be either int or str"
            dims[name] = value if isinstance(value, int) else getattr(self, value)

        _specs = []
        for name, info in specs.items():
            shape = info["shape"]

            if not isinstance(shape, omegaconf.listconfig.ListConfig):
                shape = [shape]

            shape = [dims[d] if isinstance(d, str) else d for d in shape]
            dim = int(np.prod(shape))

            _specs.append(Spec(name, dim, **info))
        return _specs

    def _configure_observation_specs(self, observation_specs: Dict) -> None:
        """Configure the observation specifications.

        All the observation specifications are stored in `self._observation_specs`

        Args:
            observation_specs (Dict): The observation specifications. (cfg["env"]["observation_specs"])
        """
        self._observation_specs = self._configure_specifications(observation_specs, "observation")

    def __configure_action_specs(self, action_specs: Dict) -> None:
        """Configure the action specifications.

        All the action specifications are stored in `self._action_specs`

        Args:
            action_specs (Dict): The action specifications. (cfg["env"]["action_specs"])
        """
        self._action_specs = self._configure_specifications(action_specs, "action")

    def export_observation_metainfo_frame(self) -> pd.DataFrame:
        """Export the observation metainfo as pandas dataframe.

        Returns:
            pd.DataFrame: The observation metainfo frame.
        """
        metainfo = self.export_observation_metainfo()
        for item in metainfo:
            item["tags"] = ",".join(item["tags"])
        return pd.DataFrame(metainfo)

    def export_observation_metainfo(self) -> List[Dict[str, Any]]:
        """Export the observation metainfo.

        Returns:
            List[Dict[str, Any]]: The observation metainfo.
        """
        metainfo = []
        current = 0
        for spec in self._observation_space:
            metainfo.append(
                {
                    "name": spec.name,
                    "dim": spec.dim,
                    "tags": spec.tags,
                    "start": current,
                    "end": current + spec.dim,
                }
            )
            current += spec.dim
        return metainfo

    def export_action_metainfo(self) -> List[Dict[str, Any]]:
        """Export the action metainfo.

        Returns:
            List[Dict[str, Any]]: The action metainfo.
        """
        metainfo = []
        current = 0
        for spec in self._action_space:
            metainfo.append(
                {
                    "name": spec.name,
                    "dim": spec.dim,
                    "start": current,
                    "end": current + spec.dim,
                }
            )
            current += spec.dim
        return metainfo

    def _get_observation_spec(self, name: str) -> ObservationSpec:
        """Get the specification of an observation.

        Args:
            name: The name of the observation.

        Returns:
            The specification of the observation.
        """
        for spec in self._observation_specs:
            if spec.name == name:
                return spec
        raise ValueError(f"Observation {name} not found.")

    def _get_observation_dim(self, name: str) -> int:
        """Get the dimension of an observation.

        Args:
            name: The name of the observation.

        Returns:
            The dimension of the observation.
        """
        return self._get_observation_spec(name).dim

    def _get_action_spec(self, name: str) -> ActionSpec:
        """Get the specification of an action.

        Args:
            name: The name of the action.

        Returns:
            The specification of the action.
        """
        for spec in self._action_specs:
            if spec.name == name:
                return spec
        raise ValueError(f"Action {name} not found.")

    def _get_action_dim(self, name: str) -> int:
        """Get the dimension of an action.

        Args:
            name: The name of the action.

        Returns:
            The dimension of the action.
        """
        return self._get_action_spec(name).dim

    def _configure_mdp_spaces(self) -> None:
        """Configure the observation, state and action spaces for the task.

        Define the scale and offset for each observation, state and action. Calculate the total number of observations,
        states and actions, and display the information to terminal.
        """
        # configure action space
        self.__configure_action_specs(self.cfg["env"]["actionSpecs"])
        self._action_space = [self._get_action_spec(name) for name in self.cfg["env"]["actionSpace"]]
        self.num_actions = sum([self._get_action_dim(name) for name in self.cfg["env"]["actionSpace"]])
        self.cfg["env"]["numActions"] = self.num_actions

        # configure observation space
        self._configure_observation_specs(self.cfg["env"]["observationSpecs"])
        observation_space = self.cfg["env"]["observationSpace"]
        observation_space_extra = self.cfg["env"]["observationSpaceExtra"]
        observation_space_extra = [] if observation_space_extra is None else observation_space_extra

        num_observations = (
            sum([self._get_observation_dim(name) for name in observation_space]) * self.stack_frame_number
        )
        self.cfg["env"]["numObservations"] = num_observations
        self.cfg["env"]["numStates"] = self.cfg["env"]["numObservations"] * self.stack_frame_number

        self._observation_space = [self._get_observation_spec(name) for name in observation_space]

        # check if observation space extra already exists in observation space
        for name in observation_space_extra:
            if name in observation_space:
                warnings.warn(f"Observation {name} already exists in the observation space.")
        observation_space_extra = [name for name in observation_space_extra if name not in observation_space]
        observation_space_extra = observation_space + observation_space_extra

        self._observation_space_extra = [self._get_observation_spec(name) for name in observation_space_extra]
        self._required_attributes = [spec.attr for spec in self._observation_space_extra]
        if self.env_info_logging:
            print_observation_space(self._observation_space)
            print_action_space(self._action_space)

    def _configure_specifications(self, specs: Dict, mdp_type: str) -> None:
        assert "__dim__" in specs, "spec must contain `__dim__`"
        assert mdp_type in ["observation", "action"], "mdp_type must be either `observation` or `action`"

        Spec = ObservationSpec if mdp_type == "observation" else ActionSpec

        dims: Dict[str, Union[str, int]] = specs.pop("__dim__")
        for name, value in dims.items():
            assert isinstance(value, int) or isinstance(value, str), "dim must be either int or str"
            dims[name] = value if isinstance(value, int) else getattr(self, value)

        _specs = []
        for name, info in specs.items():
            shape = info["shape"]

            if not isinstance(shape, omegaconf.listconfig.ListConfig):
                shape = [shape]

            shape = [dims[d] if isinstance(d, str) else d for d in shape]
            dim = int(np.prod(shape))

            _specs.append(Spec(name, dim, **info))
        return _specs

    def _get_observation_spec(self, name: str) -> ObservationSpec:
        for spec in self._observation_specs:
            if spec.name == name:
                return spec
        raise ValueError(f"Observation {name} not found.")

    def _get_observation_dim(self, name: str) -> int:
        return self._get_observation_spec(name).dim

    def _get_action_spec(self, name: str) -> ActionSpec:
        for spec in self._action_specs:
            if spec.name == name:
                return spec
        raise ValueError(f"Action {name} not found.")

    # -------------------------------------------------------------------------
    # Observation computation
    # -------------------------------------------------------------------------
    def compute_observations(self, reset_env_ids: Optional[torch.LongTensor] = None) -> None:
        """Compute the observations.

        The observations required for the task training are stored in `self.obs_buf`.

        Args:
            reset_env_ids (Optional[torch.LongTensor], optional): The indices of the environments to reset. Defaults to None.
                corresponding envs will be reset to the initial state if self.stack_frame_number > 1.
        """
        observation_dict: OrderedDict = self.retrieve_observation_dict()

        # only fetch the observations required for the task training
        observations: torch.Tensor = torch.cat(
            [observation_dict[spec.name].reshape(self.num_envs, -1) for spec in self._observation_space], dim=-1
        )

        if self.stack_frame_number > 1:
            if len(self.frames) == 0:
                self.frames.extend([observations.clone() for _ in range(self.stack_frame_number)])
            else:
                self.frames.append(observations.clone())
                if reset_env_ids is not None:
                    for frame in self.frames:
                        frame[reset_env_ids] = observations[reset_env_ids]

            self.obs_buf[:] = torch.cat(list(self.frames), 1)
        else:
            self.obs_buf[:] = observations

        if self.cfg["env"].get("returnCuriosityInfo", False):
            if self.curiosity_state_type == "policy_state":
                self.extras['curiosity_states'] = self.obs_buf[:].clone()
            elif self.curiosity_state_type == "contact_force":
                self.extras['curiosity_states'] = torch.cat([self.left_keypoint_contact_forces, self.right_keypoint_contact_forces], dim=-1).clone()
            elif self.curiosity_state_type == "contact_distance":
                left_top_relative_position = self.left_keypoint_positions_with_offset - self.top_part_position.unsqueeze(1)
                right_top_relative_position = self.right_keypoint_positions_with_offset - self.top_part_position.unsqueeze(1)
                left_bottom_relative_position = self.left_keypoint_positions_with_offset - self.bottom_part_position.unsqueeze(1)
                right_bottom_relative_position = self.right_keypoint_positions_with_offset - self.bottom_part_position.unsqueeze(1)

                rel_pos = torch.cat([left_top_relative_position, right_top_relative_position, left_bottom_relative_position, right_bottom_relative_position], dim=1).clone() # concat keypoint dim
                self.extras['curiosity_states'] = rel_pos.clone()
            else:
                raise ValueError(f"Unknown curiosity state type: {self.curiosity_state_type}")
            
    def retrieve_observation_dict(self) -> OrderedDict:
        """Retrieve the observation dict.

        Returns:
            OrderedDict[str, torch.Tensor]: The observation dict.
        """
        self._refresh_sim_tensors()

        observations = OrderedDict()
        for spec in self._observation_space_extra:
            observation: torch.Tensor = getattr(self, spec.attr)

            if "dof" in spec.tags and "position" in spec.tags:
                observation = normalize(
                    observation,
                    self.gym_assets["current"]["robot"]["limits"]["lower"],
                    self.gym_assets["current"]["robot"]["limits"]["upper"],
                )
            elif "velocity" in spec.tags:
                observation = observation * self.velocity_observation_scale
            elif "orientation" in spec.tags:
                observation = quat_to_6d(observation)

            observations[spec.name] = observation

            # if add_noise:
            #     if "object_position_wrt_palm" == spec.name:
            #         observations[spec.name] = self.observed_object_positions_wrt_palm.clone()
            #     if "object_orientation_wrt_palm" == spec.name:
            #         observations[spec.name] = self.observed_object_orientations_wrt_palm.clone()

        return observations

    def _refresh_sim_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)


        left_forearm_rb_idx = self.forearm_rigid_body_id["right"]  # actor-local body index
        right_forearm_rb_idx = self.forearm_rigid_body_id["left"]

        self.right_eef_jacobian = self.jacobians_right[:, right_forearm_rb_idx - 1, :, :6]
        self.left_eef_jacobian = self.jacobians_left[:, left_forearm_rb_idx - 1, :, :6]
                
        root_states = self.root_states.view(self.num_envs, self.num_actors, 13)
        net_contact_forces = self.net_contact_forces.view(self.num_envs, self.num_rigid_bodies, 3)
        
        
        left_hand_rigid_body_handle_env_slice = slice(self.left_hand_rigid_body_start, self.left_hand_rigid_body_end)
        right_hand_rigid_body_handle_env_slice = slice(self.right_hand_rigid_body_start, self.right_hand_rigid_body_end)
        object_rigid_body_handle_env_slice = slice(self.object_rigid_body_start, self.object_rigid_body_end)
        left_arm_links_slice = slice(self.left_hand_rigid_body_start, self.left_hand_rigid_body_start + 7) # include world link
        right_arm_links_slice = slice(self.right_hand_rigid_body_start, self.right_hand_rigid_body_start + 7) # include world link
        self.left_arm_contact_forces = net_contact_forces[:, left_arm_links_slice, :]
        self.right_arm_contact_forces = net_contact_forces[:, right_arm_links_slice, :]
        self.lh_contact_forces = net_contact_forces[:, left_hand_rigid_body_handle_env_slice, :]                      # (N, L, 3)
        self.rh_contact_forces = net_contact_forces[:, right_hand_rigid_body_handle_env_slice, :]                      # (N, R, 3)
        self.object_contact_forces = net_contact_forces[:, object_rigid_body_handle_env_slice, :]                 # (N, O, 3)
        
        self.left_fingertip_contact_forces = self.lh_contact_forces[:, self.left_fingertip_indices, :]
        self.right_fingertip_contact_forces = self.rh_contact_forces[:, self.right_fingertip_indices, :]
        self.left_keypoint_contact_forces = self.lh_contact_forces[:, self.left_keypoint_indices, :]
        self.right_keypoint_contact_forces = self.rh_contact_forces[:, self.right_keypoint_indices, :]
        self.table_contact_forces = net_contact_forces[:, self.table_rigid_body_indices[0], :] # assure the actor in same order
        
        
        # Actor root (already sliced during setup)
        # Right hand
        self.right_hand_root_positions = root_states[:, self.hand_actor_ids_env["right"], 0:3]
        self.right_hand_root_orientations = root_states[:, self.hand_actor_ids_env["right"], 3:7]
        self.right_hand_root_linear_velocities = root_states[:, self.hand_actor_ids_env["right"], 7:10]
        self.right_hand_root_angular_velocities = root_states[:, self.hand_actor_ids_env["right"], 10:13]
        # Left hand
        self.left_hand_root_positions = root_states[:, self.hand_actor_ids_env["left"], 0:3]
        self.left_hand_root_orientations = root_states[:, self.hand_actor_ids_env["left"], 3:7]
        self.left_hand_root_linear_velocities = root_states[:, self.hand_actor_ids_env["left"], 7:10]
        self.left_hand_root_angular_velocities = root_states[:, self.hand_actor_ids_env["left"], 10:13]
        # Object (assumes single tracked object per env)
        self.object_root_positions = root_states[:, self.object_handles_env[0], 0:3].view(self.num_envs, 3)
        self.object_root_orientations = root_states[:, self.object_handles_env[0], 3:7].view(self.num_envs, 4)
        self.object_root_linear_velocities = root_states[:, self.object_handles_env[0], 7:10].view(self.num_envs, 3)
        self.object_root_angular_velocities = root_states[:, self.object_handles_env[0], 10:13].view(self.num_envs, 3)

        # Mirror to generic names used by the YAML to keep things explicit
        self.rh_root_position = self.right_hand_rigid_body_positions[:, self.hand_wrist_rigid_body_id["right"]]
        self.rh_root_orientation = self.right_hand_rigid_body_orientations[:, self.hand_wrist_rigid_body_id["right"]]
        self.rh_root_linear_velocity = self.right_hand_rigid_body_linear_velocities[:, self.hand_wrist_rigid_body_id["right"]]
        self.rh_root_angular_velocity = self.right_hand_rigid_body_angular_velocities[:, self.hand_wrist_rigid_body_id["right"]]

        self.lh_root_position = self.left_hand_rigid_body_positions[:, self.hand_wrist_rigid_body_id["left"]]
        self.lh_root_orientation = self.left_hand_rigid_body_orientations[:, self.hand_wrist_rigid_body_id["left"]]
        self.lh_root_linear_velocity = self.left_hand_rigid_body_linear_velocities[:, self.hand_wrist_rigid_body_id["left"]]
        self.lh_root_angular_velocity = self.left_hand_rigid_body_angular_velocities[:, self.hand_wrist_rigid_body_id["left"]]
        
        self.left_fingertip_positions = self.left_hand_rigid_body_positions[:, self.left_fingertip_indices]
        self.right_fingertip_positions = self.right_hand_rigid_body_positions[:, self.right_fingertip_indices]
        
        self.left_keypoint_positions = self.left_hand_rigid_body_positions[:, self.left_keypoint_indices]
        self.left_keypoint_orientations = self.left_hand_rigid_body_orientations[:, self.left_keypoint_indices]
        self.right_keypoint_positions = self.right_hand_rigid_body_positions[:, self.right_keypoint_indices]
        self.right_keypoint_orientations = self.right_hand_rigid_body_orientations[:, self.right_keypoint_indices]
        
        self.left_keypoint_positions_with_offset = self.left_keypoint_positions + quat_apply(self.left_keypoint_orientations, self.left_keypoint_offsets.repeat(self.num_envs, 1, 1))
        self.right_keypoint_positions_with_offset = self.right_keypoint_positions + quat_apply(self.right_keypoint_orientations, self.right_keypoint_offsets.repeat(self.num_envs, 1, 1))
        
        # build per-finger keypoint (with offset) positions / orientations / forces
        def _slice_finger_keypoints(side: str, finger: str, idx_attr: str):
            idx_list = getattr(self, idx_attr, None)
            if idx_list is None or len(idx_list) == 0:
                return
            idx = torch.as_tensor(idx_list, dtype=torch.long, device=self.device)
            if side == "left":
                kp_pos = self.left_keypoint_positions_with_offset[:, idx, :]      # (N, K_f, 3)
                kp_ori = self.left_keypoint_orientations[:, idx, :]              # (N, K_f, 4)
                kp_force = self.left_keypoint_contact_forces[:, idx, :]          # (N, K_f, 3)
                setattr(self, f"left_{finger}_finger_keypoint_positions_with_offset", kp_pos)
                setattr(self, f"left_{finger}_finger_keypoint_orientations", kp_ori)
                setattr(self, f"left_{finger}_finger_keypoint_forces", kp_force)
            else:
                kp_pos = self.right_keypoint_positions_with_offset[:, idx, :]
                kp_ori = self.right_keypoint_orientations[:, idx, :]
                kp_force = self.right_keypoint_contact_forces[:, idx, :]
                setattr(self, f"right_{finger}_finger_keypoint_positions_with_offset", kp_pos)
                setattr(self, f"right_{finger}_finger_keypoint_orientations", kp_ori)
                setattr(self, f"right_{finger}_finger_keypoint_forces", kp_force)

        # left hand fingers
        _slice_finger_keypoints("left", "index",  "left_index_link_indices_among_keypoints")
        _slice_finger_keypoints("left", "middle", "left_middle_link_indices_among_keypoints")
        _slice_finger_keypoints("left", "ring",   "left_ring_link_indices_among_keypoints")
        _slice_finger_keypoints("left", "thumb",  "left_thumb_link_indices_among_keypoints")
        # right hand fingers
        _slice_finger_keypoints("right", "index",  "right_index_link_indices_among_keypoints")
        _slice_finger_keypoints("right", "middle", "right_middle_link_indices_among_keypoints")
        _slice_finger_keypoints("right", "ring",   "right_ring_link_indices_among_keypoints")
        _slice_finger_keypoints("right", "thumb",  "right_thumb_link_indices_among_keypoints")
        
        
        self.left_fingertip_states = self.left_hand_rigid_body_states[:, self.left_virtual_tip_indices, :]
        self.left_fingertip_positions = self.left_fingertip_states[..., 0:3]
        self.left_fingertip_orientations = self.left_fingertip_states[..., 3:7]
        self.left_fingertip_linear_velocities = self.left_fingertip_states[..., 7:10]
        self.left_fingertip_angular_velocities = self.left_fingertip_states[..., 10:13]
        
        self.right_fingertip_states = self.right_hand_rigid_body_states[:, self.left_virtual_tip_indices, :]
        self.right_fingertip_positions = self.right_fingertip_states[..., 0:3]
        self.right_fingertip_orientations = self.right_fingertip_states[..., 3:7]
        self.right_fingertip_linear_velocities = self.right_fingertip_states[..., 7:10]
        self.right_fingertip_angular_velocities = self.right_fingertip_states[..., 10:13]

        rb_top = self._object_parts["top"]["rigid_body_indices"][0]
        rb_bottom = self._object_parts["bottom"]["rigid_body_indices"][0]
        self.top_part_position = self.object_rigid_body_positions[:, rb_top]
        self.top_part_orientation = self.object_rigid_body_orientations[:, rb_top]
        self.top_part_linear_velocity = self.object_rigid_body_linear_velocities[:, rb_top]
        self.top_part_angular_velocity = self.object_rigid_body_angular_velocities[:, rb_top]
        self.bottom_part_position = self.object_rigid_body_positions[:, rb_bottom]
        self.bottom_part_orientation = self.object_rigid_body_orientations[:, rb_bottom]
        self.bottom_part_linear_velocity = self.object_rigid_body_linear_velocities[:, rb_bottom]
        self.bottom_part_angular_velocity = self.object_rigid_body_angular_velocities[:, rb_bottom]
        
        # DOF state slices (already shaped in setup)
        # right: self.right_hand_dof_positions / self.right_hand_dof_velocities
        # left:  self.left_hand_dof_positions  / self.left_hand_dof_velocities
        # These are updated by refresh_dof_state_tensor and view set in __init__.

        # Relative object pose w.r.t. palms (use actor roots as palm frames)
        rh_q_conj = quat_conjugate(self.right_hand_root_orientations)
        lh_q_conj = quat_conjugate(self.left_hand_root_orientations)

        self.object_positions_wrt_rh_palm = quat_rotate(
            rh_q_conj, self.object_root_positions - self.right_hand_root_positions
        )
        self.object_orientations_wrt_rh_palm = quat_mul(
            rh_q_conj, self.object_root_orientations
        )

        self.object_positions_wrt_lh_palm = quat_rotate(
            lh_q_conj, self.object_root_positions - self.left_hand_root_positions
        )
        self.object_orientations_wrt_lh_palm = quat_mul(
            lh_q_conj, self.object_root_orientations
        )

        # Inter-hand relative pose (left w.r.t. right)
        self.lh_wrt_rh_position = quat_rotate(
            rh_q_conj, self.left_hand_root_positions - self.right_hand_root_positions
        )
        self.lh_wrt_rh_orientation = quat_mul(
            rh_q_conj, self.left_hand_root_orientations
        )
        
        # overwrite fingertip states with virtual tip heads
        self.left_fingertip_states = self.left_hand_rigid_body_states[:, self.left_virtual_tip_indices, :]
        self.left_fingertip_positions = self.left_fingertip_states[..., 0:3]
        self.left_fingertip_orientations = self.left_fingertip_states[..., 3:7]
        self.left_fingertip_linear_velocities = self.left_fingertip_states[..., 7:10]
        self.left_fingertip_angular_velocities = self.left_fingertip_states[..., 10:13]
        
        self.right_fingertip_states = self.right_hand_rigid_body_states[:, self.left_virtual_tip_indices, :]
        self.right_fingertip_positions = self.right_fingertip_states[..., 0:3]
        self.right_fingertip_orientations = self.right_fingertip_states[..., 3:7]
        self.right_fingertip_linear_velocities = self.right_fingertip_states[..., 7:10]
        self.right_fingertip_angular_velocities = self.right_fingertip_states[..., 10:13]
        
        self._update_parts_pointclouds()
        
    def _update_parts_pointclouds(self):
        """更新物体各部分在世界坐标系下的点云"""
        self._object_parts_world_pointclouds = {}
        self._object_parts_world_normals = {}
        
        for part_name, part_info in self._object_parts.items():
            if part_info["pointcloud"] is None:
                continue
                
            # 获取该部分对应的link的rigid body states
            part_points = []
            part_normals = []

            for rb_idx in part_info["rigid_body_indices"]:
                part_pos = self.object_rigid_body_positions[:, rb_idx]
                part_rot = self.object_rigid_body_orientations[:, rb_idx]
                
                # 将canonical点云变换到世界坐标系
                pc_local = part_info["pointcloud"].unsqueeze(0).repeat(self.num_envs, 1, 1)
                normals_local = part_info["pointcloud_normals"].unsqueeze(0).repeat(self.num_envs, 1, 1)
                
                # 关键修复：扩展part_rot以匹配pc_local的点数维度
                part_rot_expanded = part_rot.unsqueeze(1).expand(-1, pc_local.size(1), -1)
                normals_rotated = quat_apply(part_rot_expanded, normals_local)
                
                # 应用旋转和位移
                pc_rotated = quat_apply(part_rot_expanded, pc_local)
                pc_world = pc_rotated + part_pos.unsqueeze(1)
                
                part_points.append(pc_world)
                part_normals.append(normals_rotated)
            
            # 合并所有link的点云
            if part_points:
                self._object_parts_world_pointclouds[part_name] = torch.cat(part_points, dim=1)
                self._object_parts_world_normals[part_name] = torch.cat(part_normals, dim=1)

    @property
    def contact_states(self) -> torch.Tensor:
        """Compute contact states (tactile information) from force sensor data.
        """
        assert self.lh_contact_forces is not None and self.rh_contact_forces is not None and self.object_contact_forces is not None

        return (torch.cat([self.lh_contact_forces.norm(dim=-1), self.rh_contact_forces.norm(dim=-1)], dim=-1).clamp_max(5.0)).float() / 5.0

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    
    def reset_idx(self, env_ids: LongTensor, first_time: bool = False) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(dtype=torch.long, device=self.device)

        # dof_states_view = self.dof_states.view(self.num_envs, self.num_dofs, 2)
        # initial_dof_states_view = self.initial_dof_states.view(self.num_envs, self.num_dofs, 2)

        # dof_states_view[env_ids] = initial_dof_states_view[env_ids]
        # dof_states_view[env_ids, :, 0] = 0.0
        # dof_states_view[env_ids, :, 1] = 0.0

        hand_indices_list = []
        for side in ["right", "left"]:
            hand_indices_list.append(self.hand_actor_handle_sim[side][env_ids].to(torch.int32))
        hand_actor_indices = torch.cat(hand_indices_list)  # (2*num_envs, )
        # sort hand actor indices
        hand_actor_indices = hand_actor_indices[hand_actor_indices.argsort()]
        
        object_indices = self.object_handles_sim[env_ids][:, 0]
        
        # reset object root states
        self.root_positions.view(self.num_envs, self.num_actors, 3)[env_ids, self.object_handles_env[0]] = torch.tensor(self.object_cfgs[0]["pose"][:3], device=self.device)
        self.root_orientations.view(self.num_envs, self.num_actors, 4)[env_ids, self.object_handles_env[0]] = torch.tensor(self.object_cfgs[0]["pose"][3:], device=self.device)
        self.root_linear_velocities.view(self.num_envs, self.num_actors, 3)[env_ids, self.object_handles_env[0]] = 0.0
        self.root_angular_velocities.view(self.num_envs, self.num_actors, 3)[env_ids, self.object_handles_env[0]] = 0.0
        
        # reset object dof states
        self.object_dof_positions[env_ids, :] = self.init_object_dof_positions[env_ids, :]
        self.object_dof_velocities[env_ids, :] = 0.0
        
        
        # reset hand root states
        self.left_hand_root_positions[env_ids] = torch.tensor(self.cfg["env"]["hands"]["left"]["initPose"]["position"], device=self.device)
        self.left_hand_root_orientations[env_ids] = torch.tensor(self.cfg["env"]["hands"]["left"]["initPose"]["orientation"], device=self.device)
        self.left_hand_root_linear_velocities[env_ids] = 0.0
        self.left_hand_root_angular_velocities[env_ids] = 0.0
        self.right_hand_root_positions[env_ids] = torch.tensor(self.cfg["env"]["hands"]["right"]["initPose"]["position"], device=self.device)
        self.right_hand_root_orientations[env_ids] = torch.tensor(self.cfg["env"]["hands"]["right"]["initPose"]["orientation"], device=self.device)
        self.right_hand_root_linear_velocities[env_ids] = 0.0
        self.right_hand_root_angular_velocities[env_ids] = 0.0
        
        
        # reset hand dof states
        self.left_hand_arm_dof_positions[env_ids, :] = self.gym_assets["current"]["hand_left"]["init"]["position"]
        self.left_hand_arm_dof_velocities[env_ids, :] = self.gym_assets["current"]["hand_left"]["init"]["velocity"]
        self.right_hand_arm_dof_positions[env_ids, :] = self.gym_assets["current"]["hand_right"]["init"]["position"]
        self.right_hand_arm_dof_velocities[env_ids, :] = self.gym_assets["current"]["hand_right"]["init"]["velocity"]
        
        left_hand_arm_dof_slice = slice(self.left_hand_arm_dof_start, self.left_hand_arm_dof_end)
        right_hand_arm_dof_slice = slice(self.right_hand_arm_dof_start, self.right_hand_arm_dof_end)
        self.prev_targets_buffer[env_ids, left_hand_arm_dof_slice] = self.left_hand_arm_dof_positions[env_ids, :]
        self.curr_targets_buffer[env_ids, left_hand_arm_dof_slice] = self.left_hand_arm_dof_positions[env_ids, :]
        self.prev_targets_buffer[env_ids, right_hand_arm_dof_slice] = self.right_hand_arm_dof_positions[env_ids, :]
        self.curr_targets_buffer[env_ids, right_hand_arm_dof_slice] = self.right_hand_arm_dof_positions[env_ids, :]
        
        hand_object_indices = torch.cat([hand_actor_indices, object_indices])
        hand_object_indices = hand_object_indices[hand_object_indices.argsort()]
        
        for side in ["right", "left"]:
            for part_name in ["top", "bottom"]:
                self._curiosity_managers[side][part_name].ensure_running_max_buffers(self.num_envs)
                self._curiosity_managers[side][part_name].reset_running_max_buffers(env_ids)
        
        # 5) 更新 target buffer
        # self.curr_targets_buffer[env_ids, :] = initial_dof_states_view[env_ids, :, 0]
        # self.prev_targets_buffer[env_ids, :] = initial_dof_states_view[env_ids, :, 0]
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(hand_object_indices),
            hand_object_indices.shape[0],
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(hand_object_indices),
            hand_object_indices.shape[0],
        )

        # self.gym.set_actor_root_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.root_states),
        #     gymtorch.unwrap_tensor(hand_actor_indices),
        #     hand_actor_indices.shape[0],
        # )
        
        # self.gym.set_dof_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.dof_states),
        #     gymtorch.unwrap_tensor(hand_actor_indices),
        #     hand_actor_indices.shape[0],
        # )
        
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.curr_targets_buffer),
            gymtorch.unwrap_tensor(hand_actor_indices),
            hand_actor_indices.shape[0],
        )


        # 6) 清零刚体力/力矩缓冲，避免残留外力
        if hasattr(self, "rb_forces"):
            self.rb_forces[env_ids, :, :] = 0.0
        if hasattr(self, "rb_torques"):
            self.rb_torques[env_ids, :, :] = 0.0
        if not hasattr(self, "cap_progress_max"):
            self.cap_progress_max = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        if not hasattr(self, "cap_progress_max_height"):
            self.cap_progress_max_height = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        self.cap_progress_max[env_ids] = 0.0
        self.cap_progress_max_height[env_ids] = 0.0
        
        if hasattr(self, "left_keypoints_to_surface_dist_min"):
            self.left_keypoints_to_surface_dist_min[env_ids] = 0.50
        if hasattr(self, "right_keypoints_to_surface_dist_min"):
            self.right_keypoints_to_surface_dist_min[env_ids] = 0.50

        # 7) PID/控制缓冲
        if self.use_pid_control:
            self._reset_pid_buffers(env_ids)

        # 8) 计数器
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_step_counter[env_ids] = 0
        self.successes[env_ids] = 0

    # -------------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------------
    
    def _refresh_action_tensors(self, actions: torch.Tensor) -> None:
        """Given a batch of actions, refresh the action tensors.

        Args:
            actions (torch.Tensor): A batch of actions. [batch_size, action_dim]
        """
        current = 0
        for spec in self._action_space:
            setattr(self, spec.attr, actions[:, current : current + spec.dim])
            current += spec.dim

    def _apply_hand_root_targets(self, side: str, translation_action: Tensor, rotation_action: Tensor) -> None:

        if side == "right":
            current_eef_pos = self.right_endeffector_positions
            current_eef_quat = self.right_endeffector_orientations
            eef_jacobian = self.right_eef_jacobian           # (N, 6, 6)
        else:
            current_eef_pos = self.left_endeffector_positions
            current_eef_quat = self.left_endeffector_orientations
            eef_jacobian = self.left_eef_jacobian

        # 动作 → EEF 平移 / 旋转速度（tangent 空间）
        eef_translations = translation_action
        # eef_rotations_axis = rot6d_to_axis_angle(rotation_action)  # (N, 3)
        eef_rotations = rotation_action

        # 调用旧环境里的相对控制 IK
        dof_delta, target_pos, target_euler = compute_relative_xarm_dof_positions(
            current_eef_positions=current_eef_pos,
            current_eef_orientations=current_eef_quat,
            eef_jacobian=eef_jacobian,
            eef_translations=eef_translations,
            eef_rotations=eef_rotations,
            max_eef_translation_speed=self.max_eef_translation_speed,
            max_eef_rotation_speed=self.max_eef_rotation_speed,
            dt=self.dt,
        )

        arm_dof_slice = self.left_arm_dof_slice if side == "left" else self.right_arm_dof_slice
        current_targets = self.curr_targets_buffer[:, arm_dof_slice]
        new_targets = current_targets + dof_delta

        self.curr_targets_buffer[:, arm_dof_slice] = new_targets


    def _apply_hand_finger_targets(self, side: str, finger_action: Tensor) -> None:
        """Apply finger position control for a specific hand using relative velocity control."""
        hand_dof_start = self.hand_dof_starts[side]
        finger_dof = self.num_hand_dofs
        
        # Reshape dof_states to [num_envs, num_dofs, 2] format before indexing
        dof_states = self.dof_states.view(self.num_envs, self.num_dofs, 2)
        
        # Get current finger DOF positions from simulation state
        finger_slice = slice(hand_dof_start, hand_dof_start + finger_dof)
        current_finger_positions = dof_states[:, hand_dof_start :hand_dof_start +  finger_dof, 0].clone()
        
        # Compute finger velocity commands
        finger_velocities = finger_action * self.finger_action_scale
        
        # Integrate velocities over time to get position changes
        finger_position_delta = finger_velocities * self.dt
        
        # Update finger targets in the buffer
        finger_target_slice = self.left_hand_dof_slice if side == "left" else self.right_hand_dof_slice
        targets = self.curr_targets_buffer[:, finger_target_slice].clone()
        targets = current_finger_positions + finger_position_delta

        
        # Apply the targets back to the buffer
        self.curr_targets_buffer[:, finger_target_slice] = targets

    def pre_physics_step(self, actions: Tensor) -> None:
        if self.training:
            self.reset_done()
        
        # Apply moving average filter to actions
        smoothed_actions = self.actions_moving_average * actions + (1.0 - self.actions_moving_average) * self.prev_actions
        self.prev_actions[:] = smoothed_actions[:]
        self.actions = smoothed_actions.clone().to(self.device)
        
        # Refresh action tensors to get individual action components
        self._refresh_action_tensors(smoothed_actions.clone().to(self.device))
        
        # --- Root motion control for both hands ---
        if self.use_pid_control:
            # Clear force/torque buffers for this step
            self.rb_forces.zero_()
            self.rb_torques.zero_()
            # PID → wrench at wrists
            self._apply_hand_root_forces("right", self.rh_root_translation_action, self.rh_root_rotation_action)
            self._apply_hand_root_forces("left", self.lh_root_translation_action, self.lh_root_rotation_action)
        else:
            # Original DOF-based root control
            self._apply_hand_root_targets("right", self.rh_root_translation_action, self.rh_eef_rotation_action)
            self._apply_hand_root_targets("left", self.lh_root_translation_action, self.lh_eef_rotation_action)
        
        # Apply finger position control for both hands
        if hasattr(self, 'rh_finger_action') and hasattr(self, 'lh_finger_action'):
            self._apply_hand_finger_targets("right", self.rh_finger_action)
            self._apply_hand_finger_targets("left", self.lh_finger_action)
            
        self.curr_targets_buffer[:, self.right_hand_arm_dof_start:self.right_hand_arm_dof_end] = torch.clamp(
            self.curr_targets_buffer[:, self.right_hand_arm_dof_start:self.right_hand_arm_dof_end], 
            min=self.hand_dof_lower_limits["right"], 
            max=self.hand_dof_upper_limits["right"]
        )
        self.curr_targets_buffer[:, self.left_hand_arm_dof_start:self.left_hand_arm_dof_end] = torch.clamp(
            self.curr_targets_buffer[:, self.left_hand_arm_dof_start:self.left_hand_arm_dof_end], 
            min=self.hand_dof_lower_limits["left"], 
            max=self.hand_dof_upper_limits["left"]
        )

        self.prev_targets_buffer = self.curr_targets_buffer.clone()
        
        # Set DOF position targets for fingers (and any remaining DOFs)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.curr_targets_buffer))

        # If PID is enabled, actually apply the accumulated wrenches at wrists
        if self.use_pid_control:
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.rb_forces),
                gymtorch.unwrap_tensor(self.rb_torques),
                gymapi.ENV_SPACE,
            )
        


    # -------------------------------------------------------------------------
    # Reward / Done
    # -------------------------------------------------------------------------
    
    def compute_done(self, is_success):
        test_sim = False
        if not test_sim:
            fall_env_mask = (
                (self.object_root_positions[:, 2] < self.gym_assets["current"]["table"]['pose'].p.z - 0.1)
            )
            
            arm_contact_mask = (self.left_arm_contact_forces.norm(p=2, dim=2) > 0).any(dim=1) | (self.right_arm_contact_forces.norm(p=2, dim=2) > 0).any(dim=1)

            failed_env_ids = (fall_env_mask | arm_contact_mask).nonzero(as_tuple=False).squeeze(-1)
                
            # if self.success_steps > 0:
            #     self.progress_buf = torch.where(is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf)
            #     self.reset_buf = torch.where(is_success > 0, 1, self.reset_buf)

            self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, 1, self.reset_buf)
            
            self.reset_buf[failed_env_ids] = 1

        # success
        succ_env_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        self.reset_buf[succ_env_ids] = 1
        self.successes[succ_env_ids] = 1
        self.successes[failed_env_ids] = 0

        # self.done_successes[failed_env_ids] = 0
        # self.done_successes[succ_env_ids] = 1

        # if "height" in self.reward_type:
        #     self.extras["final_object_height"] = self.delta_obj_height[
        #         self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        #     ].clone()
        self.extras["success_num"] = torch.sum(self.successes).unsqueeze(-1)

    @property
    def task_joint_position(self) -> Tensor:
        joint_idx = self._task_joint_indices[self.object_cfgs[0]["taskType"]]
        return self.object_dof_positions[:, joint_idx]
    
    @property
    def task_joint_velocity(self) -> Tensor:
        joint_idx = self._task_joint_indices[self.object_cfgs[0]["taskType"]]
        return self.object_dof_velocities[:, joint_idx]
    
    def _check_bimanual_contact_criteria(self) -> torch.Tensor:

        device = self.device
        N = self.num_envs


        left_ok  = torch.zeros(N, dtype=torch.bool, device=device)
        right_ok = torch.zeros(N, dtype=torch.bool, device=device)

        if not hasattr(self, "_object_parts_world_pointclouds"):
            self._object_parts_world_pointclouds = {}

        top_pc    = self._object_parts_world_pointclouds.get("top", None)      # (N, P_top, 3)
        bottom_pc = self._object_parts_world_pointclouds.get("bottom", None)   # (N, P_bot, 3)

        if (top_pc is None) or (bottom_pc is None):
            contact_satisfied = left_ok & right_ok
            self.extras["left_hand_contact_ok"] = left_ok
            self.extras["right_hand_contact_ok"] = right_ok
            self.extras["bimanual_contact_satisfied"] = contact_satisfied
            return contact_satisfied
        
        all_pc = torch.cat([top_pc, bottom_pc], dim=1)


        right_kp_pos   = self.right_keypoint_positions_with_offset          # (N, K_r, 3)
        right_kp_force = self.right_keypoint_contact_forces                # (N, K_r, 3)

        # dists_r = torch.cdist(right_kp_pos, top_pc)                      # (N, K_r, P_top)
        dists_r = torch.cdist(right_kp_pos, all_pc)                        # (N, K_r, P_top + P_bot)
        min_dists_r, _ = dists_r.min(dim=-1)                               # (N, K_r)
        force_mag_r = right_kp_force.norm(dim=-1, p=2)                     # (N, K_r)

        near_surface_r = (min_dists_r < 0.012)                             # 同 curiosity 的阈值
        has_force_r    = (force_mag_r > 0.01) & (force_mag_r < 50)
        kp_contact_r   = near_surface_r & has_force_r                      # (N, K_r) bool
        right_ok       = kp_contact_r.any(dim=-1)                          # (N,)


        left_kp_pos   = self.left_keypoint_positions_with_offset           # (N, K_l, 3)
        left_kp_force = self.left_keypoint_contact_forces                  # (N, K_l, 3)

        # dists_l = torch.cdist(left_kp_pos, bottom_pc)                    # (N, K_l, P_bot)
        dists_l = torch.cdist(left_kp_pos, all_pc)                         # (N, K_l, P_top + P_bot)
        min_dists_l, _ = dists_l.min(dim=-1)                               # (N, K_l)
        force_mag_l = left_kp_force.norm(dim=-1, p=2)                      # (N, K_l)

        near_surface_l = (min_dists_l < 0.012)
        has_force_l    = (force_mag_l > 0.01) & (force_mag_l < 50)
        kp_contact_l   = near_surface_l & has_force_l                      # (N, K_l) bool
        left_ok        = kp_contact_l.any(dim=-1)                          # (N,)


        table_force_mag = self.table_contact_forces.norm(dim=-1)
        bottom_table_contact = (table_force_mag > 1e-2) & (table_force_mag < 50)
        bottom_table_contact &= (bottom_pc[:, :, 2].amin(dim=-1) - self.table_height) < 0.01

        # contact_satisfied = right_ok & left_ok
        
        contact_satisfied = bottom_table_contact
        
        upper_surface = (bottom_pc[:, :, 2] > self.table_height - 0.01).all(dim=-1)
        
        contact_satisfied = contact_satisfied & upper_surface

        self.extras["left_kp_contact_mask"]   = kp_contact_l.sum(dim=-1)
        self.extras["right_kp_contact_mask"]  = kp_contact_r.sum(dim=-1)
        self.extras["left_hand_contact_ok"]   = left_ok
        self.extras["right_hand_contact_ok"]  = right_ok
        self.extras["bottom_table_contact"]   = bottom_table_contact
        self.extras["upper_surface"]          = upper_surface
        self.extras["bimanual_contact_satisfied"] = contact_satisfied

        return contact_satisfied

    def compute_reward(self, actions: Tensor) -> None:
        self.rew_buf[:] = 0.0

        device = self.device
        N = self.num_envs
        
        self.compute_reach_reward_keypoints()
        self.rew_buf += self.reach_rew_scaled_keypoints

        contact_satisfied = self._check_bimanual_contact_criteria()
        contact_gate = contact_satisfied.float()

        cap_joint_positions = self.task_joint_position

        is_success_raw = torch.zeros((N,), device=device, dtype=torch.bool)
        is_success = is_success_raw
        progress = torch.zeros((N,), device=device, dtype=torch.float)
        delta_reward = torch.zeros_like(progress)
        delta = torch.zeros_like(progress)
        maintain_bonus = torch.zeros_like(progress)

        obj_pose_cfg = self.object_cfgs[0]["pose"]
        target_pos = torch.tensor(obj_pose_cfg[:3], device=device, dtype=torch.float)
        target_pos[2] += 0.10
        target_rot = torch.tensor(obj_pose_cfg[3:], device=device, dtype=torch.float)
        open_threshold = 1.57

        target_pos_dist = torch.norm(self.object_root_positions - target_pos.unsqueeze(0), dim=-1)
        target_rot_expanded = target_rot.unsqueeze(0).expand(N, 4)
        target_rot_dist = quat_diff_rad(self.object_root_orientations, target_rot_expanded)
        target_joint_dist = torch.abs(open_threshold - cap_joint_positions)

        pos_threshold = 0.05
        rot_threshold = 0.20
        joint_threshold = 0.10
        near_pos = target_pos_dist < pos_threshold
        good_rot = target_rot_dist < rot_threshold
        near_joint = target_joint_dist < joint_threshold
        
        
        contact_satisfied = contact_satisfied & good_rot
        contact_gate = contact_satisfied.float()

        self.near_pos = near_pos
        sigma_pos = 0.10
        sigma_rot = 0.10
        sigma_joint = 1.0
        pos_weight = torch.exp(-target_pos_dist / sigma_pos)
        rot_weight = torch.exp(-target_rot_dist / sigma_rot)
        joint_weight = torch.exp(-target_joint_dist / sigma_joint)
        pose_weight = rot_weight * joint_weight
        self.pose_weight = pose_weight
        progress = torch.clamp(cap_joint_positions / open_threshold, 0.0, 1.0)
        # progress = pose_weight

        if not hasattr(self, "cap_progress_max"):
            self.cap_progress_max = torch.zeros((N,), device=device, dtype=torch.float)
        if not hasattr(self, "cap_progress_max_height"):
            self.cap_progress_max_height = torch.zeros((N,), device=device, dtype=torch.float)

        prev_max = self.cap_progress_max
        delta = torch.clamp(progress - prev_max, min=0.0)

        is_open = cap_joint_positions > open_threshold
        # near_goal = near_joint & good_rot & contact_satisfied
        near_goal = near_joint & contact_satisfied
        self.near_goal = near_goal

        self.success_step_counter = torch.where(
            near_goal,
            self.success_step_counter + 1,
            torch.zeros_like(self.success_step_counter),
        )


        delta_scale = 100.0
        # delta_reward = delta * (~is_success).float() * contact_gate * pose_weight
        delta_reward = delta * (~is_success).float() * contact_gate
        self.rew_buf += delta_reward * delta_scale
        # self.rew_buf += self.pose_weight * delta_scale * (~is_success).float() * contact_gate

        per_step_bonus = self.success_bonus / float(self.success_steps_required)
        maintain_bonus = near_goal.float() * per_step_bonus * contact_gate
        self.rew_buf += maintain_bonus

        self.cap_progress_max = torch.where(
            contact_satisfied,
            torch.maximum(prev_max, progress),
            prev_max,
        )

        is_success_raw = self.success_step_counter >= self.success_steps_required
        is_success = is_success_raw & contact_satisfied


        # success_reward = is_success * 4000 * pose_weight
        success_reward = is_success * 4000
        self.rew_buf += success_reward

        if self.terminate_on_success:
            self.reset_buf |= torch.where(is_success, torch.ones_like(self.reset_buf), self.reset_buf)

        self.reset_buf |= torch.where(self.progress_buf >= self.max_episode_length - 1, 1, self.reset_buf)

        self.extras["progress"] = progress.clone()
        self.extras["contact_satisfied"] = contact_satisfied.clone()
        self.extras["cap_progress_max"] = self.cap_progress_max.clone()
        self.extras["target_pos_dist"] = target_pos_dist.clone()
        self.extras["target_rot_dist"] = target_rot_dist.clone()
        self.extras["near_pos"] = near_pos.clone()
        self.extras["good_rot"] = good_rot.clone()
        self.extras["pose_weight"] = self.pose_weight.clone()
        self.extras["delta_reward"] = delta_reward.clone()
        self.extras["delta"] = delta.clone()
        self.extras["near_goal"] = self.near_goal.clone()
        self.extras["success_step_counter"] = self.success_step_counter.clone()
        self.extras["maintain_bonus"] = maintain_bonus.clone()
        self.extras["is_success_raw"] = is_success_raw.clone()
        self.extras["is_success"] = is_success.clone()

        if "ccge" in self.reward_type and self.training:
            # self._compute_curiosity_rewards(contact_satisfied)
            self._compute_curiosity_rewards_v2(contact_satisfied)
        self.compute_done(is_success)


    def compute_reach_reward_keypoints(self):
        device = self.device
        N = self.num_envs

        top_pc = self._object_parts_world_pointclouds.get("top", None)      # for right hand
        bottom_pc = self._object_parts_world_pointclouds.get("bottom", None)  # for left hand

        right_rew = torch.zeros((N,), device=device)
        left_rew = torch.zeros((N,), device=device)

        if top_pc is not None:
            right_kp_pos = self.right_keypoint_positions_with_offset  # (N, K_r, 3)
            dists_r = torch.cdist(right_kp_pos, top_pc)               # (N, K_r, P_top)
            cur_min_r, _ = torch.min(dists_r, dim=2)                  # (N, K_r)

            if (not hasattr(self, "right_keypoints_to_surface_dist_min")
                or self.right_keypoints_to_surface_dist_min.shape != cur_min_r.shape):
                self.right_keypoints_to_surface_dist_min = torch.full_like(cur_min_r, 0.50)

            delta_r = (self.right_keypoints_to_surface_dist_min - cur_min_r).clamp_min(0.0)
            self.right_keypoints_to_surface_dist_min = torch.min(self.right_keypoints_to_surface_dist_min, cur_min_r)

            right_rew = delta_r.mean(dim=1)  # (N,)

            # logging
            self.extras["right_keypoint_surface_distances"] = cur_min_r.clone()
            self.extras["right_keypoints_to_surface_dist_min"] = self.right_keypoints_to_surface_dist_min.clone()


        if bottom_pc is not None:
            left_kp_pos = self.left_keypoint_positions_with_offset   # (N, K_l, 3)
            dists_l = torch.cdist(left_kp_pos, bottom_pc)            # (N, K_l, P_bot)
            cur_min_l, _ = torch.min(dists_l, dim=2)                 # (N, K_l)

            if (not hasattr(self, "left_keypoints_to_surface_dist_min")
                or self.left_keypoints_to_surface_dist_min.shape != cur_min_l.shape):
                self.left_keypoints_to_surface_dist_min = torch.full_like(cur_min_l, 0.50)

            delta_l = (self.left_keypoints_to_surface_dist_min - cur_min_l).clamp_min(0.0)
            self.left_keypoints_to_surface_dist_min = torch.min(self.left_keypoints_to_surface_dist_min, cur_min_l)

            left_rew = delta_l.mean(dim=1)  # (N,)

            # logging
            self.extras["left_keypoint_surface_distances"] = cur_min_l.clone()
            self.extras["left_keypoints_to_surface_dist_min"] = self.left_keypoints_to_surface_dist_min.clone()

        reach_rew = 0.5 * (left_rew + right_rew)
        self.reach_rew_keypoints = reach_rew
        self.reach_rew_scaled_keypoints = self.reach_rew_keypoints * 5.0

        self.extras["reach_rew_keypoints"] = self.reach_rew_scaled_keypoints.clone()


        
    def _compute_curiosity_rewards(self, contact_satisfied: Tensor):
        """Compute curiosity rewards for each hand and corresponding object part"""
        # Get hand keypoint positions and forces
        fingertip_keypoints = {
            "right": {
                "positions": self.right_fingertip_positions,
                "orientations": self.right_fingertip_orientations,
                "forces": self.right_fingertip_contact_forces
            },
            "left": {
                "positions": self.left_fingertip_positions,
                "orientations": self.left_fingertip_orientations,
                "forces": self.left_fingertip_contact_forces
            }
        }
        
        # For each hand, compute curiosity reward
        for hand_side in ["right", "left"]:
            reach_curiosity_rew_scaled = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
            contact_coverage_rew_scaled = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
            for target_part, occ_name in [("top", "bottom"), ("bottom", "top")]:
                
                if target_part not in self._object_parts_world_pointclouds:
                    assert False
                    
                # Get part's rigid body state
                part_rb_idx = self._object_parts[target_part]["rigid_body_indices"][0]  # Use first rigid body
                part_position = self.object_rigid_body_positions[:, part_rb_idx]
                part_orientation = self.object_rigid_body_orientations[:, part_rb_idx]
                
                occ_rb_idx = self._object_parts[occ_name]["rigid_body_indices"][0]
                occ_position = self.object_rigid_body_positions[:, occ_rb_idx]
                occ_orientation = self.object_rigid_body_orientations[:, occ_rb_idx]
                
                # Get current hand fingertip positions
                keypoints_pos = fingertip_keypoints[hand_side]["positions"]
                keypoints_ori = fingertip_keypoints[hand_side]["orientations"]
                keypoints_forces = fingertip_keypoints[hand_side]["forces"]
                
                # Get target part point cloud
                part_pointcloud = self._object_parts_world_pointclouds[target_part]
                # state feature from world pointcloud (N,12)
                state_features_world = torch.cat(
                    [
                        part_pointcloud.mean(dim=1),
                        part_pointcloud.std(dim=1),
                        part_pointcloud.amin(dim=1),
                        part_pointcloud.amax(dim=1),
                    ],
                    dim=-1,
                )
                
                # Compute nearest point indices
                dists = torch.cdist(keypoints_pos, part_pointcloud)
                contact_indices = dists.argmin(dim=-1)
                min_dists = dists.min(dim=-1)[0]
                
                # Create contact mask (near surface and has force)
                near_surface = (min_dists < 0.012)
                has_force = (keypoints_forces.norm(dim=-1, p=2) > 0.01) & (keypoints_forces.norm(dim=-1, p=2) < 50)
                contact_mask = near_surface & has_force
                
                # Compute direction vectors (local x-axis in world frame)
                axis_local = torch.tensor([-1.0, 0.0, 0.0], device=self.device, dtype=keypoints_pos.dtype).view(1, 1, 3)
                axis_local = axis_local.expand_as(keypoints_pos)
                dir_world = quat_apply(keypoints_ori.reshape(-1, 4), axis_local.reshape(-1, 3)).view_as(keypoints_pos)
                
                # Compute curiosity reward
                _, info = self._curiosity_managers[hand_side][target_part].compute_reward_from_canonical(
                    object_positions=part_position,  # Use specific part position
                    object_orientations=part_orientation,  # Use specific part orientation
                    keypoint_positions_world=keypoints_pos,
                    contact_indices=contact_indices,
                    contact_mask=contact_mask,
                    # task_contact_satisfied=torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
                    task_contact_satisfied=contact_satisfied,
                    contact_forces_local=dir_world,
                    keypoint_palm_dirs_world=dir_world,
                    object_occlusion_positions=occ_position,
                    object_occlusion_orientations=occ_orientation,
                    state_features_world=state_features_world,
                )
                
                # Extract rewards from info
                reach_curiosity_rew = info["potential_field_reward"]
                contact_coverage_rew = info["cluster_novelty_reward"]
                
                # Scale rewards
                reach_curiosity_rew_scaled += reach_curiosity_rew * 5.12
                contact_coverage_rew_scaled += contact_coverage_rew * 800
                
            # Log rewards
            self.extras[f"{hand_side}_reach_curiosity_rew"] = reach_curiosity_rew_scaled
            self.extras[f"{hand_side}_contact_coverage_rew"] = contact_coverage_rew_scaled
            
            # Sum rewards across fingertips to get a single scalar per environment
            total_reward = reach_curiosity_rew_scaled + contact_coverage_rew_scaled
            
            # Scale the reward
            scaled_reward = total_reward * 0.5
            
            # Add to total reward
            self.rew_buf += scaled_reward
            
            # Log rewards
            self.extras[f"{hand_side}_curiosity_reward"] = scaled_reward
            
            
    def _compute_curiosity_rewards_v2(self, contact_satisfied: Tensor):

        state_type = str(self.cfg["env"].get("stateType", "pcd")).lower()

        for hand_side in ["right", "left"]:
            reach_curiosity_rew_scaled = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
            contact_coverage_rew_scaled = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)


            state_features_world_hash = None
            if state_type == "hash":
                if ("top" in self._object_parts_world_pointclouds) and ("bottom" in self._object_parts_world_pointclouds):
                    mgr_top = self._curiosity_managers[hand_side]["top"]
                    mgr_bot = self._curiosity_managers[hand_side]["bottom"]

                    idx_top_full = getattr(mgr_top, "_state_point_indices", None)
                    idx_bot_full = getattr(mgr_bot, "_state_point_indices", None)
                    if (idx_top_full is not None) and (idx_bot_full is not None):
                        # keep total dim unchanged: 3*P, split P points across parts
                        P = int(idx_top_full.numel())
                        p_top = P // 2
                        p_bot = P - p_top

                        top_pc = self._object_parts_world_pointclouds["top"]       # (N,Pt,3)
                        bot_pc = self._object_parts_world_pointclouds["bottom"]    # (N,Pb,3)

                        feat_top = top_pc.index_select(1, idx_top_full[:p_top]).reshape(self.num_envs, -1)
                        feat_bot = bot_pc.index_select(1, idx_bot_full[:p_bot]).reshape(self.num_envs, -1)
                        state_features_world_hash = torch.cat([feat_top, feat_bot], dim=-1).to(torch.float32)
        
            for target_part in ["top", "bottom"]:
                if target_part not in self._object_parts_world_pointclouds:
                    assert False

                occ_name = "top" if target_part == "bottom" else "bottom"
                occ_rb_idx = self._object_parts[occ_name]["rigid_body_indices"][0]
                occ_position = self.object_rigid_body_positions[:, occ_rb_idx]
                occ_orientation = self.object_rigid_body_orientations[:, occ_rb_idx]
                
                # object part pose & pc
                part_rb_idx = self._object_parts[target_part]["rigid_body_indices"][0]
                part_position = self.object_rigid_body_positions[:, part_rb_idx]        # (N,3)
                part_orientation = self.object_rigid_body_orientations[:, part_rb_idx]  # (N,4)
                part_pointcloud = self._object_parts_world_pointclouds[target_part]     # (N,P,3)

                if hand_side == "right":
                    finger_pos_tensors = [
                        self.right_index_finger_keypoint_positions_with_offset,
                        self.right_middle_finger_keypoint_positions_with_offset,
                        self.right_ring_finger_keypoint_positions_with_offset,
                        self.right_thumb_finger_keypoint_positions_with_offset,
                    ]
                    finger_ori_tensors = [
                        self.right_index_finger_keypoint_orientations,
                        self.right_middle_finger_keypoint_orientations,
                        self.right_ring_finger_keypoint_orientations,
                        self.right_thumb_finger_keypoint_orientations,
                    ]
                    finger_force_tensors = [
                        self.right_index_finger_keypoint_forces,
                        self.right_middle_finger_keypoint_forces,
                        self.right_ring_finger_keypoint_forces,
                        self.right_thumb_finger_keypoint_forces,
                    ]
                    finger_names_all = ["index", "middle", "ring", "thumb"]
                else:
                    finger_pos_tensors = [
                        self.left_index_finger_keypoint_positions_with_offset,
                        self.left_middle_finger_keypoint_positions_with_offset,
                        self.left_ring_finger_keypoint_positions_with_offset,
                        self.left_thumb_finger_keypoint_positions_with_offset,
                    ]
                    finger_ori_tensors = [
                        self.left_index_finger_keypoint_orientations,
                        self.left_middle_finger_keypoint_orientations,
                        self.left_ring_finger_keypoint_orientations,
                        self.left_thumb_finger_keypoint_orientations,
                    ]
                    finger_force_tensors = [
                        self.left_index_finger_keypoint_forces,
                        self.left_middle_finger_keypoint_forces,
                        self.left_ring_finger_keypoint_forces,
                        self.left_thumb_finger_keypoint_forces,
                    ]
                    finger_names_all = ["index", "middle", "ring", "thumb"]

                finger_positions = []      # (N,3)
                finger_orientations = []   # (N,4)
                finger_forces = []         # (N,3)
                finger_contact_indices = []  # (N,)
                finger_contact_masks = []    # (N,)

                for fname, kp_pos_f, kp_ori_f, kp_force_f in zip(
                    finger_names_all, finger_pos_tensors, finger_ori_tensors, finger_force_tensors
                ):
                    if kp_pos_f is None or kp_ori_f is None or kp_force_f is None:
                        assert False
                    # kp_pos_f: (N,Kf,3), kp_ori_f: (N,Kf,4), kp_force_f: (N,Kf,3)
                    dists = torch.cdist(kp_pos_f, part_pointcloud)    # (N,Kf,P)
                    N_env, Kf, P = dists.shape
                    dists_flat = dists.view(N_env, -1)               # (N,Kf*P)
                    min_flat_idx = dists_flat.argmin(dim=-1)         # (N,)
                    min_dists = dists_flat.gather(
                        1, min_flat_idx.unsqueeze(-1)
                    ).squeeze(-1)                                    # (N,)

                    best_point_idx = (min_flat_idx % P)              # (N,)
                    best_k_idx = (min_flat_idx // P)                 # (N,)

                    # keypoint position & orientation
                    gather_idx_pos = best_k_idx.view(N_env, 1, 1).expand(-1, 1, 3)
                    gather_idx_ori = best_k_idx.view(N_env, 1, 1).expand(-1, 1, 4)
                    kp_pos_rep = torch.gather(kp_pos_f, 1, gather_idx_pos).squeeze(1)  # (N,3)
                    kp_ori_rep = torch.gather(kp_ori_f, 1, gather_idx_ori).squeeze(1)  # (N,4)

                    # rough
                    finger_force_world = kp_force_f.sum(dim=1)        # (N,3)
                    force_mag_links = kp_force_f.norm(dim=-1, p=2)    # (N,Kf)

                    near_surface = (min_dists < 0.012)                # (N,)
                    has_force = ((force_mag_links > 0.01) & (force_mag_links < 50.0)).any(dim=-1)  # (N,)
                    contact_mask_f = near_surface & has_force         # (N,)

                    finger_positions.append(kp_pos_rep)
                    finger_orientations.append(kp_ori_rep)
                    finger_forces.append(finger_force_world)
                    finger_contact_indices.append(best_point_idx)
                    finger_contact_masks.append(contact_mask_f)

                if not finger_positions:
                    continue

                keypoints_pos = torch.stack(finger_positions, dim=1)           # (N,F,3)
                keypoints_ori = torch.stack(finger_orientations, dim=1)        # (N,F,4)
                keypoints_forces = torch.stack(finger_forces, dim=1)           # (N,F,3)
                contact_indices = torch.stack(finger_contact_indices, dim=1)   # (N,F)
                contact_mask = torch.stack(finger_contact_masks, dim=1)        # (N,F)

                mgr = self._curiosity_managers[hand_side][target_part]

                if state_type == "hash":
                    state_features_world = state_features_world_hash
                else:
                    idx = mgr._state_point_indices
                    state_features_world = part_pointcloud.index_select(1, idx).reshape(self.num_envs, -1).to(torch.float32)

                # local x 轴到 world，作为 contact_forces_local 的方向
                axis_local = torch.tensor(
                    [-1.0, 0.0, 0.0],
                    device=self.device,
                    dtype=keypoints_pos.dtype,
                ).view(1, 1, 3).expand_as(keypoints_pos)
                dir_world = quat_apply(
                    keypoints_ori.reshape(-1, 4),
                    axis_local.reshape(-1, 3),
                ).view_as(keypoints_pos)

                _, info = self._curiosity_managers[hand_side][target_part].compute_reward_from_canonical(
                    object_positions=part_position,
                    object_orientations=part_orientation,
                    keypoint_positions_world=keypoints_pos,
                    goal_positions=torch.zeros((self.num_envs, 3), device=self.device, dtype=keypoints_pos.dtype),
                    goal_orientations=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=keypoints_pos.dtype).unsqueeze(0).expand(self.num_envs, -1),
                    contact_indices=contact_indices,
                    contact_mask=contact_mask,
                    task_contact_satisfied=contact_satisfied,
                    contact_forces_local=dir_world,
                    keypoint_palm_dirs_world=dir_world,
                    object_occlusion_positions=occ_position,
                    object_occlusion_orientations=occ_orientation,
                    state_features_world=(state_features_world),
                )

                reach_curiosity_rew = info["potential_field_reward"]
                contact_coverage_rew = info["cluster_novelty_reward"]

                reach_curiosity_rew_scaled += reach_curiosity_rew * 1.28
                contact_coverage_rew_scaled += contact_coverage_rew * 200

            self.extras[f"{hand_side}_reach_curiosity_rew"] = reach_curiosity_rew_scaled
            self.extras[f"{hand_side}_contact_coverage_rew"] = contact_coverage_rew_scaled

            total_reward = reach_curiosity_rew_scaled + contact_coverage_rew_scaled
            scaled_reward = total_reward * 0.5

            self.rew_buf += scaled_reward
            self.extras[f"{hand_side}_curiosity_reward"] = scaled_reward
            
    def reset(self, dones=None, first_time=False):
        """Is called only once when environment starts to provide the first observations.

        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        if dones is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = dones.nonzero(as_tuple=False).flatten()

        # reset idx
        if env_ids.shape[0] > 0:
            self.reset_idx(env_ids, first_time=first_time)

        self.compute_observations(env_ids)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def compute_observation_space(self) -> OrderedDict:
        observations = OrderedDict()
        for spec in self._observation_space_extra:
            observations[spec.name] = getattr(self, spec.attr)
        return observations

    def export_observation_metainfo(self) -> List[Dict[str, Any]]:
        metainfo: List[Dict[str, Any]] = []
        current = 0
        for spec in self._observation_space:
            metainfo.append(
                {
                    "name": spec.name,
                    "dim": spec.dim,
                    "tags": spec.tags,
                    "start": current,
                    "end": current + spec.dim,
                }
            )
            current += spec.dim
        return metainfo

    def post_physics_step(self) -> None:
        self.progress_buf += 1
        self.randomize_buf += 1
        
        self.compute_observations()
        
        self.compute_reward(self.actions)


        if self.device.startswith("cuda"):
            gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info(device=self.device)
            gpu_mem_occupied = torch.tensor([gpu_mem_total - gpu_mem_free], device=self.device)
            self.extras["gpu_mem_occupied_MB"] = gpu_mem_occupied / 1024 / 1024
            self.extras["gpu_mem_occupied_GB"] = gpu_mem_occupied / 1024 / 1024 / 1024
            self.extras["gpu_mem_occupied_ratio"] = gpu_mem_occupied / gpu_mem_total

        # self.extras["max_jacobian_det"] = torch.max(torch.det(self.j_eef).abs()).reshape(1)
        
        
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            origin_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            origin_orientations = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
            origin_orientations[:, 3] = 1
            # draw_axes(self.gym, self.viewer, self.envs, origin_positions, origin_orientations, 0.5)
            # draw_axes(self.gym, self.viewer, self.envs, self.object_root_positions, self.object_root_orientations, 0.1)

            # if self.enable_rendered_pointcloud_observation:
            #     self.draw_camera_axes()

            # if self.enable_contact_sensors:
            #     self.draw_force_sensor_axes()
                
            # self.draw_link_keypoints()
            
            draw_points(self.gym, self.viewer, self.envs, self.left_fingertip_positions, radius=0.012, num_segments=10, color=(1.0, 0.0, 0.0))
            # draw_points(self.gym, self.viewer, self.envs, self.right_fingertip_positions, radius=0.012, num_segments=10, color=(1.0, 0.0, 0.0))
            # draw_points(self.gym, self.viewer, self.envs, self.left_keypoint_positions_with_offset, radius=0.01, num_segments=10, color=(0.0, 1.0, 0.0))
            # draw_points(self.gym, self.viewer, self.envs, self.right_keypoint_positions_with_offset, radius=0.01, num_segments=10, color=(0.0, 1.0, 0.0))
            # draw_points(self.gym, self.viewer, self.envs, self.left_keypoint_positions, radius=0.01, num_segments=10, color=(0.0, 0.0, 1.0))
            # draw_points(self.gym, self.viewer, self.envs, self.right_keypoint_positions, radius=0.01, num_segments=10, color=(0.0, 0.0, 1.0))
            
            if "top" in self._object_parts_world_pointclouds:
                top_pc = self._object_parts_world_pointclouds["top"]
                draw_points(self.gym, self.viewer, self.envs, top_pc,
                            radius=0.003, num_segments=2, color=(0.0, 1.0, 0.0))

            if "bottom" in self._object_parts_world_pointclouds:
                bottom_pc = self._object_parts_world_pointclouds["bottom"]
                draw_points(self.gym, self.viewer, self.envs, bottom_pc,
                            radius=0.003, num_segments=2, color=(0.0, 0.0, 1.0))

    # -------------------------------------------------------------------------
    # PID helpers
    # -------------------------------------------------------------------------
    def _allocate_pid_buffers(self):
        kwargs = {"device": self.device, "dtype": torch.float}
        zeros = torch.zeros((self.num_envs, 3), **kwargs)
        self.dt = self.sim_cfg["dt"] * self.sim_cfg.get("substeps", 1)
        self.pid_state: Dict[str, Dict[str, Tensor]] = {
            "right": {
                "pos_prev": zeros.clone(),
                "rot_prev": zeros.clone(),
                "pos_integral": zeros.clone(),
                "rot_integral": zeros.clone(),
                "force_target": zeros.clone(),
                "torque_target": zeros.clone(),
            },
            "left": {
                "pos_prev": zeros.clone(),
                "rot_prev": zeros.clone(),
                "pos_integral": zeros.clone(),
                "rot_integral": zeros.clone(),
                "force_target": zeros.clone(),
                "torque_target": zeros.clone(),
            },
        }

    def _reset_pid_buffers(self, env_ids: Tensor) -> None:
        for side in ("right", "left"):
            for key in ("pos_prev", "rot_prev", "pos_integral", "rot_integral", "force_target", "torque_target"):
                self.pid_state[side][key][env_ids] = 0.0
