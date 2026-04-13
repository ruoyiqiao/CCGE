import enum
import math
import os
import random
import warnings
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import cv2
import numpy as np
import omegaconf
import open3d as o3d
import pandas as pd
import pytorch3d
import torch
import trimesh
import json
from dotenv import find_dotenv
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import matrix_to_quaternion
from torch import LongTensor, Tensor

from .dataset import OakInkDataset, point_to_mesh_distance
from .isaacgym_utils import (
    ActionSpec,
    ObservationSpec,
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
from .curiosity_reward_manager import CuriosityRewardManager
# for debug
test_ik = False
test_sim = False
test_rel = False
test_pcl = False
fix_wrist = False
wrist_zero_action = False
test = False
local_test = False

success_tolerance = 0.1
height_success_tolerance = 0.1
negative_part_reward = False
trans_scale = 10
hand_pcl_num = 1024
batch_size = 1000
high_thumb_reward = False
set_arm_pose_according_to_object = False

video_pose = [0.0, -0.3, -0.3]

add_noise = False
STATIC_TARGET = False


class XArmAllegroHandUnderarmDimensions(enum.Enum):
    """Dimension constants for Isaac Gym with xArm6 + Allegro Hand."""

    # general state
    # cartesian position (3) + quaternion orientation (4)
    POSE_DIM = 7
    # linear velocity (3) + angular velocity (3)
    VELOCITY_DIM = 6
    # pose (7) + velocity (6)
    STATE_DIM = 13
    # force (3) + torque (3)
    WRENCH_DIM = 6

    NUM_FINGERTIPS = 4  # Allegro hand has 4 fingertips
    NUM_DOFS = 22  # xArm6 (6 DOF) + Allegro hand (16 DOF)

    WRIST_TRAN = 3
    WRIST_ROT = 3

    # Allegro hand actuated dimensions
    HAND_ACTUATED_DIM = 16
    NUM_KEYPOINTS = 21


class ForceSensorSpec:
    name: str
    index: int
    rigid_body_name: str
    rigid_body_index: int
    pose: gymapi.Transform

    def __init__(
        self,
        name: str,
        rigid_body_name: str,
        translation: Optional[Sequence[float]] = None,
        rotation: Optional[Sequence[float]] = None,
        *,
        index: int = -1,
        rigid_body_index: int = -1,
        pose: Optional[gymapi.Transform] = None,
    ) -> None:
        assert not (((translation is not None) or (rotation is not None)) and (pose is not None))
        if pose is not None:
            pass
        elif (translation is not None) or (rotation is not None):
            pose = gymapi.Transform()
            if translation is not None:
                assert len(translation) == 3
                pose.p = gymapi.Vec3(*translation)
            if rotation is not None:
                assert len(rotation) == 4
                pose.r = gymapi.Quat(*rotation)
        else:
            pose = gymapi.Transform()

        self.name = name
        self.index = index
        self.rigid_body_name = rigid_body_name
        self.rigid_body_index = rigid_body_index
        self.pose = pose
        self.translation, self.rotation = position(pose), orientation(pose)


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


class XArmAllegroHandFunctionalManipulationUnderarm(VecTask):
    # constants
    _grasp_task: bool = True
    _asset_root: os.PathLike = os.path.join(os.path.dirname(find_dotenv()), "assets/urdf")
    _data_root: os.PathLike = os.path.join(os.path.dirname(find_dotenv()), "data")
    _allegro_hand_right_asset_file: os.PathLike = os.path.join("hands", "allegro_hand", "allegro_hand_right.urdf")
    _allegro_hand_left_asset_file: os.PathLike = os.path.join("hands", "allegro_hand", "allegro_hand_left.urdf")
    _xarm_allegro_hand_right_asset_file: str = "xarm6_allegro_right.urdf"
    _xarm_allegro_hand_left_asset_file: str = "xarm6_allegro_left.urdf"

    # fmt: off
    _xarm_dof_names: List[str] = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]

    # Allegro hand DOF names (16 DOF total)
    _allegro_hand_dof_names: List[str] = [
        "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",  # finger 0 (index)
        "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",  # finger 1 (middle)
        "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",  # finger 2 (ring)
        "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",  # thumb
    ]

    # Group allegro hand DOF names by finger
    _allegro_finger0_dof_names: List[str] = ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"]
    _allegro_finger1_dof_names: List[str] = ["joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0"]
    _allegro_finger2_dof_names: List[str] = ["joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"]
    _allegro_thumb_dof_names: List[str] = ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"]

    _allegro_fingers_dof_names: List[str] = (
        _allegro_finger0_dof_names + _allegro_finger1_dof_names + _allegro_finger2_dof_names
    )
    _allegro_digits_dof_names: List[str] = _allegro_fingers_dof_names + _allegro_thumb_dof_names
    # fmt: on

    _arm_links: List[str] = ["link_base", "link1", "link2", "link3", "link4", "link5", "link6"]
    _hand_links: List[str] = [
        "base_link", "palm", "wrist",
        "link_0.0", "link_1.0", "link_2.0", "link_3.0",
        "link_4.0", "link_5.0", "link_6.0", "link_7.0",
        "link_8.0", "link_9.0", "link_10.0", "link_11.0",
        "link_12.0", "link_13.0", "link_14.0", "link_15.0",
    ]
    _xarm_eef_link: str = "link6"
    _fingertips: List[str] = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"] # index, middle, ring, thumb
    _index_finger_links: List[str] = ["link_1.0", "link_2.0", "link_3.0", "link_3.0_tip"]
    _middle_finger_links: List[str] = ["link_5.0", "link_6.0", "link_7.0", "link_7.0_tip"]
    _ring_finger_links: List[str] = ["link_9.0", "link_10.0", "link_11.0", "link_11.0_tip"]
    _thumb_links: List[str] = ["link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"]
    _allegro_hand_center_prim: str = "base_link"
    _allegro_hand_palm_prim: str = "palm"
    # fmt: off
    _keypoints: List[str] = [
        "base_link",
        "link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip",  # thumb
        "link_0.0", "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip",     # finger 0 (index)
        "link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip",     # finger 1 (middle)
        "link_8.0", "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip",   # finger 2 (ring)
    ]
    _keypoints_info_path: str = "assets/urdf/xarm6_allegro_right_keypoints.json" # XXX: one keypointper link for now.
    _keypoints_info: Dict[str, List[List[float]]] = json.load(open(_keypoints_info_path, "r"))
    # fmt: on

    _xarm_right_init_dof_positions: Dict[str, float] = {
        "joint1": 0.0,
        "joint2": -1.0,
        "joint3": -0.5,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
    }
    _xarm_left_init_dof_positions: Dict[str, float] = {
        "joint1": 0.0,
        "joint2": -1.0,
        "joint3": -0.5,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
    }
    allegro_hand_init_dof_positions: Dict[str, float] = {
        "joint_0.0": -0.14,
        "joint_1.0": 1.0,
        "joint_2.0": 1.8,
        "joint_3.0": 0.9,
        "joint_4.0": 0.0,
        "joint_5.0": 0.0,
        "joint_6.0": 0.15,
        "joint_7.0": 0.29,
        "joint_8.0": 0.38,
        "joint_9.0": 1.3,
        "joint_10.0": 1.4,
        "joint_11.0": 1.0,
        "joint_12.0": 0.46,
        "joint_13.0": 0.08,
        "joint_14.0": 1.0,
        "joint_15.0": 0.53,
    }
    allegro_hand_init_dof_positions: Dict[str, float] = {
        "joint_0.0": 0,
        "joint_1.0": 0.72,
        "joint_2.0": 0.78,
        "joint_3.0": 0.70,
        "joint_4.0": 0.0,
        "joint_5.0": 0.72,
        "joint_6.0": 0.78,
        "joint_7.0": 0.68,
        "joint_8.0": 0.,
        "joint_9.0": 0.70,
        "joint_10.0": 0.75,
        "joint_11.0": 0.66,
        "joint_12.0": 0.85,
        "joint_13.0": 0.54,
        "joint_14.0": 0.72,
        "joint_15.0": 0.77,
    }

    _xarm_right_init_position = [0.00, 0.55, 0.00]
    _xarm_right_init_orientation = [0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)]
    _allegro_hand_predef_qpos = [0] * 16  # reset if use predef qpos
    _target_hand_palm_pose = [-0.4, 0.053, 0.810, 0.0, -0.707, 0.707, 0.0]
    _current_hand_palm_pose = [0.021, 0.052, 0.608, 0.0, -0.707, 0.707, 0.0]
    _hand_geo_center = [0.0, 0.0, 0.0]
    _object_z = 0.5
    _object_nominal_orientation = [0.0, 0.0, 1.0, 0.0]
    _table_x_length = 0.36 - 0.016
    _table_y_length = 0.26
    _table_thickness = 0.02
    _table_pose = [0.0, 0.0, 0.50]
    _upper_shelf_pos = [0.0, 0.0, 1.05]
    # _table_thickness = 0.41
    # _table_pose = [0.0, 0.0, 0.205]

    _max_xarm_endeffector_pos_vel = 0.4
    _max_xarm_endeffector_rot_vel = torch.pi / 2

    # HACK: hardcoded palm->forearm transform
    _palm2forearm_quat = [0.0, 0.0, 0.0, 1.0]
    _palm2forearm_pos = [0.0, -0.01, 0.247]
    
    _hand_base_link2forearm_quat = [0.0, 0.0, -1.0, 0.0]
    _hand_base_link2forearm_pos = [0.0, 0.0, -0.095]
    
    _wrist2forearm_quat = [0.0, 0.0, -1.0, 0.0]
    _wrist2forearm_pos = [0.0, -0.01, 0.247]

    _dims = XArmAllegroHandUnderarmDimensions
    _observation_specs: Sequence[ObservationSpec] = []
    _action_specs: Sequence[ActionSpec] = []
    _force_sensor_specs: Sequence[ForceSensorSpec] = [
        ForceSensorSpec("link_3.0_tip", "link_3.0_tip"),
        ForceSensorSpec("link_7.0_tip", "link_7.0_tip"),
        ForceSensorSpec("link_11.0_tip", "link_11.0_tip"),
        ForceSensorSpec("link_15.0_tip", "link_15.0_tip"),
    ]

    # TODO: add description about tensor shapes
    allegro_hand_index: int

    allegro_hand_dof_lower_limits: Tensor
    allegro_hand_dof_upper_limits: Tensor
    allegro_hand_dof_init_positions: Tensor
    allegro_hand_dof_init_velocities: Tensor

    allegro_hand_dof_start: int
    allegro_hand_dof_end: int
    target_allegro_hand_dof_start: int
    target_allegro_hand_dof_end: int

    # buffers to hold intermediate results
    root_states: Tensor
    root_positions: Tensor
    root_orientations: Tensor
    root_linear_velocities: Tensor
    root_angular_velocities: Tensor

    allegro_hand_root_states: Tensor
    allegro_hand_root_positions: Tensor
    allegro_hand_root_orientations: Tensor
    allegro_hand_root_linear_velocities: Tensor
    allegro_hand_root_angular_velocities: Tensor

    scene_object_root_states: Tensor
    scene_object_root_positions: Tensor
    scene_object_root_orientations: Tensor
    scene_object_root_linear_velocities: Tensor
    scene_object_root_angular_velocities: Tensor

    allegro_hand_dof_positions: Tensor
    allegro_hand_dof_velocities: Tensor

    target_allegro_hand_dof_positions: Tensor
    target_allegro_hand_dof_velocities: Tensor

    # tensors need to be refreshed manually
    fingertip_states: Tensor
    fingertip_positions: Tensor
    fingertip_orientations: Tensor
    fingertip_positions_wrt_palm: Tensor
    fingertip_orientations_wrt_palm: Tensor
    fingertip_linear_velocities: Tensor
    fingertip_angular_velocities: Tensor

    object_root_states: Tensor
    object_root_positions: Tensor
    object_root_orientations: Tensor
    object_positions_wrt_palm: Tensor
    object_orientations_wrt_palm: Tensor

    prev_targets: Tensor
    curr_targets: Tensor
    prev_allegro_dof_speeds: Tensor

    successes: Tensor
    consecutive_successes: Tensor

    object_spacing: float
    num_objects_per_env: int

    _obj_width: float = 0.04
    _obj_depth: float = 0.16
    _obj_height: float = 0.24
    _grid_rows: int = 1
    _grid_cols: int = 5
    _grid_layers: int = 1
    _obj_spacing: float = 0.005
    
    # _obj_width: float = 0.045
    # _obj_depth: float = 0.11
    # _obj_height: float = 0.16
    # _grid_rows: int = 1
    # _grid_cols: int = 7
    # _grid_layers: int = 1
    # _obj_spacing: float = 0.002


    # _obj_width: float = 0.06
    # _obj_depth: float = 0.20
    # _obj_height: float = 0.30
    # _grid_rows: int = 1
    # _grid_cols: int = 5
    # _grid_layers: int = 1
    # _obj_spacing: float = 0.002

    VISUAL_TARGET_COLLISION_FILTER = 0x7FFFFFFF
    

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        seed = cfg["env"]["seed"]
        torch.manual_seed(seed)  # cpu
        random.seed(seed)
        np.random.seed(seed)

        self.cfg = cfg

        self.method = self.cfg["env"]["method"]

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.sub_steps = self.cfg["sim"]["substeps"]
        self.dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.use_absolute_joint_control = self.cfg["env"]["useAbsoluteJointControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.enable_contact_sensors = self.cfg["env"]["enableContactSensors"] or self.cfg["env"]["tactileObs"]
        self.contact_sensor_fingertip_only = self.cfg["env"]["contactSensorFingertipOnly"]
        self.contact_sensor_fingertip_from_all = self.cfg["env"]["contactSensorFingertipFromAll"]
        self.contact_sensor_threshold = self.cfg["env"]["contactSensorThreshold"]

        # Section for functional grasping dataset
        # self.dataset_dir = self.cfg["env"]["datasetDir"]
        # self.dataset_metainfo_path = self.cfg["env"]["datasetMetainfoPath"]
        # self.dataset_skipcode_path = self.cfg["env"]["datasetSkipcodePath"]
        # self.dataset_pose_level_sampling = self.cfg["env"]["datasetPoseLevelSampling"]
        # self.dataset_queries = self.cfg["env"]["datasetQueries"]

        self.object_spacing = self.cfg["env"]["objectSpacing"]
        self.num_objects = self.cfg["env"]["numObjects"]
        self.num_objects_per_env = self.cfg["env"]["numObjectsPerEnv"]

        self.reset_obj_ori_noise = self.cfg["env"]["resetObjOriNoise"]

        self.velocity_observation_scale = self.cfg["env"]["velocityObservationScale"]
        self.reward_type = self.cfg["env"]["rewardType"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.tran_reward_scale = self.cfg["env"]["tranRewardScale"]
        self.contact_reward_scale = self.cfg["env"]["contactRewardScale"]
        # if "curr" in self.reward_type:
        #     self.tran_reward_scale = 1.0

        self.action_noise = self.cfg["env"]["actionNoise"]
        self.action_noise_level = self.cfg["env"]["actionNoiseLevel"]
        self.action_noise_ratio = self.cfg["env"]["actionNoiseRatio"]
        self.action_noise_sigma = self.cfg["env"]["actionNoiseSigma"]
        self.action_noise_max_times = self.cfg["env"]["actionNoiseMaxTimes"]
        assert self.action_noise_level in ["step", "value"]

        self.relative_part_reward = self.cfg["env"]["relativePartReward"]
        self.part_reward_scale = self.cfg["env"]["partRewardScale"]
        self.height_reward_scale = self.cfg["env"]["heightRewardScale"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.contact_eps = self.cfg["env"]["contactEps"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        if fix_wrist or wrist_zero_action:
            self.wrist_action_penalty_scale = 0
        else:
            self.wrist_action_penalty_scale = self.cfg["env"]["wristActionPenaltyScale"]
        self.arm_action_penalty_scale = self.cfg["env"]["armActionPenaltyScale"]
        self.similarity_reward_scale = self.cfg["env"]["similarityRewardScale"]
        self.similarity_reward_freq = self.cfg["env"]["similarityRewardFreq"]

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.height_scale = self.cfg["env"]["heightScale"]
        self.time_step_penatly = self.cfg["env"]["timeStepPenatly"]
        self.manipulability_penalty_scale = self.cfg["env"]["manipulabilityPenaltyScale"]

        # Singulation-specific reward parameters
        self.tilt_reward_scale = self.cfg["env"].get("tiltRewardScale", 1.0)
        self.slide_reward_scale = self.cfg["env"].get("slideRewardScale", 1.0)
        

        # Random wrench perturbation parameters
        self.force_scale = float(self.cfg["env"].get("forceScale", 0.0))
        self.torque_scale = float(self.cfg["env"].get("torqueScale", 0.0))
        self.wrench_prob = float(self.cfg["env"].get("wrenchProb", 0.0))

        # Goal pose parameters for singulation task
        self.goal_translation_y = self.cfg["env"]["goalTranslationY"]
        self.goal_rotation_x = self.cfg["env"]["goalRotationX"]
        self.goal_tolerance_position = self.cfg["env"]["goalTolerancePosition"]
        self.goal_tolerance_rotation = self.cfg["env"]["goalToleranceRotation"]
        self.success_steps = self.cfg["env"]["successSteps"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.env_info_logging = self.cfg["logging"]["envInfo"]


        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        # Section for rendered point cloud observation
        self.real_pcl_obs = self.cfg["env"]["realPclObs"]
        self.enable_rendered_pointcloud_observation = (
            self.cfg["env"]["enableRenderedPointCloud"]
            or self.cfg["env"]["realPclObs"]
            or "rendered_pointcloud" in self.cfg["env"]["observationSpace"]
        )

        self.enable_rendered_pointcloud_target_mask = self.cfg["env"].get(
            "enableRenderedPointCloudTargetMask", False
        ) & self.enable_rendered_pointcloud_observation
        self.target_segmentation_id = int(self.cfg["env"].get("targetSegmentationId", 1))

        self.num_rendered_points = self.cfg["env"]["numRenderedPointCloudPoints"]
        self.rendered_pointcloud_multiplier = self.cfg["env"]["renderedPointCloudMultiplier"]
        self.rendered_pointcloud_sample_method = self.cfg["env"]["renderedPointCloudSampleMethod"]
        self.rendered_pointcloud_gaussian_noise = self.cfg["env"]["renderedPointCloudGaussianNoise"]
        self.rendered_pointcloud_gaussian_noise_sigma = self.cfg["env"]["renderedPointCloudGaussianNoiseSigma"]
        self.rendered_pointcloud_gaussian_noise_ratio = self.cfg["env"]["renderedPointCloudGaussianNoiseRatio"]
        assert self.rendered_pointcloud_sample_method in ["farthest", "random"]

        if self.enable_rendered_pointcloud_observation and not self.cfg["env"].get("enableCameraSensors", False):
            warnings.warn("enableRenderedPointCloud is set to True but enableCameraSensors is set to False.")
            warnings.warn("overriding enableCameraSensors to True.")
            self.cfg["env"]["enableCameraSensors"] = True

        self.vis_env_num = self.cfg["env"]["visEnvNum"]
        self.vis_image_size = self.cfg["env"]["visImageSize"]
        if self.vis_env_num > 0:
            self.cfg["env"]["enableCameraSensors"] = True
            self.save_video = True
        else:
            self.save_video = False

        self.img_pcl_obs = self.cfg["env"]["imgPclObs"]
        self.num_imagined_points = self.cfg["env"]["numImaginedPointCloudPoints"]
        self.enable_imagined_pointcloud_observation = (
            self.cfg["env"]["enableImaginedPointCloud"] or self.cfg["env"]["imgPclObs"]
        )

        self.num_object_points = self.cfg["env"]["numObjectPointCloudPoints"]

        self.num_nearest_non_targets = self.cfg['env']['observationSpecs']['__dim__']['num_nearest_non_targets']

        self.up_axis = "z"

        self.mode = self.cfg["env"]["mode"]
        self.curriculum_mode = self.cfg["env"]["curriculumMode"]

        self.render_target = self.cfg["env"].get("renderTarget", False)

        self.manipulated_object_codes = None
        self.resample_object = self.cfg["env"]["resampleObject"]
        
        # Use upper shelf
        self.use_upper_shelf = self.cfg["env"].get("useUpperShelf", True)
        self.use_back_wall = self.cfg["env"].get("useBackWall", True)
        self.use_side_walls = self.cfg["env"].get("useSideWalls", True)
        self.add_visual_target_object = self.cfg["env"].get("addVisualTargetObject", False)

        self.random_object_position = self.cfg["env"].get("randomObjectPosition", False)
        self.random_object_position_on_reset = self.cfg["env"].get("randomObjectPositionOnReset", False)
        self.reset_randomize_scene_xy = bool(self.cfg["env"].get("randomSceneXYOnReset", False))
        self.reset_randomize_scene_x = self.cfg["env"].get("randomSceneXOnReset", False)
        self.reset_randomize_scene_orientation = bool(self.cfg["env"].get("randomSceneOrientationOnReset", False))
        self.reset_randomize_orientation_mode = str(self.cfg["env"].get("randomSceneOrientationMode", "yaw"))  # "yaw" or "full"
        self.apply_random_removal = self.cfg["env"].get("applyRandomRemoval", False)
        self.object_near_edge = self.cfg["env"].get("objectNearEdge", False)

        self.dof_reset_noise_scale = self.cfg["env"].get("dofResetNoiseScale", 0.0) # 0.05

        self.hand_type = self.cfg["env"].get("handType", "leap")
        assert self.hand_type in ["leap", "allegro"]

        self._setup_domain_rand_cfg(self.cfg['domain_rand'])

        self.aggregate_tracker = AggregateTracker()

        self.object_targets = torch.zeros(self.cfg["env"]["numEnvs"], 3 + 4 + 18, device=sim_device)
        # use default xarm init dof positions
        # self._xarm_right_init_dof_positions = {
        #     "joint1": 0.0,
        #     "joint2":-1,
        #     "joint3":-0.5,
        #     "joint4": 0.0,
        #     "joint5": 0.0,
        #     "joint6": 0.0,
        # }

        self._hand_geo_center = [0, 0, 0]
        self._object_z = 0.01 + self._table_thickness / 2
        self._current_hand_palm_pose = [0.02, 0.3, 0.6, 0.707, 0.0, 0.0, 0.707]
        self.arm_control_type = "osc"

        if self.relative_part_reward:
            self.prev_pos_dist = torch.ones(self.cfg["env"]["numEnvs"], device=sim_device) * -1
            self.prev_rot_dist = torch.ones(self.cfg["env"]["numEnvs"], device=sim_device) * -1
            self.prev_contact_dist = torch.ones(self.cfg["env"]["numEnvs"], device=sim_device) * -1
            self.prev_nominal_dist = torch.ones(self.cfg["env"]["numEnvs"], device=sim_device) * -1

        self.curriculum_thres = 0.9
        if "stage" in self.curriculum_mode:
            self.height_scale = 0
            if self.relative_part_reward:
                self.part_reward_scale = 1.0
            else:
                self.part_reward_scale = 0.3
            self.nominal_env_ratio = 0.2
        elif "pose" in self.curriculum_mode:
            self.height_scale = 0
            if self.relative_part_reward:
                self.part_reward_scale = 1.0
            else:
                self.part_reward_scale = 0.3
            self.nominal_env_ratio = 1.0
        else:
            self.nominal_env_ratio = 0.2

        self.stack_frame_number = self.cfg["env"]["stackFrameNumber"]
        self.frames = deque([], maxlen=self.stack_frame_number)
        self.goal_position = self.cfg["env"].get("goalPosition", [0.4, 0.4, 0.7])
        self.goal_position = torch.tensor(self.goal_position, device=sim_device, dtype=torch.float)
        self.goal_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=sim_device, dtype=torch.float)
        # non-target object
        self.max_non_targets = self.num_objects_per_env - 1  # Maximum possible non-target objects per env
        self.k_nearest = min(self.num_nearest_non_targets, self.max_non_targets)

        # TODO: define structure to hold all the indices
        # mapping from name to asset instance
        self.gym_assets = {}
        self.gym_assets["current"] = {}
        self.gym_assets["target"] = {}

        self.num_fingertips = len(self._fingertips)

        # self.__create_functional_grasping_dataset(device=sim_device)
        self._create_box_grid_dataset(device=sim_device)
        self._configure_mdp_spaces()

        super().__init__(  # create_sim
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        # reconfig viewer
        self._configure_viewer()
        # HACK: not used
        # self.__reset_grasping_joint_indices()
        self._reset_action_indices()

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
        _jacobians: torch.Tensor = self.gym.acquire_jacobian_tensor(self.sim, "allegro_hand")

        if self.env_info_logging:
            print("root_states.shape: ", _root_states.shape)
            print("dof_states.shape: ", _dof_states.shape)
            print("rigid_body_states.shape: ", _rigid_body_states.shape)
            print("net_contact_forces.shape: ", _net_contact_forces.shape)
            print("force_sensor_states.shape: ", _force_sensor_states.shape)
            print("dof_forces.shape: ", _dof_forces.shape)
            print("jacobians.shape: ", _jacobians.shape)

        self.num_actors: int = self.gym.get_sim_actor_count(self.sim) // self.num_envs
        self.num_dofs: int = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.num_force_sensors: int = self.gym.get_sim_force_sensor_count(self.sim) // self.num_envs
        self.num_rigid_bodies: int = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs

        if self.env_info_logging:
            print("num_actors: ", self.num_actors)
            print("num_dofs: ", self.num_dofs)
            print("num_force_sensors: ", self.num_force_sensors)
            print("num_rigid_bodies: ", self.num_rigid_bodies)

        # Wrap tensors with gymtorch
        self.root_states: torch.Tensor = gymtorch.wrap_tensor(_root_states)
        self.dof_states: torch.Tensor = gymtorch.wrap_tensor(_dof_states)
        self.dof_forces: torch.Tensor = gymtorch.wrap_tensor(_dof_forces)
        self.rigid_body_states: torch.Tensor = gymtorch.wrap_tensor(_rigid_body_states)
        self.net_contact_forces: torch.Tensor = gymtorch.wrap_tensor(_net_contact_forces)
        self.jacobians: torch.Tensor = gymtorch.wrap_tensor(_jacobians)

        if self.num_force_sensors > 0:
            self.force_sensor_states: torch.Tensor = gymtorch.wrap_tensor(_force_sensor_states)
        else:
            self.force_sensor_states: Optional[torch.Tensor] = None

        eef_link = getattr(self, "_xarm_eef_link", "link6")
        eef_rb_index = self.gym.find_asset_rigid_body_index(self.gym_assets["current"]["robot"]["asset"], eef_link)
        
        arm_dof_cols = len(self._xarm_dof_names)
        self.j_eef = self.jacobians[:, eef_rb_index - 1, :, :arm_dof_cols]
        # forearm_index = self.gym.find_asset_rigid_body_index(self.gym_assets["current"]["robot"]["asset"], "link6")
        # jacobian entries corresponding to link6
        # self.j_eef = self.jacobians[:, forearm_index - 1, :, :6]

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

        self.allegro_hand_root_states = root_states[:, self.allegro_hand_index, :]
        self.allegro_hand_root_positions = self.allegro_hand_root_states[:, 0:3]
        self.allegro_hand_root_orientations = self.allegro_hand_root_states[:, 3:7]
        self.allegro_hand_root_linear_velocities = self.allegro_hand_root_states[:, 7:10]
        self.allegro_hand_root_angular_velocities = self.allegro_hand_root_states[:, 10:13]
        
        self.object_root_states = self.root_states[self.occupied_object_indices]
        self.object_root_positions = self.object_root_states[..., 0:3]
        self.object_root_orientations = self.object_root_states[..., 3:7]
        self.object_root_linear_velocities = self.object_root_states[..., 7:10]
        self.object_root_angular_velocities = self.object_root_states[..., 10:13]
        
        self.surr_object_root_states = self.root_states[self.surr_object_indices]
        
        self.surr_object_root_positions = self.surr_object_root_states[..., 0:3].view(self.num_envs, self.max_non_targets, 3)
        self.surr_object_root_orientations = self.surr_object_root_states[..., 3:7].view(self.num_envs, self.max_non_targets, 4)
        self.surr_object_root_linear_velocities = self.surr_object_root_states[..., 7:10].view(self.num_envs, self.max_non_targets, 3)
        self.surr_object_root_angular_velocities = self.surr_object_root_states[..., 10:13].view(self.num_envs, self.max_non_targets, 3)
        self.prev_surr_object_root_positions = self.surr_object_root_positions
        self.prev_surr_object_root_orientations = self.surr_object_root_orientations
        self.prev_surr_object_root_linear_velocities = self.surr_object_root_linear_velocities
        self.prev_surr_object_root_angular_velocities = self.surr_object_root_angular_velocities
        

        self.scene_object_root_positions = self.root_positions[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)
        self.scene_object_root_orientations = self.root_orientations[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 4)
        self.scene_object_root_linear_velocities = self.root_linear_velocities[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)
        self.scene_object_root_angular_velocities = self.root_angular_velocities[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)

        self.init_scene_object_root_positions = self.scene_object_root_positions.clone()
        self.init_scene_object_root_orientations = self.scene_object_root_orientations.clone()

        # for randomize purpose
        if not hasattr(self, "scene_xy_offsets") or self.scene_xy_offsets is None:
            self.scene_xy_offsets = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.nominal_scene_object_root_positions = self.init_scene_object_root_positions.clone()
        self.nominal_scene_object_root_positions[..., 0] -= self.scene_xy_offsets[:, 0].view(-1, 1)  # x
        self.nominal_scene_object_root_positions[..., 1] -= self.scene_xy_offsets[:, 1].view(-1, 1)  # y

        dof_states = self.dof_states.view(self.num_envs, self.num_dofs, 2)

        self.allegro_hand_dof_positions = dof_states[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end, 0]
        self.allegro_hand_dof_velocities = dof_states[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end, 1]


        rigid_body_states = self.rigid_body_states.view(self.num_envs, self.num_rigid_bodies, 13)

        self.allegro_hand_rigid_body_states = rigid_body_states[
            :, self.allegro_hand_rigid_body_start : self.allegro_hand_rigid_body_end, :
        ]
        self.allegro_hand_rigid_body_positions = self.allegro_hand_rigid_body_states[..., 0:3]
        self.allegro_hand_rigid_body_orientations = self.allegro_hand_rigid_body_states[..., 3:7]
        self.allegro_hand_rigid_body_linear_velocities = self.allegro_hand_rigid_body_states[..., 7:10]
        self.allegro_hand_rigid_body_angular_velocities = self.allegro_hand_rigid_body_states[..., 10:13]

        self.allegro_hand_center_states = self.allegro_hand_rigid_body_states[:, self.allegro_center_index, :]
        self.allegro_hand_center_positions = self.allegro_hand_center_states[:, 0:3]
        self.allegro_hand_center_orientations = self.allegro_hand_center_states[:, 3:7]


        endeffector_index = self.gym.find_asset_rigid_body_index(
            self.gym_assets["current"]["robot"]["asset"], eef_link
        )
        self.endeffector_states = self.allegro_hand_rigid_body_states[:, endeffector_index, :]
        self.endeffector_positions = self.allegro_hand_rigid_body_positions[:, endeffector_index, :]
        self.endeffector_orientations = self.allegro_hand_rigid_body_orientations[:, endeffector_index, :]
        self.endeffector_linear_velocities = self.allegro_hand_rigid_body_linear_velocities[:, endeffector_index, :]
        self.endeffector_angular_velocities = self.allegro_hand_rigid_body_angular_velocities[:, endeffector_index, :]

        self.nearest_non_target_object_positions = torch.zeros((self.num_envs, self.num_nearest_non_targets, 3), device=self.device)
        self.nearest_non_target_object_orientations = torch.zeros((self.num_envs, self.num_nearest_non_targets, 4), device=self.device)
        
        self.keypoint_offset = torch.tensor(self.keypoint_offset, device=self.device).reshape(1, -1, 3)
        self.fingertip_link_indices_among_keypoints = torch.tensor(self.fingertip_link_indices_among_keypoints, device=self.device)
        self.index_link_indices_among_keypoints = torch.tensor(self.index_link_indices_among_keypoints, device=self.device)
        self.thumb_link_indices_among_keypoints = torch.tensor(self.thumb_link_indices_among_keypoints, device=self.device)
        self.middle_link_indices_among_keypoints = torch.tensor(self.middle_link_indices_among_keypoints, device=self.device)
        self.ring_link_indices_among_keypoints = torch.tensor(self.ring_link_indices_among_keypoints, device=self.device)


        # Intermediate tensors for _refresh_sim_tensors
        self._target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self._gather_indices_pos = torch.zeros((self.num_envs, self.max_non_targets, 3), dtype=torch.long, device=self.device)
        self._gather_indices_ori = torch.zeros((self.num_envs, self.max_non_targets, 4), dtype=torch.long, device=self.device)
        self._non_target_positions = torch.zeros((self.num_envs, self.max_non_targets, 3), device=self.device)
        self._non_target_orientations = torch.zeros((self.num_envs, self.max_non_targets, 4), device=self.device)
        self._distances = torch.zeros((self.num_envs, self.max_non_targets), device=self.device)
        self._sorted_distances = torch.zeros((self.num_envs, self.max_non_targets), device=self.device)
        self._sorted_indices = torch.zeros((self.num_envs, self.max_non_targets), dtype=torch.long, device=self.device)
        self._nearest_indices = torch.zeros((self.num_envs, self.k_nearest), dtype=torch.long, device=self.device)
        self._batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.k_nearest)
        self._valid_mask = torch.zeros((self.num_envs, self.max_non_targets), dtype=torch.bool, device=self.device)
        self._valid_nearest = torch.zeros((self.num_envs, self.k_nearest), dtype=torch.bool, device=self.device)
        self._invalid_mask_pos = torch.zeros((self.num_envs, self.k_nearest, 3), dtype=torch.bool, device=self.device)
        self._invalid_mask_ori = torch.zeros((self.num_envs, self.k_nearest, 4), dtype=torch.bool, device=self.device)
        self._inf_tensor = torch.full((self.num_envs, self.max_non_targets), torch.inf, device=self.device)


        net_contact_forces = self.net_contact_forces.view(self.num_envs, self.num_rigid_bodies, 3)

        self.allegro_hand_net_contact_forces = net_contact_forces[
            :, self.allegro_hand_rigid_body_start : self.allegro_hand_rigid_body_end, :
        ]

        # allocate buffers to hold intermediate results

        # render_target - false mode
        kwargs = {"dtype": torch.float, "device": self.device}
        self._r_target_object_root_positions = torch.zeros((self.num_envs, 3), **kwargs)
        self._r_target_object_root_orientations = torch.zeros((self.num_envs, 4), **kwargs)
        self._r_target_allegro_dof_positions = torch.zeros((self.num_envs, self.num_dofs), **kwargs)  # 6 arm + 16 hand
        self._r_target_allegro_digits_actuated_dof_positions = torch.zeros((self.num_envs, self._dims.HAND_ACTUATED_DIM.value), **kwargs)
        self._r_target_allegro_fingers_actuated_dof_positions = torch.zeros((self.num_envs, 12), **kwargs)  # 3 fingers × 4 DOF
        self._r_target_allegro_thumb_actuated_dof_positions = torch.zeros((self.num_envs, 4), **kwargs)  # 4 thumb DOF
        self._r_target_object_positions_wrt_palm = torch.zeros((self.num_envs, 3), **kwargs)
        self._r_target_object_orientations_wrt_palm = torch.zeros((self.num_envs, 4), **kwargs)
        self._r_target_palm_positions_wrt_object = torch.zeros((self.num_envs, 3), **kwargs)
        self._r_target_palm_orientations_wrt_object = torch.zeros((self.num_envs, 4), **kwargs)

        self.prev_targets_buffer = torch.zeros((self.num_envs, self.num_dofs), **kwargs)
        self.curr_targets_buffer = torch.zeros((self.num_envs, self.num_dofs), **kwargs)
        self.prev_allegro_dof_speeds = torch.zeros((self.num_envs, self._dims.HAND_ACTUATED_DIM.value), **kwargs)

        # create slices from above buffer
        self.prev_targets = self.prev_targets_buffer[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end]
        self.curr_targets = self.curr_targets_buffer[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end]


        self.rb_forces = torch.zeros((self.num_envs, self.num_rigid_bodies, 3), **kwargs)
        self.rb_torques = torch.zeros((self.num_envs, self.num_rigid_bodies, 3), **kwargs)
        self.occupied_object_init_root_positions = self.root_positions[self.occupied_object_indices, :].view(self.num_envs, 3)
        self.occupied_object_init_root_orientations = self.root_orientations[self.occupied_object_indices, :].view(self.num_envs, 4)
        self.surr_object_init_root_positions = self.root_positions[self.surr_object_indices, :].view(self.num_envs, self.max_non_targets, 3)
        self.surr_object_init_root_orientations = self.root_orientations[self.surr_object_indices, :].view(self.num_envs, self.max_non_targets, 4)
        
        self.robot_init_dof = torch.zeros((self.num_envs, self.num_dofs), **kwargs)
        self._hand_geo_center = torch.tensor(self._hand_geo_center, **kwargs)
        self._table_pose_tensor = torch.tensor(self._table_pose, **kwargs)
        self._target_hand_palm_pose = torch.tensor(self._target_hand_palm_pose, **kwargs)
        self._current_hand_palm_pose = torch.tensor(self._current_hand_palm_pose, **kwargs)
        self._xarm_right_init_position = torch.tensor(self._xarm_right_init_position, **kwargs)
        self._xarm_right_init_orientation = torch.tensor(self._xarm_right_init_orientation, **kwargs)
        self._palm2forearm_quat = torch.tensor(self._palm2forearm_quat, **kwargs)
        self._palm2forearm_pos = torch.tensor(self._palm2forearm_pos, **kwargs)
        self._hand_base_link2forearm_quat = torch.tensor(self._hand_base_link2forearm_quat, **kwargs)
        self._hand_base_link2forearm_pos = torch.tensor(self._hand_base_link2forearm_pos, **kwargs)
        self._object_nominal_orientation = torch.tensor(self._object_nominal_orientation, **kwargs)

        if self.enable_full_pointcloud_observation:
            self.pointclouds = torch.zeros((self.num_envs, self.num_object_points, 3), **kwargs)
            self.pointclouds_wrt_palm = torch.zeros((self.num_envs, self.num_object_points, 3), **kwargs)

        self.__init_meta_data()
        self.preprocess_allegro_pointcloud()

        self.successes = torch.zeros(self.num_envs, **kwargs)
        self.done_successes = torch.zeros(self.num_envs, **kwargs)
        self.current_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_successes = torch.zeros(1, **kwargs)
        self.unused_object_init_root_positions = torch.stack(
            [position(pose, self.device) for pose in self.gym_assets["current"]["objects"]["poses"]], dim=0
        )
        self.picked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.picked_curr = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.near_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        

        state_include_goal = bool(self.cfg["env"].get("stateIncludeGoal", False))
        self.reach_curiosity_mgr = CuriosityRewardManager(
            num_keypoints=self._dims.NUM_FINGERTIPS.value,
            device=self.device,
            canonical_pointcloud=self.grasping_dataset._pointclouds[0], #NOTE: hardcode here, not per-env
            kernel_param=0.03,
            # cluster parameters for contact reward
            cluster_k=32,
            max_clustering_iters=10,
            
            canonical_normals=self.grasping_dataset._pointcloud_normals[0],
            mask_backface_points=self.cfg["env"]["maskBackfacePoints"],
            mask_palm_inward_points=self.cfg["env"]["maskPalmInwardPoints"],
            use_normal_in_clustering=self.cfg["env"].get("useNormalInClustering", True),
            num_envs=self.num_envs,

            # 
            state_feature_dim=self.cfg["env"].get("stateFeatureDim", None),
            num_key_states=int(self.cfg["env"].get("numKeyStates", 32)),
            state_counter_mode=str(self.cfg["env"].get("stateCounterMode", "cluster")),
            state_include_goal=state_include_goal, # learned hash state exclusive config


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
        )
        
        self.use_object_mask = self.cfg["env"].get("useObjectMask", False)
        if self.cfg["env"].get("returnCuriosityInfo", False):
            self.curiosity_state_type = self.cfg["env"]["CuriosityInfo"].get("curiosityStateType", "policy_state")  # or "contact_force" or "contact_distance"
        
        if cfg["env"]["curiosity"]["enable_occlusion"]:
            self._occlusion_aabb_getter = self.get_singulation_occluder_aabbs
            self.reach_curiosity_mgr.set_occlusion_module(self._occlusion_aabb_getter())
        
        self.obj_max_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # for DPM
        self.object_bboxes = torch.zeros((self.num_envs, 6), **kwargs)
        self.object_categories = torch.zeros((self.num_envs, self.grasping_dataset._category_matrix.shape[1]), **kwargs)
        self.object_bboxes_wrt_world = torch.zeros((self.num_envs, 6), **kwargs)
        self.object_bboxes_wrt_palm = torch.zeros((self.num_envs, 6), **kwargs)

        self.training = True

        self.max_J = torch.ones(self.num_envs, device=self.device) * -torch.inf

        # for evaluation-only mode
        self.occupied_object_codes: np.ndarray = np.array(["" for _ in range(self.num_envs)])
        self.occupied_object_grasps: np.ndarray = np.array(["" for _ in range(self.num_envs)])
        self.occupied_object_cluster_ids: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # for action noise times tracking
        if self.action_noise and self.action_noise_level == "step" and self.action_noise_max_times > 0:
            self.action_noise_times = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)


        # Initialize singulation reward variables
        self.tilt_reward_scaled = torch.zeros(self.num_envs, device=self.device)
        self.slide_reward_scaled = torch.zeros(self.num_envs, device=self.device)

        # init state has collide with table, so we need to first reset to get robot to a valid pose, then continue simulation
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.reset_arm(first_time=True)

        if self.cfg["env"].get("returnCuriosityInfo", False):
            self.curiosity_state_dim = self.extras["curiosity_states"].shape[1:] # remove env dim

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

    # def test_pcl(self, env_ptr, env_id):
    #     object_asset_options = gymapi.AssetOptions()
    #     asset_sphere = self.gym.create_sphere(self.sim, 0.002, object_asset_options)
    #     pose = gymapi.Transform()
    #     pose.r = gymapi.Quat(0, 0, 0, 1)
    #     pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    #     pcl = torch.from_numpy(np.load("/home/thwu/Projects/func-mani/test.npy")).to("cuda:0")
    #     for (i, point) in enumerate(pcl):
    #         pose.p = gymapi.Vec3(point[0], point[1], point[2])
    #         capsule_handle = self.gym.create_actor(env_ptr, asset_sphere, pose, "actor{}", i+1000, 0)

    def __init_meta_data(self):
        self.observation_info = {}
        observation_space = self.cfg["env"]["observationSpace"]
        for name in observation_space:
            self.observation_info[name] = self._get_observation_dim(name)

        self.object_codes = []
        for object_codes_each_env in self.object_names:
            for object_code_each_env in object_codes_each_env:
                self.object_codes.append(object_code_each_env)
        self.object_codes = list(set(self.object_codes))
        self.object_cat = self.grasping_dataset.object_cat if self.grasping_dataset.object_cat is not None else "all"
        self.max_per_cat = self.grasping_dataset.max_per_cat if self.grasping_dataset.max_per_cat is not None else "all"
        self.object_geo_level = (
            self.grasping_dataset.object_geo_level if self.grasping_dataset.object_geo_level is not None else "all"
        )
        self.object_scale = (
            self.grasping_dataset.object_scale if self.grasping_dataset.object_scale is not None else "all"
        )
        self.label_paths = self.grasping_dataset.label_paths.copy()

    #######################
    # Imagined Pointcloud #
    #######################

    def preprocess_allegro_pointcloud(self):
        """Preprocess allegro-hand pointcloud.

        Load original allegro-hand pointcloud, apply farthest point sampling, store the result in `self._cached_pointclouds`.

            0.0-3.0 index finger
            4.0-7.0 middle finger
            8.0-11.0 ring finger
            12.0-15.0 thumb
        """


        original_mesh_dir = os.path.join(self._asset_root, "hands", "allegro_hand", "meshes", "visual")
        original_mesh_filepaths: OrderedDict = OrderedDict(
            [
                # Index finger (0.0-3.0)
                ("ffproximal", "link_0.0.glb"),
                ("ffmiddle", "link_1.0.glb"),
                ("ffdistal", "link_2.0.glb"),
                ("fftip", "link_tip.glb"),
                # Middle finger (4.0-7.0)
                ("mfproximal", "link_1.0.glb"),
                ("mfmiddle", "link_2.0.glb"),
                ("mfdistal", "link_3.0.glb"),
                ("mftip", "link_tip.glb"),
                # Ring finger (8.0-11.0)
                ("rfproximal", "link_1.0.glb"),
                ("rfmiddle", "link_2.0.glb"),
                ("rfdistal", "link_3.0.glb"),
                ("rftip", "link_tip.glb"),
                # Thumb (12.0-15.0)
                ("thproximal", "link_12.0_right.glb"),
                ("thmiddle", "link_13.0.glb"),
                ("thdistal", "link_14.0.glb"),
                ("thtip", "link_tip.glb"),
            ]
        )

        # load original mesh
        components = OrderedDict()
        for name, filepath in original_mesh_filepaths.items():
            # Map to actual link names used in Allegro hand
            if name.startswith("ff"):  # Index finger (finger 0)
                if "proximal" in name:
                    link_name = "link_0.0"
                elif "middle" in name:
                    link_name = "link_1.0"
                elif "distal" in name:
                    link_name = "link_2.0"
                else:  # tip
                    link_name = "link_3.0_tip"
            elif name.startswith("mf"):  # Middle finger (finger 1)
                if "proximal" in name:
                    link_name = "link_4.0"
                elif "middle" in name:
                    link_name = "link_5.0"
                elif "distal" in name:
                    link_name = "link_6.0"
                else:  # tip
                    link_name = "link_7.0_tip"
            elif name.startswith("rf"):  # Ring finger (finger 2)
                if "proximal" in name:
                    link_name = "link_8.0"
                elif "middle" in name:
                    link_name = "link_9.0"
                elif "distal" in name:
                    link_name = "link_10.0"
                else:  # tip
                    link_name = "link_11.0_tip"
            elif name.startswith("th"):  # Thumb
                if "proximal" in name:
                    link_name = "link_12.0"
                elif "middle" in name:
                    link_name = "link_13.0"
                elif "distal" in name:
                    link_name = "link_14.0"
                else:  # tip
                    link_name = "link_15.0_tip"
            else:
                link_name = name

            components[link_name] = {}
            components[link_name]["mesh"] = trimesh.load(
                os.path.join(original_mesh_dir, filepath), process=False, force="mesh"
            )

            area = components[link_name]["mesh"].area
            if "proximal" in name:
                area *= 0.3
            elif "middle" in name:
                area *= 0.6
            components[link_name]["area"] = area

        # compute number of samples for each component
        area = sum([item["area"] for item in components.values()])
        num_samples = self.num_imagined_points
        for name in components:
            components[name]["num_samples"] = int(round(components[name]["area"] / area * num_samples))
            area -= components[name]["area"]
            num_samples -= components[name]["num_samples"]
        assert sum([item["num_samples"] for item in components.values()]) == self.num_imagined_points

        # apply farthest point sampling
        pointclouds = {}
        for name in components:
            vertices = torch.tensor(components[name]["mesh"].vertices, dtype=torch.float, device=self.device)
            vertices *= 0.001  # convert to meter
            pcd = pytorch3d.ops.sample_farthest_points(vertices[None, ...], K=components[name]["num_samples"])[0][0]
            pointclouds[name] = pcd

            components[name]["pointcloud"] = pcd
            components[name]["contact"] = self.extract_contact_region(pcd)

        # find rigid body index for each component
        current_robot_asset = self.gym_assets["current"]["robot"]["asset"]
        # target_robot_asset = self.gym_assets["target"]["robot"]["asset"]

        if self.enable_contact_sensors:
            for name in components:
                components[name]["current_index"] = self.gym.find_asset_rigid_body_index(current_robot_asset, name)
                # components[name]["target_index"] = self.gym.find_asset_rigid_body_index(target_robot_asset, name)

                # For Allegro hand, map to force sensor names at fingertips
                if name in ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]:
                    sensor_name = f"sensor_{name}"
                    components[name]["sensor_index"] = (
                        self.force_sensor_names.index(sensor_name) if sensor_name in self.force_sensor_names else -1
                    )
                else:
                    components[name]["sensor_index"] = -1  # No sensor for non-tip links
                print(f"Link: {name}, Sensor: {components[name]['sensor_index']}")

        self._cached_pointclouds = pointclouds
        self.imagined_pointcloud_components = components
        # print(self.imagined_pointcloud_components)

    def extract_contact_region(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """Split the allegro-hand pointcloud to `front` and `back` side."""
        x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
        return (x.abs() < 0.9 * x.abs().max()) & (z.abs() < 0.9 * z.abs().max()) & (y < 0)

    def compute_imagined_pointclouds(
        self,
        stage: str,
        return_finger_index: bool = False,
        return_part_index: bool = False,
        return_binary_contact: bool = False,
    ) -> torch.Tensor:
        """Compute imagined pointclouds.

        Args:
            stage (str): "current" or "target"
            return_finger_index (bool, optional): _description_. Defaults to False.
            return_part_index (bool, optional): _description_. Defaults to False.
            return_binary_contact (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor: imagined pointclouds w.r.t. world frame (num_envs, num_imagined_points, 3)
        """
        assert stage in ["current", "target"], "stage must be either `current` or `target`"

        if stage == "current":
            rigid_body_positions = self.allegro_hand_rigid_body_positions
            rigid_body_orientations = self.allegro_hand_rigid_body_orientations
        else:
            rigid_body_positions = self.target_allegro_hand_rigid_body_positions
            rigid_body_orientations = self.target_allegro_hand_rigid_body_orientations

        imagined_pointclouds = torch.zeros((self.num_envs, self.num_imagined_points, 3), device=self.device)
        cursor = 0
        for name in self.imagined_pointcloud_components:
            component = self.imagined_pointcloud_components[name]
            i = component["current_index"] if stage == "current" else component["target_index"]

            pcd = component["pointcloud"].clone()
            if pcd.size(0) == 0:
                continue

            position = rigid_body_positions[:, i]
            rotation = rigid_body_orientations[:, i]

            num_points = pcd.size(0)

            pcd = transformation_apply(rotation[:, None, :], position[:, None, :], pcd[None, :, :])
            imagined_pointclouds[:, cursor : cursor + num_points, :] = pcd

            if return_binary_contact:
                # create binary contact
                contact = torch.zeros((self.num_envs, num_points), device=self.device)
                mask = component["contact"]
                if component["sensor_index"] != -1:
                    contact[:] = mask[None, :] * self.contact_forces[:, component["sensor_index"]][:, None]

            if return_finger_index:
                # create finger index
                finger_indices = torch.zeros((self.num_envs, num_points), device=self.device)
                finger_names = ["_th", "_ff", "_mf", "_rf", "_lf"]
                for i, finger in enumerate(finger_names):
                    if finger in name:
                        finger_indices[:] = i
                        break
                else:
                    raise ValueError(f"Unknown finger name: {name}")

            if return_part_index:
                # create part index
                part_indices = torch.zeros((self.num_envs, num_points), device=self.device)
                part_names = ["proximal", "middle", "distal"]
                for i, part in enumerate(part_names):
                    if part in name:
                        part_indices[:] = i
                        break
                else:
                    raise ValueError(f"Unknown part name: {name}")

            cursor += num_points
        return imagined_pointclouds

    def train(self):
        self.training = True

    def eval(self, vis=False):
        self.training = False

    def _configure_viewer(self):
        """Viewer setup."""
        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.5, -0.3, 0.8)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.4)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def compute_object_pointclouds(self, stage: str) -> torch.Tensor:
        """Compute the pointclouds of the objects w.r.t. the world frame.

        Args:
            stage (str): "current" or "target"

        Returns:
            torch.Tensor: pointclouds of the objects w.r.t. the world frame (num_envs, num_points, 3)
        """
        assert stage in ["current", "target"], "stage must be either `current` or `target`"

        if stage == "current":
            positions = self.object_root_positions
            orientations = self.object_root_orientations
        else:
            positions = self._r_target_object_root_positions
            orientations = self._r_target_object_root_orientations

        pcd = self.pointclouds.clone()
        pcd = transformation_apply(orientations[:, None, :], positions[:, None, :], pcd)
        return pcd

    def _refresh_sim_tensors(self) -> None:
        """Refresh the tensors for the simulation."""
        # TODO: only refresh tensors that are used in the task to save computation
        # TODO: only allocate once and reuse the tensors

        # refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # self.gym.find_actor_rigid_body_index(self.envs[0], self.allegro_hand_indices[0], 'link_15.0_tip', gymapi.DOMAIN_SIM)
        net_contact_forces = self.net_contact_forces.view(self.num_envs, self.num_rigid_bodies, 3)
        self.arm_contact_forces = net_contact_forces[:, self.arm_link_indices, :]
        self.hand_contact_forces = net_contact_forces[:, self.hand_link_indices, :]
        self.index_link_contact_forces = net_contact_forces[:, self.index_finger_link_indices, :]
        self.middle_link_contact_forces = net_contact_forces[:, self.middle_finger_link_indices, :]
        self.ring_link_contact_forces = net_contact_forces[:, self.ring_finger_link_indices, :]
        self.thumb_link_contact_forces = net_contact_forces[:, self.thumb_link_indices, :]
        self.fingertip_contact_forces = net_contact_forces[:, self.fingertip_indices, :]
        self.keypoint_contact_forces = net_contact_forces[:, self.keypoint_indices, :]

        # Target object contact forces [num_envs, 3]
        self.target_object_contact_forces = self.net_contact_forces[self.target_object_rigid_body_indices, :].view(self.num_envs, 3)
        
        # Surrounding objects contact forces [num_envs, max_surr_objects, 3]
        self.surr_object_contact_forces = self.net_contact_forces[self.surr_object_rigid_body_indices.flatten(), :].view(self.num_envs, self.max_non_targets, 3)

        self.table_contact_forces = self.net_contact_forces[self.table_rigid_body_indices, :].view(self.num_envs, 3)
        if self.use_back_wall:
            self.back_wall_contact_forces = self.net_contact_forces[self.back_wall_rigid_body_indices, :].view(self.num_envs, 3)
        if self.use_side_walls:
            self.side_wall_pos_x_contact_forces = self.net_contact_forces[self.side_wall_pos_x_rigid_body_indices, :].view(self.num_envs, 3)
            self.side_wall_neg_x_contact_forces = self.net_contact_forces[self.side_wall_neg_x_rigid_body_indices, :].view(self.num_envs, 3)

        self.fingertip_states = self.allegro_hand_rigid_body_states[:, self.fingertip_indices, :]
        self.fingertip_positions = self.fingertip_states[..., 0:3]
        self.fingertip_orientations = self.fingertip_states[..., 3:7]
        self.fingertip_linear_velocities = self.fingertip_states[..., 7:10]
        self.fingertip_angular_velocities = self.fingertip_states[..., 10:13]
        
        self.index_fingertip_positions = self.fingertip_positions[:, 0, :]
        self.middle_fingertip_positions = self.fingertip_positions[:, 1, :]
        self.ring_fingertip_positions = self.fingertip_positions[:, 2, :]
        self.thumb_fingertip_positions = self.fingertip_positions[:, 3, :]
        
        self.index_links_states = self.allegro_hand_rigid_body_states[:, self.index_finger_link_indices, :]
        self.middle_links_states = self.allegro_hand_rigid_body_states[:, self.middle_finger_link_indices, :]
        self.ring_links_states = self.allegro_hand_rigid_body_states[:, self.ring_finger_link_indices, :]
        self.thumb_links_states = self.allegro_hand_rigid_body_states[:, self.thumb_link_indices, :]

        # midpoint positions: the midpoint between the thumb tip and the third joint of the middle finger
        self.midpoint_positions = (self.thumb_fingertip_positions + self.middle_links_states[:, 2, 0:3]) / 2

        self.index_links_positions = self.index_links_states[..., 0:3] # (num_envs, 4, 3)
        self.middle_links_positions = self.middle_links_states[..., 0:3] # (num_envs, 4, 3)
        self.ring_links_positions = self.ring_links_states[..., 0:3] # (num_envs, 4, 3)
        self.thumb_links_positions = self.thumb_links_states[..., 0:3] # (num_envs, 4, 3)
        self.index_links_orientations = self.index_links_states[..., 3:7] # (num_envs, 4, 4)
        self.middle_links_orientations = self.middle_links_states[..., 3:7] # (num_envs, 4, 4)
        self.ring_links_orientations = self.ring_links_states[..., 3:7] # (num_envs, 4, 4)
        self.thumb_links_orientations = self.thumb_links_states[..., 3:7] # (num_envs, 4, 4)
        

        self.keypoint_positions = self.allegro_hand_rigid_body_positions[:, self.keypoint_indices, :]
        self.keypoint_orientations = self.allegro_hand_rigid_body_orientations[:, self.keypoint_indices, :]
        self.keypoint_positions_with_offset = self.keypoint_positions + quat_apply(self.keypoint_orientations, self.keypoint_offset.repeat(self.num_envs, 1, 1))
        self.fingertip_positions_with_offset = self.keypoint_positions_with_offset[:, self.fingertip_link_indices_among_keypoints, :]


        pcl_world = self._get_target_surface_points_world()  # (N, P, 3)
        # Pairwise distances: (N, K, P) where K = number of keypoints
        dists_kp = torch.cdist(self.keypoint_positions_with_offset, pcl_world)
        min_dists_kp, _ = torch.min(dists_kp, dim=2)  # (N, K)
        kp_force_mag = self.keypoint_contact_forces.norm(dim=-1, p=2)
        near_surface = (min_dists_kp < 0.010)
        has_force = (kp_force_mag > 0.01)
        self.keypoint_contact_mask = (near_surface & has_force)  # (N, K) bool

        _idx_any = self.keypoint_contact_mask[:, self.index_link_indices_among_keypoints].any(dim=1)
        _mid_any = self.keypoint_contact_mask[:, self.middle_link_indices_among_keypoints].any(dim=1)
        _ring_any = self.keypoint_contact_mask[:, self.ring_link_indices_among_keypoints].any(dim=1)
        _thumb_any = self.keypoint_contact_mask[:, self.thumb_link_indices_among_keypoints].any(dim=1)

        self.index_keypoint_contact_mask = _idx_any
        self.middle_keypoint_contact_mask = _mid_any
        self.ring_keypoint_contact_mask = _ring_any
        self.thumb_keypoint_contact_mask = _thumb_any

        self.fingers_keypoint_contact_mask = torch.stack([
            _idx_any, _mid_any, _ring_any, _thumb_any
        ], dim=1)

        self.object_root_states = self.root_states[self.occupied_object_indices]
        self.object_root_positions = self.object_root_states[..., 0:3]
        self.object_root_orientations = self.object_root_states[..., 3:7]
        self.object_root_linear_velocities = self.object_root_states[..., 7:10]
        self.object_root_angular_velocities = self.object_root_states[..., 10:13]
        
        # surrounding object
        self.surr_object_root_states = self.root_states[self.surr_object_indices]
        
        self.prev_surr_object_root_positions = self.surr_object_root_positions.clone()
        self.prev_surr_object_root_orientations = self.surr_object_root_orientations.clone()
        self.prev_surr_object_root_linear_velocities = self.surr_object_root_linear_velocities.clone()
        self.prev_surr_object_root_angular_velocities = self.surr_object_root_angular_velocities.clone()
        
        self.surr_object_root_positions = self.surr_object_root_states[..., 0:3].view(self.num_envs, self.max_non_targets, 3)
        self.surr_object_root_orientations = self.surr_object_root_states[..., 3:7].view(self.num_envs, self.max_non_targets, 4)
        self.surr_object_root_linear_velocities = self.surr_object_root_states[..., 7:10].view(self.num_envs, self.max_non_targets, 3)
        self.surr_object_root_angular_velocities = self.surr_object_root_states[..., 10:13].view(self.num_envs, self.max_non_targets, 3)

        # scene object
        self.scene_object_root_positions = self.root_positions[ self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)
        self.scene_object_root_orientations = self.root_orientations[ self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 4)
        self.scene_object_root_linear_velocities = self.root_linear_velocities[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)
        self.scene_object_root_angular_velocities = self.root_angular_velocities[self.object_indices, :].view(self.num_envs, self.num_objects_per_env, 3)

        # Compute nearest non-target objects for each environment
        self.nearest_non_target_object_positions.zero_()
        self.nearest_non_target_object_orientations.zero_()

        if self.non_occupied_object_indices.numel() > 0:
            # occupied_object_relative_indices shape: (num_envs,)
            # scene_object_root_positions shape: (num_envs, num_objects_per_env, 3)
            torch.gather(
                self.scene_object_root_positions,
                1,
                self.occupied_object_relative_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3),
                out=self._target_positions.unsqueeze(1)
            )
            self._target_positions.squeeze_(1)  # Shape: (num_envs, 3)

            # Create expanded indices for gathering (use pre-allocated tensors)
            self._gather_indices_pos[:] = self.non_occupied_object_indices.unsqueeze(-1).expand(-1, -1, 3)
            self._gather_indices_ori[:] = self.non_occupied_object_indices.unsqueeze(-1).expand(-1, -1, 4)

            # Gather non-target positions and orientations (use pre-allocated tensors)
            torch.gather(self.scene_object_root_positions, 1, self._gather_indices_pos, out=self._non_target_positions)
            torch.gather(self.scene_object_root_orientations, 1, self._gather_indices_ori, out=self._non_target_orientations)

            # Compute distances from target to all non-target objects (use pre-allocated tensor)
            # target_positions: (num_envs, 3) -> (num_envs, 1, 3)
            # non_target_positions: (num_envs, max_non_targets, 3)
            torch.norm(
                self._non_target_positions - self._target_positions.unsqueeze(1),
                dim=2,
                out=self._distances
            )  # Shape: (num_envs, max_non_targets)

            # Set distance to infinity for padded/invalid objects (use pre-allocated mask)
            torch.logical_and(
                self.non_occupied_object_indices >= 0,
                self.non_occupied_object_indices < self.num_objects_per_env,
                out=self._valid_mask
            )
            torch.where(self._valid_mask, self._distances, self._inf_tensor, out=self._distances)

            # Sort distances and get indices of nearest objects (use pre-allocated tensors)
            torch.sort(self._distances, dim=1, out=(self._sorted_distances, self._sorted_indices))

            # Take only the k nearest (use pre-allocated tensor)
            self._nearest_indices[:] = self._sorted_indices[:, :self.k_nearest]

            # Gather the nearest positions and orientations (use pre-allocated batch_indices)
            self.nearest_non_target_object_positions[:, :self.k_nearest] = self._non_target_positions[self._batch_indices, self._nearest_indices]
            self.nearest_non_target_object_orientations[:, :self.k_nearest] = self._non_target_orientations[self._batch_indices, self._nearest_indices]

            # Handle case where some environments have invalid nearest objects (use pre-allocated tensors)
            torch.lt(self._sorted_distances[:, :self.k_nearest], torch.inf, out=self._valid_nearest)

            # Create expanded invalid masks (use pre-allocated tensors)
            torch.logical_not(self._valid_nearest.unsqueeze(-1).expand(-1, -1, 3), out=self._invalid_mask_pos)
            torch.logical_not(self._valid_nearest.unsqueeze(-1).expand(-1, -1, 4), out=self._invalid_mask_ori)

            # Apply masks to zero out invalid entries
            self.nearest_non_target_object_positions[:, :self.k_nearest, :][self._invalid_mask_pos] = 0.0
            self.nearest_non_target_object_orientations[:, :self.k_nearest, :][self._invalid_mask_ori] = 0.0


        self.object_bboxes_wrt_world[:, :3] = transformation_apply(
            self.object_root_orientations, self.object_root_positions, self.object_bboxes[:, :3]
        )
        self.object_bboxes_wrt_world[:, 3:] = transformation_apply(
            self.object_root_orientations, self.object_root_positions, self.object_bboxes[:, 3:]
        )

        world_to_palm_rotation, world_to_palm_translation = transformation_inverse(
            self.allegro_hand_center_orientations, self.allegro_hand_center_positions
        )

        self.object_bboxes_wrt_palm[:, :3] = transformation_apply(
            world_to_palm_rotation, world_to_palm_translation, self.object_bboxes_wrt_world[:, :3]
        )
        self.object_bboxes_wrt_palm[:, 3:] = transformation_apply(
            world_to_palm_rotation, world_to_palm_translation, self.object_bboxes_wrt_world[:, 3:]
        )

        self.palm_orientations_wrt_object, self.palm_positions_wrt_object = compute_relative_pose(
            self.allegro_hand_center_orientations,
            self.allegro_hand_center_positions,
            self.object_root_orientations,
            self.object_root_positions,
        )

        self.fingertip_orientations_wrt_palm, self.fingertip_positions_wrt_palm = compute_relative_pose(
            self.fingertip_orientations,
            self.fingertip_positions,
            self.allegro_hand_center_orientations[:, None, :],
            self.allegro_hand_center_positions[:, None, :],
        )

        
        self.fingertip_orientations_wrt_object, self.fingertip_positions_wrt_object = compute_relative_pose(
            self.fingertip_orientations,
            self.fingertip_positions_with_offset, # NOTE: offset definition required
            self.object_root_orientations[:, None, :],
            self.object_root_positions[:, None, :],
        )
        
        pcl_world = self._get_target_surface_points_world()   # (N, P, 3)

        # pairwise distances (batched): (N, 4, P)
        dists = torch.cdist(self.fingertip_positions_with_offset, pcl_world)
        min_dists_per_finger, idx_p = torch.min(dists, dim=2)  # (N,4), (N,4)

        # gather nearest surface point for each fingertip
        idx_p_exp = idx_p.unsqueeze(-1).expand(-1, -1, 3)      # (N,4,3)
        obj_pts = torch.gather(pcl_world, 1, idx_p_exp)        # (N,4,3)
        u = obj_pts - self.fingertip_positions_with_offset                                     # (N,4,3)
        self.fingertip_geometric_distances = u.norm(dim=-1) # (N, 4)
        self.fingertip_geometric_directions = u / self.fingertip_geometric_distances.unsqueeze(-1) # (N, 4, 3)
    
        if add_noise:
            obj_pos_estimation_nosie = torch.clamp(
                torch.randn_like(self.object_root_positions.clone()) * np.sqrt(0.0004), -0.02, 0.02
            )
            obj_quat_estimation_noise = np.sqrt(2 / 57.3)

            self.observed_object_positions = self.object_root_positions.clone() + obj_pos_estimation_nosie
            self.observed_object_orientations = random_orientation_within_angle(
                self.object_root_orientations.size(0),
                self.device,
                self.object_root_orientations.clone(),
                obj_quat_estimation_noise,
            )
            self.observed_object_orientations_wrt_palm, self.observed_object_positions_wrt_palm = compute_relative_pose(
                self.observed_object_orientations,
                self.observed_object_positions,
                self.allegro_hand_center_orientations,
                self.allegro_hand_center_positions,
            )

        self.object_orientations_wrt_palm, self.object_positions_wrt_palm = compute_relative_pose(
            self.object_root_orientations,
            self.object_root_positions,
            self.allegro_hand_center_orientations,
            self.allegro_hand_center_positions,
        )

        self.object_positions_wrt_keypoints = self.keypoint_positions - self.object_root_positions[:, None, :]




        self.position_distances = self.object_positions_wrt_palm - self._r_target_object_positions_wrt_palm
        self.orientation_distances = quat_mul(
            self.object_orientations_wrt_palm, quat_conjugate(self._r_target_object_orientations_wrt_palm)
        )
        self.dof_distances = (
            self.allegro_hand_dof_positions[:, self.allegro_digits_actuated_dof_indices]
            - self._r_target_allegro_digits_actuated_dof_positions
        )

        if self.enable_contact_sensors:
            contact_forces = self.allegro_hand_net_contact_forces[:, self.force_sensor_rigid_body_indices, :]
            contact_forces = torch.norm(contact_forces, dim=-1)
            # binary contact sensor
            self.contact_forces = torch.where(contact_forces >= self.contact_sensor_threshold, 1.0, 0.0)
            self.fingertip_contact_forces = self.contact_forces[:, self.fingertip_contact_mask]

            # visualize
            # for (env_id, each_env_contacts) in enumerate(self.contact_forces):
            #     for (contact_idx, each_env_contact) in enumerate(each_env_contacts):
            #         self.gym.set_rigid_body_color(self.envs[env_id], 0, self.force_sensor_parent_rigid_body_indices[contact_idx], gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(each_env_contact, 0.0, 0.0))
            # import time
            # time.sleep(0.1)
        if self.enable_full_pointcloud_observation:
            self.obj_pointclouds_wrt_world = self.compute_object_pointclouds("current")
            self.target_obj_pointclouds_wrt_world = self.compute_object_pointclouds("target")

            self.object_pointclouds = self.obj_pointclouds_wrt_world
            self.pointclouds_wrt_palm = compute_relative_position(
                self.obj_pointclouds_wrt_world,
                self.allegro_hand_center_orientations[:, None, :],
                self.allegro_hand_center_positions[:, None, :],
            )
            self.object_pointclouds_wrt_palm = self.pointclouds_wrt_palm


        if self.enable_imagined_pointcloud_observation:
            self.imagined_pointclouds = self.compute_imagined_pointclouds("current")
            self.imagined_pointclouds_wrt_palm = compute_relative_position(
                self.imagined_pointclouds,
                self.allegro_hand_center_orientations[:, None, :],
                self.allegro_hand_center_positions[:, None, :],
            )



        if self.enable_rendered_pointcloud_observation:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            depth = torch.stack(self.camera_tensors).view(
                self.num_envs, self.num_cameras_per_env, self.camera_properties.height, self.camera_properties.width
            )

            pointclouds, mask = pointcloud_from_depth(
                depth,
                inv_view_matrix=self.camera_inv_view_matrices,
                proj_matrix=self.camera_proj_matrices,
                width=self.camera_properties.width,
                height=self.camera_properties.height,
                u=self.camera_u2,
                v=self.camera_v2,
            )
            corner_min, corner_max = self.render_pointcloud_bbox_corners
            mask = mask & (pointclouds > corner_min).all(dim=-1) & (pointclouds < corner_max).all(dim=-1)

            num_points_per_env = self.num_cameras_per_env * self.camera_properties.height * self.camera_properties.width
            pointclouds = pointclouds.view(self.num_envs, num_points_per_env, 3)
            mask = mask.view(self.num_envs, num_points_per_env)

            if self.enable_rendered_pointcloud_target_mask:
                seg = torch.stack(self.camera_seg_tensors).view(
                    self.num_envs, self.num_cameras_per_env, self.camera_properties.height, self.camera_properties.width
                )
                seg = seg.view(self.num_envs, num_points_per_env)
                is_target = (seg == self.target_segmentation_id) & mask
            else:
                is_target = None

            counts = mask.sum(dim=1)

            rand = torch.rand((self.num_envs, num_points_per_env), device=self.device)
            keys = rand.masked_fill(~mask, -1.0)

            indices_1d = torch.argsort(keys, dim=1, descending=True)
            indices_3d = indices_1d.unsqueeze(-1).expand(-1, -1, 3)

            pointclouds = pointclouds.gather(1, indices_3d)
            mask = mask.gather(1, indices_1d)  # keep mask aligned with permuted pointclouds

            if is_target is not None:
                is_target = is_target.gather(1, indices_1d)

            if self.rendered_pointcloud_sample_method == "random":
                # random sampling
                location = torch.rand((self.num_envs, self.num_rendered_points), device=self.device)
                indices = torch.floor(location * counts.unsqueeze(-1)).long()
                indices = indices.unsqueeze(-1).expand(-1, -1, 3)
                rendered_pointclouds = pointclouds.gather(1, indices)
                if is_target is not None:
                    rendered_is_target = is_target.gather(1, indices[..., 0])
                else:
                    rendered_is_target = None
            else:
                # farthest point sampling
                maximum_rendered_candidates = self.num_rendered_points * self.rendered_pointcloud_multiplier
                cand = min(int(maximum_rendered_candidates), int(counts.max().item()))
                pointclouds = pointclouds[:, :cand]
                counts = torch.clamp(counts, max=maximum_rendered_candidates)
                rendered_pointclouds, fps_idx = sample_farthest_points(pointclouds, counts, K=self.num_rendered_points)

                if is_target is not None:
                    is_target_cand = is_target[:, :cand]
                    rendered_is_target = is_target_cand.gather(1, fps_idx)
                else:
                    rendered_is_target = None


            if (counts == 0).any():
                rendered_pointclouds[counts == 0] = 0.0

            if self.rendered_pointcloud_gaussian_noise:
                noise = (
                    torch.randn(rendered_pointclouds.shape, device=self.device)
                    * self.rendered_pointcloud_gaussian_noise_sigma
                )
                mask = (
                    torch.rand((self.num_envs, self.num_rendered_points, 1), device=self.device)
                    < self.rendered_pointcloud_gaussian_noise_ratio
                )
                noise *= mask
                rendered_pointclouds += noise

            self.rendered_pointclouds = rendered_pointclouds

            if rendered_is_target is not None:
                self.rendered_pointclouds_w_target_mask = torch.cat(
                    [rendered_pointclouds, rendered_is_target.unsqueeze(-1).float()], dim=-1
                )

            # import open3d as o3d

            # o3d_pointcloud = o3d.geometry.PointCloud()
            # o3d_pointcloud.points = o3d.utility.Vector3dVector(rendered_pointclouds[0].to("cpu").numpy())

            # origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # camera_frame_0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=self.camera_positions[0, 0].to("cpu").numpy())
            # camera_frame_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=self.camera_positions[0, 1].to("cpu").numpy())
            # # camera_frame_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=self.camera_positions[0, 2].to("cpu").numpy())

            # o3d.visualization.draw_geometries([o3d_pointcloud, origin_axis, camera_frame_0, camera_frame_1])

            # segmented_pointcloud = o3d.geometry.PointCloud()
            # segmented_pointcloud.points = o3d.utility.Vector3dVector(rendered_pointclouds[0].cpu().numpy())

            # # Color points based on segmentation mask if available
            # if self.enable_rendered_pointcloud_target_mask:
            #     assert rendered_is_target is not None, "rendered_is_target must be provided when enable_rendered_pointcloud_target_mask is True"
            #     is_target = rendered_is_target[0].cpu().numpy()
            #     colors = np.zeros((len(rendered_pointclouds[0].cpu().numpy()), 3))
            #     colors[is_target] = [1.0, 0.0, 0.0]
            #     colors[~is_target] = [0.0, 0.0, 1.0]
            #     segmented_pointcloud.colors = o3d.utility.Vector3dVector(colors)

            # # Create coordinate frames
            # origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            
            # camera_frames = []
            # for cam_idx in range(self.num_cameras_per_env):
            #     cam_pos = self.camera_positions[0, cam_idx].cpu().numpy()
            #     cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            #         size=0.05,
            #         origin=cam_pos
            #     )
            #     camera_frames.append(cam_frame)

            # geometries = [segmented_pointcloud, origin_axis] + camera_frames
            # o3d.visualization.draw_geometries(geometries, window_name="Segmented Point Cloud", width=1280, height=720)
            self.gym.end_access_image_tensors(self.sim)

        # compute tip-tip and tip-mid equidistant points - (CASE2023 Baseline)
        if self.method == "case":
            thtip_index = self.gym.find_asset_rigid_body_index(self.gym_assets["current"]["robot"]["asset"], "rh_thtip")
            mfmid_index = self.gym.find_asset_rigid_body_index(
                self.gym_assets["current"]["robot"]["asset"], "rh_mfmiddle"
            )
            mftip_index = self.gym.find_asset_rigid_body_index(self.gym_assets["current"]["robot"]["asset"], "rh_mftip")

            thtip_positions = self.allegro_hand_rigid_body_positions[:, thtip_index]
            mfmid_positions = self.allegro_hand_rigid_body_positions[:, mfmid_index]
            mftip_positions = self.allegro_hand_rigid_body_positions[:, mftip_index]

            alpha = (torch.arange(1, 4, device=self.device) / 4.0).reshape(1, 3, 1)
            tiptip_points = alpha * thtip_positions[:, None, :] + (1 - alpha) * mftip_positions[:, None, :]
            tipmid_points = alpha * thtip_positions[:, None, :] + (1 - alpha) * mfmid_positions[:, None, :]
            kpoint_positions = torch.cat([tiptip_points, tipmid_points], dim=1)
            kpoint_positions_wrt_object = compute_relative_position(
                kpoint_positions,
                self.object_root_orientations[:, None, :],
                self.object_root_positions[:, None, :],
            )
            self.kpoint_distances = point_to_mesh_distance(
                kpoint_positions_wrt_object,
                self.grasping_dataset._sdf_fields,
                self.occupied_mesh_indices,
            )

            fingertip_positions_wrt_object = compute_relative_position(
                self.fingertip_positions,
                self.object_root_orientations[:, None, :],
                self.object_root_positions[:, None, :],
            )
            self.fingertip_distances = point_to_mesh_distance(
                fingertip_positions_wrt_object,
                self.grasping_dataset._sdf_fields,
                self.occupied_mesh_indices,
            )

            norm_object_orientation = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            self.norm_object_orientation_wrt_palm = quat_mul(
                quat_conjugate(self.allegro_hand_center_orientations), norm_object_orientation
            )

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

        # check imagined pointcloud observation
        if any([("imagined" in spec.tags and "pointcloud" in spec.tags) for spec in self._observation_space_extra]):
            if not self.enable_imagined_pointcloud_observation:
                warnings.warn("imagined pointcloud observation is enabled but not configured")
                warnings.warn("automatically overwrite `enable_imagined_pointcloud_observation` to `True`")
                self.enable_imagined_pointcloud_observation = True

        # check rendered pointcloud observation
        if any([("rendered" in spec.tags and "pointcloud" in spec.tags) for spec in self._observation_space_extra]):
            if not self.enable_rendered_pointcloud_observation:
                warnings.warn("rendered pointcloud observation is enabled but not configured")
                warnings.warn("automatically overwrite `enable_rendered_pointcloud_observation` to `True`")
                self.enable_rendered_pointcloud_observation = True

        # TODO: configure it from observation space
        self.pcl_obs = self.cfg["env"]["pclObs"]
        self.enable_full_pointcloud_observation = (
            "pointcloud_wrt_palm" in observation_space
            or "pclcontact" in self.reward_type
            or "stage" in self.curriculum_mode
            or ("no" in self.curriculum_mode and self.height_scale == 1.0)
            or self.pcl_obs
        )

        if any([("perfect" in spec.tags and "pointcloud" in spec.tags) for spec in self._observation_space_extra]):
            if not self.enable_full_pointcloud_observation:
                warnings.warn("perfect pointcloud observation is enabled but not configured")
                warnings.warn("automatically overwrite `enable_full_pointcloud_observation` to `True`")
                self.enable_full_pointcloud_observation = True

    def _create_ground_plane(self, static_friction: float = 1.0, dynamic_friction: float = 1.0) -> None:
        """Create a ground plane for the simulation.

        The ground plane is created using the `gymapi.PlaneParams` class.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0
        plane_params.static_friction = static_friction
        plane_params.dynamic_friction = dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

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
        

    def _define_upper_shelf(self) -> Dict[str, Any]:
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
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

        num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
        num_rigid_shapes = self.gym.get_asset_rigid_shape_count(asset)

        import copy
        pose = gymapi.Transform()
        _upper_shelf_pose = copy.deepcopy(self._upper_shelf_pos)
        pose.p = gymapi.Vec3(*_upper_shelf_pose)

        return {
            "asset": asset,
            "pose": pose,
            "name": "upper_shelf",
            "num_rigid_bodies": num_rigid_bodies,
            "num_rigid_shapes": num_rigid_shapes,
        }

    def _define_back_wall(self) -> Dict[str, Any]:
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        wall_thickness_y = self._table_thickness

        table_top_z = self._table_pose[2] + self._table_thickness * 0.5
        if self.use_upper_shelf:
            # upper shelf bottom z equals: shelf_center_z - shelf_thickness/2
            upper_shelf_bottom_z = self._upper_shelf_pos[2] - self._table_thickness * 0.5

            wall_height_z = max(upper_shelf_bottom_z - table_top_z, 1e-3)
            wall_center_z = table_top_z + wall_height_z * 0.5
            wall_height_z += self._table_thickness * 2
        else:
            # fallback height if no shelf is used
            upper_height_z = self._table_pose[2] + self._obj_height * 1.25
            wall_height_z = max(upper_height_z - table_top_z, 1e-3)
            wall_center_z = table_top_z + wall_height_z * 0.5
            wall_height_z += self._table_thickness * 2

        wall_size_x = self._table_x_length
        wall_size_y = wall_thickness_y
        wall_size_z = wall_height_z

        asset = self.gym.create_box(self.sim, wall_size_x, wall_size_y, wall_size_z, asset_options)

        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        for shape in rigid_shape_props:
            shape.friction = 1.0
            shape.restitution = 0.0
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

        num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
        num_rigid_shapes = self.gym.get_asset_rigid_shape_count(asset)

        # Place wall flush with the negative-y edge of the table
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(
            self._table_pose[0],
            self._table_pose[1] - (self._table_y_length * 0.5 + wall_thickness_y * 0.5) - 0.002,
            wall_center_z,
        )
        pose.r = gymapi.Quat(0, 0, 0, 1)

        return {
            "asset": asset,
            "pose": pose,
            "name": "back_wall",
            "num_rigid_bodies": num_rigid_bodies,
            "num_rigid_shapes": num_rigid_shapes,
        }

    def _define_side_wall(self, sign_x: float, name: str) -> Dict[str, Any]:
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        wall_margin = 0.002

        # Match back wall height logic (so top height matches)
        table_top_z = self._table_pose[2] + self._table_thickness * 0.5
        if self.use_upper_shelf:
            upper_shelf_bottom_z = self._upper_shelf_pos[2] - self._table_thickness * 0.5

            wall_height_z = max(upper_shelf_bottom_z - table_top_z, 1e-3)
            wall_center_z = table_top_z + wall_height_z * 0.5
            wall_height_z += self._table_thickness * 2
        else:
            upper_height_z = self._table_pose[2] + self._obj_height * 1.25
            wall_height_z = max(upper_height_z - table_top_z, 1e-3)
            wall_center_z = table_top_z + wall_height_z * 0.5
            wall_height_z += self._table_thickness * 2

        # Side wall geometry: thin in x, long in y
        wall_thickness_x = self._table_thickness
        wall_size_x = wall_thickness_x
        wall_size_y = self._table_y_length
        wall_size_z = wall_height_z

        asset = self.gym.create_box(self.sim, wall_size_x, wall_size_y, wall_size_z, asset_options)

        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        for shape in rigid_shape_props:
            shape.friction = 1.0
            shape.restitution = 0.0
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

        num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
        num_rigid_shapes = self.gym.get_asset_rigid_shape_count(asset)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(
            self._table_pose[0] + sign_x * (self._table_x_length * 0.5 + wall_thickness_x * 0.5 + wall_margin),
            self._table_pose[1],
            wall_center_z,
        )
        pose.r = gymapi.Quat(0, 0, 0, 1)

        return {
            "asset": asset,
            "pose": pose,
            "name": name,
            "num_rigid_bodies": num_rigid_bodies,
            "num_rigid_shapes": num_rigid_shapes,
        }

    def __define_contact_sensors(self, allegro_hand_asset: gymapi.Asset) -> None:
        """Configure the contact sensors.

        All the contact sensors are attached to the allegro Hand. The corresponding link names should start with `sensor_`.

        Args:
            allegro_hand_asset (gymapi.Asset): The allegro Hand asset to configure.
        """
        indices = []
        fingertip_indices = []
        parent_indices = []

        print("Contact sensors:")
        for name, index in self.gym.get_asset_rigid_body_dict(allegro_hand_asset).items():
            if name.startswith("sensor_"):
                indices.append(index)
                if "distal" in name:
                    fingertip_indices.append(index)
                print(f"- {name} ({index})")
        fingertip_contact_mask = [(i in fingertip_indices) for i in indices]

        assert len(indices) > 0, "No contact sensors found in the allegro Hand asset."
        self.force_sensor_rigid_body_indices = torch.tensor(indices).long().sort().values.to(self.device)

        self.force_sensor_names = []
        for i in self.force_sensor_rigid_body_indices:
            name = self.gym.get_asset_rigid_body_name(allegro_hand_asset, i)
            self.force_sensor_names.append(name)

            parent_name = name.replace("sensor", "rh")
            parent_indices.append(self.gym.find_asset_rigid_body_index(allegro_hand_asset, parent_name))

        self.force_sensor_parent_rigid_body_indices = torch.tensor(parent_indices).long().to(self.device)
        self.fingertip_contact_mask = torch.tensor(fingertip_contact_mask).bool().to(self.device)
        self.num_tactile_sensors = self.force_sensor_rigid_body_indices.size(0)
        # find same element in two lists
        self.ft_idx_in_all = [i for (i, index) in enumerate(indices) if index in fingertip_indices]

    @property
    def contact_states(self) -> torch.Tensor:
        """Compute contact states (tactile information) from force sensor data.
        """
        assert self.hand_contact_forces is not None
        
        # print(f">>> hand_contact_forces: {self.hand_contact_forces.norm(dim=-1)[0]}")
        return (torch.cat([self.hand_contact_forces.norm(dim=-1), self.fingertip_contact_forces.norm(dim=-1)], dim=-1).clamp_max(5.0)).float() / 5.0

    def __configure_robot_dof_indices(self, allegro_hand_asset: gymapi.Asset) -> None:
        """Configure the xArm6 + Allegro Hand DOFs.

        Args:
            allegro_hand_asset (gymapi.Asset): The xArm6 + Allegro Hand asset to configure.
        """
        dof_dict = self.gym.get_asset_dof_dict(allegro_hand_asset)

        actuated_dof_indices = []
        xarm_actuated_dof_indices = []
        allegro_actuated_dof_indices = []
        allegro_digits_actuated_dof_indices = []
        allegro_fingers_actuated_dof_indices = []
        allegro_thumb_actuated_dof_indices = []

        for name, index in dof_dict.items():
            if any([dof in name for dof in self._xarm_dof_names]):
                xarm_actuated_dof_indices.append(index)
            elif any([dof in name for dof in self._allegro_hand_dof_names]):
                allegro_actuated_dof_indices.append(index)
                allegro_digits_actuated_dof_indices.append(index)

                if any([dof in name for dof in self._allegro_fingers_dof_names]):
                    allegro_fingers_actuated_dof_indices.append(index)
                elif any([dof in name for dof in self._allegro_thumb_dof_names]):
                    allegro_thumb_actuated_dof_indices.append(index)

            actuated_dof_indices.append(index)

        def _torchify(indices: List[int]) -> torch.LongTensor:
            return torch.tensor(sorted(indices)).long().to(self.device)

        self.actuated_dof_indices = _torchify(actuated_dof_indices)
        self.xarm_actuated_dof_indices = _torchify(xarm_actuated_dof_indices)
        self.allegro_actuated_dof_indices = _torchify(allegro_actuated_dof_indices)
        self.allegro_digits_actuated_dof_indices = _torchify(allegro_digits_actuated_dof_indices)
        self.allegro_fingers_actuated_dof_indices = _torchify(allegro_fingers_actuated_dof_indices)
        self.allegro_thumb_actuated_dof_indices = _torchify(allegro_thumb_actuated_dof_indices)

    def _define_allegro_hand_with_arm(
        self, asset_name: str = "allegro Hand + xarm"
    ) -> Dict[str, Any]:
        """Define & load the allegro Hand + xarm asset.

        Args:
            asset_name (str, optional): Asset name for logging. Defaults to "allegro Hand + xarm".

        Returns:
            Dict[str, Any]: The configuration of the robot.
        """
        print(">>> Loading allegro Hand + xarm for current scene")
        config = {"name": "allegro_hand"}

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        # asset_options.linear_damping = 0.1

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        if self.env_info_logging:
            print_asset_options(asset_options, asset_name)

        if self.enable_contact_sensors:
            if self.contact_sensor_fingertip_only:
                asset_filename = self._xarm_allegro_hand_right_asset_file.replace(".urdf", "_contact_fingertip.urdf")
            else:
                asset_filename = self._xarm_allegro_hand_right_asset_file.replace(".urdf", "_contact.urdf")
        else:
            asset_filename = self._xarm_allegro_hand_right_asset_file

        asset = self.gym.load_asset(self.sim, self._asset_root, asset_filename, asset_options)
        if self.env_info_logging:
            print_links_and_dofs(self.gym, asset, asset_name)

        config["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(asset)
        config["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(asset)
        config["num_dofs"] = self.gym.get_asset_dof_count(asset)
        config["num_actuators"] = self.gym.get_asset_actuator_count(asset)
        config["num_tendons"] = self.gym.get_asset_tendon_count(asset)

        num_dofs = config["num_dofs"]



        dof_props = self.gym.get_asset_dof_properties(asset)
        hand_dof_idx = 0

        # set rigid-shape properties for allegro-hand
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        for shape in rigid_shape_props:
            shape.friction = 0.8
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

        for i in range(num_dofs):
            name = self.gym.get_asset_dof_name(asset, i)
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if name.endswith(".0"):
                dof_props["stiffness"][i] = 30
                dof_props["damping"][i] = 1
                dof_props["velocity"][i] = 3.0
                dof_props["effort"][i] = 0.7
                hand_dof_idx += 1
            else:
                dof_props["stiffness"][i] = 4000
                dof_props["damping"][i] = 80
                # dof_props["stiffness"][i] = 1e6
                # dof_props["damping"][i] = 1e2

        if self.env_info_logging:
            print_dof_properties(self.gym, asset, dof_props, asset_name)

        dof_lower_limits = [dof_props["lower"][i] for i in range(num_dofs)]
        dof_upper_limits = [dof_props["upper"][i] for i in range(num_dofs)]
        dof_init_positions = [0.0 for _ in range(num_dofs)]
        dof_init_velocities = [0.0 for _ in range(num_dofs)]

        # reset xarm initial dof positions
        for name, value in self._xarm_right_init_dof_positions.items():
            dof_init_positions[self.gym.find_asset_dof_index(asset, name)] = value
        for name, value in self.allegro_hand_init_dof_positions.items():
            dof_init_positions[self.gym.find_asset_dof_index(asset, name)] = value
            self._allegro_hand_predef_qpos[
                self.gym.find_asset_dof_index(asset, name) - 6
            ] = value  # substract 6 for xarm dofs

        config["limits"] = {}
        config["limits"]["lower"] = torch.tensor(dof_lower_limits).float().to(self.device)
        config["limits"]["upper"] = torch.tensor(dof_upper_limits).float().to(self.device)

        config["init"] = {}
        config["init"]["position"] = torch.tensor(dof_init_positions).float().to(self.device)
        config["init"]["velocity"] = torch.tensor(dof_init_velocities).float().to(self.device)
        # print("dof_init_positions:", dof_init_positions)
        # print("dof_init_velocities:", dof_init_velocities)

        if self.enable_contact_sensors:
            self.__define_contact_sensors(asset)
        self.__configure_robot_dof_indices(asset)

        # fmt: off
        close_dof_names = [
            "joint_2.0", "joint_3.0",  # finger 0 (index)
            "joint_6.0", "joint_7.0",  # finger 1 (middle)
            "joint_10.0", "joint_11.0",  # finger 2 (ring)
            "joint_14.0", "joint_15.0",  # thumb
        ]
        # fmt: on

        self.close_dof_indices = torch.tensor(
            [self.gym.find_asset_dof_index(asset, name) for name in close_dof_names],
            dtype=torch.long,
            device=self.device,
        )

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self._xarm_right_init_position)
        pose.r = gymapi.Quat(*self._xarm_right_init_orientation)

        self.allegro_center_index = self.gym.find_asset_rigid_body_index(asset, self._allegro_hand_center_prim)
        self.allegro_palm_index = self.gym.find_asset_rigid_body_index(asset, self._allegro_hand_palm_prim)
        self.fingertip_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._fingertips]
        self.keypoint_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._keypoints]
        self.arm_link_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._arm_links]
        self.hand_link_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._hand_links]
        self.index_finger_link_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._index_finger_links]
        self.middle_finger_link_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._middle_finger_links]
        self.ring_finger_link_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._ring_finger_links]
        self.thumb_link_indices = [self.gym.find_asset_rigid_body_index(asset, prim) for prim in self._thumb_links]
        
        self.hand_link_indices_map = {link_name: self.gym.find_asset_rigid_body_index(asset, link_name) for link_name in self._keypoints}
        self.keypoint_offset = [self._keypoints_info[link_name] for link_name in self._keypoints] # (#key_link, 1, 3)
        self.thumb_link_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._thumb_links]
        self.index_link_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._index_finger_links]
        self.middle_link_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._middle_finger_links]
        self.ring_link_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._ring_finger_links]
        self.fingertip_link_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._fingertips]

        config["asset"] = asset
        config["pose"] = pose
        config["dof_props"] = dof_props

        print(">>> xArm6 + Allegro Hand loaded")
        return config

    def _define_object(self, dataset: str = "boxes") -> Dict[str, Any]:
        """Define & load objects for the current scene.

        For singulation task, we create a grid of boxes instead of loading dataset objects.

        Args:
            dataset (str, optional): Dataset type. Defaults to 'boxes'.

        Returns:
            Dict[str, Any]: The configuration of the objects.
        """
        return self._create_box_grid()

    def _define_object_deprecated(self, dataset: str = "boxes") -> Dict[str, Any]:
        """Define & load objects for the current scene.

        For singulation task, we create a grid of boxes instead of loading dataset objects.

        Args:
            dataset (str, optional): Dataset type. Defaults to 'boxes'.

        Returns:
            Dict[str, Any]: The configuration of the objects.
        """
        print(">>> Loading objects for current scene")
        config = {}
        config["warehouse"] = []

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        # asset_options.override_com = True
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params.resolution = 300000
        # asset_options.vhacd_params.max_convex_hulls = 10
        # asset_options.vhacd_params.max_num_vertices_per_ch = 64

        # load assets to memory

        if self.resample_object:
            # resample to original distribution
            if self.manipulated_object_codes is None:
                object_codes = self.grasping_dataset.resample(self.num_envs * self.num_objects_per_env)
                self.manipulated_object_codes = object_codes
            else:
                object_codes = self.manipulated_object_codes
        else:
            # select the first-k objects
            object_codes = self.grasping_dataset.manipulated_codes

        loaded = {}
        for i, name in enumerate(object_codes):
            if name in loaded:
                cfg = config["warehouse"][loaded[name]].copy()
            else:
                loaded[name] = i
                asset_filename = os.path.join(dataset, name, "decomposed.urdf")
                asset = self.gym.load_asset(self.sim, self._asset_root, asset_filename, asset_options)

                # set rigid-shape properties
                rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
                for shape in rigid_shape_props:
                    shape.friction = 3.0
                self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

                cfg = {"name": name, "asset": asset}
                cfg["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(asset)
                cfg["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(asset)
            config["warehouse"].append(cfg)
        config["count"] = len(config["warehouse"])

        num_rigid_bodies = [cfg["num_rigid_bodies"] for cfg in config["warehouse"]]
        num_rigid_shapes = [cfg["num_rigid_shapes"] for cfg in config["warehouse"]]
        config["num_rigid_bodies"] = sum(sorted(num_rigid_bodies, reverse=True)[: self.num_objects_per_env])
        config["num_rigid_shapes"] = sum(sorted(num_rigid_shapes, reverse=True)[: self.num_objects_per_env])

        # define object poses (unused and occupied)
        unused_pose = gymapi.Transform()
        unused_pose.p = gymapi.Vec3(0.0, 0.0, 0.1)

        occupied_pose = gymapi.Transform()
        if test_sim:
            occupied_pose.p = gymapi.Vec3(0.0, 0.2, 0.7)
        else:
            occupied_pose.p = gymapi.Vec3(0.0, 0.0, 0.7)

        num_objects_per_row = int(np.sqrt(self.num_objects_per_env))

        config["poses"] = []
        for i in range(self.num_objects_per_env):
            row, col = i // num_objects_per_row, i % num_objects_per_row

            x = unused_pose.p.x
            y = unused_pose.p.y
            z = unused_pose.p.z

            x += col * self.object_spacing
            y += row * self.object_spacing

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, y, z)
            config["poses"].append(pose)
        config["occupied_pose"] = occupied_pose

        print(">>> Objects loaded")
        return config

    def _create_box_grid(self) -> Dict[str, Any]:
        """Create a grid of boxes for singulation task.

        Returns:
            Dict[str, Any]: Configuration for the box grid
        """
        print(">>> Creating box grid for singulation task")

        config = {}
        config["warehouse"] = {
            "targ_obj": [],
            "surr_obj": [],
        }

        target_asset_options = gymapi.AssetOptions()
        target_asset_options.density = 1000.0
        target_asset_options.convex_decomposition_from_submeshes = True
        target_asset_options.override_com = True
        target_asset_options.override_inertia = True
        
        surrounding_asset_options = gymapi.AssetOptions()
        surrounding_asset_options.density = 500.0
        surrounding_asset_options.convex_decomposition_from_submeshes = True
        surrounding_asset_options.override_com = True
        surrounding_asset_options.override_inertia = True
        surrounding_asset_options.disable_gravity = True
        surrounding_asset_options.fix_base_link = True

        target_box_asset = self.gym.create_box(self.sim, self._obj_width, self._obj_depth, self._obj_height, target_asset_options)
        surrounding_box_asset = self.gym.create_box(self.sim, self._obj_width, self._obj_depth, self._obj_height, surrounding_asset_options)

        _targ_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(target_box_asset)
        _surr_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(surrounding_box_asset)
        for shape in _targ_rigid_shape_props:
            shape.friction = 0.8
            shape.restitution = 0.1
        for shape in _surr_rigid_shape_props:
            shape.friction = 1e-5
            shape.restitution = 0.1
        self.gym.set_asset_rigid_shape_properties(target_box_asset, _targ_rigid_shape_props)
        self.gym.set_asset_rigid_shape_properties(surrounding_box_asset, _surr_rigid_shape_props)

        config["warehouse"]["targ_obj"].append({
            "name": "target_box",
            "asset": target_box_asset,
            "num_rigid_bodies": self.gym.get_asset_rigid_body_count(target_box_asset),
            "num_rigid_shapes": self.gym.get_asset_rigid_shape_count(target_box_asset),
        })
        config["warehouse"]["surr_obj"].append({
            "name": "surrounding_box",
            "asset": surrounding_box_asset,
            "num_rigid_bodies": self.gym.get_asset_rigid_body_count(surrounding_box_asset),
            "num_rigid_shapes": self.gym.get_asset_rigid_shape_count(surrounding_box_asset),
        })
        config["count"] = len(config["warehouse"]["targ_obj"])

        num_rigid_bodies = [cfg_targ["num_rigid_bodies"] + cfg_surr["num_rigid_bodies"] for cfg_targ, cfg_surr in zip(config["warehouse"]["targ_obj"], config["warehouse"]["surr_obj"])]
        num_rigid_shapes = [cfg_targ["num_rigid_shapes"] + cfg_surr["num_rigid_shapes"] for cfg_targ, cfg_surr in zip(config["warehouse"]["targ_obj"], config["warehouse"]["surr_obj"])]
        config["num_rigid_bodies"] = sum(sorted(num_rigid_bodies, reverse=True)[: self.num_objects_per_env])
        config["num_rigid_shapes"] = sum(sorted(num_rigid_shapes, reverse=True)[: self.num_objects_per_env])

        config["poses"] = self.__generate_object_poses()

        print(f">>> Box grid created with {len(config['warehouse'])} box assets")
        return config

    def __generate_object_poses(self) -> List[gymapi.Transform]:
        """Generate poses for boxes in a grid pattern on the table.

        Returns:
            List[gymapi.Transform]: List of poses for each box
        """
        poses = []

        # Calculate grid center position on table
        table_center_x = self._table_pose[0]
        table_center_y = self._table_pose[1]
        table_top_z = self._table_pose[2] + (self._table_thickness + self._obj_height) / 2 + 1e-3

        # Calculate grid dimensions
        grid_width = self._grid_cols * self._obj_width + (self._grid_cols - 1) * self._obj_spacing
        grid_depth = self._grid_rows * self._obj_depth + (self._grid_rows - 1) * self._obj_spacing
        grid_height = self._grid_layers * self._obj_height + (self._grid_layers - 1) * self._obj_spacing

        self.grid_width = grid_width
        self.grid_depth = grid_depth
        self.grid_height = grid_height

        # Starting position (top-left corner of grid)
        start_x = table_center_x - grid_width / 2 + self._obj_width / 2
        start_y = table_center_y - grid_depth / 2 + self._obj_depth / 2
        if self.object_near_edge:
            start_y = table_center_y + self._table_y_length / 2 - self._obj_depth / 2 - 0.01
        start_z = table_top_z
        
        # Generate poses for each box in the grid
        for i in range(self.num_objects_per_env):
            col = i % self._grid_cols
            row = (i // self._grid_cols) % self._grid_rows
            layer = i // (self._grid_cols * self._grid_rows)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                start_x + col * (self._obj_width + self._obj_spacing),
                start_y + row * (self._obj_depth + self._obj_spacing),
                start_z + layer * (self._obj_height + self._obj_spacing)
            )
            pose.r = gymapi.Quat(0, 0, 0, 1)  # No rotation

            poses.append(pose)

        return poses

    def __define_target_allegro_hand(self, asset_name: str = "Target allegro Hand") -> Dict[str, Any]:
        """Define & load the target allegro Hand.

        Args:
            asset_name (str, optional): Asset name for logging. Defaults to "Target allegro Hand".

        Returns:
            Dict[str, Any]: The configuration of the target allegro Hand.
        """
        print(">>> Loading allegro Hand for target scene")
        config = {"name": "target_allegro_hand"}

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        if self.env_info_logging:
            print_asset_options(asset_options, asset_name)

        asset = self.gym.load_asset(self.sim, self._asset_root, self._allegro_hand_right_asset_file, asset_options)
        if self.env_info_logging:
            print_links_and_dofs(self.gym, asset, asset_name)

        config["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(asset)
        config["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(asset)
        config["num_dofs"] = self.gym.get_asset_dof_count(asset)
        config["num_actuators"] = self.gym.get_asset_actuator_count(asset)
        config["num_tendons"] = self.gym.get_asset_tendon_count(asset)

        dof_props = self.gym.get_asset_dof_properties(asset)
        for i in range(config["num_dofs"]):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = 3.0
            dof_props["damping"][i] = 0.0
        if self.env_info_logging:
            print_dof_properties(self.gym, asset, dof_props, asset_name)

        self.target_allegro_center_index = self.gym.find_asset_rigid_body_index(asset, self._allegro_hand_center_prim)
        self.target_fingertip_indices = [
            self.gym.find_asset_rigid_body_index(asset, f"rh_{prim}") for prim in self._fingertips
        ]

        pose = gymapi.Transform()

        if self.save_video:
            pose.p = gymapi.Vec3(-0.4 + video_pose[0], 0.3 + video_pose[1], 0.8 + video_pose[2])
        else:
            pose.p = gymapi.Vec3(-0.4, 0.3, 0.8)

        pose.r = gymapi.Quat(0.0, -np.sqrt(0.5), np.sqrt(0.5), 0.0)

        config["asset"] = asset
        config["pose"] = pose
        config["dof_props"] = dof_props

        print(">>> Target allegro Hand loaded")

        return config

    def _define_visual_target_object(self, asset_name: str = "Visual Target Object") -> Dict[str, Any]:
        """Define a visual-only asset to represent the goal position in the environment.

        Args:
            asset_name (str, optional): Name for the asset. Defaults to "Visual Target Object".

        Returns:
            Dict[str, Any]: Configuration dictionary for the visual target object.
        """
        print(f">>> Loading {asset_name}")
        config = {"name": "visual_target_object"}

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True

        # asset = self.gym.create_box(self.sim, self._obj_width, self._obj_depth, self._obj_height, asset_options)
        asset = self.gym.create_sphere(self.sim, 0.02, asset_options)

        num_rigid_bodies = self.gym.get_asset_rigid_body_count(asset)
        num_rigid_shapes = self.gym.get_asset_rigid_shape_count(asset)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.goal_position[0], self.goal_position[1], self.goal_position[2])   
        pose.r = gymapi.Quat(self.goal_orientation[0], self.goal_orientation[1], self.goal_orientation[2], self.goal_orientation[3])

        config["asset"] = asset
        config["pose"] = pose
        config["num_rigid_bodies"] = num_rigid_bodies
        config["num_rigid_shapes"] = num_rigid_shapes

        print(f">>> {asset_name} loaded")
        return config

    def _define_camera(self) -> None:
        """Define the cameras for the rendering."""
        if not self.enable_rendered_pointcloud_observation and not self.save_video:
            return

        self._camera_positions = [
            gymapi.Vec3(0.330799, 0.661600, 0.371473), # left
            gymapi.Vec3(-0.016071, 0.462980, 1.126893), # top
        ]

        self._camera_target_locations = [
            gymapi.Vec3(-0.067626, -0.169646, 0.759142), # left
            gymapi.Vec3(-0.025861, -0.219982, 0.396504), # top
        ]


        # self._camera_quaternions = [
        #     gymapi.Quat(-0.149580, -0.532721, 0.820501, 0.143569), # left
        #     gymapi.Quat(0.929895, -0.022131, 0.003473, 0.367142), # top
        # ]

        self._camera_quaternions = [
            gymapi.Quat(0.140885, 0.146895, 0.823185, -0.530037), # left
            gymapi.Quat(-0.639190, 0.657847, 0.268574, 0.294178), # top
        ]


        assert len(self._camera_positions) == len(self._camera_target_locations)
        self.num_cameras_per_env = len(self._camera_positions)

        # allocate tensors for camera data
        self.cameras = [[] for _ in range(self.num_envs)]
        self.camera_tensors = []
        self.camera_seg_tensors = []
        self.camera_positions = torch.zeros((self.num_envs, self.num_cameras_per_env, 3), device=self.device)
        self.camera_orientations = torch.zeros((self.num_envs, self.num_cameras_per_env, 4), device=self.device)
        self.camera_inv_view_matrices = torch.zeros((self.num_envs, self.num_cameras_per_env, 4, 4), device=self.device)
        self.camera_proj_matrices = torch.zeros((self.num_envs, self.num_cameras_per_env, 4, 4), device=self.device)

        # define camera properties
        self.camera_properties = gymapi.CameraProperties()
        self.camera_properties.horizontal_fov = 58.0
        self.camera_properties.width = 640
        self.camera_properties.height = 480
        self.camera_properties.enable_tensors = True
        self.camera_properties.use_collision_geometry = True

        # define related indices for pointcloud computation
        self.camera_u = torch.arange(0, self.camera_properties.width, device=self.device)
        self.camera_v = torch.arange(0, self.camera_properties.height, device=self.device)
        self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing="ij")

        # define bounding box corners for pointcloud computation
        self.render_pointcloud_bbox_corners = (
            torch.tensor([-self._table_x_length / 2, -self._table_y_length / 2, 0.34], device=self.device),
            torch.tensor([self._table_x_length / 2, self._table_y_length / 2, 1.20], device=self.device),
        )
        
        self.render_pointcloud_bbox_corners = (
            torch.tensor([
                -self._table_x_length / 2, 
                -self._table_y_length / 2, 
                self._table_pose[2] + self._table_thickness * 0.5], 
            device=self.device) + 5e-3,
            torch.tensor([
                self._table_x_length / 2, 
                self._table_y_length / 2 + 0.5, 
                self._upper_shelf_pos[2] - self._table_thickness * 0.5], 
            device=self.device) - 5e-3
        )

    def _create_box_grid_dataset(self, device=None) -> None:
        # Create simple box grid dataset for singulation task
        from .dataset import BoxGridDataset

        self.grasping_dataset = BoxGridDataset(
            grid_rows=self._grid_rows,
            grid_cols=self._grid_cols,
            grid_layers=self._grid_layers,
            box_width=self._obj_width,
            box_depth=self._obj_depth,
            box_height=self._obj_height,
            device=device,
        )

        self.num_categories = self.grasping_dataset._category_matrix.shape[1]

    def __reset_grasping_joint_indices(self) -> None:
        # if "target" in self.gym_assets and "robot" in self.gym_assets["target"]:
        #     asset = self.gym_assets["target"]["robot"]["asset"]
        # else:
        #     asset = self.__define_target_allegro_hand()["asset"]

        asset = self.gym_assets["target"]["robot"]["asset"]

        indices = [self.gym.find_asset_dof_index(asset, name) for name in self.grasping_dataset.dof_names]
        print("grasping dataset joints:", self.grasping_dataset.dof_names)
        self.grasping_joint_indices = torch.tensor(indices).long().to(self.device)

    def _reset_action_indices(self) -> None:
        (
            self.arm_trans_action_indices,
            self.arm_rot_action_indices,
            self.arm_roll_action_indices,
            self.hand_action_indices,
        ) = get_action_indices(self._action_space, device=self.device)

    def _create_sim_actor(
        self,
        env: gymapi.Env,
        config: Dict[str, Any],
        group: int,
        name: Optional[str] = None,
        pose: Optional[gymapi.Transform] = None,
        color: Optional[gymapi.Vec3] = None,
        actor_handle: Optional[bool] = False,
        filter:int = 0
    ) -> int:
        """Create an `Actor` in the simulator.

        Args:
            env (gymapi.Env): The environment to create the actor in.
            config (Dict[str, Any]): The configuration of the actor.
            group (int): The collision group of the actor.
            name (Optional[str], optional): The name of the actor. Defaults to None.
            pose (Optional[gymapi.Transform], optional): The pose of the actor. Defaults to None.
            color (Optional[gymapi.Vec3], optional): The color of the actor. Defaults to None.

        Returns:
            int: The index of the actor. (Domain: gymapi.DOMAIN_SIM)
        """
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

    def compute_maximum_aggregate_bodies_and_shapes(self, gym_assets: Optional[Dict] = None) -> Tuple[int, int]:
        """Compute the maximum number of rigid bodies and shapes in the environment.

        fetch `num_rigid_bodies` and `num_rigid_shapes` from the `gym_assets` dict.
        Args:
            gym_assets (Optional[Dict], optional): The gym assets to compute. Defaults to None.
                if None, use `self.gym_assets`.

        Returns:
            Tuple[int, int]: The maximum number of rigid bodies and shapes.
        """
        max_aggregate_bodies, max_aggregate_shapes = 0, 0
        for i in range(self.num_envs):
            num_bodies, num_shapes = self.compute_aggregate_bodies_and_shapes(i, gym_assets)
            max_aggregate_bodies = max(max_aggregate_bodies, num_bodies)
            max_aggregate_shapes = max(max_aggregate_shapes, num_shapes)
        return max_aggregate_bodies, max_aggregate_shapes

    def compute_aggregate_bodies_and_shapes(self, env: int, gym_assets: Optional[Dict] = None) -> Tuple[int, int]:
        """Compute the number of rigid bodies and shapes in the environment.

        Args:
            env (int): The index of the environment.
            gym_assets (Optional[Dict], optional): The gym assets to compute. Defaults to None.
                if None, use `self.gym_assets`.

        Returns:
            Tuple[int, int]: The number of rigid bodies and shapes in the environment.
        """
        if gym_assets is None:
            gym_assets = self.gym_assets

        num_bodies, num_shapes = 0, 0

        num_bodies += gym_assets["current"]["robot"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["robot"]["num_rigid_shapes"]
        num_current_objects = gym_assets["current"]["objects"]["count"]
        
        num_bodies += gym_assets["current"]["objects"]["warehouse"]["targ_obj"][(env * self.num_objects_per_env) % num_current_objects]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["objects"]["warehouse"]["targ_obj"][(env * self.num_objects_per_env) % num_current_objects]["num_rigid_shapes"]

        for i in range(1, self.num_objects_per_env):
            cur = (env * self.num_objects_per_env + i) % num_current_objects

            num_bodies += gym_assets["current"]["objects"]["warehouse"]["surr_obj"][cur]["num_rigid_bodies"]
            num_shapes += gym_assets["current"]["objects"]["warehouse"]["surr_obj"][cur]["num_rigid_shapes"]

        num_bodies += gym_assets["current"]["table"]["num_rigid_bodies"]
        num_shapes += gym_assets["current"]["table"]["num_rigid_shapes"]
        
        if self.add_visual_target_object:
            num_bodies += gym_assets["current"]["visual_target_object"]["num_rigid_bodies"]
            num_shapes += gym_assets["current"]["visual_target_object"]["num_rigid_shapes"]

        if self.use_upper_shelf:
            num_bodies += gym_assets["current"]["upper_shelf"]["num_rigid_bodies"]
            num_shapes += gym_assets["current"]["upper_shelf"]["num_rigid_shapes"]

        if self.use_back_wall:
            num_bodies += gym_assets["current"]["back_wall"]["num_rigid_bodies"]
            num_shapes += gym_assets["current"]["back_wall"]["num_rigid_shapes"]

        if self.use_side_walls:
            num_bodies += gym_assets["current"]["side_wall_pos_x"]["num_rigid_bodies"]
            num_shapes += gym_assets["current"]["side_wall_pos_x"]["num_rigid_shapes"]
            num_bodies += gym_assets["current"]["side_wall_neg_x"]["num_rigid_bodies"]
            num_shapes += gym_assets["current"]["side_wall_neg_x"]["num_rigid_shapes"]

        # num_bodies += gym_assets["current"]["objects"]["warehouse"][0]["num_rigid_bodies"]
        # num_shapes += gym_assets["current"]["objects"]["warehouse"][0]["num_rigid_shapes"]

        return num_bodies, num_shapes

    def _create_envs(self, num_envs: int, spacing: float, num_objects_per_env: int = 1):
        print(">>> Setting up %d environments" % num_envs)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(num_envs))

        print(">>> Defining gym assets")

        self.gym_assets["current"]["robot"] = self._define_allegro_hand_with_arm()
        self.gym_assets["current"]["objects"] = self._define_object()
        self.gym_assets["current"]["table"] = self._define_table()
        if self.add_visual_target_object:
            self.gym_assets["current"]["visual_target_object"] = self._define_visual_target_object()
        if self.use_upper_shelf:
            self.gym_assets["current"]["upper_shelf"] = self._define_upper_shelf()
        if self.use_back_wall:
            self.gym_assets["current"]["back_wall"] = self._define_back_wall()
        if self.use_side_walls:
            self.gym_assets["current"]["side_wall_pos_x"] = self._define_side_wall(+1.0, "side_wall_pos_x")
            self.gym_assets["current"]["side_wall_neg_x"] = self._define_side_wall(-1.0, "side_wall_neg_x")

        # self.gym_assets["target"]["robot"] = self.__define_target_allegro_hand()


        self._define_camera()

        print(">>> Done defining gym assets")

        max_aggregate_bodies, max_aggregate_shapes = self.compute_maximum_aggregate_bodies_and_shapes()

        self.envs = []
        self.cameras_handle = []

        allegro_hand_indices = []
        table_indices = []
        upper_shelf_indices = [] if self.use_upper_shelf else None
        back_wall_indices = [] if self.use_back_wall else None
        side_wall_pos_x_indices = [] if self.use_side_walls else None
        side_wall_neg_x_indices = [] if self.use_side_walls else None
        visual_target_object_indices = [] if self.add_visual_target_object else None
        object_indices = [[] for _ in range(num_envs)]
        object_encodings = [[] for _ in range(num_envs)]
        object_names = [[] for _ in range(num_envs)]
        occupied_object_indices = []
        surr_object_indices = []
        non_occupied_object_indices = [[] for _ in range(num_envs)]
        scene_object_indices = [[] for _ in range(num_envs)]
        occupied_object_indices_per_env = [random.randint(0, self.num_objects_per_env - 1) for _ in range(num_envs)]
        # occupied_object_indices_per_env = [1 for _ in range(num_envs)]

        print(">>> Creating environments")
        print("    - max_aggregate_bodies: ", max_aggregate_bodies)
        print("    - max_aggregate_shapes: ", max_aggregate_shapes)

        self.scene_xy_offsets = torch.zeros((num_envs, 2), device=self.device)

        if self.random_object_position:
            # assert self.num_objects_per_env <= 3
            self.rand_x = (torch.rand(num_envs, device=self.device) - 0.5) * (self._table_x_length - self.grid_width)
            self.rand_y = (torch.rand(num_envs, device=self.device) - 0.5) * (self._table_y_length - self.grid_depth)

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.aggregate_tracker.reset()

            if self.aggregate_mode != 0:
                num_bodies, num_shapes = self.compute_aggregate_bodies_and_shapes(i)
                agg_success = self.gym.begin_aggregate(env, max_aggregate_bodies * 1, max_aggregate_shapes * 1, True)
                if not agg_success:
                    raise RuntimeError("begin_aggregate failed")

            # add allegro hand to the environment
            actor_index, actor_handle = self._create_sim_actor(
                env, self.gym_assets["current"]["robot"], i, actor_handle=True, filter=-1
            )
            allegro_hand_indices.append(actor_index)
            if self.randomize:
                self.apply_domain_rand(env_ptr=env, actor_handle=actor_handle, friction=True, com=False)

            poses = self.gym_assets["current"]["objects"]["poses"]
            surr_obj_cur_idx = 0

            

            for k in range(self.num_objects_per_env):
                is_target = (k == occupied_object_indices_per_env[i])
                cfg = self.gym_assets["current"]["objects"]["warehouse"]["targ_obj"][k % len(self.gym_assets["current"]["objects"]["warehouse"]["targ_obj"])] if is_target else self.gym_assets["current"]["objects"]["warehouse"]["surr_obj"][k % len(self.gym_assets["current"]["objects"]["warehouse"]["surr_obj"])]
                pose = poses[k]

                if self.random_object_position:

                    rand_x_i = self.rand_x[i]
                    rand_y_i = self.rand_y[i]
                    self.scene_xy_offsets[i, 0] = rand_x_i
                    self.scene_xy_offsets[i, 1] = rand_y_i
                    base = poses[k]
                    pose = gymapi.Transform()
                    pose.p = gymapi.Vec3(base.p.x + rand_x_i, base.p.y + rand_y_i, base.p.z)
                    pose.r = base.r
                
                surr_obj_color = gymapi.Vec3(0.9, 0.0, 0.0)
                targ_obj_color = gymapi.Vec3(0.9, 0.9, 0.9)

                if is_target:
                    # actor_index = self._create_sim_actor(env, cfg, i, "targ_obj", pose, color=targ_obj_color)
                    actor_index, targ_actor_handle = self._create_sim_actor(
                        env, cfg, i, "targ_obj", pose, color=targ_obj_color, actor_handle=True, filter=-1
                    )

                    randomize_mass_lower, randomize_mass_upper = self.randomize_mass_lower, self.randomize_mass_upper

                    prop = self.gym.get_actor_rigid_body_properties(env, targ_actor_handle)
                    # print(f"Current mass: {prop[0].mass}")
                    for p in prop:
                        p.mass = np.random.uniform(randomize_mass_lower, randomize_mass_upper)
                    # print(f"Applying random mass: {prop[0].mass}")
                    self.gym.set_actor_rigid_body_properties(env, targ_actor_handle, prop)
                    # print(f"new mass: {self.gym.get_actor_rigid_body_properties(env, targ_actor_handle)[0].mass}")

                    num_rb = self.gym.get_actor_rigid_body_count(env, targ_actor_handle)
                    for rb_i in range(num_rb):
                        self.gym.set_rigid_body_segmentation_id(
                            env, targ_actor_handle, rb_i, self.target_segmentation_id
                        )

                    if self.randomize:
                        self.apply_domain_rand(env, targ_actor_handle, friction=True, com=True)
                
                else:
                    surr_obj_name = f"sur_obj_{surr_obj_cur_idx}"
                    surr_obj_cur_idx += 1
                    actor_index = self._create_sim_actor(env, cfg, i, surr_obj_name, pose, color=surr_obj_color, filter=-1)


                object_indices[i].append(actor_index)
                object_names[i].append(cfg["name"])
                object_encodings[i].append(k)

                if is_target:
                    occupied_object_indices.append(actor_index)  # global actor index for root_states access
                else:
                    non_occupied_object_indices[i].append(k)  #relative index within environment
                    surr_object_indices.append(actor_index)

                scene_object_indices[i].append(k)  # relative index within environment

            # add table to the environment
            actor_index, actor_handle = self._create_sim_actor(
                env, self.gym_assets["current"]["table"], i, actor_handle=True, filter=-1
            )
            table_indices.append(actor_handle)
            if self.randomize:
                self.apply_domain_rand(env_ptr=env, actor_handle=actor_handle, friction=True, com=False)


            if self.use_side_walls:
                actor_index, actor_handle = self._create_sim_actor(
                    env, self.gym_assets["current"]["side_wall_pos_x"], i, actor_handle=True, filter=-1
                )
                side_wall_pos_x_indices.append(actor_handle)

                actor_index, actor_handle = self._create_sim_actor(
                    env, self.gym_assets["current"]["side_wall_neg_x"], i, actor_handle=True, filter=-1
                )
                side_wall_neg_x_indices.append(actor_handle)


            if self.use_back_wall:
                actor_index, actor_handle = self._create_sim_actor(
                    env, self.gym_assets["current"]["back_wall"], i, actor_handle=True, filter=-1
                )
                back_wall_indices.append(actor_handle)
            
            if self.use_upper_shelf:
                actor_index, actor_handle = self._create_sim_actor(
                    env, self.gym_assets["current"]["upper_shelf"], i, actor_handle=True, filter=-1
                )
                upper_shelf_indices.append(actor_handle)

            if self.add_visual_target_object:
                # add visual target object to the environment
                actor_index, actor_handle = self._create_sim_actor(
                    env, self.gym_assets["current"]["visual_target_object"], i + self.num_envs, actor_handle=True, color=gymapi.Vec3(0.6, 0.72, 0.98)
                )
                visual_target_object_indices.append(actor_handle)


            if self.enable_rendered_pointcloud_observation or self.save_video:
                for k in range(self.num_cameras_per_env):
                    camera = self.gym.create_camera_sensor(env, self.camera_properties)
                    self.cameras_handle.append(camera)

                    transform = gymapi.Transform()
                    transform.p = self._camera_positions[k] 
                    transform.r = self._camera_quaternions[k]
                    self.gym.set_camera_transform(
                        camera, env, transform
                    )
                    # self.gym.set_camera_location(
                    #     camera, env, self._camera_positions[k], self._camera_target_locations[k]
                    # )
                    depth_image = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera, gymapi.IMAGE_DEPTH)
                    depth_image = gymtorch.wrap_tensor(depth_image)

                    seg_image = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera, gymapi.IMAGE_SEGMENTATION)
                    seg_image = gymtorch.wrap_tensor(seg_image)

                    view_matrix = self.gym.get_camera_view_matrix(self.sim, env, camera)
                    proj_matrix = self.gym.get_camera_proj_matrix(self.sim, env, camera)

                    view_matrix = torch.tensor(view_matrix).to(self.device)
                    proj_matrix = torch.tensor(proj_matrix).to(self.device)
                    inv_view_matrix = torch.inverse(view_matrix)

                    origin: gymapi.Vec3 = self.gym.get_env_origin(env)
                    inv_view_matrix[3][0] -= origin.x
                    inv_view_matrix[3][1] -= origin.y
                    inv_view_matrix[3][2] -= origin.z

                    # the `inv_view_matrix` is a transposed version of transformation matrix
                    # the quaternions are in the order of (w, x, y, z) in pytorch3d, need to be converted to (x, y, z, w)
                    camera_position = inv_view_matrix[3, :3]
                    camera_orientation = matrix_to_quaternion(inv_view_matrix[:3, :3].T)
                    camera_orientation = torch.cat([camera_orientation[1:], camera_orientation[:1]])

                    self.cameras[i].append(camera)
                    self.camera_tensors.append(depth_image)
                    self.camera_seg_tensors.append(seg_image)
                    self.camera_inv_view_matrices[i, k] = inv_view_matrix
                    self.camera_proj_matrices[i, k] = proj_matrix
                    self.camera_positions[i, k] = camera_position
                    self.camera_orientations[i, k] = camera_orientation
                    if self.env_info_logging:
                        print("view_matrix: ", view_matrix)
                        print("proj_matrix: ", proj_matrix)
                        print("image.shape: ", depth_image.shape)

            # if i==0:
            #     self.test_pcl(env, 0)

            if self.aggregate_mode != 0:
                agg_success = self.gym.end_aggregate(env)
                if not agg_success:
                    raise RuntimeError("end_aggregate failed")

                assert self.aggregate_tracker.aggregate_bodies == num_bodies
                assert self.aggregate_tracker.aggregate_shapes == num_shapes

            self.envs.append(env)

        print(f">>> Done creating {num_envs} environments")

        allegro_hand = self.gym.find_actor_handle(env, "allegro_hand")
        self.allegro_hand_index = self.gym.get_actor_index(env, allegro_hand, gymapi.DOMAIN_ENV)
        # breakpoint()
        
        # Object Rigid Body Index Tracking
        self.target_object_rigid_body_indices = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.surr_object_rigid_body_indices = torch.zeros((num_envs, self.max_non_targets), dtype=torch.long, device=self.device)
        
        for i in range(num_envs):
            env = self.envs[i]
            
            # Get target object rigid body index
            target_obj_handle = self.gym.find_actor_handle(env, "targ_obj")
            target_object_actor = self.gym.get_actor_index(env, target_obj_handle, gymapi.DOMAIN_ENV)
            target_rb_index = self.gym.get_actor_rigid_body_index(
                env, target_object_actor, 0, gymapi.DOMAIN_SIM
            )
            self.target_object_rigid_body_indices[i] = target_rb_index
            
            # Get surrounding object rigid body indices surr_object_indices to [env_id, max_non_targets]
            for j in range(self.max_non_targets):
                surr_obj_handle = self.gym.find_actor_handle(env, f"sur_obj_{j}")
                surr_object_actor = self.gym.get_actor_index(env, surr_obj_handle, gymapi.DOMAIN_ENV)
                surr_rb_index = self.gym.get_actor_rigid_body_index(
                    env, surr_object_actor, 0, gymapi.DOMAIN_SIM
                )
                self.surr_object_rigid_body_indices[i, j] = surr_rb_index

        # define start and end indices for allegro hand DOFs to create contiguous slices
        self.allegro_hand_dof_start = self.gym.get_actor_dof_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_dof_end = self.allegro_hand_dof_start + self.gym_assets["current"]["robot"]["num_dofs"]
        self.allegro_hand_indices = torch.tensor(allegro_hand_indices).long().to(self.device)
        self.allegro_hand_rigid_body_start = self.gym.get_actor_rigid_body_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_rigid_body_end = (
            self.allegro_hand_rigid_body_start + self.gym_assets["current"]["robot"]["num_rigid_bodies"]
        )


        self.table_indices = torch.tensor(table_indices).long().to(self.device)
        self.visual_target_object_indices = torch.tensor(visual_target_object_indices).long().to(self.device) if self.add_visual_target_object else None
        self.upper_shelf_indices = torch.tensor(upper_shelf_indices).long().to(self.device) if self.use_upper_shelf else None
        self.back_wall_indices = torch.tensor(back_wall_indices).long().to(self.device) if self.use_back_wall else None
        self.side_wall_pos_x_indices = torch.tensor(side_wall_pos_x_indices).long().to(self.device) if self.use_side_walls else None
        self.side_wall_neg_x_indices = torch.tensor(side_wall_neg_x_indices).long().to(self.device) if self.use_side_walls else None
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

        if self.use_back_wall:
            self.back_wall_rigid_body_indices = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
            for i in range(self.num_envs):
                env_i = self.envs[i]
                wall_handle_i = int(self.back_wall_indices[i].item())
                wall_actor_index = self.gym.get_actor_index(env_i, wall_handle_i, gymapi.DOMAIN_ENV)
                wall_rb_index = self.gym.get_actor_rigid_body_index(env_i, wall_actor_index, 0, gymapi.DOMAIN_SIM)
                self.back_wall_rigid_body_indices[i] = wall_rb_index

        if self.use_side_walls:
            self.side_wall_pos_x_rigid_body_indices = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
            self.side_wall_neg_x_rigid_body_indices = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

            for i in range(self.num_envs):
                env_i = self.envs[i]

                pos_handle_i = int(self.side_wall_pos_x_indices[i].item())
                pos_actor_index = self.gym.get_actor_index(env_i, pos_handle_i, gymapi.DOMAIN_ENV)
                pos_rb_index = self.gym.get_actor_rigid_body_index(env_i, pos_actor_index, 0, gymapi.DOMAIN_SIM)
                self.side_wall_pos_x_rigid_body_indices[i] = pos_rb_index

                neg_handle_i = int(self.side_wall_neg_x_indices[i].item())
                neg_actor_index = self.gym.get_actor_index(env_i, neg_handle_i, gymapi.DOMAIN_ENV)
                neg_rb_index = self.gym.get_actor_rigid_body_index(env_i, neg_actor_index, 0, gymapi.DOMAIN_SIM)
                self.side_wall_neg_x_rigid_body_indices[i] = neg_rb_index

        self.occupied_object_indices = (torch.tensor(occupied_object_indices).long().to(self.device))  # (env_id) - global actor indices
        self.occupied_object_relative_indices = (torch.tensor(occupied_object_indices_per_env).long().to(self.device))  # (env_id) - relative indices 0 to num_objects_per_env-1
        self.non_occupied_object_indices = (torch.tensor(non_occupied_object_indices).long().to(self.device))  # (env_id, max_non_targets)
        self.surr_object_indices = (torch.tensor(surr_object_indices).long().to(self.device))  # (env_id, max_non_targets)
        self.scene_object_indices = (torch.tensor(scene_object_indices).long().to(self.device))  # (env_id, object_id)
        # fmt off

    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.up_axis == "z" else 1

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"])

        # Cache target object mass and inertia (diagonal) for random wrench scaling
        self.target_masses = torch.zeros((self.num_envs,), device=self.device)
        self.target_inertias = torch.zeros((self.num_envs, 3), device=self.device)
        for i in range(self.num_envs):
            env = self.envs[i]
            targ_handle = self.gym.find_actor_handle(env, "targ_obj")
            rb_props = self.gym.get_actor_rigid_body_properties(env, targ_handle)
            # HACK: only apply to object with index 0
            self.target_masses[i] = float(rb_props[0].mass)

            # inertia diagonal (Ixx, Iyy, Izz) from Mat33 rows (Vec3)
            inertia = rb_props[0].inertia
            Ixx = float(inertia.x.x)
            Iyy = float(inertia.y.y)
            Izz = float(inertia.z.z)

            self.target_inertias[i, 0] = Ixx
            self.target_inertias[i, 1] = Iyy
            self.target_inertias[i, 2] = Izz

        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

    def _setup_domain_rand_cfg(self, rand_cfg):
        self.randomize_mass = rand_cfg['randomizeMass']
        self.randomize_mass_lower = rand_cfg['randomizeMassLower']
        self.randomize_mass_upper = rand_cfg['randomizeMassUpper']
        self.randomize_com = rand_cfg['randomizeCOM']
        self.randomize_com_lower = rand_cfg['randomizeCOMLower']
        self.randomize_com_upper = rand_cfg['randomizeCOMUpper']
        self.randomize_friction = rand_cfg['randomizeFriction']
        self.randomize_friction_lower = rand_cfg['randomizeFrictionLower']
        self.randomize_friction_upper = rand_cfg['randomizeFrictionUpper']
        self.randomize_scale = rand_cfg['randomizeScale']
        self.scale_list_init = rand_cfg['scaleListInit']
        self.randomize_scale_list = rand_cfg['randomizeScaleList']
        self.randomize_scale_lower = rand_cfg['randomizeScaleLower']
        self.randomize_scale_upper = rand_cfg['randomizeScaleUpper']
        self.randomize_pd_gains = rand_cfg['randomizePDGains']
        self.randomize_p_gain_lower = rand_cfg['randomizePGainLower']
        self.randomize_p_gain_upper = rand_cfg['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_cfg['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_cfg['randomizeDGainUpper']

    def apply_domain_rand(self, env_ptr, actor_handle, friction=False, com=False) -> None:

        obj_com = [0, 0, 0]
        if com:
            prop = self.gym.get_actor_rigid_body_properties(env_ptr, actor_handle)
            assert len(prop) == 1
            obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                    np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                    np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
            prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
            # print(f"current com: {self.gym.get_actor_rigid_body_properties(env_ptr, actor_handle)[0].com}")
            # print(f"Applying random com: {obj_com}")
            self.gym.set_actor_rigid_body_properties(env_ptr, actor_handle, prop)
            # print(f"new com: {self.gym.get_actor_rigid_body_properties(env_ptr, actor_handle)[0].com}")
        
        obj_friction = 1.0
        if friction:
            rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
            # print(f"current friction: {self.gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)[0].friction}")
            # print(f"Applying random friction: {rand_friction}")
            object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)
            for p in object_props:
                p.friction = rand_friction
            self.gym.set_actor_rigid_shape_properties(env_ptr, actor_handle, object_props)
            # print(f"new friction: {self.gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)[0].friction}")


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
                self.extras['curiosity_states'] = self.keypoint_contact_forces.clone()
            elif self.curiosity_state_type == "contact_distance":
                rel_pos = self.keypoint_positions_with_offset - self.object_root_positions.unsqueeze(1)
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

            if add_noise:
                if "object_position_wrt_palm" == spec.name:
                    observations[spec.name] = self.observed_object_positions_wrt_palm.clone()
                if "object_orientation_wrt_palm" == spec.name:
                    observations[spec.name] = self.observed_object_orientations_wrt_palm.clone()
        return observations

    def _get_target_surface_points_world(self) -> torch.Tensor:
        # (num_envs, P, 3)
        # canonical = self.grasping_dataset._pointclouds[self.occupied_object_relative_indices]  # (N,P,3)
        canonical = self.grasping_dataset._pointclouds[[0] * self.num_envs]  # (N,P,3)
        pc_world = quat_rotate(self.object_root_orientations[:, None, :], canonical) + self.object_root_positions[:, None, :]
        return pc_world
    
    def compute_curiosity_observations_surface_all_fingertips(self) -> torch.Tensor:
        # Features per fingertip: [u_hat(3), r_log_norm(1)] → total 4 fingertips × 4 = 16
        pcl_world = self._get_target_surface_points_world()   # (N, P, 3)
        tips = self.fingertip_positions                        # (N, 4, 3)
        offseted_tips = self.fingertip_positions_with_offset
        
        if self.cfg["env"]["use_center_collision"]:
            _dists = torch.cdist(tips, pcl_world)
            _min_dists_per_finger, _idx_p = torch.min(_dists, dim=2)
            _idx_p_exp = _idx_p.unsqueeze(-1).expand(-1, -1, 3)
            _obj_pts = torch.gather(pcl_world, 1, _idx_p_exp)
            _u = _obj_pts - tips
            _r = torch.norm(_u, dim=2, keepdim=True).clamp_min(1e-6)

        # pairwise distances (batched): (N, 4, P)
        dists = torch.cdist(offseted_tips, pcl_world)
        min_dists_per_finger, idx_p = torch.min(dists, dim=2)  # (N,4), (N,4)

        # gather nearest surface point for each fingertip
        idx_p_exp = idx_p.unsqueeze(-1).expand(-1, -1, 3)      # (N,4,3)
        obj_pts = torch.gather(pcl_world, 1, idx_p_exp)        # (N,4,3)
        u = obj_pts - offseted_tips                                     # (N,4,3)
        self.fingertip_geometric_distances = u.norm(dim=-1) # (N, 4)
        

        r = torch.norm(u, dim=2, keepdim=True).clamp_min(1e-6) # (N,4,1)
        u_hat = u / r                                          # (N,4,3)

        r0 = self.cfg["env"].get("curiosity", {}).get("r0", 0.02)
        r_max = self.cfg["env"].get("curiosity", {}).get("r_max", 0.20)
        r_log = torch.log1p(r / r0)                            # (N,4,1)
        r_log_norm = (r_log / math.log1p(r_max / r0)).clamp(0.0, 1.0)
        
        r = _r if self.cfg["env"]["use_center_collision"] else r

        feat = torch.cat([u_hat, r_log_norm], dim=-1)          # (N,4,4)
        return feat.reshape(self.num_envs, 16), r                 # (N,16)
    
    def compute_curiosity_observations_surface_all_keypoints(self) -> torch.Tensor:
        # Features per fingertip: [u_hat(3), r_log_norm(1)] → total 4 fingertips × 4 = 16
        pcl_world = self._get_target_surface_points_world()   # (N, P, 3)
        offseted_keypoints = self.keypoint_positions_with_offset

        # pairwise distances (batched): (N, 4, P)
        dists = torch.cdist(offseted_keypoints, pcl_world)
        min_dists_per_finger, idx_p = torch.min(dists, dim=2)  # (N,4), (N,4)

        # gather nearest surface point for each fingertip
        idx_p_exp = idx_p.unsqueeze(-1).expand(-1, -1, 3)      # (N,4,3)
        obj_pts = torch.gather(pcl_world, 1, idx_p_exp)        # (N,4,3)
        u = obj_pts - offseted_keypoints                                     # (N,4,3)

        r = torch.norm(u, dim=2, keepdim=True).clamp_min(1e-6) # (N,4,1)
        u_hat = u / r                                          # (N,4,3)

        r0 = self.cfg["env"].get("curiosity", {}).get("r0", 0.02)
        r_max = self.cfg["env"].get("curiosity", {}).get("r_max", 0.20)
        r_log = torch.log1p(r / r0)                            # (N,4,1)
        r_log_norm = (r_log / math.log1p(r_max / r0)).clamp(0.0, 1.0)

        feat = torch.cat([u_hat, r_log_norm], dim=-1)          # (N,4,4)
        return feat.reshape(self.num_envs, -1), r                 # (N,16)
    
    def compute_contact_filtered_fingertips_relative_pos(self):
        """
        Compute fingertip positions relative to the target object's center, filtered by contact.
        Returns:
            filtered_rel (Tensor): (N, 4, 3)
            has_contact (BoolTensor): (N,) any fingertip satisfied both conditions
        """
        # Relative fingertip positions to object center: (N,4,3)
        rel_pos = self.fingertip_positions - self.object_root_positions.unsqueeze(1)

        # Distance to nearest surface point per fingertip: (N,4,1)
        _, r = self.compute_curiosity_observations_surface_all_fingertips()  # r: (N,4,1)


        contact_mag = self.fingertip_contact_forces.norm(dim=-1, p=2)

        # Contact filters
        if self.cfg["env"]["use_center_collision"]:
            near_surface = (r.squeeze(-1) < 0.012)
        else:
            near_surface = (r.squeeze(-1) < 0.005)

        has_force = (contact_mag > 0.01) # (N,4)
        contact_mask = near_surface & has_force # (N,4)

        # Apply mask
        filtered_rel = rel_pos * contact_mask.unsqueeze(-1)  # (N,4,3)

        has_contact = contact_mask.any(dim=1)  # (N,)
        return filtered_rel, has_contact
    
    def compute_contact_filtered_keypoints_relative_pos(self):
        """
        Compute keypoint positions relative to the target object's center, filtered by contact.
        Returns:
            filtered_rel (Tensor): (N, 4, 3)
            has_contact (BoolTensor): (N,) any keypoint satisfied both conditions
        """
        # Relative keypoint positions to object center: (N,4,3)
        rel_pos = self.keypoint_positions_with_offset - self.object_root_positions.unsqueeze(1)

        # Distance to nearest surface point per keypoint: (N,4,1)
        _, r = self.compute_curiosity_observations_surface_all_keypoints()  # r: (N,4,1)


        contact_mag = self.keypoint_contact_forces.norm(dim=-1, p=2)

        # Contact filters
        near_surface = (r.squeeze(-1) < 0.005)           # (N,4)
        has_force = (contact_mag > 1e-2)                 # (N,4)
        contact_mask = near_surface & has_force         # (N,4)

        # Apply mask
        filtered_rel = rel_pos * contact_mask.unsqueeze(-1)  # (N,4,3)

        has_contact = contact_mask.any(dim=1)  # (N,)
        return filtered_rel, has_contact


    def set_table_color(self, env_ids, color=[0, 0, 0]):
        for succ_env_id in env_ids:
            self.gym.set_rigid_body_color(
                self.envs[succ_env_id], self.table_indices[succ_env_id], 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color)
            )

    def compute_action_reward(self, actions):
        if self.action_penalty_scale < 0:
            action_penalty = torch.sum(actions**2, dim=-1)
            self.action_penalty_scaled = action_penalty * self.action_penalty_scale
        elif self.wrist_action_penalty_scale < 0:
            action_penalty = torch.sum(self.allegro_dof_speeds[:2] ** 2, dim=-1)
            self.action_penalty_scaled = action_penalty * self.wrist_action_penalty_scale
        elif self.arm_action_penalty_scale < 0:
            ur_action = torch.cat([self.eef_translation, self.eef_rotation], dim=1)
            action_penalty = torch.sum(ur_action**2, dim=-1)
            self.action_penalty_scaled = action_penalty * self.arm_action_penalty_scale
        else:
            action_penalty = torch.sum(actions**2, dim=-1)
            self.action_penalty_scaled = action_penalty * 0

        
    def compute_reach_reward_keypoints(self):
        """Reaching reward using keypoint-to-object-surface distances with historical minima."""
        pcl_world = self._get_target_surface_points_world()
        keypoints_w = self.keypoint_positions_with_offset

        # Current nearest distances from each keypoint to the object surface: (N, K)
        # torch.cdist: (N, K, P) → min over P
        dists = torch.cdist(keypoints_w, pcl_world)
        cur_min_dist, _ = torch.min(dists, dim=2)  # (N, K)
        
        self.cur_keypoints_to_obj_surface_dist = cur_min_dist
        self.cur_index_keypoint_to_obj_surface_dist = cur_min_dist[:, self.index_link_indices_among_keypoints]
        self.cur_thumb_keypoint_to_obj_surface_dist = cur_min_dist[:, self.thumb_link_indices_among_keypoints]

        if not hasattr(self, "keypoints_to_surface_dist_min"):
            self.keypoints_to_surface_dist_min = torch.full_like(cur_min_dist, 0.30)  # meters

        delta = (self.keypoints_to_surface_dist_min - cur_min_dist).clamp_min(0.0)  # (N, K)
        self.keypoints_to_surface_dist_min = torch.min(self.keypoints_to_surface_dist_min, cur_min_dist)

        reach_rew_keypoints = delta.mean(dim=1)  # (N,)
        self.reach_rew_keypoints = reach_rew_keypoints
        self.reach_rew_scaled_keypoints = self.reach_rew_keypoints * 20.0

        # Logging
        self.extras["keypoint_surface_distances"] = cur_min_dist.clone()
        self.extras["keypoints_to_surface_dist_min"] = self.keypoints_to_surface_dist_min.clone()
        self.extras["reach_rew_keypoints"] = self.reach_rew_scaled_keypoints.clone()
        
        
    def refresh_contact_mask(self):
        filtered_rel, _ = self.compute_contact_filtered_fingertips_relative_pos()  # (N,4,3), (N,)

        contact_mask = (filtered_rel.abs().sum(dim=-1) > 0)  # (N,4) bool
        self.contact_mask = contact_mask
        
    def compute_curiosity_informed_reach_reward(self):
        """Compute reach reward - curiosity-informed version using fingertip positions.
        """
        from .torch_utils import quat_conjugate, quat_apply
        # compute per-fingertip nearest canonical index in world frame
        canonical = self.reach_curiosity_mgr.canonical_pointcloud
        pc_world = quat_rotate(self.object_root_orientations[:, None, :], canonical.unsqueeze(0).expand(self.num_envs, -1, -1)) \
                + self.object_root_positions[:, None, :]  # (N,M,3)

        # construct point cloud feature from world pointcloud
        mgr = self.reach_curiosity_mgr
        goal_positions = self.goal_position.repeat(self.num_envs, 1)
        goal_orientations = self.goal_orientation.repeat(self.num_envs, 1)
        state_features_world = mgr.build_state_features_from_world_pc(
            pc_world,
            goal_positions=goal_positions,
            goal_orientations=goal_orientations,
        )
        
        # World fingertip positions
        kp_pos_world = self.fingertip_positions_with_offset
        f_world = self.fingertip_contact_forces


        d = torch.cdist(kp_pos_world, pc_world)        # (N,4,M)
        contact_indices = d.argmin(dim=-1)                         # (N,4)
        self.keypoint_to_surface_dist = d.min(dim=-1)[0]                 # (N,4)
        
        q_conj = quat_conjugate(self.object_root_orientations)                      # (N,4)
        q_conj_exp = q_conj.unsqueeze(1).expand(-1, kp_pos_world.shape[1], -1)      # (N,L,4)
        contact_forces_canonical = quat_apply(q_conj_exp, -f_world)                 # (N,L,3)
        
        # self.contact_mask = contact_mask
        if self.hand_type == "leap":
            axis_local = torch.tensor([-1.0, 0.0, 0.0], device=self.device, dtype=kp_pos_world.dtype).view(1, 1, 3)
        elif self.hand_type == "allegro":
            axis_local = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=kp_pos_world.dtype).view(1, 1, 3)
        else:
            raise ValueError(f"Invalid hand type: {self.hand_type}")

        axis_local = axis_local.expand_as(kp_pos_world)
        kp_ori_world = self.fingertip_orientations
        dir_world = quat_apply(kp_ori_world.reshape(-1, 4), axis_local.reshape(-1, 3)).view_as(kp_pos_world)
        contact_dirs_canonical = quat_apply(q_conj_exp, -dir_world)

        reward, info = self.reach_curiosity_mgr.compute_reward_from_canonical(
            object_positions=self.object_root_positions,
            object_orientations=self.object_root_orientations,
            keypoint_positions_world=kp_pos_world,
            goal_positions=goal_positions,
            goal_orientations=goal_orientations,
            contact_indices=contact_indices,
            contact_mask=self.contact_mask,
            task_contact_satisfied=self.contact_satisfied,
            # contact_forces_local=contact_forces_canonical,
            contact_forces_local=contact_dirs_canonical,
            keypoint_palm_dirs_world=dir_world,
            state_features_world=state_features_world,
        )
        
        multiplier = (self.contact_mask[:, :-1].any(dim=-1).float() + self.contact_mask[:, -1].float()) / 2.0
        
        # self.reach_curiosity_rew = reward
        # self.reach_curiosity_rew_scaled = self.reach_curiosity_rew * 10
        self.reach_curiosity_rew = info["potential_field_reward"].clone()
        # W0: single state & global running max
        # self.reach_curiosity_rew_scaled = self.reach_curiosity_rew * 5.12 # 5.12 1.28 0.01, 0.02, 0.04
        # self.contact_coverage_rew = info["cluster_novelty_reward"].clone()
        # self.contact_coverage_rew_scaled = self.contact_coverage_rew * 800

        # W1: multiple states & state-wise running max
        self.reach_curiosity_rew_scaled = self.reach_curiosity_rew * 1.28 # 5.12 1.28 0.01, 0.02, 0.04
        self.contact_coverage_rew = info["cluster_novelty_reward"].clone()
        self.contact_coverage_rew_scaled = self.contact_coverage_rew * 200
        self.contact_coverage_rew_scaled = self.contact_coverage_rew_scaled
        self.extras["reach_curiosity_rew"] = self.reach_curiosity_rew_scaled.clone()
        self.extras["avg_potential"] = info["avg_potential"].clone()
        self.extras["contact_count"] = info["contact_count"].clone()
        self.extras["cluster_novelty_reward"] = self.contact_coverage_rew_scaled.clone()


        if self.reach_curiosity_mgr.state_type == "hash":
            self.extras["hash_recon_loss"] = info["hash_recon_loss"].clone().detach().repeat(self.num_envs)
            self.extras["hash_binary_reg"] = info["hash_binary_reg"].clone().detach().repeat(self.num_envs)
            self.extras["stateid_entropy"] = info["stateid_entropy"].clone().detach().repeat(self.num_envs)


    def check_contact_criteria(self):

        tip_contact = self.contact_mask[:, 3] & self.contact_mask[:, 0:3].any(dim=-1)
        tip_contact = self.keypoint_contact_mask.any(dim=-1)
        opposing_ok = torch.ones_like(tip_contact, dtype=torch.bool)

        table_contact = (self.table_contact_forces.norm(dim=-1) > 1e-2)
        lift_thresh  = 0.08
        in_max_displacement   = (self.y_displacement < lift_thresh)

        env_contact = torch.zeros_like(table_contact)
        if self.use_back_wall:
            env_contact |= (self.back_wall_contact_forces.norm(dim=-1) > 1e-2)
        if self.use_side_walls:
            env_contact |= (self.side_wall_pos_x_contact_forces.norm(dim=-1) > 1e-2)
            env_contact |= (self.side_wall_neg_x_contact_forces.norm(dim=-1) > 1e-2)
        
        env_contact |= (self.surr_object_contact_forces.norm(dim=-1) > 1e-2).any(dim=-1)
        
        hand_contact = (tip_contact & opposing_ok) | (table_contact)
        
        hand_contact = ((tip_contact & opposing_ok) | (table_contact)) & in_max_displacement
        hand_contact |= ((tip_contact & opposing_ok) | (~table_contact)) & ~in_max_displacement

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

    def compute_done(self, is_success):
        if not test_sim:
            fall_env_mask = (
                (self.object_root_positions[:, 2] < self._table_pose[2] - 0.1)
            )
            arm_contact_mask = (self.arm_contact_forces.norm(p=2, dim=2) > 1).any(dim=1)
            
            # if self.use_back_wall:
            #     fall_env_mask = fall_env_mask | (self.back_wall_contact_forces.norm(p=2, dim=-1) > 1.5e1)

            # fall_env_mask |= (~self.lifted) & (self.object_root_positions[:, 1] > self._table_y_length / 1.5)

            failed_env_ids = (fall_env_mask | arm_contact_mask).nonzero(as_tuple=False).squeeze(-1)
                
            if self.success_steps > 0:
                self.progress_buf = torch.where(is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf)
                self.reset_buf = torch.where(is_success > 0, 1, self.reset_buf)

            self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, 1, self.reset_buf)
            
            self.reset_buf[failed_env_ids] = 1

        # success
        succ_env_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        self.reset_buf[succ_env_ids] = 1
        self.successes[succ_env_ids] = 1
        self.successes[failed_env_ids] = 0

        self.done_successes[failed_env_ids] = 0
        self.done_successes[succ_env_ids] = 1

        if "height" in self.reward_type:
            self.extras["final_object_height"] = self.delta_obj_height[
                self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            ].clone()
        self.extras["success_num"] = torch.sum(self.successes).unsqueeze(-1)

    def compute_reward(self, actions: Tensor) -> None:


        self.y_displacement = self.object_root_positions[:, 1] - self.occupied_object_init_root_positions[:, 1]

        reward_types = self.reward_type.strip().split("+")
        assert len(reward_types) > 0, f"No reward type specified, reward_type from cfg is {self.reward_type}"

        self.refresh_contact_mask() # self.contact_mask
        self.contact_satisfied = self.check_contact_criteria()

        self.compute_reach_reward_keypoints(); self.reach_rew_scaled = self.reach_rew_scaled_keypoints.clone()
        # self.compute_curiosity_informed_reach_reward()
        if self.training:
            self.compute_curiosity_informed_reach_reward()
        else:
            self.reach_curiosity_rew_scaled = torch.zeros_like(self.rew_buf)
            self.contact_coverage_rew_scaled = torch.zeros_like(self.rew_buf)
        # self.compute_pre_grasp_reward()
        
        if "target" in reward_types:
            self.compute_targ_reward()

        
        self.near_goal_steps += self.near_goal.to(torch.long)
        self.near_goal_steps *= self.near_goal.to(torch.long) # avoid swing behavior
        is_success = self.near_goal_steps >= self.success_steps
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
        
        self.rew_buf[:] = self.task_reward

        self.compute_done(is_success)

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
        # print("self.endeffector_positions:", self.endeffector_positions[0])
        # print("self._xarm_allegro_hand_right_asset_file", self._xarm_allegro_hand_right_asset_file)
        # print("self._xarm_right_init_dof_positions", self._xarm_right_init_dof_positions)
        # breakpoint()

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_idx(self, env_ids: LongTensor, first_time=False) -> None:
        num_reset_envs: int = env_ids.shape[0]

        # if self.randomize and self.randomize_mass:
        #     lower, upper = self.randomize_mass_lower, self.randomize_mass_upper

            # for env_id in env_ids:
            #     env = self.envs[env_id]
            #     handle = self.gym.find_actor_handle(env, 'targ_obj')
            #     prop = self.gym.get_actor_rigid_body_properties(env, handle)
            #     print(f"Current mass: {prop[0].mass}")
            #     for p in prop:
            #         p.mass = np.random.uniform(lower, upper)
            #     print(f"Applying random mass: {prop[0].mass}")
            #     # self.gym.set_actor_rigid_body_properties(env, handle, prop)
            #     print(f"new mass: {self.gym.get_actor_rigid_body_properties(env, handle)[0].mass}")

        if self.mode == "eval" and local_test:
            self.set_table_color(env_ids, color=[1.0, 1.0, 1.0])

        noise = torch.rand(env_ids.shape[0], 3, device=self.device) * 2.0 - 1.0

        if self.relative_part_reward:
            self.prev_pos_dist[env_ids] = -1
            self.prev_rot_dist[env_ids] = -1
            self.prev_contact_dist[env_ids] = -1
            self.prev_nominal_dist[env_ids] = -1
        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset action noise times tracker
        if self.action_noise and self.action_noise_level == "step" and self.action_noise_max_times > 0:
            self.action_noise_times[env_ids] = 0

        occupied_object_relative_indices = self.occupied_object_relative_indices[env_ids]

        if self.env_info_logging:
            for i, env_id in enumerate(env_ids):
                # print(env_id, self.object_names[env_id][occupied_object_relative_indices[i]])
                pass

        object_indices = self.object_encodings[env_ids, occupied_object_relative_indices]
        examples = self.grasping_dataset.sample(object_indices)

        pointclouds = examples["pointcloud"]
        bbox = examples["bbox"]
        onehot = examples["category_onehot"]
        clutser_ids = examples["cluster"]

        if self.enable_full_pointcloud_observation:
            self.pointclouds[env_ids] = pointclouds

        self.occupied_object_cluster_ids[env_ids] = torch.from_numpy(clutser_ids).to(self.device).to(torch.long)


        # TODO: add noise to the initial DOF positions
        dof_init_positions = self.gym_assets["current"]["robot"]["init"]["position"].clone()
        dof_init_velocities = self.gym_assets["current"]["robot"]["init"]["velocity"].clone()

        dof_init_positions = dof_init_positions.unsqueeze(0).repeat(num_reset_envs, 1)
        dof_init_velocities = dof_init_velocities.unsqueeze(0).repeat(num_reset_envs, 1)

        arm_noise = (torch.rand(num_reset_envs, self.xarm_actuated_dof_indices.numel(), device=self.device) * 2.0 - 1.0) * self.dof_reset_noise_scale
        hand_noise = (torch.rand(num_reset_envs, self.allegro_actuated_dof_indices.numel(), device=self.device) * 2.0 - 1.0) * self.dof_reset_noise_scale

        dof_init_positions[:, self.xarm_actuated_dof_indices] += arm_noise
        dof_init_positions[:, self.allegro_actuated_dof_indices] += hand_noise

        lower = self.gym_assets["current"]["robot"]["limits"]["lower"]
        upper = self.gym_assets["current"]["robot"]["limits"]["upper"]
        dof_init_positions = torch.max(torch.min(dof_init_positions, upper), lower)

        # print("Reset DOF positions:", dof_init_positions[0, :])

        self.allegro_hand_dof_positions[env_ids, :] = dof_init_positions
        self.allegro_hand_dof_velocities[env_ids, :] = dof_init_velocities

        self.prev_targets[env_ids] = dof_init_positions
        self.curr_targets[env_ids] = dof_init_positions
        self.prev_allegro_dof_speeds[env_ids, :] = 0.0

        # random object orientation
        # if self.reset_obj_ori_noise > 0:
        #     occupied_object_init_root_orientation = random_orientation_within_angle(
        #         num_reset_envs, self.device, object_orientation, self.reset_obj_ori_noise / (180 / torch.pi)
        #     )
        # else:
        #     occupied_object_init_root_orientation = random_orientation(num_reset_envs, self.device)

        # For singulation task, use nominal orientation (no rotation) for all boxes
        occupied_object_init_root_orientation = self._object_nominal_orientation.clone().detach().repeat(num_reset_envs, 1)

        # Compute statastics of object pointclouds
        pointclouds_wrt_world = quat_rotate(occupied_object_init_root_orientation[:, None, :], pointclouds)

        obj_min_z = torch.min(pointclouds_wrt_world[:, :, 2], dim=1)[0]
        obj_x_length = torch.max(pointclouds[:, :, 0], dim=1)[0] - torch.min(pointclouds[:, :, 0], dim=1)[0]
        obj_y_length = torch.max(pointclouds[:, :, 1], dim=1)[0] - torch.min(pointclouds[:, :, 1], dim=1)[0]
        obj_z_length = torch.max(pointclouds[:, :, 2], dim=1)[0] - torch.min(pointclouds[:, :, 2], dim=1)[0]
        obj_max_length = torch.max(torch.stack([obj_x_length, obj_y_length, obj_z_length]), dim=0)[0]
        self.obj_max_length[env_ids] = obj_max_length.clone()
        self.object_bboxes[env_ids] = bbox.clone()
        self.object_categories[env_ids] = onehot.clone()

        if hasattr(self, "fingertips_to_obj_dist_min"):
            self.fingertips_to_obj_dist_min[env_ids] = 0.25
        if hasattr(self, "keypoints_to_obj_dist_min"):
            self.keypoints_to_obj_dist_min[env_ids] = 0.3
        if hasattr(self, "keypoints_to_surface_dist_min"):
            self.keypoints_to_surface_dist_min[env_ids] = 0.3
        if hasattr(self, "goal_position_dist_min"):
            self.goal_position_dist_min[env_ids] = self.cfg["env"].get("goalPositionDistMin", 0.6)
        if hasattr(self, "thumb_to_marker_dist_min"):
            self.thumb_to_marker_dist_min[env_ids] = 0.15
        if hasattr(self, "index_to_marker_dist_min"):
            self.index_to_marker_dist_min[env_ids] = 0.15
        if hasattr(self, "ring_to_marker_dist_min"):
            self.ring_to_marker_dist_min[env_ids] = 0.15
        if hasattr(self, "obj_height_displacement_min"):
            self.obj_height_displacement_min[env_ids] = 0.15
        if hasattr(self, "fingertips_to_obj_dist_surface_min"):
            self.fingertips_to_obj_dist_surface_min[env_ids] = torch.tensor([0.2, 0.2, 0.2, 0.5], device=self.device).unsqueeze(0).unsqueeze(-1)
            
        # reset curiosity running-max buffers (supports both legacy and state-conditioned shapes)
        self.reach_curiosity_mgr.ensure_running_max_buffers(self.num_envs)
        self.reach_curiosity_mgr.reset_running_max_buffers(env_ids)


        # Set occupied object root positions & orientations
        obj_ids = self.object_indices.view(self.num_envs, -1)[env_ids]
        self.root_positions[obj_ids, :] = self.init_scene_object_root_positions[env_ids, :]
        self.root_orientations[obj_ids, :] = self.init_scene_object_root_orientations[env_ids, :]
        self.root_linear_velocities[obj_ids, :] = 0.0
        self.root_angular_velocities[obj_ids, :] = 0.0

        if self.reset_randomize_scene_xy:
            self._ensure_scene_xy_offsets()
            dx, dy = self._sample_scene_xy_offsets(env_ids)
            self._apply_scene_xy_offsets(env_ids, dx, dy, apply_random_removal=self.apply_random_removal)
        
        if self.reset_randomize_scene_x:
            self._ensure_scene_xy_offsets()
            dx, dy = self._sample_scene_xy_offsets(env_ids)
            self._apply_scene_xy_offsets(env_ids, dx, dy=torch.zeros_like(dy), apply_random_removal=self.apply_random_removal)

        if self.reset_randomize_scene_orientation:
            q = self._sample_object_orientation(env_ids.shape[0], mode=self.reset_randomize_orientation_mode)
            self._apply_target_object_orientation(env_ids, q)

        if self.cfg["env"]["curiosity"]["enable_occlusion"] and hasattr(self, "_occlusion_aabb_getter"):
            aabbs = self._occlusion_aabb_getter()
            self.reach_curiosity_mgr.update_occlusion_aabbs(aabbs, env_ids=env_ids)

        self.robot_init_dof[env_ids, :] = dof_init_positions.clone()

        # Set dof-position-targets & dof-states
        indices = self.allegro_hand_indices[env_ids]
        indices = indices.flatten().to(torch.int32)

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.curr_targets_buffer),
            gymtorch.unwrap_tensor(indices),
            indices.shape[0],
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(indices),
            indices.shape[0],
        )

        # Set actor-root-states
        indices = self.object_indices[env_ids]
        indices = indices.flatten().to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(indices),
            indices.shape[0],
        )

        # Reset progress-buffer, reset-buffer, success-buffer
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        
        self.near_goal_steps[env_ids] = 0
        self.reset_goal_buf[env_ids] = 0
        self.picked[env_ids] = 0
        self.picked_curr[env_ids] = 0
        self.near_goal[env_ids] = 0
        # self.goal_position_dist_min[env_ids] = 1.0
        
        self.actions[env_ids, :] = 0
        self.prev_actions[env_ids, :] = 0

        if hasattr(self, "lifted"):
            self.lifted[env_ids] = 0
        if hasattr(self, "clearance_max"):
            self.clearance_max[env_ids] = 0.0



    def _ensure_scene_xy_offsets(self):
        if not hasattr(self, "scene_xy_offsets") or self.scene_xy_offsets is None:
            #NOTE: cube in box task and grasp task should enter this branch
            self.scene_xy_offsets = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def _sample_scene_xy_offsets(self, env_ids: torch.Tensor):
        dx_max = float(self._table_x_length - self.grid_width)
        dy_max = float(self._table_y_length - self.grid_depth)
        dx = (torch.rand((env_ids.shape[0],), device=self.device) - 0.5) * dx_max
        dy = (torch.rand((env_ids.shape[0],), device=self.device) - 0.5) * dy_max
        return dx, dy

    @torch.no_grad()
    def _apply_scene_xy_offsets(self, env_ids: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, apply_random_removal: bool = False):
        obj_ids = self.object_indices.view(self.num_envs, -1)[env_ids]  # (R, num_objects_per_env)
        new_pos = self.nominal_scene_object_root_positions[env_ids].clone()
        new_pos[..., 0] += dx.view(-1, 1)
        new_pos[..., 1] += dy.view(-1, 1)

        self.root_positions[obj_ids, :] = new_pos
        self.root_orientations[obj_ids, :] = self.init_scene_object_root_orientations[env_ids, :]
        self.root_linear_velocities[obj_ids, :] = 0.0
        self.root_angular_velocities[obj_ids, :] = 0.0


        if apply_random_removal and hasattr(self, "surr_object_indices"):
            reset_non_target_obj_idx = self.surr_object_indices.view(self.num_envs, -1)[env_ids].flatten()
            removal_selection = torch.rand(reset_non_target_obj_idx.shape[0], device=self.device) < 0.2
            self.root_positions[reset_non_target_obj_idx, 2] += removal_selection * 3.0
        self.scene_xy_offsets[env_ids, 0] = dx
        self.scene_xy_offsets[env_ids, 1] = dy

    @torch.no_grad()
    def _sample_object_orientation(self, n: int, mode: str = "yaw"):
        """return (n,4) xyzw quaternions"""
        if mode == "yaw":
            yaw = (torch.rand((n,), device=self.device) * 2.0 - 1.0) * math.pi
            half = 0.5 * yaw
            q = torch.zeros((n, 4), device=self.device, dtype=torch.float32)
            q[:, 2] = torch.sin(half)  # z
            q[:, 3] = torch.cos(half)  # w
            return q
        elif mode == "full":
            from .torch_utils import random_orientation
            return random_orientation(n, self.device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def _apply_target_object_orientation(self, env_ids: torch.Tensor, q_xyzw: torch.Tensor):
        actor_ids = self.occupied_object_indices[env_ids]  # (R,)
        self.root_orientations[actor_ids, :] = q_xyzw
        self.root_angular_velocities[actor_ids, :] = 0.0


    def set_states(
        self,
        robot_dof,
        object_targets=None,
        obj_pos=None,
        obj_orn=None,
        env_ids=None,
        step_time=-1,
        denomalize_robot_dof=False,
        set_dof_state=True,
        arm_ik=False,
    ):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device)

        if object_targets is not None and obj_pos is not None and obj_orn is not None:
            self.object_targets[env_ids] = object_targets
            # these are used for reward computation
            self._r_target_object_positions_wrt_palm[env_ids] = object_targets[:, :3]
            self._r_target_object_orientations_wrt_palm[env_ids] = object_targets[:, 3:7]
            ii, jj = torch.meshgrid(env_ids, self.allegro_digits_actuated_dof_indices - 6, indexing="ij")
            self._r_target_allegro_dof_positions[ii, jj] = object_targets[:, 7:25]

            self.occupied_object_init_root_positions[env_ids, :] = obj_pos
            self.occupied_object_init_root_orientations[env_ids, :] = obj_orn

            self.root_positions[self.occupied_object_indices[env_ids], :] = obj_pos
            self.root_orientations[self.occupied_object_indices[env_ids], :] = obj_orn

            indices = torch.unique((self.object_indices[env_ids]).flatten().to(torch.int32))
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(indices),
                indices.shape[0],
            )

        if arm_ik:
            targets = self.prev_targets.clone()

            cur_pos = self.endeffector_positions.clone()
            cur_quat = self.endeffector_orientations.clone()
            arm_pos = robot_dof[:, self.arm_trans_action_indices].clone()
            arm_rot_euler = robot_dof[:, self.arm_rot_action_indices].clone()
            arm_rot_quat = quat_from_euler_xyz(arm_rot_euler[:, 0], arm_rot_euler[:, 1], arm_rot_euler[:, 2])
            hand_target_dof = robot_dof[:, self.hand_action_indices]
            delta_joint_move = ik(
                self.j_eef,
                cur_pos,
                cur_quat,
                arm_pos,
                arm_rot_quat,
            )
            arm_target_dof = targets[:, self.ur_actuated_dof_indices] + delta_joint_move

            current_dof = targets.clone()
            current_dof[:, self.allegro_actuated_dof_indices] = hand_target_dof
            current_dof[:, self.allegro_tendon_dof_indices] = saturate(
                current_dof[:, self.allegro_coupled_dof_indices]
                - self.gym_assets["current"]["robot"]["limits"]["upper"][self.allegro_coupled_dof_indices],
                self.gym_assets["current"]["robot"]["limits"]["lower"][self.allegro_tendon_dof_indices],
                self.gym_assets["current"]["robot"]["limits"]["upper"][self.allegro_tendon_dof_indices],
            )
            robot_dof = current_dof.clone()
            robot_dof[:, self.ur_actuated_dof_indices] = arm_target_dof

            robot_dof = saturate(
                robot_dof,
                self.gym_assets["current"]["robot"]["limits"]["lower"],
                self.gym_assets["current"]["robot"]["limits"]["upper"],
            )

        if denomalize_robot_dof:
            robot_dof = denormalize(
                robot_dof,
                self.gym_assets["current"]["robot"]["limits"]["lower"],
                self.gym_assets["current"]["robot"]["limits"]["upper"],
            )

        if set_dof_state:
            self.allegro_hand_dof_positions[env_ids, :] = robot_dof
        self.prev_targets[env_ids] = robot_dof
        self.curr_targets[env_ids] = robot_dof

        indices = torch.unique((self.allegro_hand_indices[env_ids]).flatten().to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.curr_targets_buffer),
            gymtorch.unwrap_tensor(indices),
            indices.shape[0],
        )

        if set_dof_state:
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_states), gymtorch.unwrap_tensor(indices), indices.shape[0]
            )

        if step_time > 0:
            self.step_simulation(step_time)

        self.compute_observations()


    def _refresh_action_tensors(self, actions: torch.Tensor) -> None:
        """Given a batch of actions, refresh the action tensors.

        Args:
            actions (torch.Tensor): A batch of actions. [batch_size, action_dim]
        """
        current = 0
        for spec in self._action_space:
            setattr(self, spec.attr, actions[:, current : current + spec.dim])
            current += spec.dim

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.training:
            self.reset_done()

        if self.action_noise:
            noise = torch.randn_like(actions) * self.action_noise_sigma
            if self.action_noise_level == "value":
                mask = torch.rand((self.num_envs, self.num_actions), device=self.device) < self.action_noise_ratio
            elif self.action_noise_level == "step":
                mask = torch.rand((self.num_envs), device=self.device) < self.action_noise_ratio
                if self.action_noise_max_times > 0:
                    mask = mask & (self.action_noise_times < self.action_noise_max_times)
                    self.action_noise_times[mask] += 1
                mask = mask.unsqueeze(-1).repeat(1, self.num_actions)
            # ignore the actions that are already zero
            zero = (actions.abs() < 1e-8).all(dim=1).unsqueeze(-1)
            mask = mask & ~zero
            # add noise
            actions[mask] += noise[mask]

        self.actions = actions.clone().to(self.device)
        self.clamped_actions = actions.clone().to(self.device)
        self.prev_endeffector_states = self.endeffector_states.clone()
        self.prev_endeffector_positions = self.endeffector_positions.clone()
        self.prev_endeffector_orientations = self.endeffector_orientations.clone()
        self.prev_endeffector_orientations_euler = torch.stack(get_euler_xyz(self.prev_endeffector_orientations), dim=1)
        self.prev_allegro_actuated_dof_positions = self.allegro_hand_dof_positions[
            :, self.allegro_actuated_dof_indices
        ].clone()
        smoothed_actions = self.act_moving_average * self.actions + (1.0 - self.act_moving_average) * self.prev_actions
        self.prev_actions[:] = smoothed_actions[:]
        self._refresh_action_tensors(smoothed_actions.clone().to(self.device))

        if self.use_relative_control:
            targets = self.prev_targets.clone()
            if self.arm_control_type == "osc":
                xarm_dof_movements, self.target_eef_pos, self.target_eef_euler = compute_relative_xarm_dof_positions(
                    self.endeffector_positions,
                    self.endeffector_orientations,
                    self.j_eef,
                    self.eef_translation,
                    self.eef_rotation,
                    self._max_xarm_endeffector_pos_vel,
                    self._max_xarm_endeffector_rot_vel,
                    self.dt,
                )
            else:
                xarm_dof_speeds = torch.cat([self.eef_translation, self.eef_rotation], dim=1)
                xarm_dof_movements = xarm_dof_speeds * self.dof_speed_scale * self.dt

            if getattr(self, "eef_translation", None) is None and getattr(self, "eef_rotation", None) is None:
                xarm_dof_movements[:] = 0

            self.curr_targets[:, self.xarm_actuated_dof_indices] = (
                targets[:, self.xarm_actuated_dof_indices] + xarm_dof_movements
            )

            if getattr(self, "allegro_dof_speeds", None) is not None:

                self.curr_targets[:, self.allegro_actuated_dof_indices] = (
                    targets[:, self.allegro_actuated_dof_indices]
                    + self.allegro_dof_speeds * self.dof_speed_scale * self.dt
                )
                
                self.curr_targets[:, self.allegro_actuated_dof_indices] = saturate(
                    self.curr_targets[:, self.allegro_actuated_dof_indices],
                    self.gym_assets["current"]["robot"]["limits"]["lower"][self.allegro_actuated_dof_indices],
                    self.gym_assets["current"]["robot"]["limits"]["upper"][self.allegro_actuated_dof_indices],
                )
        elif self.use_absolute_joint_control:
            self.curr_targets[:] = actions
        else:
            # simulate the tendon coupling
            self.curr_targets[:, self.actuated_dof_indices] = self.actions
            self.curr_targets[:, self.allegro_tendon_dof_indices] = (
                torch.clamp_min(self.curr_targets[:, self.allegro_coupled_dof_indices], 0.0) * 2.0 - 1.0
            )
            self.curr_targets[:, self.allegro_coupled_dof_indices] = (
                torch.clamp_max(self.curr_targets[:, self.allegro_coupled_dof_indices], 0.0) * 2.0 + 1.0
            )
            # denormalize & saturate the targets
            self.curr_targets[:] = denormalize(
                self.curr_targets,
                self.gym_assets["current"]["robot"]["limits"]["lower"],
                self.gym_assets["current"]["robot"]["limits"]["upper"],
            )
            self.curr_targets[:] = (
                self.act_moving_average * self.curr_targets + (1.0 - self.act_moving_average) * self.prev_targets
            )

        self.curr_targets[:] = saturate(
            self.curr_targets,
            self.gym_assets["current"]["robot"]["limits"]["lower"],
            self.gym_assets["current"]["robot"]["limits"]["upper"],
        )

        # return
        if test_sim:
            self.curr_targets[:, 6:] = self.gym_assets["current"]["robot"]["limits"]["lower"][6:]
        self.prev_targets[:] = self.curr_targets[:]

        indices = self.allegro_hand_indices
        indices = indices.flatten().to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.curr_targets_buffer),
            gymtorch.unwrap_tensor(indices),
            indices.shape[0],
        )

        # Apply random wrench perturbations to the target object with probability per env
        if self.randomize and (self.force_scale > 0.0 or self.torque_scale > 0.0) and self.wrench_prob > 0.0:
            self.rb_forces.zero_()
            self.rb_torques.zero_()

            with torch.no_grad():
                mask = (torch.rand((self.num_envs,), device=self.device) < self.wrench_prob)
                if mask.any():
                    f_dir = torch.randn((self.num_envs, 3), device=self.device)
                    f_dir = f_dir / (f_dir.norm(dim=-1, keepdim=True) + 1e-8)
                    t_dir = torch.randn((self.num_envs, 3), device=self.device)
                    t_dir = t_dir / (t_dir.norm(dim=-1, keepdim=True) + 1e-8)

                    # scale forces by mass and torques by inertia diagonal (approx)
                    f_mag = self.force_scale * self.target_masses.view(-1, 1)
                    t_mag = self.torque_scale * self.target_inertias

                    f_perturb = f_dir * f_mag
                    t_perturb = t_dir * t_mag

                    # write only for masked envs and target rigid body index
                    trg_rb_idx = self.target_object_rigid_body_indices
                    local_trg_rb_idx = trg_rb_idx[mask] % self.num_rigid_bodies
                    env_ids_masked = torch.arange(self.num_envs, device=self.device)[mask]
                    self.rb_forces[env_ids_masked, local_trg_rb_idx, :] = f_perturb[mask]
                    self.rb_torques[env_ids_masked, local_trg_rb_idx, :] = t_perturb[mask]

                    self.gym.apply_rigid_body_force_tensors(
                        self.sim,
                        gymtorch.unwrap_tensor(self.rb_forces),
                        gymtorch.unwrap_tensor(self.rb_torques),
                        gymapi.ENV_SPACE,
                    )

    def get_singulation_occluder_aabbs(self) -> torch.Tensor:
        """
        Returns per-env AABBs in world coordinates.
        """
        device = self.device
        N = self.num_envs


        table_pos = self.rigid_body_states[self.table_rigid_body_indices, 0:3]  # [N,3]

        X = float(self._table_x_length)
        Y = float(self._table_y_length)
        Thk = float(self._table_thickness)
        table_half = torch.tensor([X * 0.5, Y * 0.5, Thk * 0.5], device=device)

        table_min = table_pos - table_half
        table_max = table_pos + table_half

        # --- surrounding objects AABBs (per env, world coords) ---
        surr_pos = self.surr_object_root_positions  # [N,S,3]
        w, d, h = float(self._obj_width), float(self._obj_depth), float(self._obj_height)
        half = torch.tensor([w * 0.5, d * 0.5, h * 0.5], device=device).view(1, 1, 3)

        surr_min = surr_pos - half  # [N,S,3]
        surr_max = surr_pos + half  # [N,S,3]

        # pack: [N, 1+S, 2, 3]
        aabbs = torch.zeros((N, 1 + self.max_non_targets, 2, 3), device=device, dtype=torch.float)
        aabbs[:, 0, 0] = table_min
        aabbs[:, 0, 1] = table_max
        aabbs[:, 1:, 0] = surr_min
        aabbs[:, 1:, 1] = surr_max
        return aabbs

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()

        if self.method == "case":
            self.compute_case2023_reward()
        else:
            self.compute_reward(self.actions)

        # track gpu memory usage
        if self.device.startswith("cuda"):
            gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info(device=self.device)
            gpu_mem_occupied = torch.tensor([gpu_mem_total - gpu_mem_free], device=self.device)
            self.extras["gpu_mem_occupied_MB"] = gpu_mem_occupied / 1024 / 1024
            self.extras["gpu_mem_occupied_GB"] = gpu_mem_occupied / 1024 / 1024 / 1024
            self.extras["gpu_mem_occupied_ratio"] = gpu_mem_occupied / gpu_mem_total

        
        JJT = self.j_eef @ self.j_eef.transpose(-1, -2)   # (B, 6, 6)
        manip = torch.sqrt(torch.det(JJT).clamp(min=1e-12))
        self.extras["max_jacobian_det"] = manip.max().reshape(1)
        # self.extras["max_jacobian_det"] = torch.max(torch.det(self.j_eef).abs()).reshape(1)
        

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            origin_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            origin_orientations = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
            origin_orientations[:, 3] = 1
            # draw_axes(self.gym, self.viewer, self.envs, origin_positions, origin_orientations, 0.5)
            # draw_axes(self.gym, self.viewer, self.envs, self.object_root_positions, self.object_root_orientations, 0.1)

            if self.enable_rendered_pointcloud_observation:
                self.draw_camera_axes()

            if self.enable_contact_sensors:
                self.draw_force_sensor_axes()
                
            self.draw_link_keypoints()
            
            draw_points(self.gym, self.viewer, self.envs, self.fingertip_positions, radius=0.012, num_segments=10, color=(1.0, 0.0, 0.0))
            draw_points(self.gym, self.viewer, self.envs, self.keypoint_positions_with_offset, radius=0.005, num_segments=10, color=(0.0, 1.0, 0.0))

            aabbs = self._occlusion_aabb_getter()

            if aabbs.dim() == 3:
                aabbs = aabbs.unsqueeze(0).repeat(self.num_envs, 1, 1, 1)

            centers = 0.5 * (aabbs[:, :, 0] + aabbs[:, :, 1])      # (N,K,3)
            sizes   = (aabbs[:, :, 1] - aabbs[:, :, 0]).clamp(min=0)  # (N,K,3)

            # identity quats
            q = torch.zeros((self.num_envs, centers.shape[1], 4), device=self.device, dtype=torch.float)
            q[..., 3] = 1.0

            K = centers.shape[1]
            for k in range(K):
                sx, sy, sz = sizes[:, k, :].mean(dim=0).tolist()
                draw_boxes(
                    self.gym, self.viewer, self.envs,
                    centers[:, k, :],
                    q[:, k, :],
                    size=(sx, sy, sz),
                    color=(1.0, 1.0, 0.0),
                    shadow_density=0,
                )

    # Visualization Utilities
    def close(self, env_ids, close_dis=0.3, close_dof_indices=None, check_contact=False):
        for i in range(50):
            if i < 30:
                targets = self.allegro_hand_dof_positions.clone()
                ii, jj = torch.meshgrid(env_ids, close_dof_indices, indexing="ij")
                self.curr_targets[ii, jj] = targets[ii, jj] + close_dis / 30
                indices = torch.unique(
                    torch.cat([self.allegro_hand_indices, self.target_allegro_hand_indices]).flatten().to(torch.int32)
                )
                self.gym.set_dof_position_target_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.curr_targets_buffer),
                    gymtorch.unwrap_tensor(indices),
                    indices.shape[0],
                )
            if self.force_render and i % 1 == 0:
                self.render()
            self.gym.simulate(self.sim)
            self._refresh_sim_tensors()

    def draw_force_sensor_axes(self) -> None:
        positions: torch.Tensor = self.allegro_hand_rigid_body_positions[:, self.force_sensor_rigid_body_indices]
        orientations: torch.Tensor = self.allegro_hand_rigid_body_orientations[:, self.force_sensor_rigid_body_indices]
        draw_boxes(self.gym, self.viewer, self.envs, positions, orientations, 0.001)

    def draw_camera_axes(self) -> None:
        for i in range(self.num_cameras_per_env):
            draw_axes(
                self.gym, self.viewer, self.envs, self.camera_positions[:, i], self.camera_orientations[:, i], 0.1
            )
            
    def draw_link_keypoints(self) -> None:
        # for link_name, link_index in self.hand_link_indices_map.items():
        #     link_positions = self.allegro_hand_rigid_body_positions[:, link_index]
        #     link_orientations = self.allegro_hand_rigid_body_orientations[:, link_index]
        #     link_keypoints_offset:torch.Tensor = to_torch(self.keypoints_info[link_name], device=self.device).repeat(self.num_envs, 1)
        #     link_keypoints_positions = link_positions + quat_apply(link_orientations, link_keypoints_offset)
        #     draw_axes(self.gym, self.viewer, self.envs, link_keypoints_positions, link_orientations, 0.02)
        for idx, link_name in enumerate(self._keypoints):
            link_positions = self.keypoint_positions_with_offset[:, idx, :]
            link_orientations = self.keypoint_orientations[:, idx, :]
            draw_axes(self.gym, self.viewer, self.envs, link_positions, link_orientations, 0.02)
        # for idx in self.fingertip_link_indices_among_keypoints:
        #     link_positions = self.keypoint_positions_with_offset[:, idx, :]
        #     link_orientations = self.keypoint_orientations[:, idx, :]
        #     draw_axes(self.gym, self.viewer, self.envs, link_positions, link_orientations, 0.02)

    def print_force_sensor_info(self, env_id: int = 0) -> None:
        force_sensor_states = self.force_sensor_states.view(self.num_envs, self.num_force_sensors, 6)
        force_sensor_state = force_sensor_states[env_id, ...]

        forces = force_sensor_state[:, 0:3]
        magnitudes = torch.norm(forces, dim=-1)
        print("force_magnitudes: ", magnitudes)
        # print("force_sensor_state: ", force_sensor_state)

    def get_images(self, img_width=1024, img_height=768, env_ids=None, simulate=True):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # vis part env
        env_ids = env_ids[: self.vis_env_num]
        # step the physics simulation
        if simulate:
            self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)

        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)

        if self.force_render:
            self.render()

        images = []
        # get rgb image
        for env_id in env_ids:
            image = self.gym.get_camera_image(
                self.sim, self.envs[env_id], self.cameras_handle[env_id], gymapi.IMAGE_COLOR
            )
            image = np.reshape(image, (np.shape(image)[0], -1, 4))[..., :3]
            image = image[:, :, (2, 1, 0)]
            image = cv2.resize(image, (img_width, img_height))
            images.append(image)

        images = np.stack(images, axis=0)
        images = to_torch(images, device=self.device)
        return images



def compute_offset_point_world(
    obj_position: torch.Tensor,
    obj_orientation: torch.Tensor,
    local_offset: torch.Tensor,
) -> torch.Tensor:
    """Compute the global position of a point defined in an object's local frame.
    
    Args:
        obj_position (torch.Tensor): Object's global position, shape (..., 3)
        obj_orientation (torch.Tensor): Object's global orientation (quaternion), shape (..., 4)
        local_offset (torch.Tensor): Offset in object's local frame, shape (3) or (..., 3)
        
    Returns:
        torch.Tensor: Global position of the offset point, shape (..., 3)
    """
    if local_offset.dim() == 1 and obj_position.dim() > 1:
        local_offset = local_offset.expand_as(obj_position)
    
    if obj_orientation.shape != obj_position.shape[:-1] + (4,):
        obj_orientation = obj_orientation.expand(*obj_position.shape[:-1], 4)
    
    rotated_offset = quat_apply(obj_orientation, local_offset)
    
    return obj_position + rotated_offset

def compute_relative_pose(
    a_orientation: torch.Tensor,
    a_position: torch.Tensor,
    b_orientation: torch.Tensor,
    b_position: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a pose in b's frame.

    Args:
        a_orientation (torch.Tensor): Orientations of a, shape (..., 4).
        a_position (torch.Tensor): Positions of a, shape (..., 3).
        b_orientation (torch.Tensor): Orientations of b, shape (..., 4).
        b_position (torch.Tensor): Positions of b, shape (..., 3).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Orientation & Position of a in b's frame.
    """
    assert a_position.dim() == b_position.dim()
    assert a_orientation.dim() == b_orientation.dim()

    w2b_rotation, w2b_translation = transformation_inverse(b_orientation, b_position)

    a_position, w2b_translation = torch.broadcast_tensors(a_position, w2b_translation)
    a_orientation, w2b_rotation = torch.broadcast_tensors(a_orientation, w2b_rotation)

    orientation, position = transformation_multiply(w2b_rotation, w2b_translation, a_orientation, a_position)
    return orientation, position


def compute_relative_position(
    a_position: torch.Tensor,
    b_orientation: torch.Tensor,
    b_position: torch.Tensor,
) -> torch.Tensor:
    """Compute a position in b's frame.

    Args:
        a_position (torch.Tensor): Positions of a, shape (..., 3).
        b_orientation (torch.Tensor): Orientations of b, shape (..., 4).
        b_position (torch.Tensor): Positions of b, shape (..., 3).

    Returns:
        torch.Tensor: Position of a in b's frame.
    """
    assert a_position.dim() == b_position.dim() == b_orientation.dim()

    w2b_rotation, w2b_translation = transformation_inverse(b_orientation, b_position)

    a_position, w2b_translation = torch.broadcast_tensors(a_position, w2b_translation)
    quaternion_shape = a_position.shape[:-1] + (4,)
    w2b_rotation = torch.broadcast_to(w2b_rotation, quaternion_shape)

    position = quat_apply(w2b_rotation, a_position) + w2b_translation
    return position


@torch.jit.script
def pointcloud_from_depth(
    depth: torch.Tensor,
    inv_view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
    width: Optional[int] = None,
    height: Optional[int] = None,
    u: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    threshold: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct point cloud from depth image.

    Args:
        depth (torch.Tensor): depth image, shape (..., height, width)
        inv_view_matrix (torch.Tensor): inverse view matrix, shape (..., 4, 4)
        proj_matrix (torch.Tensor): projection matrix, shape (..., 4, 4)
        width (Optional[int]): width of depth image. Defaults to depth.shape[1].
        height (Optional[int]): height of depth image. Defaults to depth.shape[0].
        u (Optional[torch.Tensor], optional): 2d grid of u coordinates. Defaults to None.
        v (Optional[torch.Tensor], optional): 2d grid of v coordinates. Defaults to None.
        threshold (float, optional): depth threshold. Defaults to 10.0.

    Returns:
        - torch.Tensor: point cloud, shape (..., height * width, 3)
        - torch.Tensor: mask, shape (..., height * width)
    """
    assert depth.ndim >= 2
    assert depth.device == inv_view_matrix.device == proj_matrix.device
    assert u is None or u.device == depth.device
    assert v is None or v.device == depth.device
    device = depth.device

    if width is None:
        width = depth.size(-1)

    if height is None:
        height = depth.size(-2)

    if u is None or v is None:
        v, u = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

    fu = 2 / proj_matrix[..., 0, 0]
    fv = 2 / proj_matrix[..., 1, 1]

    fu = fu.unsqueeze(-1).unsqueeze(-1)
    fv = fv.unsqueeze(-1).unsqueeze(-1)

    center_u = width / 2
    center_v = height / 2

    z = depth
    finite = torch.isfinite(z)
    invalid_z = -(threshold + 1.0)
    z = torch.where(finite, z, torch.full_like(z, invalid_z))

    x = -(u - center_u) / width * z * fu
    y = (v - center_v) / height * z * fv

    x, y, z = x.flatten(-2), y.flatten(-2), z.flatten(-2)

    mask = finite.flatten(-2) & (z > -threshold)
    points = torch.stack((x, y, z), dim=-1)

    rotation = inv_view_matrix[..., 0:3, 0:3].unsqueeze(-3)
    translation = inv_view_matrix[..., 3, 0:3].unsqueeze(-2).unsqueeze(-2)

    points.unsqueeze_(-2)
    points = (points @ rotation) + translation
    points.squeeze_(-2)

    return points, mask



# tangent space delta
def compute_relative_xarm_dof_positions(
    current_eef_positions: torch.Tensor,
    current_eef_orientations: torch.Tensor,
    eef_jacobian: torch.Tensor,
    eef_translations: torch.Tensor,
    eef_rotations: torch.Tensor,
    max_eef_translation_speed: float,
    max_eef_rotation_speed: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_trans = max_eef_translation_speed * dt
    max_rot = max_eef_rotation_speed * dt

    # Translation
    trans = eef_translations * max_trans
    trans_norm = torch.norm(trans, dim=-1, keepdim=True)
    trans = torch.where(trans_norm > max_trans, trans / trans_norm * max_trans, trans)
    target_pos = current_eef_positions + trans

    # Rotation: SO(3) tangent space → local quaternion delta
    rot = eef_rotations * max_rot
    rot_norm = torch.norm(rot, dim=-1, keepdim=True)
    rot = torch.where(rot_norm > max_rot, rot / rot_norm * max_rot, rot)

    # Exponential map: so(3) → quat
    angle = rot_norm.squeeze(-1)  # (N,)
    axis = rot / (rot_norm + 1e-8)  # (N, 3)
    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)
    delta_quat = torch.cat([axis * sin_half.unsqueeze(-1), cos_half.unsqueeze(-1)], dim=-1)
    delta_quat = torch.nn.functional.normalize(delta_quat, dim=-1)

    # Local-frame composition
    target_quat = quat_mul(current_eef_orientations, delta_quat)

    theta = quat_diff_rad_normalized(current_eef_orientations, target_quat)
    if (theta > max_rot).any():
        mask = theta > max_rot
        t = max_rot / theta[mask]
        # SLERP: q = slerp(q0, q1, t)
        q0 = current_eef_orientations[mask]
        q1 = target_quat[mask]
        sin_theta = torch.sin(theta[mask])
        sin_t_theta = torch.sin(t * theta[mask])
        sin_1mt_theta = torch.sin((1 - t) * theta[mask])
        target_quat[mask] = (sin_1mt_theta.unsqueeze(-1) * q0 + sin_t_theta.unsqueeze(-1) * q1) / sin_theta.unsqueeze(-1)

    # For logging
    target_euler = torch.stack(get_euler_xyz(target_quat), dim=1)

    dof_delta = ik(eef_jacobian, current_eef_positions, current_eef_orientations, target_pos, target_quat)
    return dof_delta, target_pos, target_euler


def normalized_entropy(counts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute normalized entropy for a non-negative count vector.

    Args:
        counts: Tensor of shape (M,), non-negative counts.
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor in [0, 1], normalized entropy.
    """
    # Ensure float
    counts = counts.float()

    M = counts.shape[1]

    total = counts.sum(dim=1, keepdim=True)  # (L, 1)

    p = counts / (total + eps)
    entropy = -(p * (p + eps).log()).sum(dim=1) # (L,)

    return entropy / torch.log(torch.tensor(float(M), device=counts.device))


def concentration(counts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Concentration score per keypoint: 1 - normalized entropy.
    """
    H = normalized_entropy(counts, eps) # (L,)
    return 1.0 - H


def topk_mean_concentration(counts: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Top-k mean concentration score across keypoints.

    Args:
        counts: Tensor of shape (L, M), non-negative counts per keypoint.
        k: Number of top keypoints to consider.
        eps: Small constant for numerical stability.
    Returns:
        Scalar tensor, top-k mean concentration score.
    """
    conc = concentration(counts, eps)  # (L,)
    topk_conc, _ = torch.topk(conc, k=k, largest=True, sorted=False)  # (k,)
    return topk_conc.mean()
