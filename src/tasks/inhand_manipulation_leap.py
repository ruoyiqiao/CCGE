from isaacgymenvs.tasks.allegro_hand import AllegroHand
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp  
import torch
import math
from .torch_utils import *
from isaacgym import gymapi
from isaacgym import gymtorch
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
import json
from .isaacgym_utils import (
    ObservationSpec,
    ActionSpec,
    print_observation_space,
    print_action_space,
    print_asset_options,
    print_links_and_dofs,
    print_dof_properties,
)
import omegaconf
import warnings
from collections import OrderedDict, deque
import random
from .curiosity_reward_manager import CuriosityRewardManager

from .inhand_manipulation_allegro import InhandManipulationAllegro
import enum
import os


class LeapHandDimensions(enum.Enum):
    """Dimension constants for Isaac Gym with xArm6 + Allegro Hand."""

    POSE_DIM = 7
    VELOCITY_DIM = 6
    STATE_DIM = 13
    WRENCH_DIM = 6

    NUM_FINGERTIPS = 4
    NUM_DOFS = 16

    HAND_ACTUATED_DIM = 16
    NUM_KEYPOINTS = 21


class InhandManipulationLeap(InhandManipulationAllegro):
    _asset_root: str = "assets"
    # Use the Leap URDF
    _xarm_allegro_hand_right_asset_file: str = "urdf/leap_hand.urdf"
    _xarm_allegro_hand_left_asset_file: str = ""
    
    _dims = LeapHandDimensions

    # xArm DOFs
    _xarm_dof_names: List[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

    # Hand reference links (neutral)
    _hand_center_link: str = "palm_lower"
    _hand_palm_link: str = "palm_lower"

    # virtual tip heads
    _tip_links: List[str] = [
        "index_tip_head",
        "middle_tip_head",
        "ring_tip_head",
        "thumb_tip_head",
    ]
    _fingertips: List[str] = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"] # index, middle, ring, thumb

    # Hand links (subset; used for mapping/debug)
    _hand_links: List[str] = [
        "palm_lower",
        "mcp_joint",
        "pip",
        "dip",
        "fingertip",
        "mcp_joint_2",
        "pip_2",
        "dip_2",
        "fingertip_2",
        "mcp_joint_3",
        "pip_3",
        "dip_3",
        "fingertip_3",
        "thumb_temp_base",
        "thumb_pip",
        "thumb_dip",
        "thumb_fingertip",
        "index_tip_head",
        "middle_tip_head",
        "ring_tip_head",
        "thumb_tip_head",
    ]

    # Per-finger link groups (for indices-among-keypoints bookkeeping)
    _index_finger_links: List[str] = ["pip", "dip", "fingertip"]
    _middle_finger_links: List[str] = ["pip_2", "dip_2", "fingertip_2"]
    _ring_finger_links: List[str] = ["pip_3", "dip_3", "fingertip_3"]
    _thumb_links: List[str] = ["thumb_dip", "thumb_fingertip"]
    
    _index_finger_keypoint_links: List[str] = ["pip", "dip", "fingertip"]
    _middle_finger_keypoint_links: List[str] = ["pip_2", "dip_2", "fingertip_2"]
    _ring_finger_keypoint_links: List[str] = ["pip_3", "dip_3", "fingertip_3"]
    _thumb_keypoint_links: List[str] = ["thumb_dip", "thumb_fingertip", "thumb_fingertip"] #double thumb_fingertip for 2 keypoints in xarm6_leap_right_keypoints.json

    # Initial DOF positions (neutral naming)
    hand_init_dof_positions: Dict[str, float] = {
        # index finger
        "index_joint_0": 0.0,
        "index_joint_1": 0.95,
        "index_joint_2": 0.66,
        "index_joint_3": 0.80,
        # middle finger
        "middle_joint_0": 0.0,
        "middle_joint_1": 0.95,
        "middle_joint_2": 0.66,
        "middle_joint_3": 0.80,
        # ring finger
        "ring_joint_0": 0.0,
        "ring_joint_1": 0.95,
        "ring_joint_2": 0.66,
        "ring_joint_3": 0.80,
        # thumb
        "thumb_joint_0": 0.85,
        "thumb_joint_1": 0.95,
        "thumb_joint_2": 0.30,
        "thumb_joint_3": 0.24,
    }

    _keypoints_info_path: str = "assets/urdf/xarm6_leap_right_keypoints.json"

     # Initial DOF positions for different objects (when useUpsideDown=True)
    # For useUpsideDown=False, hand_init_dof_positions will be set to all 0.0
    _object_hand_init_dof_positions: Dict[str, Dict[str, float]] = {
        "elephant": { #scale = 0.7
        "index_joint_0": 0.0, "index_joint_1": 1.3, "index_joint_2": 0.7, "index_joint_3": 1,
        "middle_joint_0": 0.0, "middle_joint_1": 1.3, "middle_joint_2": 0.7, "middle_joint_3": 1,
        "ring_joint_0": 0.0, "ring_joint_1": 1.3, "ring_joint_2": 0.7, "ring_joint_3": 1,
        "thumb_joint_0": 1.7, "thumb_joint_1": 0.6, "thumb_joint_2": 0, "thumb_joint_3": 0.90,
        },
         "mouse": { #OK
        "index_joint_0": 0.0, "index_joint_1": 0.95, "index_joint_2": 0.75, "index_joint_3": 0.80,
        "middle_joint_0": 0.0, "middle_joint_1": 0.95, "middle_joint_2": 0.75, "middle_joint_3": 0.80,
        "ring_joint_0": 0.0, "ring_joint_1": 0.95, "ring_joint_2": 0.75, "ring_joint_3": 0.80,
        "thumb_joint_0": 1.1, "thumb_joint_1": 0.95, "thumb_joint_2": 0.90, "thumb_joint_3": 0.90,
        },
        # "mug": { #OK
        # "index_joint_0": 0.0, "index_joint_1": 0.90, "index_joint_2": 0.90, "index_joint_3": 1,
        # "middle_joint_0": 0.0, "middle_joint_1": 0.90, "middle_joint_2": 0.90, "middle_joint_3": 1,
        # "ring_joint_0": 0.0, "ring_joint_1": 0.90, "ring_joint_2": 0.90, "ring_joint_3": 1,
        # "thumb_joint_0": 1.7, "thumb_joint_1": 0.3143, "thumb_joint_2": 0.0638, "thumb_joint_3": 1.0791,
        # },
        "mug": { #scale = 0.68
        "index_joint_0": 0.0, "index_joint_1": 0.90, "index_joint_2": 0.90, "index_joint_3": 1,
        "middle_joint_0": 0.0, "middle_joint_1": 0.90, "middle_joint_2": 0.90, "middle_joint_3": 1,
        "ring_joint_0": 0.0, "ring_joint_1": 0.90, "ring_joint_2": 0.90, "ring_joint_3": 1,
        "thumb_joint_0": 1.7, "thumb_joint_1": 0.3143, "thumb_joint_2": 1.0, "thumb_joint_3": 0.15,
        },
        "rubber_duck": { #OK
        "index_joint_0": 0.0, "index_joint_1": 1.25, "index_joint_2": 0.80, "index_joint_3": 0.95,
        "middle_joint_0": 0.0, "middle_joint_1": 1.15, "middle_joint_2": 0.75, "middle_joint_3": 0.90,
        "ring_joint_0": 0.0, "ring_joint_1": 1.25, "ring_joint_2": 0.80, "ring_joint_3": 0.95,
        "thumb_joint_0": 1.57, "thumb_joint_1": 1.1, "thumb_joint_2": 0.38, "thumb_joint_3": 0.60,
        },
        "stanford_bunny": { #scale = 0.68
        "index_joint_0": 0.0, "index_joint_1": 1.3, "index_joint_2": 0.6, "index_joint_3": 0.6,
        "middle_joint_0": 0.0, "middle_joint_1": 1.3, "middle_joint_2": 0.6, "middle_joint_3": 0.6,
        "ring_joint_0": 0.0, "ring_joint_1": 1.3, "ring_joint_2": 0.6, "ring_joint_3": 0.6,
        "thumb_joint_0": 1.7, "thumb_joint_1": 0.6, "thumb_joint_2": 0.3, "thumb_joint_3": 0.85,
        },
        # "utah_teapot": { #scale=0.7 OK
        #     "index_joint_0": 0.0, "index_joint_1": 0.90, "index_joint_2": 0.90, "index_joint_3": 1,
        #     "middle_joint_0": 0.0, "middle_joint_1": 0.90, "middle_joint_2": 0.90, "middle_joint_3": 1,
        #     "ring_joint_0": 0.0, "ring_joint_1": 0.90, "ring_joint_2": 0.90, "ring_joint_3": 1,
        #     "thumb_joint_0": 1.7, "thumb_joint_1": 0.3143, "thumb_joint_2": 0.0638, "thumb_joint_3": 1.0791,
        # },
        "utah_teapot": { #scale=0.55
            "index_joint_0": 0.0, "index_joint_1": 0.90, "index_joint_2": 0.90, "index_joint_3": 1,
            "middle_joint_0": 0.0, "middle_joint_1": 0.90, "middle_joint_2": 0.90, "middle_joint_3": 1,
            "ring_joint_0": 0, "ring_joint_1": 0.90, "ring_joint_2": 0.90, "ring_joint_3": 1,
            "thumb_joint_0": 1.7, "thumb_joint_1": 0.3143, "thumb_joint_2": 0.28, "thumb_joint_3": 1.0791,
        },
        "cell_phone": { #OK
            "index_joint_0": 0.0, "index_joint_1": 0.90, "index_joint_2": 0.90, "index_joint_3": 1,
            "middle_joint_0": 0.0, "middle_joint_1": 0.90, "middle_joint_2": 0.90, "middle_joint_3": 1,
            "ring_joint_0": 0, "ring_joint_1": 0.90, "ring_joint_2": 0.90, "ring_joint_3": 1,
            "thumb_joint_0": 1.7, "thumb_joint_1": 0.3143, "thumb_joint_2": 0.28, "thumb_joint_3": 1.0791,
        },
        "cube_16x3x3": { #scale=0.75
            "index_joint_0": -0.15, "index_joint_1": 1.51, "index_joint_2": 0.5, "index_joint_3": -0.15,
            "middle_joint_0": 0.0, "middle_joint_1": 1.6, "middle_joint_2": 0.5, "middle_joint_3": -0.2,
            "ring_joint_0": 0.15, "ring_joint_1": 1.51, "ring_joint_2": 0.5, "ring_joint_3": -0.15,
            "thumb_joint_0": 1.9, "thumb_joint_1": 0.0, "thumb_joint_2": 0.17, "thumb_joint_3": -0.2,
        },
        "cube_16x3x3_2": { #scale=0.75 (same as cube_16x3x3)
            "index_joint_0": -0.15, "index_joint_1": 1.51, "index_joint_2": 0.5, "index_joint_3": -0.15,
            "middle_joint_0": 0.0, "middle_joint_1": 1.6, "middle_joint_2": 0.5, "middle_joint_3": -0.2,
            "ring_joint_0": 0.15, "ring_joint_1": 1.51, "ring_joint_2": 0.5, "ring_joint_3": -0.15,
            "thumb_joint_0": 1.9, "thumb_joint_1": 0.0, "thumb_joint_2": 0.17, "thumb_joint_3": -0.2,
        },
    }
    
    
    # Initial object pose (position offset, rotation, and scale) for different objects (when useUpsideDown=True)
    # For useUpsideDown=False, object pose is set in _create_envs with default values
    # Structure: {object_name: {"position_offset": (dx, dy, dz), "rotation_quat": (x, y, z, w), "scale": float}}
    _object_initial_pose: Dict[str, Dict[str, Union[Tuple[float, ...], float]]] = {
        # "elephant": {
        #     "position_offset": (-0.03, 0.02, -0.09),  
        #     "rotation_quat": (0.0, 0.0, 0.0, 1.0),  
        #     "scale": 1.0, 
        # },
        "elephant": {
        "position_offset": (-0.035, 0.04, -0.08), 
        "rotation_quat": (0.0, 0.0, 0.0, 1.0), 
        "scale": 0.7, 
        },
        "mouse": {
            "position_offset": (-0.03, 0.02, -0.07), 
            "rotation_quat": (0.0, 0.0, 0.0, 1.0),  
            "scale": 1.0, 
        },
        # "mug": {
        #     "position_offset": (-0.032, 0.02, -0.08), 
        #     "rotation_quat": (0.0, 0.707, 0.0, 0.707),
        #     "scale": 1.0,  
        # },
        "mug": {
            "position_offset": (-0.032, 0.02, -0.07), 
            "rotation_quat": (0.0, 0.707, 0.0, 0.707),  
            "scale": 0.68,  
        },
        "rubber_duck": {
            "position_offset": (-0.03, 0.02, -0.07),  
            "rotation_quat": (0.0, 0.0, 0.0, 1.0),  
            "scale": 1.0,  
        },
        "stanford_bunny": {
            "position_offset": (-0.03, 0.02, -0.08), 
            "rotation_quat": (0.0, 0.0, 0.0, 1.0),  
            "scale": 0.68,  
        },
        # "utah_teapot": { #scale=0.7 OK 
        #     "position_offset": (-0.03, 0.02, -0.09),  
        #     "rotation_quat": (0.0, 0.0, 0.0, 1.0), 
        #     "scale": 0.7,  
        # },
        "utah_teapot": { #scale=0.55 
            "position_offset": (-0.04, 0.02, -0.07), 
            "rotation_quat": (0.0, 0.0, 0.0, 1.0),  
            "scale": 0.55,  
        },
        "cell_phone": { #scale=0.85
            "position_offset": (-0.04, 0.02, -0.07), 
            "rotation_quat": (0.0, 0.0, 0.0, 1.0), 
            "scale": 0.80,  
        },
        "cube_16x3x3": { #scale=0.55
            "position_offset": (-0.04, 0.05, -0.12), 
            "rotation_quat": (0.0, 0.0, 0.0, 1.0),  
            "scale": 1,  
        },
        "cube_16x3x3_2": { #scale=0.55 (same as cube_16x3x3)
            "position_offset": (-0.04, 0.05, -0.12), 
            "rotation_quat": (0.0, 0.0, 0.0, 1.0),  
            "scale": 1,  
        },
    }
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        kp_json = os.path.join(project_root, self._keypoints_info_path)
        self._keypoints_info: Dict[str, List[List[float]]] = {}
        if os.path.exists(kp_json):
            with open(kp_json, "r") as f:
                self._keypoints_info = json.load(f)

        # Build flattened keypoint pairs (link, sub-index)
        self._flattened_keypoint_pairs: List[Tuple[str, int]] = []
        for link_name, offsets in self._keypoints_info.items():
            for k in range(len(offsets)):
                self._flattened_keypoint_pairs.append((link_name, k))

        self._keypoints = [ln for (ln, _) in self._flattened_keypoint_pairs]
        
        if cfg["env"].get("returnCuriosityInfo", False):
            self.curiosity_state_type = cfg["env"]["CuriosityInfo"].get("curiosityStateType", "policy_state")  # or "contact_force" or "contact_distance"

        # Auto-select hand_init_dof_positions based on object and use_upside_down
        # MUST be called BEFORE super().__init__() because _create_envs (which calls _define_allegro_hand) 
        # is called inside super().__init__()
        self._setup_hand_init_dof_positions(cfg)

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        if self.cfg["env"].get("returnCuriosityInfo", False):
            self.curiosity_state_dim = self.extras["curiosity_states"].shape[1:] # remove env dim

    def compute_observations(self, reset_env_ids: Optional[torch.LongTensor] = None):
        super().compute_observations(reset_env_ids)
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
            
    def _setup_hand_init_dof_positions(self, cfg):
        """Automatically select hand_init_dof_positions based on object type and use_upside_down.
        
        - If use_upside_down=False: Set all DOF positions to 0.0 (default)
        - If use_upside_down=True: Select from _object_hand_init_dof_positions based on object name
        
        Note: This must be called BEFORE super().__init__() because _create_envs is called inside it.
        """
        # Get use_upside_down from cfg (not self.use_upside_down, which is set in super().__init__)
        use_upside_down = cfg["env"].get("useUpsideDown", False)
        
        if not use_upside_down:
            # 手心向上：所有 DOF 位置设为 0.0
            self.hand_init_dof_positions = {}
            print(">>> useUpsideDown=False: Using default hand_init_dof_positions (all 0.0)")
        else:
            # 手心向下：根据物体名称选择对应的 hand_init_dof_positions
            object_urdf_path = cfg["env"].get("objectUrdfPath", "")
            if not object_urdf_path:
                print(">>> Warning: objectUrdfPath not found, using default hand_init_dof_positions")
                self.hand_init_dof_positions = {}
                return
            
            # Extract object name from path (e.g., "contactdb/elephant/elephant.urdf" -> "elephant")
            # Also handles paths like "contactdb/stah_teapot/stah_teapot.urdf" -> "stah_teapot"
            object_name = None
            path_parts = object_urdf_path.split('/')
            # Try to find object name in path parts
            for part in path_parts:
                # Remove .urdf extension if present
                part_clean = part.replace('.urdf', '')
                if part_clean in self._object_hand_init_dof_positions:
                    object_name = part_clean
                    break

            if object_name and object_name in self._object_hand_init_dof_positions:
                self.hand_init_dof_positions = self._object_hand_init_dof_positions[object_name].copy()
                print(f">>> useUpsideDown=True: Auto-selected hand_init_dof_positions for object '{object_name}'")
            else:
                print(f">>> Warning: Object name not found in _object_hand_init_dof_positions (path: {object_urdf_path})")
                print(f">>> Available objects: {list(self._object_hand_init_dof_positions.keys())}")
                print(">>> Using default hand_init_dof_positions (all 0.0)")
                self.hand_init_dof_positions = {}

    def _create_envs(self, num_envs, spacing, num_per_row):
        print(">>> Setting up %d environments" % num_envs)
        num_per_row = int(np.sqrt(num_envs))

        # super()._create_envs(num_envs, spacing, num_per_row)
        
        # =========================================================== AllegroHand Task _create_envs() ===================================================
        # directly calling parent class's _create_envs() will make wrong number of rigid bodies
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        print(">>> Defining gym assets")
        
        asset_root = self._asset_root
        object_mode = self.cfg["env"]["objectMode"]
        if object_mode == "xml":
            object_asset_file = self.asset_files_dict[self.object_type]
        elif object_mode == "urdf":
            urdf_rel = str(self.cfg["env"]["objectUrdfPath"]).strip()
            if urdf_rel == "":
                raise ValueError("env.objectUrdfPath must be set when objectMode == 'urdf'")
            # asset = self.gym.load_asset(self.sim, self._asset_root, urdf_rel, target_asset_options)
            object_asset_file = urdf_rel
            print(">>> Loaded URDF asset: ", urdf_rel)
        else:
            raise ValueError(f"Unsupported objectMode: {object_mode}")

        self.gym_assets["current"]["robot"] = self._define_allegro_hand()
        allegro_hand_asset = self.gym_assets["current"]["robot"]["asset"]
        shadow_hand_dof_props = self.gym_assets["current"]["robot"]["dof_props"]
        
        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))
        # shadow_hand_start_pose.p.y = shadow_hand_start_pose.p.y - 0.5
        if not self.use_upside_down:
            shadow_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0 * np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.0 * np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.5 * np.pi)
        else:
            shadow_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 1.0 * np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.0 * np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.5 * np.pi)
        
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()

        # 初始化物体缩放（默认1.0）
        object_scale = 1.0
        
        if not self.use_upside_down:
            # ========== 手心向上：使用默认配置 ==========
            # Extract object name from path to check object type
            object_name = None
            object_urdf_path = self.cfg["env"].get("objectUrdfPath", "")
            if object_urdf_path:
                path_parts = object_urdf_path.split('/')
                for part in path_parts:
                    part_clean = part.replace('.urdf', '')
                    # Check for objects that need special handling
                    if part_clean in ["cell_phone", "cube_16x3x3", "cube_16x3x3_2", "letter_N", "letter_S", "letter_U", "letter_R"]:
                        object_name = part_clean
                        break
            
            # Set initial position based on object type
            # letter_N, letter_S, letter_U use (0.00, 0.03, 0.15)
            # All other objects use (0.05, 0.0, 0.15)
            if object_name in ["letter_N", "letter_S", "letter_U", "letter_R"]:
                pose_dx, pose_dy, pose_dz = 0.00, 0.03, 0.15
            else:
                # Default configuration for other objects (including cell_phone and cube_16x3x3)
                pose_dx, pose_dy, pose_dz = 0.05, 0.0, 0.15



            object_start_pose.p.x = shadow_hand_start_pose.p.x + pose_dx
            object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
            object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

            if self.object_type == "pen":
                object_start_pose.p.z = shadow_hand_start_pose.p.z + 0.02
            
            # Set fixed object orientation based on object type
            # - letter_N, letter_S, letter_U: use (0.707, 0.0, 0.0, 0.707)
            # - Other objects: use identity quaternion [0, 0, 0, 1]
            # Note: In reset_idx:
            #   - cell_phone, cube_16x3x3, cube_16x3x3_2, letter_N, letter_S, letter_U will keep fixed rotation (no random rotation)
            #   - Other objects will get random rotation
            if object_name in ["letter_N", "letter_S", "letter_U", "letter_R"]:
                object_start_pose.r = gymapi.Quat(0.707, 0.0, 0.0, 0.707)
            else:
                object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            # Save object name for use in reset_idx
            self.object_name_upside = object_name
        else:
            # ========== 手心向下：根据物体类型选择初始位姿 ==========
            # Extract object name from path (e.g., "contactdb/elephant/elephant.urdf" -> "elephant")
            object_name = None
            object_urdf_path = self.cfg["env"].get("objectUrdfPath", "")
            if object_urdf_path:
                path_parts = object_urdf_path.split('/')
                for part in path_parts:
                    part_clean = part.replace('.urdf', '')
                    if part_clean in self._object_initial_pose:
                        object_name = part_clean
                        break
            
            if object_name and object_name in self._object_initial_pose:
                # 使用配置中的初始位姿
                pose_config = self._object_initial_pose[object_name]
                pose_dx, pose_dy, pose_dz = pose_config["position_offset"]
                rotation_quat = pose_config["rotation_quat"]
                object_scale = pose_config.get("scale", 1.0)  # 获取缩放配置，默认1.0
                
                object_start_pose.p.x = shadow_hand_start_pose.p.x + pose_dx
                object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
                object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz
                
                # 使用配置中的旋转四元数
                object_start_pose.r = gymapi.Quat(*rotation_quat)
                
                print(f">>> useUpsideDown=True: Auto-selected initial pose for object '{object_name}'")
                print(f"    position_offset: {pose_config['position_offset']}")
                print(f"    rotation_quat: {rotation_quat}")
                print(f"    scale: {object_scale}")
                
                # Save object name for use in reset_target_pose (for special goal orientation handling)
                self.object_name_downside = object_name
            else:
                pose_dx, pose_dy, pose_dz = -0.03, 0.02, -0.07
                object_start_pose.p.x = shadow_hand_start_pose.p.x + pose_dx
                object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
                object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz
                object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                object_scale = 1.0  
                
                if object_urdf_path:
                    print(f">>> Warning: Object '{object_name}' not found in _object_initial_pose (path: {object_urdf_path})")
                    print(f">>> Available objects: {list(self._object_initial_pose.keys())}")
                    print(">>> Using default object pose")
                else:
                    print(">>> Warning: objectUrdfPath not found, using default object pose")
                
                # Set object_name_downside to None if not found
                self.object_name_downside = None
        
        self.object_scale = object_scale

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + 2
        max_agg_shapes = self.num_shadow_hand_shapes + 2

        self.allegro_hands = []
        self.envs = []

        self.cameras_handle = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        # self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        shadow_hand_rb_count = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(shadow_hand_rb_count, shadow_hand_rb_count + object_rb_count))


        self._define_camera()

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            allegro_hand_actor = self.gym.create_actor(env_ptr, allegro_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)

            # apply object scale (if configured)
            if hasattr(self, 'object_scale') and self.object_scale != 1.0:
                self.gym.set_actor_scale(env_ptr, object_handle, self.object_scale)

            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            
            # apply goal object scale (same as object)
            if hasattr(self, 'object_scale') and self.object_scale != 1.0:
                self.gym.set_actor_scale(env_ptr, goal_handle, self.object_scale)

            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.enable_rendered_pointcloud_observation:
                for k in range(self.num_cameras_per_env):
                    camera = self.gym.create_camera_sensor(env_ptr, self.camera_properties)
                    self.cameras_handle.append(camera)

                    self.gym.set_camera_location(
                        camera, env_ptr, self._camera_positions[k], self._camera_target_locations[k]
                    )
                    depth_image = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera, gymapi.IMAGE_DEPTH)
                    depth_image = gymtorch.wrap_tensor(depth_image)

                    seg_image = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera, gymapi.IMAGE_SEGMENTATION)
                    seg_image = gymtorch.wrap_tensor(seg_image)

                    view_matrix = self.gym.get_camera_view_matrix(self.sim, env_ptr, camera)
                    proj_matrix = self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera)

                    view_matrix = torch.tensor(view_matrix).to(self.device)
                    proj_matrix = torch.tensor(proj_matrix).to(self.device)
                    inv_view_matrix = torch.inverse(view_matrix)

                    origin: gymapi.Vec3 = self.gym.get_env_origin(env_ptr)
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

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_hand_actor)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        
        # ===================================================== Allegro Hand _create_envs END =====================================================
        
        env = self.envs[-1] # it seems we can use the last env to get indices
        
        allegro_hand = self.gym.find_actor_handle(env, "hand")
        self.allegro_hand_index = self.gym.get_actor_index(env, allegro_hand, gymapi.DOMAIN_ENV)
        
        # define start and end indices for allegro hand DOFs to create contiguous slices
        self.allegro_hand_dof_start = self.gym.get_actor_dof_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_dof_end = self.allegro_hand_dof_start + self.gym_assets["current"]["robot"]["num_dofs"]
        self.allegro_hand_indices = torch.tensor(self.hand_indices).long().to(self.device)
        self.allegro_hand_rigid_body_start = self.gym.get_actor_rigid_body_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_rigid_body_end = (
            self.allegro_hand_rigid_body_start + self.gym_assets["current"]["robot"]["num_rigid_bodies"]
        )
        
    def _define_allegro_hand(
        self, asset_name: str = "hand"
    ) -> Dict[str, Any]:
        """Define & load the allegro Hand  asset.

        Args:
            asset_name (str, optional): Asset name for logging. Defaults to "hand".

        Returns:
            Dict[str, Any]: The configuration of the robot.
        """
        print(">>> Loading allegro Hand for current scene")
        config = {"name": "hand"}

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
            
        asset_filename = self._xarm_allegro_hand_right_asset_file # self.cfg["env"]["asset"].get("assetFileName")

        allegro_hand_asset = self.gym.load_asset(self.sim, self._asset_root, asset_filename, asset_options)
        if self.env_info_logging:
            print_links_and_dofs(self.gym, allegro_hand_asset, asset_name)

        config["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        config["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        config["num_dofs"] = self.gym.get_asset_dof_count(allegro_hand_asset)
        config["num_actuators"] = self.gym.get_asset_actuator_count(allegro_hand_asset)
        config["num_tendons"] = self.gym.get_asset_tendon_count(allegro_hand_asset)

        num_dofs = config["num_dofs"]

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        print("Num dofs: ", self.num_shadow_hand_dofs)
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

            print("Max effort: ", shadow_hand_dof_props['effort'][i])
            shadow_hand_dof_props['effort'][i] = 0.5
            shadow_hand_dof_props['stiffness'][i] = 3
            shadow_hand_dof_props['damping'][i] = 0.1
            shadow_hand_dof_props['friction'][i] = 0.01
            shadow_hand_dof_props['armature'][i] = 0.001

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # hand_dof_idx = 0

        # # set rigid-shape properties for allegro-hand
        # rigid_shape_props = self.gym.get_asset_rigid_shape_properties(allegro_hand_asset)
        # for shape in rigid_shape_props:
        #     print("shape.friction:", shape.friction)
        #     shape.friction = 0.8
        
        # self.gym.set_asset_rigid_shape_properties(allegro_hand_asset, rigid_shape_props)
        
        # for i in range(num_dofs):
        #     name = self.gym.get_asset_dof_name(allegro_hand_asset, i)
        #     shadow_hand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
        #     if name.endswith(".0"):
        #         dof_props["stiffness"][i] = 30
        #         dof_props["damping"][i] = 1
        #         dof_props["velocity"][i] = 3.0
        #         dof_props["effort"][i] = 5
        #         hand_dof_idx += 1
        #     else:
        #         dof_props["stiffness"][i] = 4000
        #         dof_props["damping"][i] = 80
        #         # dof_props["stiffness"][i] = 1e6
        #         # dof_props["damping"][i] = 1e2

        if self.env_info_logging:
            print_dof_properties(self.gym, allegro_hand_asset, shadow_hand_dof_props, asset_name)

        dof_lower_limits = [shadow_hand_dof_props["lower"][i] for i in range(num_dofs)]
        dof_upper_limits = [shadow_hand_dof_props["upper"][i] for i in range(num_dofs)]
        dof_init_positions = [0.0 for _ in range(num_dofs)]
        dof_init_velocities = [0.0 for _ in range(num_dofs)]

        config["limits"] = {}
        config["limits"]["lower"] = torch.tensor(dof_lower_limits).float().to(self.device)
        config["limits"]["upper"] = torch.tensor(dof_upper_limits).float().to(self.device)

        config["init"] = {}
        config["init"]["position"] = torch.tensor(dof_init_positions).float().to(self.device)
        config["init"]["velocity"] = torch.tensor(dof_init_velocities).float().to(self.device)

        # Apply hand_init_dof_positions if defined LXW
        if hasattr(self, 'hand_init_dof_positions') and self.hand_init_dof_positions:
            for name, value in self.hand_init_dof_positions.items():
                idx = self.gym.find_asset_dof_index(allegro_hand_asset, name)
                if idx != -1:
                    dof_init_positions[idx] = value
                    print(f"  Set initial DOF {name} (index {idx}) = {value}")
                else:
                    print(f"  Warning: DOF '{name}' not found in asset")
        
        # Save initial DOF positions tensor for reset_idx LXW
        self.hand_init_dof_positions_tensor = torch.tensor(dof_init_positions, dtype=torch.float, device=self.device)
        
        # fmt: off
        close_dof_names = [
            "joint_2.0", "joint_3.0",  # finger 0 (index)
            "joint_6.0", "joint_7.0",  # finger 1 (middle)
            "joint_10.0", "joint_11.0",  # finger 2 (ring)
            "joint_14.0", "joint_15.0",  # thumb
        ]
        # fmt: on

        self.close_dof_indices = torch.tensor(
            [self.gym.find_asset_dof_index(allegro_hand_asset, name) for name in close_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        
        # Keypoints: flatten multi-kp per link
        keypoint_indices: List[int] = []
        keypoint_offsets: List[List[float]] = []
        link_name_to_index: Dict[str, int] = {}
        for link in self._hand_links:
            li = self.gym.find_asset_rigid_body_index(allegro_hand_asset, link)
            if li != -1:
                link_name_to_index[link] = li
        for link_name, sub_k in self._flattened_keypoint_pairs:
            if link_name not in link_name_to_index:
                continue
            keypoint_indices.append(link_name_to_index[link_name])
            keypoint_offsets.append(self._keypoints_info[link_name][sub_k])

        self.allegro_center_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, self._allegro_hand_center_prim)
        # self.keypoint_indices = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, prim) for prim in self._keypoints]
        self.keypoint_indices = keypoint_indices
        self.fingertip_indices = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, prim) for prim in self._fingertips]
        self.virtual_tip_indices = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, name) for name in self._tip_links]
        self.hand_link_indices = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, prim) for prim in self._hand_links]
        # self.keypoint_offset = [self._keypoints_info[link_name] for link_name in self._keypoints] # (#key_link, 1, 3)
        self.keypoint_offset = keypoint_offsets
        self.fingertip_link_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._fingertips]
        self.index_finger_links_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._index_finger_links]
        self.middle_finger_links_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._middle_finger_links]
        self.ring_finger_links_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._ring_finger_links]
        self.thumb_finger_links_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._thumb_links]
        
        self.index_finger_links_indices_among_hand_links = [self._hand_links.index(link_name) for link_name in self._index_finger_links]
        self.middle_finger_links_indices_among_hand_links = [self._hand_links.index(link_name) for link_name in self._middle_finger_links]
        self.ring_finger_links_indices_among_hand_links = [self._hand_links.index(link_name) for link_name in self._ring_finger_links]
        self.thumb_finger_links_indices_among_hand_links = [self._hand_links.index(link_name) for link_name in self._thumb_links]  
        
        config["asset"] = allegro_hand_asset
        config["dof_props"] = shadow_hand_dof_props

        print(">>> Allegro Hand loaded")
        return config
    
    def _refresh_sim_tensors(self) -> None:
        super()._refresh_sim_tensors()

        # overwrite fingertip states with virtual tip heads
        self.fingertip_states = self.allegro_hand_rigid_body_states[:, self.virtual_tip_indices, :]
        self.fingertip_positions = self.fingertip_states[..., 0:3]
        self.fingertip_orientations = self.fingertip_states[..., 3:7]
        self.fingertip_linear_velocities = self.fingertip_states[..., 7:10]
        self.fingertip_angular_velocities = self.fingertip_states[..., 10:13]
        
        self.index_fingertip_positions = self.fingertip_positions[:, 0, :]
        self.middle_fingertip_positions = self.fingertip_positions[:, 1, :]
        self.ring_fingertip_positions = self.fingertip_positions[:, 2, :]
        self.thumb_fingertip_positions = self.fingertip_positions[:, 3, :]
        
        self.hand_link_states = self.allegro_hand_rigid_body_states[:, self.hand_link_indices, :]
        self.hand_link_positions = self.hand_link_states[..., 0:3]
        self.hand_link_orientations = self.hand_link_states[..., 3:7]
        
        self.index_finger_keypoint_positions_with_offset = self.hand_link_positions[:, self.index_finger_links_indices_among_hand_links, :]
        self.middle_finger_keypoint_positions_with_offset = self.hand_link_positions[:, self.middle_finger_links_indices_among_hand_links, :]
        self.ring_finger_keypoint_positions_with_offset = self.hand_link_positions[:, self.ring_finger_links_indices_among_hand_links, :]
        self.thumb_finger_keypoint_positions_with_offset = self.hand_link_positions[:, self.thumb_finger_links_indices_among_hand_links, :]
        
        self.index_finger_keypoint_orientations_with_offset = self.hand_link_orientations[:, self.index_finger_links_indices_among_hand_links, :]
        self.middle_finger_keypoint_orientations_with_offset = self.hand_link_orientations[:, self.middle_finger_links_indices_among_hand_links, :]
        self.ring_finger_keypoint_orientations_with_offset = self.hand_link_orientations[:, self.ring_finger_links_indices_among_hand_links, :]
        self.thumb_finger_keypoint_orientations_with_offset = self.hand_link_orientations[:, self.thumb_finger_links_indices_among_hand_links, :]
        
        net_contact_forces = self.net_contact_forces.view(self.num_envs, self.num_rigid_bodies, 3)
        self.hand_link_contact_forces = net_contact_forces[:, self.hand_link_indices, :]
        self.index_finger_keypoint_forces = self.hand_link_contact_forces[:, self.index_finger_links_indices_among_hand_links, :]
        self.middle_finger_keypoint_forces = self.hand_link_contact_forces[:, self.middle_finger_links_indices_among_hand_links, :]
        self.ring_finger_keypoint_forces = self.hand_link_contact_forces[:, self.ring_finger_links_indices_among_hand_links, :]
        self.thumb_finger_keypoint_forces = self.hand_link_contact_forces[:, self.thumb_finger_links_indices_among_hand_links, :]

    def reset_idx(self, env_ids: torch.LongTensor, goal_env_ids: torch.LongTensor, first_time=False) -> None:
        """Reset environments with different behavior for upside_down mode.
        
        - For use_upside_down=False: Use AllegroHand.reset_idx logic (copy.py behavior)
        - For use_upside_down=True: Custom reset with fixed object pose (no noise)
        """

        if self.randomize and self.randomize_mass:
            lower, upper = self.randomize_mass_lower, self.randomize_mass_upper
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)
                for p in prop:
                    p.mass = np.random.uniform(lower, upper)
                self.gym.set_actor_rigid_body_properties(env, handle, prop)
    
        if self.run_consective_goals:
            if not self.use_upside_down:
                # ========== 手心向上：严格按照 AllegroHand.reset_idx 的逻辑（copy.py配置） ==========
                # 完全复制 AllegroHand.reset_idx 的实现，不覆盖物体位姿
                num_reset_envs = len(env_ids)
                
                # generate random values
                rand_floats = torch_rand_float(-1.0, 1.0, (num_reset_envs, self.num_shadow_hand_dofs * 2 + 5), device=self.device)

                # randomize start object poses
                self.reset_target_pose(env_ids)

                # reset rigid body forces
                self.rb_forces[env_ids, :, :] = 0.0

                # reset object (with position and rotation noise from parent)
                self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
                self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
                    self.reset_position_noise * rand_floats[:, 0:2]
                self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
                    self.reset_position_noise * rand_floats[:, self.up_axis_idx]

                # Special handling for rotation:
                # - cell_phone, cube_16x3x3, cube_16x3x3_2, letter_N, letter_S, letter_U: no random rotation, use fixed initial rotation
                # - Other objects: random rotation
                if hasattr(self, 'object_name_upside') and self.object_name_upside in ["cell_phone", "cube_16x3x3", "cube_16x3x3_2", "letter_N", "letter_S", "letter_U", "letter_R"]:
                    # Use fixed initial rotation for these objects
                    # - cell_phone, cube_16x3x3, cube_16x3x3_2: identity quaternion (0, 0, 0, 1)
                    # - letter_N, letter_S, letter_U: (0.707, 0.0, 0.0, 0.707)
                    self.root_state_tensor[self.object_indices[env_ids], 3:7] = self.object_init_state[env_ids, 3:7]
                else:
                    # Apply random rotation for other objects
                    new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
                    if self.object_type == "pen":
                        rand_angle_y = torch.tensor(0.3)
                        new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                                self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])
                    self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
                
                self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

                object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                        self.goal_object_indices[env_ids],
                                                        self.goal_object_indices[goal_env_ids]]).to(torch.int32))
                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.root_state_tensor),
                                                            gymtorch.unwrap_tensor(object_indices), len(object_indices))

                # reset random force probabilities
                self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                            * torch.rand(num_reset_envs, device=self.device) + torch.log(self.force_prob_range[1]))

                # reset shadow hand (with noise from parent)
                delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
                delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
                rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5:5+self.num_shadow_hand_dofs] + 1)

                pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
                self.shadow_hand_dof_pos[env_ids, :] = pos
                self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
                    self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]
                self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
                self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

                hand_indices = self.hand_indices[env_ids].to(torch.int32)
                self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                                gymtorch.unwrap_tensor(self.prev_targets),
                                                                gymtorch.unwrap_tensor(hand_indices), len(env_ids))

                self.gym.set_dof_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.dof_state),
                                                    gymtorch.unwrap_tensor(hand_indices), len(env_ids))

                self.progress_buf[env_ids] = 0
                self.reset_buf[env_ids] = 0
                self.successes[env_ids] = 0
                self.actions[env_ids] = 0.0

            else:
                # ========== 手心向下：完全自定义reset，确保所有物体位姿相同（无噪声） ==========
                num_reset_envs = len(env_ids)
                
                # Refresh state tensors
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)
                
                # ========== 1. Reset object: 固定位姿，无噪声 ==========
                object_indices_reset = self.object_indices[env_ids].to(torch.int32)
                # 使用 object_init_state 中的的固定位姿（所有环境相同）
                self.root_state_tensor[object_indices_reset, 0:3] = self.object_init_state[env_ids, 0:3]
                self.root_state_tensor[object_indices_reset, 3:7] = self.object_init_state[env_ids, 3:7]  # 固定orientation
                self.root_state_tensor[object_indices_reset, 7:10] = 0.0  # Zero linear velocity
                self.root_state_tensor[object_indices_reset, 10:13] = 0.0  # Zero angular velocity
                
                # ========== 2. Reset goal: 调用 reset_target_pose ==========
                self.reset_target_pose(env_ids)
                
                # ========== 3. Reset hand DOF: 使用 hand_init_dof_positions ==========
                if hasattr(self, 'hand_init_dof_positions_tensor') and self.hand_init_dof_positions_tensor is not None:
                    hand_init_pos = self.hand_init_dof_positions_tensor.unsqueeze(0).expand(num_reset_envs, -1)
                    self.shadow_hand_dof_pos[env_ids] = hand_init_pos
                    self.shadow_hand_dof_vel[env_ids] = 0.0
                    
                    global_hand_dof_indices = torch.arange(
                        self.allegro_hand_dof_start, 
                        self.allegro_hand_dof_end, 
                        dtype=torch.long, 
                        device=self.device
                    )
                    self.cur_targets[env_ids][:, global_hand_dof_indices] = hand_init_pos
                    self.prev_targets[env_ids][:, global_hand_dof_indices] = hand_init_pos
                    
                    dof_state_view = self.dof_state.view(self.num_envs, -1, 2)
                    hand_dof_start = self.allegro_hand_dof_start
                    hand_dof_end = self.allegro_hand_dof_end
                    
                    dof_state_view[env_ids, hand_dof_start:hand_dof_end, 0] = hand_init_pos
                    dof_state_view[env_ids, hand_dof_start:hand_dof_end, 1] = 0.0
                    
                    hand_indices = self.hand_indices[env_ids].to(torch.int32)
                    self.gym.set_dof_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(self.dof_state),
                        gymtorch.unwrap_tensor(hand_indices),
                        len(hand_indices),
                    )
                    
                    self.gym.set_dof_position_target_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(self.cur_targets),
                        gymtorch.unwrap_tensor(hand_indices),
                        len(hand_indices),
                    )
                
                # ========== 4. Reset rigid body forces ==========
                self.rb_forces[env_ids, :, :] = 0.0
                
                # ========== 5. Reset random force probabilities ==========
                if hasattr(self, 'random_force_prob'):
                    self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                                * torch.rand(num_reset_envs, device=self.device) + torch.log(self.force_prob_range[1]))
                
                # ========== 6. Apply all state changes to simulation ==========
                # Combine object and goal indices
                all_object_indices = torch.unique(torch.cat([
                    object_indices_reset,
                    self.goal_object_indices[env_ids].to(torch.int32),
                    self.goal_object_indices[goal_env_ids].to(torch.int32)
                ]))
                
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.root_state_tensor),
                    gymtorch.unwrap_tensor(all_object_indices),
                    len(all_object_indices)
                )
                
                # !! we have new api now 
                # ========== 7. Reset curiosity manager states ==========
                # if hasattr(self, 'reach_curiosity_mgr'):
                #     if self.reach_curiosity_mgr.potential_per_kp_max is None:
                #         self.reach_curiosity_mgr.potential_per_kp_max = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
                #     self.reach_curiosity_mgr.potential_per_kp_max[env_ids] = 0
                #     if self.reach_curiosity_mgr.use_contact_coverage_max_delta and self.reach_curiosity_mgr.contact_coverage_per_kp_max is not None:
                #         self.reach_curiosity_mgr.contact_coverage_per_kp_max[env_ids] = 0
                
                # ========== 8. Reset buffers ==========
                self.progress_buf[env_ids] = 0
                self.reset_buf[env_ids] = 0
                self.successes[env_ids] = 0
                # self.reset_goal_buf[env_ids] = 0
                self.actions[env_ids] = 0.0
        else:

            # generate random values
            rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

            # randomize start object poses
            self.reset_target_pose(env_ids)

            # reset rigid body forces
            self.rb_forces[env_ids, :, :] = 0.0

            # reset object
            self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
            self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
                self.reset_position_noise * rand_floats[:, 0:2]
            self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
                self.reset_position_noise * rand_floats[:, self.up_axis_idx]

            new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
            if self.object_type == "pen":
                rand_angle_y = torch.tensor(0.3)
                new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                        self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

            
            self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
            self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

            object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                    self.goal_object_indices[env_ids],
                                                    self.goal_object_indices[goal_env_ids]]).to(torch.int32))
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state_tensor),
                                                        gymtorch.unwrap_tensor(object_indices), len(object_indices))

            # reset random force probabilities
            self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                        * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

            # reset shadow hand
            delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
            delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
            rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5:5+self.num_shadow_hand_dofs] + 1)

            pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
            self.shadow_hand_dof_pos[env_ids, :] = pos
            self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
                self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]
            self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
            self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

            hand_indices = self.hand_indices[env_ids].to(torch.int32)
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.prev_targets),
                                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(hand_indices), len(env_ids))

            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
            self.successes[env_ids] = 0
            self.actions[env_ids] = 0.0

        # reset curiosity running-max buffers (supports both legacy and state-conditioned shapes)
        self.reach_curiosity_mgr.ensure_running_max_buffers(self.num_envs)
        self.reach_curiosity_mgr.reset_running_max_buffers(env_ids)

    def reset_target_pose(self, env_ids, apply_reset=False):
        """Reset target (goal) pose with random orientations.
        
        Uses the same method as AllegroHand.reset_target_pose: randomize_rotation
        For cube_16x3x3_2 in upside_down mode: only rotate around z-axis (horizontal plane)
        """
        num_reset_envs = len(env_ids)
        
        # Check if this is cube_16x3x3_2 in upside_down mode (needs horizontal plane rotation only)
        if (hasattr(self, 'use_upside_down') and self.use_upside_down and 
            hasattr(self, 'object_name_downside') and self.object_name_downside == "cube_16x3x3_2"):
            # For cube_16x3x3_2: only rotate around z-axis (horizontal plane), keep object flat
            # Generate random angle around z-axis (0 to 2*pi)
            rand_floats = torch_rand_float(-1.0, 1.0, (num_reset_envs, 1), device=self.device)
            # Random angle from 0 to 2*pi
            z_angle = rand_floats[:, 0] * np.pi
            # Create quaternion rotation around z-axis only
            new_rot = quat_from_angle_axis(z_angle, self.z_unit_tensor[env_ids])
        else:
            # Use the same method as AllegroHand: randomize_rotation with random floats
            rand_floats = torch_rand_float(-1.0, 1.0, (num_reset_envs, 4), device=self.device)
            new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        # Reset goal position to initial state
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]

        self.goal_states[env_ids, 3:7] = new_rot
        
        # Update root_state_tensor for goal objects
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = (
            self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        )
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
        )
        
        # Apply reset to simulation if requested
        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(goal_object_indices),
                len(goal_object_indices)
            )
        
            # Clear reset goal buffer
            self.reset_goal_buf[env_ids] = 0

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot