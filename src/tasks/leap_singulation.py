import os
import json
from typing import Any, Dict, List, Tuple
import enum

import torch

from isaacgym import gymapi

from .allegro_singulation import (
    XArmAllegroHandFunctionalManipulationUnderarm as _LeapImplBase,
    ForceSensorSpec,
)
from .isaacgym_utils import (
    print_asset_options,
    print_dof_properties,
)

class XArmLeapHandUnderarmDimensions(enum.Enum):
    """Dimension constants for Isaac Gym with xArm6 + Allegro Hand."""

    POSE_DIM = 7
    VELOCITY_DIM = 6
    STATE_DIM = 13
    WRENCH_DIM = 6

    NUM_FINGERTIPS = 4
    NUM_DOFS = 22

    WRIST_TRAN = 3
    WRIST_ROT = 3

    HAND_ACTUATED_DIM = 16
    NUM_KEYPOINTS = 21


class XArm7LeapHandUnderarmDimensions(enum.Enum):
    POSE_DIM = 7
    VELOCITY_DIM = 6
    STATE_DIM = 13
    WRENCH_DIM = 6
    NUM_FINGERTIPS = 4
    NUM_DOFS = 23
    WRIST_TRAN = 3
    WRIST_ROT = 3
    HAND_ACTUATED_DIM = 16
    NUM_KEYPOINTS = 21

class XArmLeapHandFunctionalManipulationUnderarm(_LeapImplBase):
    """Leap Hand + xArm6 task using the same control method as the Allegro base task.

    This subclass swaps the robot URDF, DOF/link names, fingertip/force sensors,
    and expands keypoints to support multiple keypoints per link.
    """

    # Use the Leap URDF
    _xarm_allegro_hand_right_asset_file: str = "xarm6_leap_vertical_moving.urdf"
    _xarm_allegro_hand_left_asset_file: str = "xarm6_leap_vertical_moving.urdf"
    
    _dims = XArmLeapHandUnderarmDimensions

    # xArm DOFs
    _xarm_dof_names: List[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

    _arm_links: List[str] = ["link_base", "link1", "link2", "link3", "link4", "link5", "link6"]

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
    _index_finger_links: List[str] = ["pip", "dip", "fingertip", "index_tip_head"]
    _middle_finger_links: List[str] = ["pip_2", "dip_2", "fingertip_2", "middle_tip_head"]
    _ring_finger_links: List[str] = ["pip_3", "dip_3", "fingertip_3", "ring_tip_head"]
    _thumb_links: List[str] = ["thumb_pip", "thumb_dip", "thumb_fingertip", "thumb_tip_head"]

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

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self._allegro_hand_center_prim = self._hand_center_link
        self._allegro_hand_palm_prim = self._hand_palm_link

        self.allegro_hand_init_dof_positions = dict(self.hand_init_dof_positions)

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
        # overwrite xarm init dof positions
        # self._xarm_right_init_dof_positions = {
        #     "joint1": 0.0,
        #     "joint2":-0.4,
        #     "joint3":-1.4,
        #     "joint4": 0.0,
        #     "joint5": 1.8,
        #     "joint6": 0.0,
        # }
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _define_allegro_hand_with_arm(self, asset_name: str = "Leap Hand + xArm6") -> Dict[str, Any]:
        """Define & load Leap Hand + xArm6 asset and configure indices, sensors, keypoints."""
        print(">>> Loading Leap Hand + xArm6 for current scene")
        # config: Dict[str, Any] = {"name": "leap_hand"}
        config: Dict[str, Any] = {"name": "allegro_hand"}

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        if self.env_info_logging:
            print_asset_options(asset_options, asset_name)        
        

        asset_filename = self._xarm_allegro_hand_right_asset_file
        asset = self.gym.load_asset(self.sim, self._asset_root, asset_filename, asset_options)

        config["num_rigid_bodies"] = self.gym.get_asset_rigid_body_count(asset)
        config["num_rigid_shapes"] = self.gym.get_asset_rigid_shape_count(asset)
        config["num_dofs"] = self.gym.get_asset_dof_count(asset)
        config["num_actuators"] = self.gym.get_asset_actuator_count(asset)
        config["num_tendons"] = self.gym.get_asset_tendon_count(asset)

        num_dofs = config["num_dofs"]

        # DOF properties
        dof_props = self.gym.get_asset_dof_properties(asset)

        # Set friction for hand shapes
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        for shape in rigid_shape_props:
            shape.friction = 0.8
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

        p_gain = torch.tensor(
            [3.52, 1.78, 2.84, 2.30, 1.94, 2.18,
             2.55, 2.01, 2.26, 2.30, 3.76, 4.64,
             1.86, 3.44, 4.82, 1.53], dtype=torch.float32 ) # 3.0
        d_gain = torch.tensor(
            [0.194, 0.106, 0.091, 0.195, 0.199, 0.192,
             0.149, 0.050, 0.088, 0.135, 0.027, 0.081,
             0.123, 0.042, 0.082, 0.068], dtype=torch.float32 ) # 0.1
        j = 0
        
        for i in range(num_dofs):
            name = self.gym.get_asset_dof_name(asset, i)
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if name in self._xarm_dof_names or name.startswith("joint"):
                # Arm DOFs
                dof_props["stiffness"][i] = 4000
                dof_props["damping"][i] = 80
            else:
                # hand DOFs
                pass
                dof_props["stiffness"][i] = 3.0
                dof_props["damping"][i] = 0.1
                dof_props["effort"][i] = 0.5
                dof_props["friction"][i] = 0.01
                dof_props["armature"][i] = 0.001
                j += 1
                # dof_props["stiffness"][i] = 30
                # dof_props["damping"][i] = 1
                # dof_props["velocity"][i] = 3.0

        if self.env_info_logging:
            print_dof_properties(self.gym, asset, dof_props, asset_name)

        dof_lower_limits = [dof_props["lower"][i] for i in range(num_dofs)]
        dof_upper_limits = [dof_props["upper"][i] for i in range(num_dofs)]
        dof_init_positions = [0.0 for _ in range(num_dofs)]
        dof_init_velocities = [0.0 for _ in range(num_dofs)]

        # xArm initial DOFs
        for name, value in self._xarm_right_init_dof_positions.items():
            dof_init_positions[self.gym.find_asset_dof_index(asset, name)] = value

        # Hand initial DOFs
        for name, value in self.allegro_hand_init_dof_positions.items():
            idx = self.gym.find_asset_dof_index(asset, name)
            if idx != -1:
                dof_init_positions[idx] = value

        config["limits"] = {}
        config["limits"]["lower"] = torch.tensor(dof_lower_limits).float().to(self.device)
        config["limits"]["upper"] = torch.tensor(dof_upper_limits).float().to(self.device)

        config["init"] = {}
        config["init"]["position"] = torch.tensor(dof_init_positions).float().to(self.device)
        config["init"]["velocity"] = torch.tensor(dof_init_velocities).float().to(self.device)

        # Sensors and DOF index groups
        if self.enable_contact_sensors:
            self.__define_contact_sensors(asset)
        self.__configure_robot_dof_indices(asset)

        # Close-DOF list (last two joints per finger)
        close_dof_names = [
            "index_joint_2",
            "index_joint_3",
            "middle_joint_2",
            "middle_joint_3",
            "ring_joint_2",
            "ring_joint_3",
            "thumb_joint_2",
            "thumb_joint_3",
        ]
        self.close_dof_indices = torch.tensor(
            [self.gym.find_asset_dof_index(asset, name) for name in close_dof_names],
            dtype=torch.long,
            device=self.device,
        )

        # Initial base pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self._xarm_right_init_position)
        pose.r = gymapi.Quat(*self._xarm_right_init_orientation)

        # Link indices
        self.hand_center_index = self.gym.find_asset_rigid_body_index(asset, self._hand_center_link)
        self.hand_palm_index = self.gym.find_asset_rigid_body_index(asset, self._hand_palm_link)

        self.allegro_center_index = self.hand_center_index
        self.allegro_palm_index = self.hand_palm_index
        self.fingertip_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._fingertips]
        self.virtual_tip_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._tip_links]

        self.arm_link_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._arm_links]
        self.hand_link_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._hand_links]
        assert (torch.tensor(self.arm_link_indices) >= 0).all(), "Arm link indices are not found in the asset"
        assert (torch.tensor(self.hand_link_indices) >= 0).all(), "Hand link indices are not found in the asset"
        
        self.index_finger_link_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._index_finger_links]
        self.middle_finger_link_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._middle_finger_links]
        self.ring_finger_link_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._ring_finger_links]
        self.thumb_link_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self._thumb_links]

        # Keypoints: flatten multi-kp per link
        keypoint_indices: List[int] = []
        keypoint_offsets: List[List[float]] = []
        link_name_to_index: Dict[str, int] = {}
        for link in self._hand_links:
            li = self.gym.find_asset_rigid_body_index(asset, link)
            if li != -1:
                link_name_to_index[link] = li
        for link_name, sub_k in self._flattened_keypoint_pairs:
            if link_name not in link_name_to_index:
                continue
            keypoint_indices.append(link_name_to_index[link_name])
            keypoint_offsets.append(self._keypoints_info[link_name][sub_k])

        # Fallback: if JSON missing
        # if len(keypoint_indices) == 0:
        #     for tip in self._tip_links:
        #         li = self.gym.find_asset_rigid_body_index(asset, tip)
        #         if li != -1:
        #             keypoint_indices.append(li)
        #             keypoint_offsets.append([0.0, 0.0, 0.0])

        self.keypoint_indices = keypoint_indices
        self.hand_link_indices_map = link_name_to_index
        # Shape: (1, K, 3)
        self.keypoint_offset = torch.tensor(keypoint_offsets, device=self.device, dtype=torch.float32).unsqueeze(0)

        # Build per-finger index lists among keypoints
        def indices_among_keypoints(links: List[str]) -> List[int]:
            idxs: List[int] = []
            for j, (ln, _) in enumerate(self._flattened_keypoint_pairs):
                if ln in links:
                    idxs.append(j)
            return idxs

        self.thumb_link_indices_among_keypoints = indices_among_keypoints(self._thumb_links)
        self.index_link_indices_among_keypoints = indices_among_keypoints(self._index_finger_links)
        self.middle_link_indices_among_keypoints = indices_among_keypoints(self._middle_finger_links)
        self.ring_link_indices_among_keypoints = indices_among_keypoints(self._ring_finger_links)

        # Choose a single representative kp per fingertip for 4-tip features
        def last_kp_for(link_name: str) -> int:
            last_idx = -1
            for j, (ln, _) in enumerate(self._flattened_keypoint_pairs):
                if ln == link_name:
                    last_idx = j
            return last_idx if last_idx >= 0 else 0

        # self.fingertip_link_indices_among_keypoints = [
        #     last_kp_for("index_tip_head"),
        #     last_kp_for("middle_tip_head"),
        #     last_kp_for("ring_tip_head"),
        #     last_kp_for("thumb_tip_head"),
        # ]
        self.fingertip_link_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._fingertips]

        config["asset"] = asset
        config["pose"] = pose
        config["dof_props"] = dof_props

        print(">>> xArm6 + Leap Hand loaded")
        return config

    def __define_contact_sensors(self, allegro_hand_asset) -> None:  # noqa: N802 (keep base naming)
        self._force_sensor_specs = [
            ForceSensorSpec("index_tip_head", "index_tip_head"),
            ForceSensorSpec("middle_tip_head", "middle_tip_head"),
            ForceSensorSpec("ring_tip_head", "ring_tip_head"),
            ForceSensorSpec("thumb_tip_head", "thumb_tip_head"),
        ]
        super()._XArmAllegroHandFunctionalManipulationUnderarm__define_contact_sensors(allegro_hand_asset)  # type: ignore[attr-defined]

    def __configure_robot_dof_indices(self, allegro_hand_asset) -> None:  # noqa: N802 (keep base naming)

        self._allegro_finger0_dof_names = [f"index_joint_{i}" for i in range(4)]
        self._allegro_finger1_dof_names = [f"middle_joint_{i}" for i in range(4)]
        self._allegro_finger2_dof_names = [f"ring_joint_{i}" for i in range(4)]
        self._allegro_thumb_dof_names = [f"thumb_joint_{i}" for i in range(4)]
        self._allegro_hand_dof_names = (
            self._allegro_finger0_dof_names
            + self._allegro_finger1_dof_names
            + self._allegro_finger2_dof_names
            + self._allegro_thumb_dof_names
        )
        self._allegro_digits_dof_names = (
            self._allegro_finger0_dof_names
            + self._allegro_finger1_dof_names
            + self._allegro_finger2_dof_names
            + self._allegro_thumb_dof_names
        )
        self._allegro_fingers_dof_names = (
            self._allegro_finger0_dof_names
            + self._allegro_finger1_dof_names
            + self._allegro_finger2_dof_names
        )

        super()._XArmAllegroHandFunctionalManipulationUnderarm__configure_robot_dof_indices(allegro_hand_asset)  # type: ignore[attr-defined]

    # def draw_link_keypoints(self) -> None:
    #     # Override to avoid relying on base _keypoints list (Allegro-specific size/order).
    #     # Optionally draw our flattened keypoints here if needed.
    #     return
    
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




class LeapFunctionalManipulationUnderarm:
    """Standalone Leap task class (no inheritance in public API).

    Internally composes the implementation and forwards attributes/methods.
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        use_xarm7 = bool(cfg["env"].get("useXarm7", False))
        Impl = XArm7LeapHandFunctionalManipulationUnderarm if use_xarm7 else XArmLeapHandFunctionalManipulationUnderarm
        self._impl = Impl(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

    def __getattr__(self, name: str):
        return getattr(self._impl, name)

    # Explicitly forward common VecTask methods for clarity
    def pre_physics_step(self, actions):
        return self._impl.pre_physics_step(actions)

    def post_physics_step(self):
        return self._impl.post_physics_step()

    def compute_observations(self, reset_env_ids=None):
        return self._impl.compute_observations(reset_env_ids)

    def compute_reward(self, actions):
        return self._impl.compute_reward(actions)

    def reset(self, dones=None, first_time=False):
        return self._impl.reset(dones, first_time)

    def reset_idx(self, env_ids, first_time=False):
        return self._impl.reset_idx(env_ids, first_time)


class XArm7LeapHandFunctionalManipulationUnderarm(XArmLeapHandFunctionalManipulationUnderarm):
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