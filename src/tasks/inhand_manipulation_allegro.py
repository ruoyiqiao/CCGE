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
    draw_points,
    draw_axes
)
import omegaconf
import warnings
from collections import OrderedDict, deque
import random
from .curiosity_reward_manager import CuriosityRewardManager
from pathlib import Path

from pytorch3d.ops import sample_farthest_points

from .allegro_singulation import XArmAllegroHandUnderarmDimensions

class InhandManipulationAllegro(AllegroHand):

    _grasp_task: bool = False
    _asset_root: str = "./assets"
    _dims = XArmAllegroHandUnderarmDimensions
    
    _observation_specs: Sequence[ObservationSpec] = []
    _action_specs: Sequence[ActionSpec] = []
    _allegro_hand_center_prim: str = "palm_link"
    _fingertips: List[str] = ["index_biotac_tip", "middle_biotac_tip", "ring_biotac_tip", "thumb_biotac_tip"] # index, middle, ring, thumb
    _keypoints: List[str] = [
        "base_link",
        "link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip",  # thumb
        "link_0.0", "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip",     # finger 0 (index)
        "link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip",     # finger 1 (middle)
        "link_8.0", "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip",   # finger 2 (ring)
    ]
    
    _fingertips: List[str] = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"] # index, middle, ring, thumb
    _index_finger_links: List[str] = ["link_1.0", "link_2.0", "link_3.0", "link_3.0_tip"]
    _middle_finger_links: List[str] = ["link_5.0", "link_6.0", "link_7.0", "link_7.0_tip"]
    _ring_finger_links: List[str] = ["link_9.0", "link_10.0", "link_11.0", "link_11.0_tip"]
    _thumb_links: List[str] = ["link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"]
    _allegro_hand_center_prim: str = "base_link"
    _allegro_hand_palm_prim: str = "palm"
    _keypoints_info_path: str = "assets/urdf/xarm6_allegro_right_keypoints.json"
    _keypoints_info: Dict[str, List[List[float]]] = json.load(open(_keypoints_info_path, "r"))
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        seed = cfg["env"]["seed"]
        torch.manual_seed(seed)  # cpu
        random.seed(seed)
        np.random.seed(seed)

        self.cfg = cfg
        
        self.randomize = self.cfg["task"]["randomize"]
        self.env_info_logging = self.cfg["logging"]["envInfo"]
        self.stack_frame_number = self.cfg["env"]["stackFrameNumber"]
        self.enable_contact_sensors = self.cfg["env"]["enableContactSensors"]
        self.reward_type = self.cfg["env"]["rewardType"]
        self.mode = self.cfg["env"]["mode"]
        self.velocity_observation_scale = self.cfg["env"]["velocityObservationScale"]
        
        self.gym_assets = {}
        self.gym_assets["current"] = {}
        
        self.__configure_mdp_spaces()
        
        # super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        # print("InhandManipulationAllegro: Overriding AllegroHand init")

        # =========================================================== AllegroHand Task __init__() ===================================================
        # directly calling parent class's __init__ will make wrong _observation_space

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.enable_rendered_pointcloud_observation = self.cfg["env"]["enableRenderedPointCloud"]
        self.enable_rendered_pointcloud_target_mask = self.cfg["env"]["enableRenderedPointCloudTargetMask"]

        self.target_segmentation_id = int(self.cfg["env"].get("targetSegmentationId", 1))

        self.num_rendered_points = self.cfg["env"]["numRenderedPointCloudPoints"]
        self.rendered_pointcloud_multiplier = self.cfg["env"]["renderedPointCloudMultiplier"]
        self.rendered_pointcloud_sample_method = self.cfg["env"]["renderedPointCloudSampleMethod"]
        self.rendered_pointcloud_gaussian_noise = self.cfg["env"]["renderedPointCloudGaussianNoise"]
        self.rendered_pointcloud_gaussian_noise_sigma = self.cfg["env"]["renderedPointCloudGaussianNoiseSigma"]
        self.rendered_pointcloud_gaussian_noise_ratio = self.cfg["env"]["renderedPointCloudGaussianNoiseRatio"]
        assert self.rendered_pointcloud_sample_method in ["farthest", "random"]


        self.run_consective_goals = self.cfg["env"].get("runConsectiveGoals", True)

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "full_no_vel": 50,
            "full": 72,
            "full_state": 88
        }

        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        self.use_upside_down = self.cfg["env"].get("useUpsideDown", False)  

        num_states = 0
        if self.asymmetric_obs:
            num_states = 88

        # self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        # self.cfg["env"]["numStates"] = num_states
        # self.cfg["env"]["numActions"] = 16

        self._setup_domain_rand_cfg(self.cfg['domain_rand'])


        super(AllegroHand, self).__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
        #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

             dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
             self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        # =================================================================== AllegroHand Task __init__() END ===================================================   
        
        dof_state = self.dof_state.view(self.num_envs, -1, 2)
        self.allegro_hand_dof_positions = dof_state[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end, 0]
        self.allegro_hand_dof_velocities = dof_state[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end, 1]
        self.allegro_hand_dof_forces = self.dof_force_tensor[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end]
        
        self.allegro_hand_rigid_body_states = self.rigid_body_states[
            :, self.allegro_hand_rigid_body_start : self.allegro_hand_rigid_body_end, :
        ]    
        self.allegro_hand_rigid_body_positions = self.allegro_hand_rigid_body_states[..., 0:3]
        self.allegro_hand_rigid_body_orientations = self.allegro_hand_rigid_body_states[..., 3:7]
        self.allegro_hand_rigid_body_linear_velocities = self.allegro_hand_rigid_body_states[..., 7:10]
        self.allegro_hand_rigid_body_angular_velocities = self.allegro_hand_rigid_body_states[..., 10:13]
        
        self.allegro_hand_center_states = self.allegro_hand_rigid_body_states[:, self.allegro_center_index, :]
        self.allegro_hand_center_positions = self.allegro_hand_center_states[:, 0:3]
        self.allegro_hand_center_orientations = self.allegro_hand_center_states[:, 3:7]
        
        self.keypoint_offset = torch.tensor(self.keypoint_offset, device=self.device).reshape(1, -1, 3)
        self.fingertip_link_indices_among_keypoints = torch.tensor(self.fingertip_link_indices_among_keypoints, device=self.device)
        self.index_finger_links_indices_among_keypoints = torch.tensor(self.index_finger_links_indices_among_keypoints, device=self.device)
        self.middle_finger_links_indices_among_keypoints = torch.tensor(self.middle_finger_links_indices_among_keypoints, device=self.device)
        self.ring_finger_links_indices_among_keypoints = torch.tensor(self.ring_finger_links_indices_among_keypoints, device=self.device)
        self.thumb_finger_links_indices_among_keypoints = torch.tensor(self.thumb_finger_links_indices_among_keypoints, device=self.device)
        

        
        self.__create_object_dataset(device=sim_device)
                
        _net_contact_forces: torch.Tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.num_rigid_bodies: int = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs
        self.net_contact_forces: torch.Tensor = gymtorch.wrap_tensor(_net_contact_forces)
        
        self.observation_info = {}
        observation_space = self.cfg["env"]["observationSpace"]
        for name in observation_space:
            self.observation_info[name] = self._get_observation_dim(name)
            
        state_include_goal = bool(self.cfg["env"].get("stateIncludeGoal", False))
        self.reach_curiosity_mgr = CuriosityRewardManager(
            num_keypoints=self._dims.NUM_FINGERTIPS.value,
            device=self.device,
            canonical_pointcloud=self.grasping_dataset._pointclouds[0], #NOTE: hardcode here, not per-env
            # cluster parameters for contact reward
            cluster_k=32,
            max_clustering_iters=10,
            
            canonical_normals=self.grasping_dataset._pointcloud_normals[0],
            mask_backface_points=self.cfg["env"]["maskBackfacePoints"],
            use_normal_in_clustering=self.cfg["env"].get("useNormalInClustering", True),
            num_envs=self.num_envs,

            # cfg for sfb
            state_feature_dim=self.cfg["env"].get("stateFeatureDim", None),
            num_key_states=int(self.cfg["env"].get("numKeyStates", 256)),
            state_counter_mode=str(self.cfg["env"].get("stateCounterMode", "cluster")),
            state_include_goal=state_include_goal,


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
            
        # meaning less attributes, only for log in ppo.py
        self.object_codes = ["all"]
        self.label_paths = ["in_hand_manipulation_allegro"]
        self.num_objects = 1
        self.object_cat = self.object_type
        self.max_per_cat = -1
        self.object_geo_level = "all"
        self.object_scale = "all"

        self.training = True

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.reset_arm(first_time=True)

    def reset_arm(self, first_time=False):
        self.reset(first_time=first_time)
        for _ in range(10):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            self.compute_observations()

    def __configure_specifications(self, specs: Dict, mdp_type: str) -> None:
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

    def __configure_observation_specs(self, observation_specs: Dict) -> None:
        """Configure the observation specifications.

        All the observation specifications are stored in `self._observation_specs`

        Args:
            observation_specs (Dict): The observation specifications. (cfg["env"]["observation_specs"])
        """
        self._observation_specs = self.__configure_specifications(observation_specs, "observation")

    def __configure_action_specs(self, action_specs: Dict) -> None:
        """Configure the action specifications.

        All the action specifications are stored in `self._action_specs`

        Args:
            action_specs (Dict): The action specifications. (cfg["env"]["action_specs"])
        """
        self._action_specs = self.__configure_specifications(action_specs, "action")
            
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
    
    def __configure_mdp_spaces(self) -> None:
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
        self.__configure_observation_specs(self.cfg["env"]["observationSpecs"])
        observation_space = self.cfg["env"]["observationSpace"]
        observation_space_extra = self.cfg["env"]["observationSpaceExtra"]
        observation_space_extra = [] if observation_space_extra is None else observation_space_extra

        num_observations = (
            sum([self._get_observation_dim(name) for name in observation_space]) * self.stack_frame_number
        )
        for name in observation_space:
            print(f"Observation: {name}, dim: {self._get_observation_dim(name)}")
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
    
    def reset(self, dones=None, first_time=False):
        if dones is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = dones.nonzero(as_tuple=False).flatten()

        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        # reset idx
        if env_ids.shape[0] > 0:
            self.reset_idx(env_ids, goal_env_ids, first_time=first_time)

        self.reset_goal_buf[goal_env_ids] = 0 # delayed reset place here
        self.compute_observations(env_ids)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_idx(self, env_ids: torch.LongTensor, goal_env_ids: torch.LongTensor, first_time=False) -> None:

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
            super().reset_idx(env_ids, goal_env_ids)
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
    
    def draw_link_keypoints(self) -> None:
        for idx, link_name in enumerate(self._keypoints):
            link_positions = self.keypoint_positions_with_offset[:, idx, :]
            link_orientations = self.keypoint_orientations[:, idx, :]
            draw_axes(self.gym, self.viewer, self.envs, link_positions, link_orientations, 0.02)
            
    def pre_physics_step(self, actions):
        # env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # # if only goals need reset, then call set API
        # if len(goal_env_ids) > 0 and len(env_ids) == 0:
        #     self.reset_target_pose(goal_env_ids, apply_reset=True)

        # # if goals need reset in addition to other envs, call set API in reset()
        # elif len(goal_env_ids) > 0:
        #     self.reset_target_pose(goal_env_ids)

        # if len(env_ids) > 0:
        #     self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        # hand_indices = self.hand_indices[:].to(torch.int32)
        # self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.cur_targets), gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)


    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])


        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
            self.reset_goal_buf[env_ids] = 0 # delay for later reset

    
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
            self.compute_observations()

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)
            # self.compute_observations() will be called in self.reset() in the training / evaluation loop

        # if len(env_ids) > 0:
        #     self.reset_idx(env_ids, goal_env_ids)
        
        # self.compute_observations()
        
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_link_keypoints()
                
            draw_points(self.gym, self.viewer, self.envs, self.fingertip_positions, radius=0.012, num_segments=10, color=(1.0, 0.0, 0.0))
            draw_points(self.gym, self.viewer, self.envs, self.keypoint_positions_with_offset, radius=0.005, num_segments=10, color=(0.0, 1.0, 0.0))
        
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
                # observation = quat_to_6d(observation)
                observation = observation

            observations[spec.name] = observation

        return observations
    
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
        
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        self.goal_ori_dist = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        
        self.object_root_positions = self.object_pos
        self.object_root_orientations = self.object_rot
        self.object_root_linear_velocities = self.object_linvel
        self.object_root_angular_velocities = self.object_angvel

        self.keypoint_positions = self.allegro_hand_rigid_body_positions[:, self.keypoint_indices, :]
        self.keypoint_orientations = self.allegro_hand_rigid_body_orientations[:, self.keypoint_indices, :]
        self.keypoint_positions_with_offset = self.keypoint_positions + quat_apply(self.keypoint_orientations, self.keypoint_offset.repeat(self.num_envs, 1, 1))
        self.keypoint_orientations_with_offset = self.keypoint_orientations # currently no orientation offset
        self.fingertip_positions_with_offset = self.keypoint_positions_with_offset[:, self.fingertip_link_indices_among_keypoints, :]
        self.fingertip_orientations_with_offset = self.keypoint_orientations_with_offset[:, self.fingertip_link_indices_among_keypoints, :]
        
        self.index_finger_keypoint_positions_with_offset = self.keypoint_positions_with_offset[:, self.index_finger_links_indices_among_keypoints, :]
        self.middle_finger_keypoint_positions_with_offset = self.keypoint_positions_with_offset[:, self.middle_finger_links_indices_among_keypoints, :]
        self.ring_finger_keypoint_positions_with_offset = self.keypoint_positions_with_offset[:, self.ring_finger_links_indices_among_keypoints, :]
        self.thumb_finger_keypoint_positions_with_offset = self.keypoint_positions_with_offset[:, self.thumb_finger_links_indices_among_keypoints, :]
        
        self.index_finger_keypoint_orientations_with_offset = self.keypoint_orientations_with_offset[:, self.index_finger_links_indices_among_keypoints, :]
        self.middle_finger_keypoint_orientations_with_offset = self.keypoint_orientations_with_offset[:, self.middle_finger_links_indices_among_keypoints, :]
        self.ring_finger_keypoint_orientations_with_offset = self.keypoint_orientations_with_offset[:, self.ring_finger_links_indices_among_keypoints, :]
        self.thumb_finger_keypoint_orientations_with_offset = self.keypoint_orientations_with_offset[:, self.thumb_finger_links_indices_among_keypoints, :]
        
        net_contact_forces = self.net_contact_forces.view(self.num_envs, self.num_rigid_bodies, 3)
        self.keypoint_contact_forces = net_contact_forces[:, self.keypoint_indices, :]
        self.fingertip_contact_forces = net_contact_forces[:, self.fingertip_indices, :]
        
        self.index_finger_keypoint_forces = self.keypoint_contact_forces[:, self.index_finger_links_indices_among_keypoints, :]
        self.middle_finger_keypoint_forces = self.keypoint_contact_forces[:, self.middle_finger_links_indices_among_keypoints, :]
        self.ring_finger_keypoint_forces = self.keypoint_contact_forces[:, self.ring_finger_links_indices_among_keypoints, :]
        self.thumb_finger_keypoint_forces = self.keypoint_contact_forces[:, self.thumb_finger_links_indices_among_keypoints, :]
        
        pcl_world = self._get_target_surface_points_world()  # (N, P, 3)
        # Pairwise distances: (N, K, P) where K = number of keypoints
        dists_kp = torch.cdist(self.keypoint_positions_with_offset, pcl_world)
        min_dists_kp, _ = torch.min(dists_kp, dim=2)  # (N, K)
        kp_force_mag = self.keypoint_contact_forces.norm(dim=-1, p=2)
        near_surface = (min_dists_kp < 0.010)
        has_force = (kp_force_mag > 0.01)
        self.keypoint_contact_mask = (near_surface & has_force)  # (N, K) bool
        
        self.fingertip_states = self.allegro_hand_rigid_body_states[:, self.fingertip_indices, :]
        self.fingertip_positions = self.fingertip_states[..., 0:3]
        self.fingertip_orientations = self.fingertip_states[..., 3:7]
        self.fingertip_linear_velocities = self.fingertip_states[..., 7:10]
        self.fingertip_angular_velocities = self.fingertip_states[..., 10:13]
        
        self.fingertip_orientations_wrt_palm, self.fingertip_positions_wrt_palm = compute_relative_pose(
            self.fingertip_orientations,
            self.fingertip_positions,
            self.allegro_hand_center_orientations[:, None, :],
            self.allegro_hand_center_positions[:, None, :],
        )
        
        self.object_orientations_wrt_palm, self.object_positions_wrt_palm = compute_relative_pose(
            self.object_root_orientations,
            self.object_root_positions,
            self.allegro_hand_center_orientations,
            self.allegro_hand_center_positions,
        )

        self.object_positions_wrt_keypoints = self.keypoint_positions - self.object_root_positions[:, None, :]

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

            assert pointclouds[mask].isfinite().all(), "pointclouds is not finite"

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
        
    def __create_object_dataset(self, device=None) -> None:
        # Create simple box grid dataset for singulation task
        from .dataset import ObjectDataset, TableTopDataset
        object_mode = str(self.cfg["env"].get("objectMode", "xml")).lower()
        pcl_num = int(self.cfg["env"].get("numObjectPointCloudPoints", 1024))
        if object_mode == "urdf":
            urdf_rel = str(self.cfg["env"].get("objectUrdfPath", "")).strip()
            if urdf_rel == "":
                raise ValueError("env.objectUrdfPath must be set when objectMode == 'urdf'")
            object_asset_root = str(Path(__file__).parent.parent.parent / "assets")
            self.grasping_dataset = TableTopDataset(
                mode="urdf",
                device=device,
                pcl_num=pcl_num,
                urdf_rel_path=urdf_rel,
                asset_root=object_asset_root,
            )
        else:
            self.grasping_dataset = ObjectDataset(
                mode=object_mode,
                device=device,
                pcl_num=pcl_num,
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
            
        asset_filename = self.cfg["env"]["asset"].get("assetFileName")

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

        self.allegro_center_index = self.gym.find_asset_rigid_body_index(allegro_hand_asset, self._allegro_hand_center_prim)
        self.keypoint_indices = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, prim) for prim in self._keypoints]
        self.fingertip_indices = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, prim) for prim in self._fingertips]
        self.keypoint_offset = [
            self._keypoints_info.get(link_name, [[0.0, 0.0, 0.0]])
            for link_name in self._keypoints
        ]  # (#key_link, 1, 3)
        self.fingertip_link_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._fingertips]
        self.index_finger_links_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._index_finger_links]
        self.middle_finger_links_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._middle_finger_links]
        self.ring_finger_links_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._ring_finger_links]
        self.thumb_finger_links_indices_among_keypoints = [self._keypoints.index(link_name) for link_name in self._thumb_links]
        
        config["asset"] = allegro_hand_asset
        config["dof_props"] = shadow_hand_dof_props

        print(">>> Allegro Hand loaded")
        return config

    def _define_camera(self) -> None:
        """Define the cameras for the rendering."""
        if not self.enable_rendered_pointcloud_observation:
            return

        self._camera_positions = [gymapi.Vec3(-0.3, -0.0, 0.8), gymapi.Vec3(0.3, 0.0, 0.8)]
        self._camera_target_locations = [gymapi.Vec3(0.0, 0.0, 0.5), gymapi.Vec3(0.0, 0.0, 0.5)]

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
        self.camera_properties.width = 256
        self.camera_properties.height = 256
        self.camera_properties.enable_tensors = True

        # define related indices for pointcloud computation
        self.camera_u = torch.arange(0, self.camera_properties.width, device=self.device)
        self.camera_v = torch.arange(0, self.camera_properties.height, device=self.device)
        self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing="ij")

        # define bounding box corners for pointcloud computation
        self.render_pointcloud_bbox_corners = (
            torch.tensor([-0.15, -0.15, 0.45], device=self.device),
            torch.tensor([0.15, 0.15, 0.6], device=self.device),
        )
        
        # self.render_pointcloud_bbox_corners = (
        #     torch.tensor([
        #         -self._table_x_length / 2, 
        #         -self._table_y_length / 2, 
        #         self._table_pose[2] + self._table_thickness * 0.5], 
        #     device=self.device) + 5e-3,
        #     torch.tensor([
        #         self._table_x_length / 2, 
        #         self._table_y_length / 2 + 0.5, 
        #         self._upper_shelf_pos[2] - self._table_thickness * 0.5], 
        #     device=self.device) - 5e-3
        # )
    
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
        # shadow_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.47 * np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.25 * np.pi)
        shadow_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -np.pi/2)
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = shadow_hand_start_pose.p.x
        # pose_dy, pose_dz = -0.2, 0.06
        pose_dy, pose_dz = 0.0, 0.15
        object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        if self.object_type == "pen":
            object_start_pose.p.z = shadow_hand_start_pose.p.z + 0.02

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
            if self.randomize:
                self.apply_domain_rand(env_ptr=env, actor_handle=hand_idx, friction=True, com=False)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            if self.randomize:
                self.apply_domain_rand(env_ptr=env, actor_handle=object_idx, friction=True, com=True)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.enable_rendered_pointcloud_observation:
                for k in range(self.num_cameras_per_env):
                    camera = self.gym.create_camera_sensor(env, self.camera_properties)
                    self.cameras_handle.append(camera)

                    self.gym.set_camera_location(
                        camera, env, self._camera_positions[k], self._camera_target_locations[k]
                    )
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
        
    def _get_target_surface_points_world(self) -> torch.Tensor:
        # (num_envs, P, 3)
        canonical = self.grasping_dataset._pointclouds  # (1,P,3)
        pc_world = quat_rotate(self.object_root_orientations[:, None, :], canonical) + self.object_root_positions[:, None, :]
        # self.visualize_pointcloud(pc_world)
        return pc_world


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
            self.gym.set_actor_rigid_body_properties(env_ptr, actor_handle, prop)
        
        obj_friction = 1.0
        if friction:
            rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)

            object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)
            for p in object_props:
                p.friction = rand_friction
            self.gym.set_actor_rigid_shape_properties(env_ptr, actor_handle, object_props)



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
        return feat.reshape(self.num_envs, 16), r    

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
        rel_pos = self.keypoint_positions - self.object_root_positions.unsqueeze(1)

        # Distance to nearest surface point per keypoint: (N,4,1)
        _, r = self.compute_curiosity_observations_surface_all_keypoints()  # r: (N,4,1)


        contact_mag = self.keypoint_contact_forces.norm(dim=-1, p=2)
        # print("contact_mag", contact_mag[0])

        # Contact filters
        near_surface = (r.squeeze(-1) < 0.01)           # (N,4)
        # print("near_surface", near_surface[0])
        has_force = (contact_mag > 0.5)                 # (N,4)
        contact_mask = near_surface & has_force         # (N,4)

        # Apply mask
        filtered_rel = rel_pos * contact_mask.unsqueeze(-1)  # (N,4,3)

        has_contact = contact_mask.any(dim=1)  # (N,)
        return filtered_rel, has_contact
        
    def compute_reach_reward_keypoints(self):
        """Reaching reward using keypoint-to-object-surface distances with historical minima."""
        pcl_world = self._get_target_surface_points_world()
        keypoints_w = self.keypoint_positions_with_offset

        # Current nearest distances from each keypoint to the object surface: (N, K)
        # torch.cdist: (N, K, P) → min over P
        dists = torch.cdist(keypoints_w, pcl_world)
        cur_min_dist, _ = torch.min(dists, dim=2)  # (N, K)

        if not hasattr(self, "keypoints_to_surface_dist_min"):
            self.keypoints_to_surface_dist_min = torch.full_like(cur_min_dist, 0.30)  # meters

        delta = (self.keypoints_to_surface_dist_min - cur_min_dist) #.clamp_min(0.0)  # (N, K)
        self.keypoints_to_surface_dist_min = torch.min(self.keypoints_to_surface_dist_min, cur_min_dist)

        reach_rew_keypoints = delta.mean(dim=1)  # (N,)
        self.reach_rew_keypoints = reach_rew_keypoints
        self.reach_rew_scaled_keypoints = self.reach_rew_keypoints * 1.0

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


        mgr = self.reach_curiosity_mgr
        state_features_world = mgr.build_state_features_from_world_pc(
            pc_world,
            goal_positions=self.goal_pos,
            goal_orientations=self.goal_rot,
        )
        
        finger_pos_tensors = [
            self.index_finger_keypoint_positions_with_offset,
            self.middle_finger_keypoint_positions_with_offset,
            self.ring_finger_keypoint_positions_with_offset,
            self.thumb_finger_keypoint_positions_with_offset,
        ]
        finger_ori_tensors = [
            self.index_finger_keypoint_orientations_with_offset,
            self.middle_finger_keypoint_orientations_with_offset,
            self.ring_finger_keypoint_orientations_with_offset,
            self.thumb_finger_keypoint_orientations_with_offset,
        ]
        finger_force_tensors = [
            self.index_finger_keypoint_forces,
            self.middle_finger_keypoint_forces,
            self.ring_finger_keypoint_forces,
            self.thumb_finger_keypoint_forces,
        ]
        finger_names_all = ["index", "middle", "ring", "thumb"]
        
        finger_positions = []      # (N,3)
        finger_orientations = []   # (N,4)
        finger_forces = []         # (N,3)
        finger_contact_indices = []  # (N,)
        finger_contact_masks = []    # (N,)
        finger_names = []

        for fname, kp_pos_f, kp_ori_f, kp_force_f in zip(
            finger_names_all, finger_pos_tensors, finger_ori_tensors, finger_force_tensors
        ):

            dists = torch.cdist(kp_pos_f, pc_world)        # (N,4,M)
            contact_indices = dists.argmin(dim=-1)                         # (N,4)
            N_env, Kf, P = dists.shape
            dists_flat = dists.view(N_env, -1)               # (N,Kf*P)
            min_flat_idx = dists_flat.argmin(dim=-1)         # (N,)
            min_dists = dists_flat.gather(
                1, min_flat_idx.unsqueeze(-1)
            ).squeeze(-1)                                    # (N,)

            best_point_idx = (min_flat_idx % P)              # (N,)
            best_k_idx = (min_flat_idx // P)                 # (N,)
            
            

            gather_idx_pos = best_k_idx.view(N_env, 1, 1).expand(-1, 1, 3)
            gather_idx_ori = best_k_idx.view(N_env, 1, 1).expand(-1, 1, 4)
            kp_pos_rep = torch.gather(kp_pos_f, 1, gather_idx_pos).squeeze(1)  # (N,3)
            kp_ori_rep = torch.gather(kp_ori_f, 1, gather_idx_ori).squeeze(1)  # (N,4)
            

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
            finger_names.append(fname)
        
        keypoints_pos = torch.stack(finger_positions, dim=1)           # (N,F,3)
        keypoints_ori = torch.stack(finger_orientations, dim=1)        # (N,F,4)
        keypoints_forces = torch.stack(finger_forces, dim=1)           # (N,F,3)
        contact_indices = torch.stack(finger_contact_indices, dim=1)   # (N,F)
        contact_mask = torch.stack(finger_contact_masks, dim=1)        # (N,F)
        
        axis_local = torch.tensor(
            [-1.0, 0.0, 0.0],
            device=self.device,
            dtype=keypoints_pos.dtype,
        ).view(1, 1, 3).expand_as(keypoints_pos)
        dir_world = quat_apply(
            keypoints_ori.reshape(-1, 4),
            axis_local.reshape(-1, 3),
        ).view_as(keypoints_pos)

        reward, info = self.reach_curiosity_mgr.compute_reward_from_canonical(
            object_positions=self.object_root_positions,
            object_orientations=self.object_root_orientations,
            keypoint_positions_world=keypoints_pos,
            goal_positions=self.goal_pos,
            goal_orientations=self.goal_rot,
            contact_indices=contact_indices,
            contact_mask=contact_mask,
            task_contact_satisfied=self.contact_satisfied,
            # contact_forces_local=contact_forces_canonical,
            contact_forces_local=dir_world,
            state_features_world=state_features_world,
        )
        
        # multiplier = (self.contact_mask[:, :-1].any(dim=-1).float() + self.contact_mask[:, -1].float()) / 2.0
        
        # self.reach_curiosity_rew = reward
        # self.reach_curiosity_rew_scaled = self.reach_curiosity_rew * 10
        self.reach_curiosity_rew = info["potential_field_reward"].clone()
        self.reach_curiosity_rew_scaled = self.reach_curiosity_rew * 2.56 # 5.12 1.28 0.01, 0.02, 0.04
        self.contact_coverage_rew = info["cluster_novelty_reward"].clone()
        self.contact_coverage_rew_scaled = self.contact_coverage_rew * 240
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
        tip_contact = self.keypoint_contact_mask.sum(dim=-1) >= 2  # (N,) bool
        tip_contact = torch.ones_like(tip_contact, dtype=torch.bool)

        hand_contact = tip_contact

        self.extras["hand_contact"]        = hand_contact.clone()
        self.extras["tip_contact"]         = tip_contact.clone()

        return hand_contact
    
    def compute_reward(self, actions):
        reward_types = self.reward_type.strip().split("+")
        assert len(reward_types) > 0, f"No reward type specified, reward_type from cfg is {self.reward_type}"
        
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        if not self.run_consective_goals:
            self.reset_buf[:] = torch.where(
                self.reset_goal_buf[:] > 0,
                (self.reset_buf[:] + self.reset_goal_buf[:]).bool().float(),
                self.reset_buf[:]
            )
        

        
        self.task_reward = self.rew_buf.clone()
        self.extras["task_reward"] = self.task_reward.clone()

        self.refresh_contact_mask() # self.contact_mask
        self.contact_satisfied = self.check_contact_criteria()
        
        self.compute_reach_reward_keypoints(); self.reach_rew_scaled = self.reach_rew_scaled_keypoints.clone()
        if self.training:
            self.compute_curiosity_informed_reach_reward()
        else:
            self.reach_curiosity_rew_scaled = torch.zeros_like(self.rew_buf)
            self.contact_coverage_rew_scaled = torch.zeros_like(self.rew_buf)
        
        if "task" in reward_types:
            self.rew_buf += self.task_reward
            reward_types.remove("task")
        if "reach" in reward_types:
            self.rew_buf += self.reach_rew_scaled
            reward_types.remove("reach")
        if "energy_reach" in reward_types:
            self.rew_buf += self.reach_curiosity_rew_scaled
            reward_types.remove("energy_reach")
        if "contact_coverage" in reward_types:
            self.rew_buf += self.contact_coverage_rew_scaled
            reward_types.remove("contact_coverage")
            
        assert len(reward_types) == 0, f"Unknown reward types {reward_types} specified."

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))
        
        self.extras["success_num"] = torch.sum(self.successes>0).unsqueeze(-1).clone()
        
    def train(self):
        self.training = True

    def eval(self, vis=False):
        self.training = False        

        
#####################################################################
###=========================jit functions=========================###
#####################################################################
                
                
@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    # print(f"{torch.sum(goal_dist >= fall_dist).item()} envs fall dist reset!")
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        # print(f"{torch.sum(successes >= max_consecutive_successes).item()} envs max consecutive success reset!")

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)
    # print(f"{torch.sum(timed_out).item()} envs timeout reset!")

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes




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



@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


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
