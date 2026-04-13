# push_box_2d.py
from __future__ import annotations

import math
from typing import Dict, Any, Tuple, List, Union

import torch
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask

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
import numpy as np
import random   
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, tensor_clamp
from .torch_utils import quat_rotate
from .curiosity_reward_manager import CuriosityRewardManager


def create_box_pointcloud_on_zplane(numpoints_per_face, box_size, zplane=0.0):
    """
    Create pointcloud on the surface of a box, but only on the z=zplane plane.
    box_size: float, length of box side
    Returns:
        pc: (P, 3) tensor
    """
    interval = box_size / numpoints_per_face
    half = box_size / 2.0
    edge = -half + interval / 2.0

    x = np.linspace(-edge, edge, numpoints_per_face)
    y = np.array([-half, half]) # lower and upper face
    xv, yv = np.meshgrid(x, y) # vertical faces (2, numpoints_per_face)
    x_n = np.zeros_like(x)
    y_n = np.array([-half, half]) # lower (down) and upper (up) face
    xv_n, yv_n = np.meshgrid(x_n, y_n) # vertical faces (2, numpoints_per_face)
    
    y = np.linspace(-edge, edge, numpoints_per_face)
    x = np.array([-half, half])  # left and right face
    xh, yh = np.meshgrid(x, y) # horizontal faces (2, numpoints_per_face)
    # breakpoint()
    x_n = np.array([-half, half])  # left (left) and right (right) face
    y_n = np.zeros_like(y)
    xh_n, yh_n = np.meshgrid(x_n, y_n) # horizontal faces (2, numpoints_per_face)

    x = np.concatenate([xv.flatten(), xh.T.flatten()]) # (P,)
    y = np.concatenate([yv.flatten(), yh.T.flatten()]) # (P,)
    z = np.ones_like(x) * zplane # (P,)

    x_n = np.concatenate([xv_n.flatten(), xh_n.T.flatten()]) # (P,)
    y_n = np.concatenate([yv_n.flatten(), yh_n.T.flatten()]) # (P,)
    z_n = np.zeros_like(x_n) * zplane # (P,)

    pc = np.stack([x, y, z])  # (P, 3)
    pc = torch.tensor(pc, dtype=torch.float32)

    normal = np.stack([x_n, y_n, z_n])  # (P, 3)
    normal = torch.tensor(normal, dtype=torch.float32)
    # breakpoint()
    return pc.t(), normal.t()


class PushBox2D(VecTask):
    """
    2D push-box toy env in IsaacGymEnvs style (VecTask subclass)
    - Agent: sphere rigid body (point-mass-like)
    - Box: cube (side=2), translation only (we hard-freeze rotation)
    - Box init center: (-3,0) or (3,0)
    - Agent init: (0,-3)
    - Goal: move box center to (0,0)
    """

    _keypoints: List[str] = [
        "fingertip"
    ]

    def __init__(
        self,
        cfg: Dict[str, Any],
        rl_device: str,
        sim_device: str,
        graphics_device_id: int,
        headless: bool,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
    ):
        seed = cfg["env"]["seed"]
        torch.manual_seed(seed)  # cpu
        random.seed(seed)
        np.random.seed(seed)

        self.cfg = cfg
        
        self.env_info_logging = self.cfg["logging"]["envInfo"]
        self.stack_frame_number = self.cfg["env"]["stackFrameNumber"]
        self.enable_contact_sensors = self.cfg["env"]["enableContactSensors"]
        self.reward_type = self.cfg["env"]["rewardType"]
        self.mode = self.cfg["env"]["mode"]
        self.velocity_observation_scale = self.cfg["env"]["velocityObservationScale"]

        env_cfg = cfg["env"]
        self.num_environments = env_cfg["numEnvs"]
        self.max_episode_length = env_cfg.get("episodeLength", 300)

        # task geometry
        self.box_half = 1.0  # side length 2
        self.goal_xy = torch.tensor([0.0, 0.0], device=rl_device, dtype=torch.float32)

        # dynamics / control
        self.dt = cfg["sim"]["dt"]
        self.agent_radius = env_cfg.get("agentRadius", 0.15)
        self.agent_max_force = env_cfg.get("agentMaxForce", 80.0)      # action -> force on agent
        self.contact_range = env_cfg.get("contactRange", 0.12)         # contact threshold
        self.contact_k = env_cfg.get("contactK", 200.0)                # contact stiffness-ish
        self.linear_damping = env_cfg.get("linearDamping", 0.5)        # optional damping factor in state projection

        # rewards
        self.progress_scale = env_cfg.get("progressScale", 600.0)
        self.reach_progress_scale = env_cfg.get("reachProgressScale", 50.0) #100.0)
        self.action_penalty = env_cfg.get("actionPenalty", 0.01)
        self.success_bonus = env_cfg.get("successBonus", 4000.0)
        self.success_thresh = env_cfg.get("successThresh", 0.5)
        self.time_penalty = env_cfg.get("timePenalty", 0.1)
        self.too_far_penalty = env_cfg.get("tooFarPenalty", -1000.0)
        self.reach_too_far_penalty = env_cfg.get("reachTooFarPenalty", -1000.0)

        # obs/action dims (你也可以改成更小)
        # obs: agent(x,y,vx,vy) + box(x,y,vx,vy) + (box-goal)(dx,dy) + (box-agent)(dx,dy)
        # self.num_obs = env_cfg.get("numObservations", 12)
        # self.num_actions = env_cfg.get("numActions", 2)

        self.__configure_mdp_spaces()

        self.observation_info = {}
        observation_space = self.cfg["env"]["observationSpace"]
        for name in observation_space:
            self.observation_info[name] = self._get_observation_dim(name)


        super().__init__(
            config=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # tensors (after create_sim / prepare_sim)
        self.root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        # shape: (num_envs * num_actors, 13) [pos(3), rot(4), linvel(3), angvel(3)]
        self.num_actors = 4
        self.root_state = self.root_state.view(self.num_envs, self.num_actors, 13)

        # force tensors for apply_rigid_body_force_tensors
        # We'll apply per-rigid-body forces; easiest is to use rigid-body state tensor.
        self.rb_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.rb_state = self.rb_state.view(self.num_envs, -1, 13)
        self.point_rb_state = self.rb_state[:, self.agent_rigid_body_start:self.agent_rigid_body_end, :]
        self.point_pos = self.point_rb_state[:, -1, 0:3]
        self.point_vel = self.point_rb_state[:, -1, 6:9]

        # self.num_point_dofs = 2
        self.point_default_dof_pos = torch.zeros(self.num_point_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.dof_state = self.dof_state.view(self.num_envs, -1, 2)
        self.point_dof_state = self.dof_state[:, :2, :]
        self.point_dof_pos = self.point_dof_state[:, :, 0]
        self.point_dof_vel = self.point_dof_state[:, :, 1]
        self.point_dof_lower_limits = torch.tensor([-8.0, -2.0], device=self.device)
        self.point_dof_upper_limits = torch.tensor([8.0, 8.0], device=self.device)
        self.act_moving_average = env_cfg.get("actMovingAverage", 0.8)
        self.point_dof_speed_scale = env_cfg.get("pointDofSpeedScale", 15.0)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.point_dof_speed_upper_limit = torch.ones(self.num_dofs, device=self.device) * env_cfg.get("pointDofSpeedUpperLimit", 1.0)
        self.point_dof_speed_lower_limit = torch.ones(self.num_dofs, device=self.device) * env_cfg.get("pointDofSpeedLowerLimit", -1.0)
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)


        self.net_contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self.net_contact_forces = self.net_contact_forces.view(self.num_envs, -1, 3)
        self.point_rb_net_contact_forces = self.net_contact_forces[:, self.agent_rigid_body_start:self.agent_rigid_body_end, :]
        self.point_net_contact_forces = self.point_rb_net_contact_forces[:, -1, :]

        # indices
        self.agent_actor_id = 0
        self.box_actor_id = 1
        self.goal_actor_id = 2
        self.table_actor_id = 3
            
        self.box_state = self.root_state[:, self.box_actor_id, :]
        self.box_pos = self.box_state[:, 0:3]
        self.box_ori = self.box_state[:, 3:7]
        self.box_vel = self.box_state[:, 7:10]
        self.box_angvel = self.box_state[:, 10:13]


        # self.prev_box_goal_dist = torch.zeros(self.num_envs, device=rl_device, dtype=torch.float32)
        # self.prev_reach_dist = torch.zeros(self.num_envs, device=rl_device, dtype=torch.float32)
        self.prev_dist = torch.ones((self.num_envs, 1), device=rl_device, dtype=torch.float32) * 2.8
        self.prev_box_goal_dist = torch.ones((self.num_envs, 1), device=rl_device, dtype=torch.float32) * 3

        # init obs/reward buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=rl_device, dtype=torch.float32)
        self.rew_buf = torch.zeros((self.num_envs,), device=rl_device, dtype=torch.float32)
        self.reset_buf = torch.zeros((self.num_envs,), device=rl_device, dtype=torch.long)
        self.progress_buf = torch.zeros((self.num_envs,), device=rl_device, dtype=torch.long)
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.enable_exploration_logging = self.cfg["env"].get("enableExplorationLogging", False)

        self.dist_min = torch.ones(self.num_envs, device=rl_device, dtype=torch.float32) * 3.0
        self.reach_dist_min = torch.ones(self.num_envs, device=rl_device, dtype=torch.float32) * 4.2
        self.keypoints_to_surface_dist_min = torch.ones(self.num_envs, device=rl_device, dtype=torch.float32) * 2.8

        # pc_world
        pointcloud, pointcloud_normal = create_box_pointcloud_on_zplane(numpoints_per_face=64,box_size=2*self.box_half,zplane=0.0)
        self._pointclouds = pointcloud.unsqueeze(0).to(self.device)  # (1, P, 3)
        self._pointcloud_normals = pointcloud_normal.unsqueeze(0).to(self.device)  # (1, P, 3)

        # curiosity reward manager
        state_include_goal = bool(self.cfg["env"].get("stateIncludeGoal", False))
        self.reach_curiosity_mgr = CuriosityRewardManager(
            num_keypoints=1,
            num_object_points=self._pointclouds.shape[1],
            device=self.device,
            canonical_pointcloud=self._pointclouds[0], #NOTE: hardcode here, not per-env
            k=16,  # highest point knn size # no use 
            # cluster parameters for contact reward
            cluster_k=self._pointclouds.shape[1]//16,
            max_clustering_iters=0,
            enable_predefined_clusters=True,
            
            canonical_normals=self._pointcloud_normals[0],
            mask_backface_points=self.cfg["env"]["maskBackfacePoints"],
            mask_palm_inward_points=self.cfg["env"]["maskPalmInwardPoints"],
            use_normal_in_clustering=self.cfg["env"].get("useNormalInClustering", True),
            num_envs=self.num_envs,

            # 
            state_feature_dim=self.cfg["env"].get("stateFeatureDim", None),
            num_key_states=int(self.cfg["env"].get("numKeyStates", 32)),
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

            state_type = self.cfg["env"].get("stateType", "predefined"),
            enable_predefined_state=True,
            state_running_max_mode=self.cfg["env"].get("stateRunningMaxMode", "state"),
            kernel_param=0.3
        )

        # do first reset
        self.reset_arm(first_time=True)

        # for ppo log
        self.use_precomputed_poses = ""
        self.use_pre_poses_init = ""
        self.use_multi_stage_curiosity = ""

        self.train()


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
        self.cfg["env"]["numObservations"] = 21 # num_observations
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

    # --------------------------
    # Sim / env creation
    # --------------------------
    # def create_sim(self):
    #     sim_params = self.sim_params  # VecTask already parsed it
    #     self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, sim_params)

    #     # ground
    #     plane_params = gymapi.PlaneParams()
    #     plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    #     plane_params.static_friction = 1.0
    #     plane_params.dynamic_friction = 1.0
    #     plane_params.restitution = 0.0
    #     self.gym.add_ground(self.sim, plane_params)

    #     self._create_envs()
    #     return self.sim

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        env_spacing = self.cfg["env"].get("envSpacing", 10.0)
        lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # assets
        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = True
        asset_opts.disable_gravity = True
        asset_opts.angular_damping = 0.01
        asset_opts.thickness = 0.001
        asset_opts.density = 100.0
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_opts.use_physx_armature = True
        asset_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # asset_options.flip_visual_attachments = False
        # asset_options.fix_base_link = True
        # asset_options.collapse_fixed_joints = True
        # asset_options.disable_gravity = True
        # asset_options.thickness = 0.001
        # asset_options.angular_damping = 0.01

        # agent sphere
        asset_filename = "urdf/point.urdf"
        agent_asset = self.gym.load_asset(self.sim, "assets", asset_filename, asset_opts)
        self.num_point_dofs = self.gym.get_asset_dof_count(agent_asset)
        print("Num point dofs: ", self.num_point_dofs)

        dof_props = self.gym.get_asset_dof_properties(agent_asset)
        for i in range(self.num_point_dofs):
            dof_props['stiffness'][i] = 4000.0
            dof_props['damping'][i] = 400.0
            dof_props['effort'][i] = 10.0
            dof_props['velocity'][i] = 1.0

        # set rigid-shape properties for point agent
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(agent_asset)
        for shape in rigid_shape_props:
            shape.friction = 0.8
        self.gym.set_asset_rigid_shape_properties(agent_asset, rigid_shape_props)

        # box cube (side=2)
        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = False
        asset_opts.disable_gravity = False
        asset_opts.density = 0.1
        asset_opts.angular_damping = 0.01
        asset_opts.linear_damping = 0.01
        asset_opts.thickness = 0.001
        box_asset = self.gym.create_box(self.sim, 2.0, 2.0, 2.0, asset_opts)
        
        # set rigid-shape properties for box
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(box_asset)
        for shape in rigid_shape_props:
            shape.friction = 0.6
        self.gym.set_asset_rigid_shape_properties(box_asset, rigid_shape_props)

        # box cube (side=2)
        asset_opts.fix_base_link = True
        goal_asset = self.gym.create_box(self.sim, 2.0, 2.0, 2.0, asset_opts)

        # table (16 * 3 * 0.2)
        table_asset = self.gym.create_box(self.sim, 12.0, 0.2, 3.0, asset_opts)
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for shape in rigid_shape_props:
            shape.friction = 0.0
        self.gym.set_asset_rigid_shape_properties(table_asset, rigid_shape_props)

        self.envs = []
        self.agent_handles = []
        self.box_handles = []
        self.goal_handles = []
        self.table_handles = []

        # We'll store rigid body indices for force application
        self.agent_rb_indices = []
        self.box_rb_indices = []
        self.goal_rb_indices = []
        self.table_rb_indices = []

        self.agent_actor_indices = []
        self.box_actor_indices = []
        self.goal_actor_indices = []
        self.table_actor_indices = []
        self.initial_indices = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, int(math.sqrt(self.num_envs)) + 1)
            self.envs.append(env)

            # agent pose
            agent_pose = gymapi.Transform()
            agent_pose.p = gymapi.Vec3(0.0, -3.0, 1.0)  # sit on ground
            agent_pose.r = gymapi.Quat(0, 0, 0, 1)

            agent_handle = self.gym.create_actor(env, agent_asset, agent_pose, "agent", i, 0, 0)
            self.gym.set_actor_dof_properties(env, agent_handle, dof_props)
            self.agent_handles.append(agent_handle)

            # box pose (init at +-3,0)
            box_pose = gymapi.Transform()
            x0 = -3.0
            self.initial_indices.append(x0 == 3.0)
            box_pose.p = gymapi.Vec3(x0, 0.0, 1.0)  # cube half-height=1
            box_pose.r = gymapi.Quat(0, 0, 0, 1)

            box_handle = self.gym.create_actor(env, box_asset, box_pose, "box", i, 0, 0)
            self.box_handles.append(box_handle)

            # goal visual (green cube at origin)
            goal_pose = gymapi.Transform()
            goal_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            goal_pose.r = gymapi.Quat(0, 0, 0, 1)
            goal_handle = self.gym.create_actor(env, goal_asset, goal_pose, "goal", i + self.num_envs, 0, 0)
            self.gym.set_rigid_body_color(
                env, goal_handle, 0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.0, 1.0, 0.0)
            )
            self.goal_handles.append(goal_handle)

            # table pose (0, 0.1, 1.5)
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 1.1, 1.5)
            table_pose.r = gymapi.Quat(0, 0, 0, 1)
            table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i, 0, 0)
            self.gym.set_rigid_body_color(
                env, table_handle, 0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.4, 0.2, 0.05)
            )
            self.table_handles.append(table_handle)

            # material-ish (optional)
            # for h in [agent_handle, box_handle]:
            #     props = self.gym.get_actor_rigid_shape_properties(env, h)
            #     for p in props:
            #         p.friction = 1.0
            #         p.restitution = 0.0
            #     self.gym.set_actor_rigid_shape_properties(env, h, props)

            # record rigid body indices for force tensors
            agent_rb = self.gym.get_actor_rigid_body_index(env, agent_handle, 0, gymapi.DOMAIN_SIM)
            box_rb = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
            goal_rb = self.gym.get_actor_rigid_body_index(env, goal_handle, 0, gymapi.DOMAIN_SIM)
            self.agent_rb_indices.append(agent_rb)
            self.box_rb_indices.append(box_rb)
            self.goal_rb_indices.append(goal_rb)

            agent_actor_idx = self.gym.get_actor_index(env, agent_handle, gymapi.DOMAIN_SIM)
            box_actor_idx = self.gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
            goal_actor_idx = self.gym.get_actor_index(env, goal_handle, gymapi.DOMAIN_SIM)
            self.agent_actor_indices.append(agent_actor_idx)
            self.box_actor_indices.append(box_actor_idx)
            self.goal_actor_indices.append(goal_actor_idx)

        self.agent_rb_indices = torch.tensor(self.agent_rb_indices, device=self.device, dtype=torch.long)
        self.box_rb_indices = torch.tensor(self.box_rb_indices, device=self.device, dtype=torch.long)
        self.goal_rb_indices = torch.tensor(self.goal_rb_indices, device=self.device, dtype=torch.long)

        self.agent_actor_indices = torch.tensor(self.agent_actor_indices, device=self.device, dtype=torch.long)
        self.box_actor_indices = torch.tensor(self.box_actor_indices, device=self.device, dtype=torch.long)
        self.goal_actor_indices = torch.tensor(self.goal_actor_indices, device=self.device, dtype=torch.long)

        self.initial_indices = torch.tensor(self.initial_indices, device=self.device, dtype=torch.long)

        agent = self.gym.find_actor_handle(env, "agent")
        self.agent_rigid_body_start = self.gym.get_actor_rigid_body_index(env, agent, 0, gymapi.DOMAIN_ENV)
        self.num_agent_rigid_body = self.gym.get_asset_rigid_body_count(agent_asset)
        self.agent_rigid_body_end = (
            self.agent_rigid_body_start + self.num_agent_rigid_body
        )

        # actor root state tensor needs refresh after creating actors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def pre_physics_step(self, actions):
        self.actions = torch.clamp(actions, -1.0, 1.0)
        # if not self.training:
        #     print(actions[0:5].cpu().numpy())

        # targets = self.prev_targets[:] + self.point_dof_speed_scale * self.dt * self.actions
        # self.cur_targets[:] = tensor_clamp(targets, self.point_dof_lower_limits, self.point_dof_upper_limits)
        
        self.cur_targets[:] = scale(self.actions, self.point_dof_lower_limits, self.point_dof_upper_limits)
        self.cur_targets[:] = self.act_moving_average * self.cur_targets[:] + (1.0 - self.act_moving_average) * self.prev_targets[:]
        self.cur_targets[:] = tensor_clamp(self.cur_targets[:], self.point_dof_lower_limits, self.point_dof_upper_limits)
        
        # F_agent_xy = self.actions * self.agent_max_force

        # num_rb_total = self.rb_state.shape[0] * self.rb_state.shape[1]
        # forces = torch.zeros((num_rb_total, 3), device=self.rl_device, dtype=torch.float32)

        # forces[self.agent_rb_idx, 0:2] = F_agent_xy

        # self.gym.apply_rigid_body_force_tensors(
        #     self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE
        # )
        self.prev_targets[:] = self.cur_targets[:]
        # print("current targets:", self.cur_targets[:5].cpu().numpy())
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        # refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # # hard-project to "2D translation only" by clamping z, quat, ang vel
        # self._project_to_2d()

        # update progress
        self.progress_buf += 1

        # compute obs + reward + resets
        self.compute_observations()
        self.compute_reward()

        # reset if needed
        # env_ids = torch.nonzero(self.reset_buf, as_tuple=False).squeeze(-1)
        # if env_ids.numel() > 0:
        #     self.reset_idx(env_ids)

        # self.gym.clear_lines(self.viewer)
        # pcl_world = self._get_target_surface_points_world()
        # draw_points(gym=self.gym, viewer=self.viewer, envs=self.envs, positions=pcl_world, radius=0.01, num_segments=10, color=(0.0, 0.0, 1.0))
        # pcl_world_w_normals = torch.concat([pcl_world, pcl_world + 0.3 * self._pointcloud_normals.expand_as(pcl_world)], dim=-1)
        # for i in range(self.num_envs):
        #     self.gym.add_lines(self.viewer, self.envs[i], pcl_world_w_normals.shape[1], pcl_world_w_normals[i].cpu().numpy(), np.array([1.0, 0.0, 0.0]).reshape(1,3).repeat(pcl_world_w_normals.shape[1], axis=0).astype(np.float32))

    def _get_target_surface_points_world(self) -> torch.Tensor:
        canonical = self._pointclouds  # (1,P,3)
        pc_world = quat_rotate(self.box_ori[:, None, :], canonical) + self.box_pos[:, None, :]
        # print("pc_world:", pc_world)
        # print("box_ori", self.box_ori)
        # print("box_pos", self.box_pos)
        return pc_world

    def compute_reach_reward_keypoints(self):
        """Reaching reward using keypoint-to-object-surface distances with historical minima."""
        pcl_world = self._get_target_surface_points_world()
        keypoints_w = self.point_pos[:, None, :]  # (N, K=1, 3)


        # Current nearest distances from each keypoint to the object surface: (N, K)
        # torch.cdist: (N, K, P) → min over P
        dists = torch.cdist(keypoints_w, pcl_world)
        cur_min_dist, _ = torch.min(dists, dim=2)  # (N, K)
        # print("cur_min_dist:", cur_min_dist[0:5].cpu().numpy())
        
        # self.cur_keypoints_to_obj_surface_dist = cur_min_dist

        delta = (self.keypoints_to_surface_dist_min - cur_min_dist).clamp_min(0.0)  # (N, K)
        self.keypoints_to_surface_dist_min = torch.min(self.keypoints_to_surface_dist_min, cur_min_dist)

        # reach_rew_keypoints = delta.mean(dim=1)  # (N,)
        # self.reach_rew_keypoints = reach_rew_keypoints
        # self.reach_rew_scaled_keypoints = self.reach_rew_keypoints * 20.0 * 5
        
        delta = 0.99 * torch.exp(-cur_min_dist/3) - torch.exp(-self.prev_dist/3)  # (N, K)
        self.prev_dist = cur_min_dist.clone()
        self.reach_rew_keypoints = delta.squeeze(-1)  # (N,)
        self.reach_rew_scaled_keypoints = self.reach_rew_keypoints * 200 * 2
        # print("reach_rew_scaled_keypoints:", self.reach_rew_scaled_keypoints[0:5].cpu().numpy())

        # Logging
        self.extras["keypoint_surface_distances"] = cur_min_dist.clone()
        self.extras["keypoints_to_surface_dist_min"] = self.keypoints_to_surface_dist_min.clone()
        self.extras["reach_rew_keypoints"] = self.reach_rew_scaled_keypoints.clone()

    def compute_surface_distance(self) -> torch.Tensor:
        # Features per fingertip: [u_hat(3), r_log_norm(1)] → total 4 fingertips × 4 = 16
        pcl_world = self._get_target_surface_points_world()   # (N, P, 3)
        tips = self.point_pos.unsqueeze(1)                        # (N, 1, 3)

        # pairwise distances (batched): (N, 1, P)
        dists = torch.cdist(tips, pcl_world)
        min_dists_per_finger, idx_p = torch.min(dists, dim=2)  # (N,1), (N,1)
        print("contact idx:", idx_p[0])
        print("contact point", self._pointclouds[0, idx_p[0,0]])
        print("contact cluster", self.reach_curiosity_mgr._point_to_cluster[idx_p[0,0]])

        # gather nearest surface point for each fingertip
        idx_p_exp = idx_p.unsqueeze(-1).expand(-1, -1, 3)      # (N,1,3)
        obj_pts = torch.gather(pcl_world, 1, idx_p_exp)        # (N,1,3)
        u = obj_pts - tips                                     # (N,1,3)

        r = torch.norm(u, dim=2, keepdim=True).clamp_min(1e-6) # (N,1,1)

        return r                 # (N,1,1)

    def compute_contact_filtered_point_relative_pos(self):
        """
        Compute point position relative to the target object's center, filtered by contact.
        Returns:
            filtered_rel (Tensor): (N, 1, 3)
            has_contact (BoolTensor): (N,) any point satisfied both conditions
        """
        # Relative fingertip positions to object center: (N,1,3)
        rel_pos = self.point_pos.unsqueeze(1) - self.box_pos.unsqueeze(1)

        # Distance to nearest surface point per fingertip: (N,1,1)
        r = self.compute_surface_distance()  # r: (N,1,1)


        contact_mag = self.point_net_contact_forces.norm(dim=-1, p=2) # (N,1)

        # Contact filters
        near_surface = (r.squeeze(-1) < self.agent_radius + 0.02)  # (N,1)
        # print("distance to surface:", r.squeeze(-1).cpu().numpy())

        has_force = (contact_mag > 0.01) # (N,1)
        contact_mask = near_surface & has_force.unsqueeze(-1) # (N,1)
        # print("contact force:", contact_mag.squeeze(-1).cpu().numpy())
        # print("net contact forces", self.net_contact_forces.cpu().numpy())

        # Apply mask
        filtered_rel = rel_pos * contact_mask.unsqueeze(-1)  # (N,1,3)

        has_contact = contact_mask.any(dim=1)  # (N,)
        return filtered_rel, has_contact

    def refresh_contact_mask(self):
        filtered_rel, _ = self.compute_contact_filtered_point_relative_pos()  # (N,1,3), (N,)

        contact_mask = (filtered_rel.abs().sum(dim=-1) > 0)  # (N,1) bool
        self.contact_mask = contact_mask

    def check_contact_criteria(self):
        point_contact = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)

        return point_contact

    def compute_curiosity_informed_reach_reward(self):
        """Compute reach reward - curiosity-informed version using fingertip positions.
        """
        from .torch_utils import quat_conjugate, quat_apply
        # compute per-fingertip nearest canonical index in world frame
        canonical = self.reach_curiosity_mgr.canonical_pointcloud
        pc_world = quat_rotate(self.box_ori[:, None, :], canonical.unsqueeze(0).expand(self.num_envs, -1, -1)) \
                + self.box_pos[:, None, :]  # (N,M,3)

        # construct point cloud feature from world pointcloud
        mgr = self.reach_curiosity_mgr
        goal_positions = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=pc_world.dtype).unsqueeze(0).expand(self.num_envs, -1)
        goal_orientations = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=pc_world.dtype).unsqueeze(0).expand(self.num_envs, -1)
        state_features_world = mgr.build_state_features_from_world_pc(
            pc_world,
            goal_positions=goal_positions,
            goal_orientations=goal_orientations,
        )
        
        kp_pos_world = self.point_pos.unsqueeze(1)      # (N,1,3)

        d = torch.cdist(kp_pos_world, pc_world)        # (N,1,M)
        contact_indices = d.argmin(dim=-1)                         # (N,1)
        self.keypoint_to_surface_dist = d.min(dim=-1)[0]                 # (N,1)
        
        q_conj = quat_conjugate(self.box_ori)                      # (N,4)
        q_conj_exp = q_conj.unsqueeze(1).expand(-1, kp_pos_world.shape[1], -1)      # (N,1,4)

        # NOTE: inward palm direction: leap if -x, allegro is +x
        axis_local = torch.tensor([-1.0, 0.0, 0.0], device=self.device, dtype=kp_pos_world.dtype).view(1, 1, 3)
        axis_local = axis_local.expand_as(kp_pos_world)

        kp_ori_world = self.box_ori.unsqueeze(1).expand_as(q_conj_exp)    # (N,1,4)

        dir_world = quat_apply(kp_ori_world.reshape(-1, 4), axis_local.reshape(-1, 3)).view_as(kp_pos_world)
        contact_dirs_canonical = quat_apply(q_conj_exp, -dir_world)

        reward, info = self.reach_curiosity_mgr.compute_reward_from_canonical(
            object_positions=self.box_pos,
            object_orientations=self.box_ori,
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
            predefined_state_ids=self.initial_indices
        )
        
        # self.reach_curiosity_rew = reward
        # self.reach_curiosity_rew_scaled = self.reach_curiosity_rew * 10
        self.reach_curiosity_rew = info["potential_field_reward"].clone()
        num_key_state_scale = 6 / self.reach_curiosity_mgr.num_key_states
        self.reach_curiosity_rew_scaled = self.reach_curiosity_rew * 5.12 * 100 * 5 * num_key_state_scale# 5.12 1.28 0.01, 0.02, 0.04
        self.contact_coverage_rew = info["cluster_novelty_reward"].clone()
        self.contact_coverage_rew_scaled = self.contact_coverage_rew * 800 * 8 * 5 * num_key_state_scale
        self.contact_coverage_rew_scaled = self.contact_coverage_rew_scaled
        self.extras["reach_curiosity_rew"] = self.reach_curiosity_rew_scaled.clone()
        self.extras["avg_potential"] = info["avg_potential"].clone()
        self.extras["contact_count"] = info["contact_count"].clone()
        self.extras["cluster_novelty_reward"] = self.contact_coverage_rew_scaled.clone()


        if self.reach_curiosity_mgr.state_type == "hash":
            self.extras["hash_recon_loss"] = info["hash_recon_loss"].clone().detach().repeat(self.num_envs)
            self.extras["hash_binary_reg"] = info["hash_binary_reg"].clone().detach().repeat(self.num_envs)
            self.extras["stateid_entropy"] = info["stateid_entropy"].clone().detach().repeat(self.num_envs)

    # --------------------------
    # Observations / rewards
    # --------------------------
    def compute_observations(self):

        agent_xy = self.point_pos[:, 0:2]
        agent_vxy = self.point_vel[:, 0:2]
        agent_dof_pos = self.point_dof_pos
        agent_dof_vel = self.point_dof_vel
        box_xy = self.box_pos[:, 0:2]
        box_ori = self.box_ori
        box_vxy = self.box_vel[:, 0:2]
        box_avz = self.box_angvel[:, 2:3]

        box_goal = box_xy - self.goal_xy.unsqueeze(0)          # (N,2)
        box_agent = box_xy - agent_xy                          # (N,2)

        obs = torch.cat([agent_xy, agent_vxy, agent_dof_pos, agent_dof_vel, box_xy, box_ori, box_vxy, box_avz, box_goal, box_agent], dim=-1) # (N,21)
        # obs = torch.cat([box_xy, box_ori, box_vxy, box_avz, box_goal, box_agent], dim=-1) # (N,13)
        # print("obs:", obs.cpu().numpy())
        self.obs_buf[:] = obs

        # print("box_xyz:", rs[:, self.box_actor_id, 0:3].cpu().numpy())
        # print("goal_xyz:", rs[:, self.goal_actor_id, 0:3].cpu().numpy())

    def compute_reward(self):
        reward_types = self.reward_type.strip().split("+")
        assert len(reward_types) > 0, f"No reward type specified, reward_type from cfg is {self.reward_type}"

        box_xy = self.box_pos[:, 0:2]

        dist = torch.norm(box_xy - self.goal_xy.unsqueeze(0), dim=-1)  # (N,)
        delta = (self.dist_min - dist).clamp_min(0.0)  # (N,)
        self.extras["box_goal_dist"] = dist
        # delta = 0.99 * torch.exp(-dist/3) - torch.exp(-self.prev_box_goal_dist/3).squeeze(-1)  # (N,)
        # self.prev_box_goal_dist = dist
        self.dist_min = torch.min(self.dist_min, dist)
        dist_rew = self.progress_scale * delta
        self.extras["dist_rew"] = dist_rew

        reach_dist = torch.norm(box_xy - self.point_pos[:, 0:2], dim=-1)
        # print("box pos", box_xy, "point agent pos", self.point_pos[:, 0:2], "reach_dist:", reach_dist)
        self.extras["reach_dist"] = reach_dist
        delta = (self.reach_dist_min - reach_dist).clamp_min(0.0)
        self.reach_dist_min = torch.min(self.reach_dist_min, reach_dist)
        reach_dist_rew = self.reach_progress_scale * delta
        self.extras["reach_dist_rew"] = reach_dist_rew

        self.refresh_contact_mask() # self.contact_mask
        self.contact_satisfied = self.check_contact_criteria()

        self.compute_reach_reward_keypoints()
        # reach_dist_rew = self.reach_rew_scaled_keypoints
        if self.training:
            self.compute_curiosity_informed_reach_reward()
        else:
            self.reach_curiosity_rew_scaled = torch.zeros_like(self.rew_buf)
            self.contact_coverage_rew_scaled = torch.zeros_like(self.rew_buf)

        # print("contact mask", self.contact_mask)
        # print("counter", self.reach_curiosity_mgr.state_bank.counts)

        # default dense shaping: progress - action cost + success bonus
        # rew = dist_rew + reach_dist_rew #- self.time_penalty #- self.action_penalty * torch.sum(self.actions ** 2, dim=-1)

        success = dist < self.success_thresh
        # rew = torch.where(success, rew + self.success_bonus, rew)
        self.successes = success.float()

        # reset conditions
        time_out = self.progress_buf >= (self.max_episode_length - 1)
        self.reset_buf[:] = torch.where(success | time_out, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))

        # reset if dist too large (optional)
        too_far = dist > 6.0
        self.reset_buf[:] = torch.where(too_far, torch.ones_like(self.reset_buf), self.reset_buf)
        # rew = torch.where(too_far, rew + self.too_far_penalty, rew)

        # reset if reach_dist too large (optional)
        reach_too_far = reach_dist > 6.0
        self.reset_buf[:] = torch.where(reach_too_far, torch.ones_like(self.reset_buf), self.reset_buf)
        # rew = torch.where(reach_too_far, rew + self.reach_too_far_penalty, rew)

        self.rew_buf[:] = 0
        if "task" in reward_types:
            task_rew = dist_rew + success.float() * self.success_bonus + too_far.float() * self.too_far_penalty + reach_too_far.float() * self.reach_too_far_penalty
            self.rew_buf[:] += task_rew
        if "reach_center" in reward_types:
            self.rew_buf[:] += reach_dist_rew
        if "reach" in reward_types:
            self.rew_buf[:] += self.reach_rew_scaled_keypoints
        if "energy_reach" in reward_types:
            self.rew_buf[:] += self.reach_curiosity_rew_scaled
        if "contact_coverage" in reward_types:
            self.rew_buf[:] += self.contact_coverage_rew_scaled

        self.extras["task_reward"] = task_rew.clone() if "task" in reward_types else torch.zeros_like(self.rew_buf)
        self.extras["reward"] = self.rew_buf.clone()
        print("reward: ", self.rew_buf[0])

        # extras for logging
        # self.extras["success"] = success.float()
        self.extras["box_goal_dist"] = dist

    # --------------------------
    # Reset
    # --------------------------
    def reset(self, dones=None, first_time=False):
        if dones is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
            # self.reach_curiosity_mgr.reset_counters()
        else:
            env_ids = dones.nonzero(as_tuple=False).flatten()

        # reset idx
        if env_ids.shape[0] > 0:
            self.reset_idx(env_ids, first_time=first_time)

        self.compute_observations()

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return self.obs_dict
    
    def reset_idx(self, env_ids: torch.Tensor, first_time: bool = False):
        if env_ids.numel() == 0:
            return

        rs = self.root_state

        # agent init
        # rs[env_ids, self.agent_actor_id, 0] = 0.0
        # rs[env_ids, self.agent_actor_id, 1] = -3.0
        # rs[env_ids, self.agent_actor_id, 2] = 1.0 #self.agent_radius
        # rs[env_ids, self.agent_actor_id, 3:7] = torch.tensor([0, 0, 0, 1], device=self.rl_device, dtype=torch.float32)
        # rs[env_ids, self.agent_actor_id, 7:13] = 0.0

        # box init at (-3,0) or (3,0): random per env
        # 使用 env_id 奇偶当随机也行；这里用 torch 随机
        r = torch.rand((env_ids.numel(),), device=self.rl_device)
        if self.cfg["env"].get("randomInit", False):
            prob_thres = 0.5
        else:
            prob_thres = 0.0
        initial_signal = r < prob_thres
        self.initial_indices[env_ids] = initial_signal.long()
        x0 = torch.where(initial_signal, torch.full_like(r, 3.0), torch.full_like(r, -3.0))
        rs[env_ids, self.box_actor_id, 0] = x0
        rs[env_ids, self.box_actor_id, 1] = 0.0
        rs[env_ids, self.box_actor_id, 2] = 1.0
        rs[env_ids, self.box_actor_id, 3:7] = torch.tensor([0, 0, 0, 1], device=self.rl_device, dtype=torch.float32)
        rs[env_ids, self.box_actor_id, 7:13] = 0.0

        # rs[env_ids, self.goal_actor_id, 0] = 0.0
        # rs[env_ids, self.goal_actor_id, 1] = 0.0
        # rs[env_ids, self.goal_actor_id, 2] = 1.0
        # rs[env_ids, self.goal_actor_id, 3:7] = torch.tensor([0, 0, 0, 1], device=self.rl_device, dtype=torch.float32)
        # rs[env_ids, self.goal_actor_id, 7:13] = 0.0
        # write back
        box_actor_indices = self.box_actor_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(rs.view(-1, 13)),
                                                     gymtorch.unwrap_tensor(box_actor_indices), len(env_ids))

        pos = self.point_default_dof_pos.clone()
        self.point_dof_pos[env_ids, :] = pos
        # self.point_dof_pos[env_ids, 0] = (initial_signal.float() * 2 -1) * 5
        # self.point_dof_pos[env_ids, 1] = 3.0

        self.point_dof_vel[env_ids, :] = 0.0
        self.prev_targets[env_ids, :self.num_point_dofs] = pos
        self.cur_targets[env_ids, :self.num_point_dofs] = pos

        agent_actor_indices = self.agent_actor_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(agent_actor_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(agent_actor_indices), len(env_ids))
        # book-keeping
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        # init prev dist
        # box_xy = rs[env_ids, self.box_actor_id, 0:2]
        # self.prev_box_goal_dist[env_ids] = torch.norm(box_xy - self.goal_xy.unsqueeze(0), dim=-1)

        # agent_xy = rs[env_ids, self.agent_actor_id, 0:2]
        # self.prev_reach_dist[env_ids] = torch.norm(box_xy - agent_xy, dim=-1)

        self.prev_dist[env_ids] = 2.8
        self.prev_box_goal_dist[env_ids] = 3.0

        self.dist_min[env_ids] = 3.0
        self.reach_dist_min[env_ids] = 4.2
        self.keypoints_to_surface_dist_min[env_ids] = 2.8

        self.reach_curiosity_mgr.ensure_running_max_buffers(self.num_envs)
        self.reach_curiosity_mgr.reset_running_max_buffers(env_ids)

    def train(self):
        self.training = True

    def eval(self, vis=False):
        self.training = False    


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
