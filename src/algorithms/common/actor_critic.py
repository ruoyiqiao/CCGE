from ipaddress import v4_int_to_packed
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Independent
import math

# from networks.pointnet import PointNetEncoder
from algorithms.common.normalization import EmpiricalNormalization
from tasks.isaacgym_utils import pack_pointcloud_observations

fuse_type = "concat" # concat or sep

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        model_cfg,
        asymmetric=False,
        pointnet_type="pt2",
        observation_info=None,
        in_pointnet_feature_dim=3,
        hand_pcl=False,
        hand_model=None,
        args=None,
        stack_frame_number=1,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        obs_groups=None,
    ):
        super(ActorCritic, self).__init__()

        # network parameter
        self.asymmetric = asymmetric
        self.pointnet_type = pointnet_type
        self.in_pointnet_feature_dim = in_pointnet_feature_dim

        self.obs_groups = obs_groups
        
        """Get network input output dim."""
        # retrival observation dim for input
        self.state_dim = 0
        self.tactile_dim = 0
        self.pcl_dim = 0
        self.grad_dim = 0
        self.stack_frame_number = stack_frame_number

        for info in observation_info:
            if "tactile" in info["tags"]:
                self.tactile_dim += info["dim"]
            elif "pointcloud" in info["tags"]:
                self.pcl_dim += info["dim"] // 3
            elif "gradient" in info["tags"]:
                self.grad_dim += info["dim"]
            else:
                self.state_dim += info["dim"]
        
        if self.stack_frame_number > 1 and fuse_type == "concat":
            self.state_dim *= self.stack_frame_number
        print(">>> Initialize ActorCritic")
        print(f"  - state_dim: {self.state_dim}")
        print(f"  - tactile_dim: {self.tactile_dim}")
        print(f"  - pcl_dim: {self.pcl_dim}")
        print(f"  - grad_dim: {self.grad_dim}")
        self.observation_info = observation_info

        if obs_groups is not None:
            num_actor_obs = self.state_dim + self.tactile_dim + self.grad_dim
            if self.pcl_dim > 0:
                num_actor_obs += model_cfg.get("pcl_feature_dim", 512)
            num_critic_obs = num_actor_obs
        else:
            num_actor_obs = obs_shape[0] if isinstance(obs_shape, (list, tuple)) else obs_shape
            num_critic_obs = num_actor_obs

        # retrival action dim
        self.action_dim = actions_shape[0]
        """
        init network: current we set self.state_base = False, only set true for pure state input
        """
        # network parameter
        activation = get_activation(model_cfg["activation"])
        self.shared_pointnet = model_cfg["shared_pointnet"]
        self.points_per_object = model_cfg["points_per_object"]
        """Actor layer."""
        # state encoder
        actor_state_encoder_hid_sizes = model_cfg["pi_state_encoder_hid_sizes"]
        actor_hidden_dim = actor_state_encoder_hid_sizes[-1]
        self.actor_state_enc = self.build_block(
            self.state_dim, actor_hidden_dim, activation, actor_state_encoder_hid_sizes
        )
        self.total_feat_num = 1
        
        if self.stack_frame_number > 1 and fuse_type == "sep":
            self.pos_emb = SinusoidalPosEmb(actor_hidden_dim*2)
            
            self.actor_state_fuser = self.build_block(
                actor_hidden_dim * 4, actor_hidden_dim, activation, actor_state_encoder_hid_sizes
            )

        # tactile feature encoder
        if self.tactile_dim > 0:
            self.actor_tactile_enc = self.build_block(self.tactile_dim, actor_hidden_dim, activation, [])
            self.total_feat_num += 1
        # pointcloud feature encoder
        if self.pcl_dim > 0:
            self.pcl_feature_dim = model_cfg["pcl_feature_dim"]
            self.actor_pcl_enc = self.build_block(self.pcl_feature_dim, actor_hidden_dim, activation, [])
            self.total_feat_num += 1
        # gradient feature encoder
        if self.grad_dim > 0:
            self.actor_grad_enc = self.build_block(self.grad_dim, actor_hidden_dim, activation, [])
            self.total_feat_num += 1

        # fuse feature
        if self.total_feat_num > 1:
            self.actor_fuse = self.build_block(actor_hidden_dim * self.total_feat_num, actor_hidden_dim, activation, [])

        # mlp output
        self.actor_output = self.build_block(
            actor_hidden_dim, self.action_dim, activation, [], activate_for_last_layer=False
        )
        
        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
            
        """Critic layer."""
        # state encoder
        critic_state_encoder_hid_sizes = model_cfg["vf_state_encoder_hid_sizes"]
        critic_hidden_dim = critic_state_encoder_hid_sizes[-1]
        self.critic_mode = model_cfg.get("critic_mode", "single")
        self.num_critics = 2 if self.critic_mode == "dual" else 1
        self.critic_state_enc = self.build_block(
            self.state_dim, critic_hidden_dim, activation, critic_state_encoder_hid_sizes
        )
        
        if self.stack_frame_number > 1 and fuse_type == "sep":
            self.critic_fuser = self.build_block(
                critic_hidden_dim * 4, critic_hidden_dim, activation, critic_state_encoder_hid_sizes
            )

        # tactile feature encoder
        if self.tactile_dim > 0:
            self.critic_tactile_enc = self.build_block(self.tactile_dim, critic_hidden_dim, activation, [])

        # pointcloud feature encoder
        if self.pcl_dim > 0:
            self.critic_pcl_enc = self.build_block(self.pcl_feature_dim, critic_hidden_dim, activation, [])

        # gradient feature encoder
        if self.grad_dim > 0:
            self.critic_grad_enc = self.build_block(self.grad_dim, critic_hidden_dim, activation, [])

        # fuse feature
        if self.total_feat_num > 1:
            # mlp output
            self.critic_fuse = self.build_block(
                critic_hidden_dim * self.total_feat_num, critic_hidden_dim, activation, []
            )

        # mlp output
        if self.critic_mode == "dual":
            self.critic_output_ext = self.build_block(critic_hidden_dim, 1, activation, [], activate_for_last_layer=False)
            self.critic_output_int = self.build_block(critic_hidden_dim, 1, activation, [], activate_for_last_layer=False)
        else:
            self.critic_output = self.build_block(critic_hidden_dim, 1, activation, [], activate_for_last_layer=False)

        
        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        if self.pcl_dim > 0:
            """Shared layer."""
            if self.shared_pointnet:
                if self.pointnet_type == "pt":
                    self.pointnet_enc = PointNetEncoder(
                        num_points=self.pcl_dim, in_dim=self.in_pointnet_feature_dim, out_dim=self.pcl_feature_dim
                    )
            else:
                if self.pointnet_type == "pt":
                    self.actor_pointnet_enc = PointNetEncoder(
                        num_points=self.pcl_dim, in_dim=self.in_pointnet_feature_dim, out_dim=self.pcl_feature_dim
                    )
                    self.critic_pointnet_enc = PointNetEncoder(
                        num_points=self.pcl_dim, in_dim=self.in_pointnet_feature_dim, out_dim=self.pcl_feature_dim
                    )

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))
        """ilad."""
        if args is not None and args.exp_name == "ilad":
            self.additional_critic_mlp1 = self.build_block(
                critic_hidden_dim + self.action_dim, 1, activation, [], activate_for_last_layer=False
            )

        self._initialize_weights()

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)

        # Actor networks
        self.init_weights(self.actor_state_enc, [np.sqrt(2)] * len(self.actor_state_enc))
        if hasattr(self, 'actor_state_fuser'):
            self.init_weights(self.actor_state_fuser, [np.sqrt(2)] * len(self.actor_state_fuser))
        if hasattr(self, 'actor_tactile_enc'):
            self.init_weights(self.actor_tactile_enc, [np.sqrt(2)] * len(self.actor_tactile_enc))
        if hasattr(self, 'actor_pcl_enc'):
            self.init_weights(self.actor_pcl_enc, [np.sqrt(2)] * len(self.actor_pcl_enc))
        if hasattr(self, 'actor_grad_enc'):
            self.init_weights(self.actor_grad_enc, [np.sqrt(2)] * len(self.actor_grad_enc))
        if hasattr(self, 'actor_fuse'):
            self.init_weights(self.actor_fuse, [np.sqrt(2)] * len(self.actor_fuse))

        # Critic networks
        self.init_weights(self.critic_state_enc, [np.sqrt(2)] * len(self.critic_state_enc))
        if hasattr(self, 'critic_fuser'):
            self.init_weights(self.critic_fuser, [np.sqrt(2)] * len(self.critic_fuser))
        if hasattr(self, 'critic_tactile_enc'):
            self.init_weights(self.critic_tactile_enc, [np.sqrt(2)] * len(self.critic_tactile_enc))
        if hasattr(self, 'critic_pcl_enc'):
            self.init_weights(self.critic_pcl_enc, [np.sqrt(2)] * len(self.critic_pcl_enc))
        if hasattr(self, 'critic_grad_enc'):
            self.init_weights(self.critic_grad_enc, [np.sqrt(2)] * len(self.critic_grad_enc))
        if hasattr(self, 'critic_fuse'):
            self.init_weights(self.critic_fuse, [np.sqrt(2)] * len(self.critic_fuse))


        actor_output_linear = [m for m in self.actor_output if isinstance(m, nn.Linear)]
        if actor_output_linear:
            torch.nn.init.orthogonal_(actor_output_linear[-1].weight, gain=0.01)  # type: ignore

        if self.critic_mode == "dual":
            lin_e = [m for m in self.critic_output_ext if isinstance(m, nn.Linear)]
            lin_i = [m for m in self.critic_output_int if isinstance(m, nn.Linear)]
            if lin_e:
                torch.nn.init.orthogonal_(lin_e[-1].weight, gain=1.0)  # type: ignore
            if lin_i:
                torch.nn.init.orthogonal_(lin_i[-1].weight, gain=1.0)  # type: ignore
        else:
            critic_output_linear = [m for m in self.critic_output if isinstance(m, nn.Linear)]
            if critic_output_linear:
                torch.nn.init.orthogonal_(critic_output_linear[-1].weight, gain=1.0)  # type: ignore

        # Additional critic for ILAD
        if hasattr(self, 'additional_critic_mlp1'):
            additional_critic_linear = [m for m in self.additional_critic_mlp1 if isinstance(m, nn.Linear)]
            if additional_critic_linear:
                torch.nn.init.orthogonal_(additional_critic_linear[-1].weight, gain=1.0)  # type: ignore

    def build_block(self, input_dim, output_dim, activation, hidden_dim, activate_for_last_layer=True):
        layers = []
        if len(hidden_dim) == 0:
            layers.append(nn.Linear(input_dim, output_dim))
            if activate_for_last_layer:
                layers.append(activation)
        else:
            layers.append(nn.Linear(input_dim, hidden_dim[0]))
            layers.append(activation)
            for l in range(len(hidden_dim)):
                if l == len(hidden_dim) - 1:
                    layers.append(nn.Linear(hidden_dim[l], output_dim))
                    if activate_for_last_layer:
                        layers.append(activation)
                else:
                    layers.append(nn.Linear(hidden_dim[l], hidden_dim[l + 1]))
                    layers.append(activation)
        return nn.Sequential(*layers)

    def forward(self):
        raise NotImplementedError

    def forward_actor(self, observations):
        """Process observation."""
        batch_size = observations.size(0)

        state_batch, tactile_batch, pcl_batch, gf_batch = self.process_observations(observations=observations)
        """forward."""
        # state encoder
        
        if self.stack_frame_number > 1 and fuse_type == "sep":
            state_feat = []
            for i in range(self.stack_frame_number):
                # from ipdb import set_trace; set_trace()
                state_feat.append(self.pos_emb(self.actor_state_enc(state_batch[:, i*self.state_dim:(i+1)*self.state_dim]))[:,0,:])
            
            state_feat = torch.cat(state_feat, dim=-1)
            state_feat = self.actor_state_fuser(state_feat)
        else:
            state_feat = self.actor_state_enc(state_batch)

        # pointcloud encoder
        if self.tactile_dim > 0:
            tactile_feat = self.actor_tactile_enc(tactile_batch)
        if self.pcl_dim > 0:
            if self.shared_pointnet:
                if self.pointnet_type == "pt":
                    pcl_batch = pcl_batch.reshape(batch_size, -1, self.in_pointnet_feature_dim)
                    pcl_batch = pcl_batch.permute(0, 2, 1)
                    pcl_feat, _, _ = self.pointnet_enc(pcl_batch)
            else:
                if self.pointnet_type == "pt":
                    pcl_batch = pcl_batch.reshape(batch_size, -1, self.in_pointnet_feature_dim)
                    pcl_batch = pcl_batch.permute(0, 2, 1)
                    pcl_feat, _, _ = self.actor_pointnet_enc(pcl_batch)
            pcl_feat = self.actor_pcl_enc(pcl_feat.reshape(batch_size, -1))  # B x 512
        if self.grad_dim > 0:
            grad_feat = self.actor_grad_enc(gf_batch)

        # fuse
        x = state_feat
        if self.tactile_dim > 0:
            x = torch.cat([x, tactile_feat], -1)
        if self.pcl_dim > 0:
            x = torch.cat([x, pcl_feat], -1)
        if self.grad_dim > 0:
            x = torch.cat([x, grad_feat], -1)
        if self.total_feat_num > 1:
            x = self.actor_fuse(x)

        # output
        x = self.actor_output(x)
        return x

    def forward_critic(self, observations):
        """Process observation."""
        batch_size = observations.size(0)

        state_batch, tactile_batch, pcl_batch, gf_batch = self.process_observations(observations=observations)
        """forward."""
        # state encoder
        if self.stack_frame_number > 1 and fuse_type == "sep":
            state_feat = []
            for i in range(self.stack_frame_number):
                state_feat.append(self.pos_emb(self.critic_state_enc(state_batch[:, i*self.state_dim:(i+1)*self.state_dim]))[:,0,:])
            
            state_feat = torch.cat(state_feat, dim=-1)
            state_feat = self.critic_fuser(state_feat)
        else:
            state_feat = self.critic_state_enc(state_batch)
        

        if self.tactile_dim > 0:
            tactile_feat = self.critic_tactile_enc(tactile_batch)
        # point cloud encoder
        if self.pcl_dim > 0:
            if self.shared_pointnet:
                if self.pointnet_type == "pt":
                    pcl_batch = pcl_batch.reshape(batch_size, -1, self.in_pointnet_feature_dim)
                    pcl_batch = pcl_batch.permute(0, 2, 1)
                    pcl_feat, _, _ = self.pointnet_enc(pcl_batch)
            else:
                if self.pointnet_type == "pt":
                    pcl_batch = pcl_batch.reshape(batch_size, -1, self.in_pointnet_feature_dim)
                    pcl_batch = pcl_batch.permute(0, 2, 1)
                    pcl_feat, _, _ = self.critic_pointnet_enc(pcl_batch)
            pcl_feat = self.critic_pcl_enc(pcl_feat.reshape(batch_size, -1))  # B x 512
        if self.grad_dim > 0:
            grad_feat = self.critic_grad_enc(gf_batch)

        # fuse
        x = state_feat
        if self.tactile_dim > 0:
            x = torch.cat([x, tactile_feat], -1)
        if self.pcl_dim > 0:
            x = torch.cat([x, pcl_feat], -1)
        if self.grad_dim > 0:
            x = torch.cat([x, grad_feat], -1)

        if self.total_feat_num > 1:
            x = self.critic_fuse(x)

        if self.critic_mode == "dual":
            v_ext = self.critic_output_ext(x)
            v_int = self.critic_output_int(x)
            return torch.cat([v_ext, v_int], -1) # ext, int
        else:
            x = self.critic_output(x)
            return x

    def forward_additional_critic(self, observations, actions):
        """Process observation."""
        batch_size = observations.size(0)

        state_batch, tactile_batch, pcl_batch, gf_batch = self.process_observations(observations=observations)
        """forward."""
        # state encoder
        state_feat = self.critic_state_enc(state_batch)

        if self.tactile_dim > 0:
            tactile_feat = self.critic_tactile_enc(tactile_batch)
        # point cloud encoder
        if self.pcl_dim > 0:
            if self.shared_pointnet:
                if self.pointnet_type == "pt":
                    pcl_batch = pcl_batch.reshape(batch_size, -1, self.in_pointnet_feature_dim)
                    pcl_batch = pcl_batch.permute(0, 2, 1)
                    pcl_feat, _, _ = self.pointnet_enc(pcl_batch)
            else:
                if self.pointnet_type == "pt":
                    pcl_batch = pcl_batch.reshape(batch_size, -1, self.in_pointnet_feature_dim)
                    pcl_batch = pcl_batch.permute(0, 2, 1)
                    pcl_feat, _, _ = self.critic_pointnet_enc(pcl_batch)
            pcl_feat = self.critic_pcl_enc(pcl_feat.reshape(batch_size, -1))  # B x 512
        if self.grad_dim > 0:
            grad_feat = self.critic_grad_enc(gf_batch)

        # fuse
        x = state_feat
        if self.tactile_dim > 0:
            x = torch.cat([x, tactile_feat], -1)
        if self.pcl_dim > 0:
            x = torch.cat([x, pcl_feat], -1)
        if self.grad_dim > 0:
            x = torch.cat([x, grad_feat], -1)

        if self.total_feat_num > 1:
            x = self.critic_fuse(x)

        # output
        x = torch.concat([x, actions], -1)
        x = self.additional_critic_mlp1(x)
        return x

    def process_observations(self, observations: torch.Tensor):
        device = observations.device
        batch_size = observations.size(0)
        state_batch = []
        tactile_batch = []
        pcl_batch = {}
        gf_batch = []
        for info in self.observation_info:
            if "tactile" in info["tags"]:
                tactile_batch.append(observations[:, info["start"] : info["end"]])
            elif "pointcloud" in info["tags"]:
                # pcl_batch.append(observations[:, info["start"] : info["end"]])
                pcl_batch[info["name"]] = {}
                pcl_batch[info["name"]]["points"] = observations[:, info["start"] : info["end"]].reshape(
                    batch_size, -1, 3
                )
            elif "gradient" in info["tags"]:
                gf_batch.append(observations[:, info["start"] : info["end"]])
            else:
                state_batch.append(observations[:, info["start"] : info["end"]])

        if self.stack_frame_number > 1:
            state_batch = observations[:, :]
        else:
            state_batch = torch.cat(state_batch, dim=-1)

        if self.tactile_dim > 0:
            tactile_batch = torch.cat(tactile_batch, dim=-1)
        else:
            tactile_batch = None

        if self.pcl_dim > 0:
            # pcl_batch = torch.cat(pcl_batch, dim=-1)
            pcl_batch = pack_pointcloud_observations(pcl_batch, device=device)
        else:
            pcl_batch = None

        if self.grad_dim > 0:
            gf_batch = torch.cat(gf_batch, dim=-1)
        else:
            gf_batch = None
        return state_batch, tactile_batch, pcl_batch, gf_batch

    def act(self, observations: torch.Tensor, states: Optional[torch.Tensor] = None):
        actor_observations = observations.clone()
        if self.actor_obs_normalization:
            actor_observations = self.actor_obs_normalizer(observations)

        actions_mean = self.forward_actor(actor_observations)

        # print(self.log_std)
        # covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        # distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        
        distribution = Independent(Normal(actions_mean, self.log_std.exp()), 1)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        if self.asymmetric:
            value = self.critic(states)
        else:
            critic_observations = observations.clone()
            if self.critic_obs_normalization:
                critic_observations = self.critic_obs_normalizer(observations)
            value = self.forward_critic(critic_observations)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
        )

    def cal_actions_log_prob(self, observations: torch.Tensor, actions: torch.Tensor):

        if self.actor_obs_normalization:
            observations = self.actor_obs_normalizer(observations)
            
        actions_mean = self.forward_actor(observations)

        # covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        # distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        distribution = Independent(Normal(actions_mean, self.log_std.exp()), 1)

        actions_log_prob = distribution.log_prob(actions)
        return actions.detach(), actions_log_prob.detach(), actions_mean.detach()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:

        if self.actor_obs_normalization:
            observations = self.actor_obs_normalizer(observations)
            
        actions_mean = self.forward_actor(observations)
        return actions_mean

    def evaluate(self, observations, states, actions):

        if self.actor_obs_normalization:
            actor_observations = self.actor_obs_normalizer(observations)
        else:
            actor_observations = observations
            
        actions_mean = self.forward_actor(actor_observations)

        # covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        # distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        
        distribution = Independent(Normal(actions_mean, self.log_std.exp()), 1)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.asymmetric:
            value = self.critic(states)
        else:
            if self.critic_obs_normalization:
                critic_observations = self.critic_obs_normalizer(observations)
            else:
                critic_observations = observations
            value = self.forward_critic(critic_observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


    def get_actor_obs(self, obs):
        """Get actor observations. Compatible with both old and new observation formats."""
        if self.obs_groups is not None and isinstance(obs, dict):
            obs_list = []
            for obs_group in self.obs_groups["policy"]:
                obs_list.append(obs[obs_group])
            return torch.cat(obs_list, dim=-1)
        else:
            return obs

    def get_critic_obs(self, obs):
        """Get critic observations. Compatible with both old and new observation formats."""
        if self.obs_groups is not None and isinstance(obs, dict):
            obs_list = []
            for obs_group in self.obs_groups["critic"]:
                obs_list.append(obs[obs_group])
            return torch.cat(obs_list, dim=-1)
        else:
            return obs

    def update_normalization(self, obs):
        """Update observation normalization statistics."""
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Invalid activation name.")
