import copy
import functools
import glob
import io
import os
import pickle
import statistics
import time
from collections import deque
from datetime import datetime

import _pickle as CPickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from gym.spaces import Space
from ipdb import set_trace
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithms.ppo.storage import RolloutStorage, RolloutStorageDual, RolloutStorageCuriosity
from algorithms.ppo.utils import AverageScalarMeter, RunningMeanStd

from tasks.torch_utils import get_euler_xyz

from ..common.actor_critic import ActorCritic
from .curiosity_model import CuriosityModel

save_video = False
img_size = 256
save_traj = False
ana = False
obs_state_dim = 208
plot_direction = False
pcl_number = 512


def images_to_video(path, images, fps=10, size=(256, 256), suffix="mp4"):
    path = path + f".{suffix}"
    out = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=fps, frameSize=size, isColor=True)
    for item in images:
        out.write(item.astype(np.uint8))
    out.release()


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


class PPO:
    def __init__(
        self,
        vec_env,
        cfg_train,
        device="cpu",
        sampler="sequential",
        log_dir="",
        is_testing=False,
        print_log=True,
        apply_reset=False,
        asymmetric=False,
        args=None,
    ):
        self.args = args
        """PPO."""
        # PPO parameters
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.cfg_train = copy.deepcopy(cfg_train)
        learn_cfg = self.cfg_train["learn"]
        self.device = device
        self.asymmetric = asymmetric
        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.step_size = learn_cfg["optim_stepsize"]
        self.init_noise_std = learn_cfg["init_noise_std"]
        self.normalize_input = learn_cfg["normalize_input"]
        self.normalize_value = learn_cfg["normalize_value"]
        self.value_bootstrap = learn_cfg.get("value_bootstrap", False)
        self.model_cfg = self.cfg_train["policy"]
        self.num_transitions_per_env = learn_cfg["nsteps"]
        self.learning_rate = learn_cfg["optim_stepsize"]

        self.clip_param = learn_cfg["cliprange"]
        self.num_learning_epochs = learn_cfg["noptepochs"]
        self.num_mini_batches = learn_cfg["nminibatches"]
        self.value_loss_coef = learn_cfg.get("value_loss_coef", 2.0)
        self.entropy_coef = learn_cfg["ent_coef"]
        self.bounds_loss_coef = learn_cfg.get("bounds_loss_coef", 0.0)
        self.reward_scale_value = learn_cfg.get("reward_scale_value", 1.0)
        self.intrinsic_reward_scale = learn_cfg.get("intrinsic_reward_scale", 1.0)
        self.gamma = learn_cfg["gamma"]
        self.lam = learn_cfg["lam"]
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False)

        # policy type
        self.action_type = self.cfg_train["setting"]["action_type"]
        self.sub_action_type = self.cfg_train["setting"]["sub_action_type"]
        self.action_clip = self.cfg_train["setting"]["action_clip"]
        self.grad_process = self.cfg_train["setting"]["grad_process"]

        self.critic_mode = self.model_cfg.get("critic_mode", "single")
        # assert self.critic_mode == "single"
        self.critic_weights = torch.tensor(self.model_cfg.get("critics_weights", [1.0, 1.0]), device=self.device).float().view(1, 1, -1)
        self.int_gamma = self.model_cfg.get("int_gamma", self.gamma)
        self.int_lam = self.model_cfg.get("int_lam", self.lam)
        self.int_non_episodic_flags = self.model_cfg.get("int_non_episodic_flags", True)
        self.model_cfg["critic_mode"] = self.critic_mode

        if self.action_type == "joint":
            if self.sub_action_type == "add+jointscale":
                action_space_shape = (vec_env.num_actions * 2,)
            elif self.sub_action_type == "addscale+add":
                action_space_shape = (vec_env.num_actions * 2 + 1,)
        else:
            action_space_shape = self.action_space.shape

        observation_space_shape = self.observation_space.shape

        self.vec_env = vec_env

        pointnet_version = self.cfg_train["policy"]["pointnet_version"]
        hand_pcl = self.cfg_train["policy"]["hand_pcl"]
        hand_model = None

        # observation_metainfo = self.vec_env.export_observation_metainfo()
        # observation_metainfo = [obs for obs in observation_metainfo if obs['name'] in obs_space]
        # PPO components
        self.actor_critic = ActorCritic(
            observation_space_shape,
            self.state_space.shape,
            action_space_shape,
            self.init_noise_std,
            self.model_cfg,
            asymmetric=asymmetric,
            pointnet_type=pointnet_version,
            observation_info=self.vec_env.export_observation_metainfo(),
            # observation_info=observation_metainfo,
            hand_pcl=hand_pcl,
            hand_model=hand_model,
            in_pointnet_feature_dim=4,  # TODO
            args=args,
            stack_frame_number=self.vec_env.stack_frame_number,
            actor_obs_normalization=self.normalize_input,
            critic_obs_normalization=self.normalize_input,
            obs_groups=None
        )

        # pointnet backbone
        if self.actor_critic.pcl_dim > 0:
            self.pointnet_finetune = self.model_cfg["finetune_pointnet"]
            self.finetune_pointnet_bz = 128
            if self.model_cfg["pretrain_pointnet"]:
                if pointnet_version == "pt2":
                    pointnet_model_dict = torch.load(
                        os.path.join(args.score_model_path, "pointnet2.pt"), map_location=self.device
                    )
                elif pointnet_version == "pt":
                    pointnet_model_dict = torch.load(
                        os.path.join(args.score_model_path, "pointnet.pt"), map_location=self.device
                    )
                if self.model_cfg["shared_pointnet"]:
                    self.actor_critic.pointnet_enc.load_state_dict(pointnet_model_dict)
                    if not self.model_cfg["finetune_pointnet"]:
                        # freeze pointnet
                        for name, param in self.actor_critic.pointnet_enc.named_parameters():
                            param.requires_grad = False
                else:
                    self.actor_critic.actor_pointnet_enc.load_state_dict(pointnet_model_dict)
                    self.actor_critic.critic_pointnet_enc.load_state_dict(pointnet_model_dict)

                    if not self.model_cfg["finetune_pointnet"]:
                        # freeze pointnet
                        for name, param in self.actor_critic.actor_pointnet_enc.named_parameters():
                            param.requires_grad = False
                        for name, param in self.actor_critic.critic_pointnet_enc.named_parameters():
                            param.requires_grad = False

        self.actor_critic.to(self.device)
        self.use_curiosity_model = self.cfg_train["setting"].get("use_curiosity_model", False)

        if self.critic_mode == "dual" and not self.use_curiosity_model:
            self.storage = RolloutStorageDual(
                self.vec_env.num_envs,
                self.num_transitions_per_env,
                observation_space_shape,
                self.state_space.shape,
                action_space_shape,
                num_critics=2,
                reward_group_weights=self.critic_weights.squeeze().tolist(),
                device=self.device,
                sampler=sampler,
            )
        elif self.critic_mode == "single" and not self.use_curiosity_model:
            self.storage = RolloutStorage(
                self.vec_env.num_envs,
                self.num_transitions_per_env,
                observation_space_shape,
                self.state_space.shape,
                action_space_shape,
                self.device,
                sampler,
            )
        elif self.critic_mode == "single" and self.use_curiosity_model:
            curiosity_state_shape = (torch.tensor(self.vec_env.curiosity_state_dim).prod().item(),)
            self.storage = RolloutStorageCuriosity(
                self.vec_env.num_envs,
                self.num_transitions_per_env,
                observation_space_shape,
                self.state_space.shape,
                action_space_shape,
                self.device,
                sampler,
                curiosity_state_shape=curiosity_state_shape,
            )

        self.obs_running_mean_std = RunningMeanStd(observation_space_shape).to(
            self.device
        )
        self.value_running_mean_std = RunningMeanStd((1,)).to(self.device) if self.critic_mode == "single" else RunningMeanStd((2,)).to(self.device)

        if self.use_curiosity_model:
            """ Add a curiosity model for exploration.
                1. rnd
                2. prediction error
                3. disagreement
                4. neural hash
            """
            self.curiosity_model = CuriosityModel(
                model_type=self.cfg_train["curiosity_model"]["model_type"],
                obs_dim=self.vec_env.num_obs,
                action_dim=self.vec_env.num_actions,
                curi_obs_dim=torch.tensor(self.vec_env.curiosity_state_dim).prod().item(),
                emb_dim=self.cfg_train["curiosity_model"]["emb_dim"],
                hidden_sizes=self.cfg_train["curiosity_model"]["hidden_sizes"],
                ensemble_size=self.cfg_train["curiosity_model"]["ensemble_size"],
                activation="relu",
                device=self.device,
                simhash_dim=self.cfg_train["curiosity_model"]["simhash_dim"],
                code_dim=self.cfg_train["curiosity_model"]["code_dim"],
                obs_act_normalization=self.cfg_train["curiosity_model"].get("obs_act_normalization", True),
                curi_obs_normalization=self.cfg_train["curiosity_model"].get("curi_obs_normalization", True)
            )

            self.curiosity_optimizer = optim.Adam(
                self.curiosity_model.model.parameters(), lr=1e-4, eps=1e-5
            )
        

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.learning_rate, 
            eps=1e-5
        )

        """Log."""
        # self.log_dir = log_dir
        if self.args.model_dir != "" and self.vec_env.mode == "train":
            time_now = self.args.model_dir.split("/")[-1].split("_")[0]
        else:
            time_now = time.strftime("%m-%d-%H-%M", time.localtime(time.time()))

        # if len(self.vec_env.object_codes) > 1:
        #     object_type = "all"
        # else:
        #     object_type = self.vec_env.object_codes[0]

        # if len(self.vec_env.label_paths) > 1:
        #     label_type = "all"
        # else:
        #     label_type = self.vec_env.label_paths[0]

        if log_dir == "":
            self.writer = None
            self.log_dir = None
        else:
            self.log_dir = f"./logs/{args.exp_name}/{time_now}_{log_dir}"

            # tensorboard logging
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            # env cfg logging
            with open(os.path.join(self.log_dir, f"{self.vec_env.cfg['name']}.yaml"), "w") as f:
                OmegaConf.save(self.vec_env.cfg, f)
            # train cfg logging
            with open(os.path.join(self.log_dir, f"{args.cfg_train}.yaml"), "w") as f:
                yaml.dump(cfg_train, f)

            # save A matrix for auto-encoder
            if hasattr(self.vec_env, "reach_curiosity_mgr") and hasattr(self.vec_env.reach_curiosity_mgr.state_bank, "ae"):
                A_matrix = self.vec_env.reach_curiosity_mgr.state_bank.A
                torch.save(A_matrix, os.path.join(self.log_dir, "A_matrix.pt"))

            if hasattr(self.vec_env, "reach_curiosity_mgr") and hasattr(self.vec_env.reach_curiosity_mgr, "_state_point_indices"):
                _state_point_indices = self.vec_env.reach_curiosity_mgr._state_point_indices
                torch.save(_state_point_indices, os.path.join(self.log_dir, "state_point_indices.pt"))

        self.print_log = print_log
        if self.print_log:
            self.episode_rewards = AverageScalarMeter(200)
            self.episode_lengths = AverageScalarMeter(200)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        if save_video:
            self.video_log_dir = os.path.join(self.log_dir, "video")
            os.makedirs(self.video_log_dir, exist_ok=True)
            self.vis_env_num = self.args.vis_env_num

        self.apply_reset = apply_reset
        """Evaluation."""
        self.eval_round = 2

        if self.vec_env.mode == "eval":
            self.eval_round = self.args.eval_times

        """ Demo """
        if self.args.collect_demo_num > 0:
            self.demo_dir = os.path.join(self.log_dir, "demo")
            os.makedirs(self.demo_dir, exist_ok=True)

            # self.total_demo_num = self.args.collect_demo_num * len(self.vec_env.object_codes)
            self.total_demo_num = self.args.collect_demo_num
            self.cur_demo_num = 0
            self.demo_obs = torch.tensor([], device="cpu")
            self.agent_pos = torch.tensor([], device="cpu")
            self.demo_action = torch.tensor([], device="cpu")
            self.demo_abs_action = torch.tensor([], device="cpu")
            self.demo_pointcloud = torch.tensor([], device="cpu")
            self.demo_init_state = {}
            self.demo_rigidbody_pose = torch.tensor([], device="cpu")
            # self.demo_init_state["obj_pos"] = self.vec_env.occupied_object_init_root_positions.cpu().numpy()
            # self.demo_init_state["obj_orn"] = self.vec_env.occupied_object_init_root_orientations.cpu().numpy()
            # self.demo_init_state["robot_dof"] = self.vec_env.robot_init_dof.cpu().numpy()
            # self.demo_init_state["target_dof"] = self.vec_env.object_targets.cpu().numpy()

    def restore_test(self, path):
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint["model"])
        self.set_test()

    def restore_train(self, path):
        if not path:
            return
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint["model"])
        if self.args.con:
            self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.set_train()

    def set_test(self, vis=False):
        self.actor_critic.eval()
        self.vec_env.eval(vis=vis)
        if self.normalize_value:
            self.value_running_mean_std.eval()  

    def set_train(self):
        self.actor_critic.train()
        self.vec_env.train()
        if self.normalize_value:
            self.value_running_mean_std.train()

    def save(self, path):
        weights = {
            "model": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.current_learning_iteration,
            "tot_timesteps": self.tot_timesteps,
            "value_rms": self.value_running_mean_std.state_dict(),
        }
        torch.save(weights, path)

        # save auto-encoder and its normalizer
        # if hasattr(self.vec_env.reach_curiosity_mgr.state_bank, "ae"):
        #     ae_path = path.replace("model", "ae_model")
        #     torch.save(self.vec_env.reach_curiosity_mgr.state_bank.ae.state_dict(), ae_path)
        #     rms_path = path.replace("model", "ae_rms")
        #     torch.save(self.vec_env.reach_curiosity_mgr.state_bank.normalizer.state_dict(), rms_path)

    def get_action(self, current_obs, mode):
        # Compute the action
        actions, grad, _ = self.compute_action(current_obs=current_obs, mode=mode)
        step_actions = self.process_actions(actions=actions.clone(), grad=grad.clone())
        return step_actions

    def eval(self, it):
        # eval initilization
        self.set_test(vis=save_video)
        test_times = 0
        success_rates = []  # s_rate for each round
        reward_all = []
        eps_len_all = []
        succ_eps_len_all = []
        # demo_per_obj = {}
        demo_per_grasp = {}

        if self.vec_env.mode == "train":
            save_time = 0  # means save all videos
        else:
            save_time = self.eval_round - 1

        if self.args.collect_demo_num > 0:
            breakout_threshold = 200
            counter = 0
            while self.cur_demo_num < self.total_demo_num:
                counter += 1
                if counter > breakout_threshold:
                    print("Breakout threshold reached")
                    print(f"# Demos: {self.cur_demo_num} / {self.total_demo_num}")
                    break
                print(f"Current Round {counter}")
                print(f"# Demos: {self.cur_demo_num} / {self.total_demo_num}")
                # reset env
                # self.vec_env.reset_arm() # redundant?
                # TODO since reset not step simulation, the current obs is actually not correct
                current_obs = self.vec_env.reset()["obs"]
                try:
                    current_rigidbody_pose = self.vec_env.rigid_body_states.clone().cpu().unsqueeze(0)
                    current_rigidbody_pose = current_rigidbody_pose.view(1, self.vec_env.num_envs, self.vec_env.num_rigid_bodies, 13)
                except:
                    print("No rigid body states info.")
                    pass

                eval_done_envs = torch.zeros(self.vec_env.num_envs, dtype=torch.long, device=self.device)

                self.demo_init_state = {}
                self.demo_obs = torch.tensor([], device="cpu")
                self.agent_pos = torch.tensor([], device="cpu")
                self.demo_action = torch.tensor([], device="cpu")
                self.demo_abs_action = torch.tensor([], device="cpu")
                self.demo_pointcloud = torch.tensor([], device="cpu")
                self.demo_rigidbody_pose = torch.tensor([], device="cpu")

                # step
                with torch.no_grad():
                    while True:
                        # Compute the action
                        actions, grad, _ = self.compute_action(current_obs=current_obs, mode="eval")
                        step_actions = self.process_actions(actions=actions.clone(), grad=grad.clone())

                        # Step the vec_environment
                        done_env_ids = (eval_done_envs > 0).nonzero(as_tuple=False).squeeze(-1)
                        step_actions[done_env_ids, :] = 0
                        # log.info(f"eval_done_envs: {eval_done_envs[0]}")

                        full_obs = current_obs.reshape(1, self.vec_env.num_envs, -1)
                        try:
                            current_rigidbody_pose = self.vec_env.rigid_body_states.clone().cpu().unsqueeze(0)
                            current_rigidbody_pose = current_rigidbody_pose.view(1, self.vec_env.num_envs, self.vec_env.num_rigid_bodies, 13)
                        except:
                            print("No rigid body states info.")
                            pass

                        if hasattr(self.vec_env, "img_pcl_obs") and self.vec_env.img_pcl_obs:
                            current_img_pcl_obs = self.vec_env.imagined_pointclouds.reshape(self.vec_env.num_envs, -1)
                            # get full obs
                            full_obs = torch.cat([
                                    full_obs,
                                    current_img_pcl_obs.reshape(1, self.vec_env.num_envs, -1),
                            ], -1)
                        
                        if self.vec_env.enable_rendered_pointcloud_observation:
                            if self.vec_env.enable_rendered_pointcloud_target_mask:
                                pc = self.vec_env.rendered_pointclouds_w_target_mask
                            else:
                                pc = self.vec_env.rendered_pointclouds
                            current_pc = pc.clone().cpu().unsqueeze(0)  # [1, num_envs, P, C]
                            self.demo_pointcloud = torch.cat([self.demo_pointcloud, current_pc], dim=0)
            
                        self.demo_obs = torch.cat([self.demo_obs, full_obs.cpu()])
                        try:
                            self.demo_rigidbody_pose = torch.cat([self.demo_rigidbody_pose, current_rigidbody_pose], dim=0)
                        except:
                            pass
                        
                        if self.vec_env._grasp_task:
                            self.agent_pos = torch.cat([
                                self.agent_pos, 
                                self.vec_env.allegro_hand_dof_positions.clone().cpu().unsqueeze(0)
                            ])
                        else:
                            dof_positions = self.vec_env.allegro_hand_dof_positions
                            goal_object_rotations = self.vec_env.goal_rot
                            proprioceptive_pos = torch.cat([dof_positions, goal_object_rotations], dim=-1)
                            self.agent_pos = torch.cat([
                                self.agent_pos, 
                                proprioceptive_pos.clone().cpu().unsqueeze(0)
                            ])


                        clamped_actions = torch.clamp(step_actions, -1.0, 1.0)
                        next_obs, rews, dones, infos = self.vec_env.step(clamped_actions)

                        if self.vec_env._grasp_task:
                            abs_action = self.vec_env.curr_targets.clone()   # (num_envs, num_dofs)
                        else:
                            abs_action = self.vec_env.cur_targets.clone()   # (num_envs, num_dofs)

                        self.demo_action = torch.cat(
                            [self.demo_action, clamped_actions.unsqueeze(0).cpu()],  # (1, num_envs, num_dofs)
                            dim=0
                        )
                        self.demo_abs_action = torch.cat(
                            [self.demo_abs_action, abs_action.unsqueeze(0).cpu()],  # (1, num_envs, num_dofs)
                            dim=0
                        )
                        
                        current_obs.copy_(next_obs["obs"])

                        # done
                        new_done_env_ids = (dones & (1 - eval_done_envs)).nonzero(as_tuple=False).squeeze(-1)
                        if len(new_done_env_ids) > 0:
                            if self.vec_env._grasp_task:
                                succ_fins = (self.vec_env.successes > 0) & dones
                                success_env_ids = (
                                    (succ_fins & (1 - eval_done_envs)).nonzero(as_tuple=False).squeeze(-1)
                                )
                            else:
                                keep_fins = dones
                                success_env_ids = (
                                    (keep_fins & (1 - eval_done_envs)).nonzero(as_tuple=False).squeeze(-1)
                                )
                            
                            test_times += len(new_done_env_ids)
                            eval_done_envs[new_done_env_ids] = 1
                            for success_env_id in success_env_ids:
                                if success_env_id in new_done_env_ids and self.cur_demo_num < self.total_demo_num:
                                    cur_demo = {}
                                    # cur_demo["env_mode"] = self.vec_env.env_mode
                                    cur_demo["obs_space"] = self.vec_env.cfg["env"]["observationSpace"]
                                    cur_demo["action_space"] = self.vec_env.cfg["env"]["actionSpace"]
                                    cur_demo["obs"] = self.demo_obs[:, success_env_id, :].cpu().numpy()
                                    try:
                                        cur_demo["rigidbody_pose"] = self.demo_rigidbody_pose[:, success_env_id, :, :].cpu().numpy()
                                    except:
                                        pass
                                    cur_demo["agent_pos"] = self.agent_pos[:, success_env_id, :].detach().cpu().numpy()
                                    cur_demo["action"] = self.demo_action[:, success_env_id, :].detach().cpu().numpy()
                                    cur_demo["action_abs"] = self.demo_abs_action[:, success_env_id, :].detach().cpu().numpy()
                                    if self.vec_env.enable_rendered_pointcloud_observation:
                                        cur_demo["point_cloud"] = self.demo_pointcloud[:, success_env_id, :, :].numpy()


                                    if self.cur_demo_num <= self.args.collect_demo_num:
                                        demo_id = int(self.cur_demo_num + 1)
                                        demo_fname = f"demo_{demo_id:06d}.npy"

                                        cur_demo["demo_id"] = demo_id
                                        cur_demo["demo_filename"] = demo_fname

                                        print("save data to", os.path.join(self.demo_dir, demo_fname))
                                        np.save(
                                            os.path.join(self.demo_dir, demo_fname),
                                            cur_demo,
                                        )
                                        self.cur_demo_num += 1

                        if eval_done_envs.sum() == self.vec_env.num_envs or self.cur_demo_num == self.total_demo_num:
                            break
        else:
            # start evaluation
            with tqdm(total=self.eval_round) as pbar:
                pbar.set_description("Validating:")
                with torch.no_grad():
                    for r in range(self.eval_round):
                        if save_video and r <= save_time:
                            all_images = torch.tensor([], device=self.device)
                        # reset env
                        current_obs = self.vec_env.reset()["obs"]
                        eval_done_envs = torch.zeros(self.vec_env.num_envs, dtype=torch.long, device=self.device)
                        successes = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
                        eps_len = torch.zeros(self.vec_env.num_envs, dtype=torch.long, device=self.device)
                        pos_dist = torch.zeros(self.vec_env.num_envs, device=self.device)
                        rot_dist = torch.zeros(self.vec_env.num_envs, device=self.device)
                        contact_dist = torch.zeros(self.vec_env.num_envs, device=self.device)
                        succ_eps_len = []
                        if plot_direction:
                            arm_diff_direction = []
                            hand_diff_direction = []

                        #     contact_dist = torch.zeros(self.vec_env.num_envs, device=self.device)
                        tran_rew = 0
                        rot_rew = 0
                        contact_rew = 0
                        height_rew = 0

                        # step
                        while True:
                            # Compute the action
                            actions, grad, _ = self.compute_action(current_obs=current_obs, mode="eval")
                            step_actions = self.process_actions(actions=actions.clone(), grad=grad.clone())


                            # Step the vec_environment
                            # print(step_actions)
                            # set done env action to be zero
                            done_env_ids = (eval_done_envs > 0).nonzero(as_tuple=False).squeeze(-1)
                            # breakpoint()
                            # print("shape of done_env_ids:", done_env_ids.shape)
                            # print("num done envs:", len(done_env_ids))
                            step_actions[done_env_ids, :] = 0

                            clamped_actions = torch.clamp(step_actions, -1.0, 1.0)
                            next_obs, rews, dones, infos = self.vec_env.step(clamped_actions)
                            
                            # print("num of dones:", dones.sum())
                            # from loguru import logger as log
                            # log.info(f"observation: {current_obs[0]}")
                            # log.info(f"step_actions: {step_actions[0]}")
                            # log.info(f"success: {self.vec_env.successes[0]}")

                            if plot_direction:
                                arm_diff_direction.append(
                                    torch.mean(infos["arm_pos_diff_direction"] + infos["arm_rot_diff_direction"]).item()
                                    / 6
                                )
                                hand_diff_direction.append(torch.mean(infos["hand_diff_direction"]).item() / 20)

                            if save_video and r <= save_time:
                                image = self.vec_env.render(
                                    rgb=True, img_size=img_size, vis_env_num=self.vis_env_num
                                ).reshape(self.vis_env_num, 1, img_size, img_size, 3)
                                all_images = torch.cat([all_images, image], 1)
                            current_obs.copy_(next_obs["obs"])

                            # done
                            # print("shape of dones:", dones.shape)
                            # print("shape of eval_done_envs:", eval_done_envs.shape)
                            new_done_env_ids = (dones & (1 - eval_done_envs)).nonzero(as_tuple=False).squeeze(-1)
                            if len(new_done_env_ids) > 0:
                                # if 0 in new_done_env_ids:
                                #     print("--")
                                if r > save_time and save_video:
                                    self.vec_env.graphics_device_id = -1
                                    self.vec_env.enable_camera_sensors = False

                                if save_video and r <= save_time:
                                    for i, images in enumerate(all_images):
                                        obj_type = self.vec_env.object_type_per_env[i]
                                        save_path = os.path.join(
                                            self.video_log_dir, f"{obj_type}_epoach:{it}_round:{r}"
                                        )
                                        images_to_video(
                                            path=save_path, images=images.cpu().numpy(), size=(img_size, img_size)
                                        )

                                test_times += len(new_done_env_ids)
                                reward_all.extend(rews[new_done_env_ids].cpu().numpy())
                                # pos_dist[new_done_env_ids] = infos["pos_dist"][new_done_env_ids]
                                # rot_dist[new_done_env_ids] = infos["rot_dist"][new_done_env_ids]
                                # contact_dist[new_done_env_ids] = infos["fj_dist"][new_done_env_ids]

                                success_env_ids = (
                                    ((self.vec_env.successes > 0) & dones & (1 - eval_done_envs)).nonzero(as_tuple=False).squeeze(-1)
                                )
                                eval_done_envs[new_done_env_ids] = 1
                                successes[success_env_ids] = 1
                                eps_len[new_done_env_ids] = self.vec_env.progress_buf[new_done_env_ids]
                                succ_eps_len.extend(self.vec_env.progress_buf[success_env_ids].cpu().numpy().tolist())

                            if test_times == (r + 1) * self.vec_env.num_envs:
                                # self.vec_env.lift_test(eval_done_envs.nonzero(as_tuple=False).squeeze(-1))
                                # for id in self.vec_env.successes.nonzero(as_tuple=False):
                                #     print(fj_dist[id], contact_dist[id], pos_dist[id], rot_dist[id])
                                assert torch.sum(eval_done_envs).item() == self.vec_env.num_envs
                                success_rates.append(torch.sum(successes).unsqueeze(-1) / self.vec_env.num_envs)
                                eps_len_all.append(eps_len.float().mean().item())
                                succ_eps_len_all.append(np.mean(succ_eps_len))

                                if plot_direction:
                                    plt.plot(arm_diff_direction, label="arm_diff_direction")
                                    plt.plot(hand_diff_direction, label="hand_diff_direction")

                                break
                        pbar.update(1)

            assert test_times == self.eval_round * self.vec_env.num_envs
            success_rates = torch.cat(success_rates)
            sr_mu, sr_std = success_rates.mean().cpu().numpy().item(), success_rates.std().cpu().numpy().item()
            print(f"|| num_envs: {self.vec_env.num_envs} || eval_times: {self.eval_round}")
            print(f"eval_success_rate % : {sr_mu*100:.2f} +- {sr_std*100:.2f}")
            eval_rews = np.mean(reward_all)
            print(f"eval_rewards: {eval_rews}")
            print(f"eval_eps_len: {np.mean(eps_len_all)}")
            print(f"eval_succ_eps_len: {np.mean(succ_eps_len_all)}")
            if self.writer is not None:
                self.writer.add_scalar("Eval/success_rate", sr_mu, it)
                self.writer.add_scalar("Eval/eval_rews", eval_rews, it)
            
            # if self.vec_env.enable_exploration_logging:
            #     self.vec_env.save_exploration_density(os.path.join(self.log_dir, f"exploration_data_{it}.pt"))

    def train(self, num_learning_iterations, log_interval=1):
        # rewbuffer = deque(maxlen=100)
        # lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros((self.vec_env.num_envs, 1), dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

        # reward_sum = []
        # episode_length = []

        # reset env
        current_obs = self.vec_env.reset()["obs"]
        current_states = self.vec_env.get_state()
        if self.use_curiosity_model:
            curiosity_states = self.vec_env.extras["curiosity_states"].reshape(self.vec_env.num_envs, -1)
        for it in range(self.current_learning_iteration, num_learning_iterations):
            start = time.time()
            ep_infos = []
            intrinsic_rewards_list = []
            scaled_rewards_list = []
            # Rollout
            for _ in range(self.num_transitions_per_env):
                if self.apply_reset:
                    current_obs = self.vec_env.reset()["obs"]
                    current_states = self.vec_env.get_state()
                    if self.use_curiosity_model:
                        curiosity_states = self.vec_env.extras["curiosity_states"].reshape(self.vec_env.num_envs, -1)

                # Compute the action
                actions, actions_log_prob, values, mu, sigma, grad, storage_obs = self.compute_action(
                    current_obs=current_obs, current_states=current_states
                )
                step_actions = self.process_actions(actions=actions.clone(), grad=grad.clone())

                # Step the vec_environment
                clamped_actions = torch.clamp(step_actions, -1.0, 1.0)
                next_obs, rews, dones, infos = self.vec_env.step(clamped_actions)
                next_states = self.vec_env.get_state()
                if self.use_curiosity_model:
                    next_curiosity_states = infos["curiosity_states"].reshape(self.vec_env.num_envs, -1)
                
                self.actor_critic.update_normalization(next_obs["obs"])
                
                origin_rewards = rews.clone()
                # rewards = rews.unsqueeze(-1)
                if self.critic_mode == "dual":
                    cur = infos["curiosity_reward"].to(self.device).view(-1)
                    rew_e = origin_rewards - cur
                    rew_i = cur
                    rewards = torch.stack([rew_e, rew_i], dim=-1)  # (N, 2)
                    # print(f"rew_e: {rew_e.mean()}, rew_i: {rew_i.mean()}")
                else:
                    rewards = origin_rewards.unsqueeze(-1)
                    
                scaled_rewards = self.reward_scale_value * rewards
                if self.use_curiosity_model:
                    self.curiosity_model.update_normalization(current_obs, actions, next_curiosity_states)
                    intrinsic_rewards = self.intrinsic_reward_scale * self.curiosity_model.compute_intrinsic_reward(current_obs, actions, next_curiosity_states).detach().unsqueeze(-1)
                    scaled_rewards_list.append(scaled_rewards.mean().item())
                    intrinsic_rewards_list.append(intrinsic_rewards.mean().item())
                    scaled_rewards = scaled_rewards + intrinsic_rewards
                    
                # add for value bootstrap if the episode ends because of timeout
                if self.value_bootstrap and 'time_outs' in infos:
                    # assert self.critic_mode != "dual", "do not support dual critic mode with value bootstrap"
                    # dual mode seems work now # !!! Now do not support dual critic mode
                    scaled_rewards += self.gamma * values * infos['time_outs'].float().unsqueeze(-1)

                # Record the transition
                if self.critic_mode == "dual" and not self.use_curiosity_model:
                    self.storage.add_transitions(
                        storage_obs, current_states, actions, scaled_rewards, dones, values, actions_log_prob, mu, sigma
                    )
                elif self.critic_mode == "single" and self.use_curiosity_model:
                    self.storage.add_transitions(
                        storage_obs, current_states, actions, scaled_rewards, dones, values, actions_log_prob, mu, sigma, next_curiosity_states
                    )
                elif self.critic_mode == "single" and not self.use_curiosity_model:
                    self.storage.add_transitions(
                        storage_obs, current_states, actions, scaled_rewards, dones, values, actions_log_prob, mu, sigma
                    )
                else:
                    raise NotImplementedError
                    
                current_obs.copy_(next_obs["obs"])
                current_states.copy_(next_states)
                if self.use_curiosity_model:
                    curiosity_states.copy_(next_curiosity_states)

                # Book keeping
                ep_infos.append(infos.copy())

                if self.print_log:
                    cur_reward_sum[:] += scaled_rewards if not self.critic_mode == "dual" else scaled_rewards[:, :1]
                    cur_episode_length[:] += 1
                    done_indices = (dones > 0).nonzero(as_tuple=False)
                    not_dones = 1.0 - dones.float()

                    # reward_sum.extend(cur_reward_sum[done_indices][:, 0].cpu().numpy().tolist())
                    # episode_length.extend(cur_episode_length[done_indices][:, 0].cpu().numpy().tolist())
                    self.episode_rewards.update(cur_reward_sum[done_indices])
                    self.episode_lengths.update(cur_episode_length[done_indices])
                    cur_reward_sum = cur_reward_sum * not_dones.unsqueeze(-1)
                    cur_episode_length = cur_episode_length * not_dones

                # done
                if torch.sum(dones) > 0:
                    current_obs = self.vec_env.reset(dones)["obs"]
                    current_states = self.vec_env.get_state()

            # if self.print_log:
            #     rewbuffer.extend(reward_sum)
            #     lenbuffer.extend(episode_length)

            _, _, last_values, _, _, _, _ = self.compute_action(
                current_obs=current_obs, current_states=current_states, mode="train"
            )
            stop = time.time()
            collection_time = stop - start
            mean_trajectory_length, mean_reward = self.storage.get_statistics()

            # Learning step
            start = stop
            # self.storage.compute_returns(last_values, self.gamma, self.lam) # original
            if self.critic_mode == "dual":
                gammas = torch.tensor([self.gamma, self.int_gamma], device=self.device, dtype=torch.float32)
                lams = torch.tensor([self.lam, self.int_lam], device=self.device, dtype=torch.float32)
                episodic_flags = torch.tensor([1, 0 if self.int_non_episodic_flags else 1], device=self.device, dtype=torch.float32)
                self.storage.compute_returns_multi(last_values, gammas, lams, episodic_flags)
            else:
                self.storage.compute_returns(last_values, self.gamma, self.lam)
            
            values = self.storage.values
            returns = self.storage.returns
            # print("before norm:")
            # print("returns:", returns.mean().item())
            # print("values:", values.mean().item())
            if self.normalize_value:
                values = self.value_running_mean_std(values)
                returns = self.value_running_mean_std(returns)
            self.storage.values = values
            self.storage.returns = returns
            
            # print("storage items:")
            # print("rewards:", self.storage.rewards.mean().item())
            # print("returns:", self.storage.returns.mean().item())
            # print("values:", self.storage.values.mean().item())
            # print("actions:", self.storage.actions.mean().item())
            # print("obses:", self.storage.observations.mean().item())

            mean_value_loss, mean_surrogate_loss, mean_kl_loss, mean_b_loss, mean_entropy, mean_curiosity_loss = self.update()
            self.storage.clear()
            stop = time.time()
            learn_time = stop - start
            if self.print_log:
                self.log(locals())
            if (it + 1) % log_interval == 0:
                self.set_test()
                self.eval(it + 1)
                self.set_train()
                if self.log_dir is not None:
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(it + 1)))

                current_obs = self.vec_env.reset()["obs"]
                current_states = self.vec_env.get_state()
                if self.use_curiosity_model:
                        curiosity_states = self.vec_env.extras["curiosity_states"].reshape(self.vec_env.num_envs, -1)
                cur_episode_length[:] = 0
                # TODO clean extras
            ep_infos.clear()
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(num_learning_iterations)))

    def log(self, locs, width=70, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if key == "success_num":
                    value = torch.sum(infotensor)
                    if self.writer is not None:
                        self.writer.add_scalar("Episode/" + "total_success_num", value, locs["it"])
                    ep_string += f"""{f'Total episode {key}:':>{pad}} {value:.4f}\n"""
                value = torch.mean(infotensor)
                if self.writer is not None:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()
        
        if self.writer is not None:
            self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
            self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
            self.writer.add_scalar("Loss/kl", locs["mean_kl_loss"], locs["it"])
            self.writer.add_scalar("Loss/curiosity_loss", locs["mean_curiosity_loss"], locs["it"])
            self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

            self.writer.add_scalar("Train/mean_reward", self.episode_rewards.get_mean(), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", self.episode_lengths.get_mean(), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", self.episode_rewards.get_mean(), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time",self.episode_lengths.get_mean(),self.tot_time,)
            self.writer.add_scalar("Train/mean_scaled_rewards", np.mean(locs["scaled_rewards_list"]), locs["it"])
            self.writer.add_scalar("Train/mean_intrinsic_rewards", np.mean(locs["intrinsic_rewards_list"]), locs["it"])
        
            self.writer.add_scalar("Train2/mean_reward/step", locs["mean_reward"], locs["it"])
            self.writer.add_scalar("Train2/mean_episode_length/episode", locs["mean_trajectory_length"], locs["it"])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if self.print_log:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'b_loss:':>{pad}} {locs['mean_b_loss']:.4f}\n"""
                f"""{'Entropy:':>{pad}} {locs['mean_entropy']:.4f}\n"""
                f"""{'Curiosity loss:':>{pad}} {locs['mean_curiosity_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {self.episode_rewards.get_mean():.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {self.episode_lengths.get_mean():.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
            f"""{'Logdir:':>{pad}} {self.log_dir if (self.log_dir is not None and self.log_dir != '') else 'None'}\n"""
        )
        print(log_string)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl_loss = 0
        mean_b_loss = 0
        mean_entropy = 0
        mean_curiosity_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)


        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:

                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                # target_values_batch = self.storage.values.view(-1, 1)[indices]
                # returns_batch = self.storage.returns.view(-1, 1)[indices]
                if self.critic_mode == "dual":
                    target_values_batch = self.storage.values.view(-1, 2)[indices]
                    returns_batch = self.storage.returns.view(-1, 2)[indices]
                else:
                    target_values_batch = self.storage.values.view(-1, 1)[indices]
                    returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                # advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                if self.critic_mode == "dual":
                    advantages_batch = self.storage.combined_advantages.view(-1, 1)[indices]
                else:
                    advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                curiosity_states_batch = self.storage.curiosity_states.view(-1, self.storage.curiosity_states.size(-1))[indices] if self.use_curiosity_model else None

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch, states_batch, actions_batch)


                # KL
                if self.desired_kl != None and self.schedule == "adaptive":
                    kl = torch.sum(
                        sigma_batch
                        - old_sigma_batch
                        + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch.exp()))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()


                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                    
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu_batch - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu_batch + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1).mean()
                else:
                    b_loss = torch.tensor(0.0, device=self.device)

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.bounds_loss_coef * b_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Gradient step for curiosity module
                if self.use_curiosity_model:
                    # normalized_obs_batch = self.actor_critic.actor_obs_normalizer(obs_batch)
                    # breakpoint()
                    curiosity_loss = self.curiosity_model.compute_loss(
                        obs_batch, actions_batch, curiosity_states_batch
                    )
                    self.curiosity_optimizer.zero_grad()
                    curiosity_loss.backward()
                    nn.utils.clip_grad_norm_(self.curiosity_model.model.parameters(), self.max_grad_norm)
                    self.curiosity_optimizer.step()
                    mean_curiosity_loss += curiosity_loss.item()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_kl_loss += kl_mean.item()
                mean_b_loss += b_loss.item()
                mean_entropy += entropy_batch.mean().item()
                mean_curiosity_loss += curiosity_loss.item() if self.use_curiosity_model else 0.0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl_loss /= num_updates
        mean_b_loss /= num_updates
        mean_entropy /= num_updates
        mean_curiosity_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, mean_kl_loss, mean_b_loss, mean_entropy, mean_curiosity_loss

    """
    utils
    """

    def process_actions(self, actions, grad=None):
        if self.action_type == "direct":
            step_actions = actions
        elif self.action_type == "joint":
            if self.sub_action_type == "add+jointscale":
                self.vec_env.extras["grad_ss_mean"] = torch.mean(abs(actions[:, : self.vec_env.num_actions]), -1)
                self.vec_env.extras["grad_ss_std"] = torch.std(abs(actions[:, : self.vec_env.num_actions]), -1)
                self.vec_env.extras["residual_mean"] = torch.mean(abs(actions[:, self.vec_env.num_actions :]), -1)
                self.vec_env.extras["residual_std"] = torch.std(abs(actions[:, self.vec_env.num_actions :]), -1)
                step_actions = grad * actions[:, : self.vec_env.num_actions] + actions[:, self.vec_env.num_actions :]
            elif self.sub_action_type == "addscale+add":
                step_actions = (
                    grad * (actions[:, :1] + actions[:, 1 : 1 + self.vec_env.num_actions])
                    + actions[:, 1 + self.vec_env.num_actions :]
                )
        return step_actions

    def compute_action(self, current_obs, current_states=None, mode="train"):
        
        grad = torch.tensor([], device=self.device)

        # pointnet fine-tuning
        if self.actor_critic.pcl_dim > 0 and self.pointnet_finetune:
            batch_num = current_obs.size(0) // self.finetune_pointnet_bz + 1
            for batch_idx in range(batch_num):
                obs_batch = current_obs[self.finetune_pointnet_bz * batch_idx : self.finetune_pointnet_bz * (batch_idx + 1)]

                if mode == "train":
                    actions_batch, actions_log_prob_batch, values_batch, mu_batch, sigma_batch = self.actor_critic.act(
                        obs_batch, current_states
                    )
                    if self.normalize_value:
                        values_batch = self.value_running_mean_std(values_batch, True)
                else:
                    actions_batch = self.actor_critic.act_inference(obs_batch)

                # merge batch results
                if batch_idx == 0:
                    if mode == "train":
                        actions, actions_log_prob, values, mu, sigma = (
                            actions_batch,
                            actions_log_prob_batch,
                            values_batch,
                            mu_batch,
                            sigma_batch,
                        )
                    else:
                        actions = actions_batch
                else:
                    if mode == "train":
                        actions = torch.cat([actions, actions_batch])
                        actions_log_prob = torch.cat([actions_log_prob, actions_log_prob_batch])
                        values = torch.cat([values, values_batch])
                        mu = torch.cat([mu, mu_batch])
                        sigma = torch.cat([sigma, sigma_batch])
                    else:
                        actions = torch.cat([actions, actions_batch])
        else:
            if mode == "train":
                actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
                if self.normalize_value:
                    values = self.value_running_mean_std(values, True)
            else:
                actions = self.actor_critic.act_inference(current_obs)

        if mode == "train":
            # return original obs for storage
            return actions, actions_log_prob, values, mu, sigma, grad, current_obs
        else:
            return actions, grad, current_obs
