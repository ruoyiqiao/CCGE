import torch
import torch.nn as nn
from ..common.mlp import MLPBackbone
from ..common.normalization import EmpiricalNormalization

from typing import Tuple

class _HashAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, code_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.code = nn.Linear(hidden_dim, code_dim)  # sigmoid applied in forward
        self.dec = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor, noise_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,F)
        h = self.enc(x)
        b = torch.sigmoid(self.code(h))  # (B,D) in (0,1)
        if noise_scale > 0:
            # Uniform noise U(-a,a)
            n = (torch.rand_like(b) * 2.0 - 1.0) * float(noise_scale)
            b_noisy = (b + n).clamp(0.0, 1.0)
        else:
            b_noisy = b
        x_hat = self.dec(b_noisy)
        return x_hat, b

@torch.no_grad()
def _bits_to_u64(bits01: torch.Tensor) -> torch.Tensor:
    bits = bits01.to(torch.long)
    K = bits.shape[1]
    shifts = torch.arange(K, device=bits.device, dtype=torch.long)
    return (bits << shifts).sum(dim=1)


class CuriosityModel:
    def __init__(self, model_type, obs_dim, action_dim, curi_obs_dim, emb_dim, hidden_sizes, activation='relu', ensemble_size=5, simhash_dim=5, code_dim=16, hash_hidden_dim=512, device='cpu', obs_act_normalization=False, curi_obs_normalization=False):
        self.model_type = model_type
        if model_type == 'prediction_error':
            self.model = MLPBackbone(in_channels=obs_dim + action_dim,
                                     out_channels=curi_obs_dim,
                                     hidden_channels=hidden_sizes,
                                     activation_name=activation).to(device)
        elif model_type == 'rnd':
            self.model = MLPBackbone(in_channels=curi_obs_dim,
                                     out_channels=emb_dim,
                                     hidden_channels=hidden_sizes,
                                     activation_name=activation).to(device)
            self.target_model = MLPBackbone(in_channels=curi_obs_dim,
                                            out_channels=emb_dim,
                                            hidden_channels=hidden_sizes,
                                            activation_name=activation).to(device)
            # Freeze target model parameters
            for param in self.target_model.parameters():
                param.requires_grad = False
        elif model_type == 'disagreement':
            self.model = nn.ModuleList([
                MLPBackbone(
                    in_channels=obs_dim + action_dim,
                    out_channels=curi_obs_dim,
                    hidden_channels=hidden_sizes,
                    activation_name=activation
                )
                for _ in range(ensemble_size)
            ]).to(device)
        elif model_type == 'neural_hash':
            self.model = _HashAE(
                in_dim=obs_dim,
                hidden_dim=hash_hidden_dim,
                code_dim=code_dim
            ).to(device)

            g = torch.Generator(device=device)
            g.manual_seed(int(42))
            self.A = torch.randn((simhash_dim, code_dim), generator=g, device=device, dtype=torch.float32)
            self.bin_cnt = torch.zeros(
                2**simhash_dim, 
                dtype=torch.long, 
                device=device, 
                requires_grad=False
            )
            self.S = 2**simhash_dim
        else:
            raise NotImplementedError(f"Curiosity model type {model_type} not implemented.")
        
        self.obs_act_normalization = obs_act_normalization
        if obs_act_normalization:
            self.obs_act_normalizer = EmpiricalNormalization(obs_dim + action_dim).to(device)
        else:
            self.obs_act_normalizer = torch.nn.Identity().to(device)

        self.curi_obs_normalization = curi_obs_normalization
        if curi_obs_normalization:
            self.curi_obs_normalizer = EmpiricalNormalization(curi_obs_dim).to(device)
        else:
            self.curi_obs_normalizer = torch.nn.Identity().to(device)

    def forward(self, obs, action, curi_obs):
        if self.model_type == 'prediction_error':
            input_tensor = torch.cat([obs, action], dim=-1)

            if self.obs_act_normalization:
                input_tensor = self.obs_act_normalizer(input_tensor)
            
            preds = self.model(input_tensor)
            return preds
        elif self.model_type == 'rnd':
            if self.curi_obs_normalization:
                curi_obs = self.curi_obs_normalizer(curi_obs)

            preds = self.model(curi_obs)
            return preds
        elif self.model_type == 'disagreement':
            input_tensor = torch.cat([obs, action], dim=-1)

            if self.obs_act_normalization:
                input_tensor = self.obs_act_normalizer(input_tensor)

            # K 个模型分别预测
            preds = torch.stack(
                [model(input_tensor) for model in self.model],
                dim=0
            )
            # preds: (K, B, D)
            return preds
        elif self.model_type == 'neural_hash':

            if self.curi_obs_normalization:
                curi_obs = self.curi_obs_normalizer(curi_obs)

            preds, _ = self.model(curi_obs, noise_scale=0.0)
            return preds
        else:
            raise NotImplementedError(f"Curiosity model type {self.model_type} not implemented.")
    
    def compute_loss(self, obs, action, curi_obs):
        if self.model_type == 'prediction_error':
            # breakpoint()
            preds = self.forward(obs, action, None)

            if self.curi_obs_normalization:
                curi_obs = self.curi_obs_normalizer(curi_obs)

            targets = curi_obs
            loss = torch.mean((preds - targets) ** 2)
            return loss
        elif self.model_type == 'rnd':
            preds = self.forward(None, None, curi_obs)

            if self.curi_obs_normalization:
                curi_obs = self.curi_obs_normalizer(curi_obs)

            with torch.no_grad():
                target_preds = self.target_model(curi_obs)
            loss = torch.mean((preds - target_preds) ** 2)
            return loss
        elif self.model_type == 'disagreement':
            preds = self.forward(obs, action, None)                  # (K, B, D)

            if self.curi_obs_normalization:
                curi_obs = self.curi_obs_normalizer(curi_obs)

            # target broadcast to (K, B, D)
            targets = curi_obs.unsqueeze(0).expand_as(preds)    # (K, B, D)

            loss = ((preds - targets) ** 2).mean()
            return loss
        elif self.model_type == 'neural_hash':

            if self.curi_obs_normalization:
                curi_obs = self.curi_obs_normalizer(curi_obs)

            preds, b = self.model(curi_obs, noise_scale=0.3)
            targets = curi_obs
            recon = torch.mean((preds - targets) ** 2)
            reg = torch.minimum((1.0 - b) ** 2, b ** 2).mean()
            loss = recon + 1.0 * reg
            # print(f"recon: {recon}, reg: {reg}")
            # loss = recon
            return loss
        else:
            raise NotImplementedError(f"Curiosity model type {self.model_type} not implemented.")
    
    def compute_intrinsic_reward(self, obs, action, curi_obs):
        if self.model_type == 'prediction_error':
            preds = self.forward(obs, action, None)

            if self.curi_obs_normalization:
                curi_obs = self.curi_obs_normalizer(curi_obs)

            targets = curi_obs
            intrinsic_reward = torch.mean((preds - targets) ** 2, dim=-1)
            return intrinsic_reward
        elif self.model_type == 'rnd':
            preds = self.forward(None, None, curi_obs)

            if self.curi_obs_normalization:
                curi_obs = self.curi_obs_normalizer(curi_obs)

            with torch.no_grad():
                target_preds = self.target_model(curi_obs)
            intrinsic_reward = torch.mean((preds - target_preds) ** 2, dim=-1)
            return intrinsic_reward
        elif self.model_type == 'disagreement':
            preds = self.forward(obs, action, None)  # (K, B, D)

            # 计算每个样本的方差作为内在奖励
            variance = torch.var(preds, dim=0)  # (B, D)
            intrinsic_reward = torch.mean(variance, dim=-1)  # (B,)
            return intrinsic_reward
        elif self.model_type == 'neural_hash':
            if self.curi_obs_normalization:
                curi_obs = self.curi_obs_normalizer(curi_obs)

            state_ids = self._state_id_from_feats(curi_obs)

            counts = torch.bincount(state_ids, minlength=self.bin_cnt.numel())
            self.bin_cnt += counts
            curiosity_reward = 1.0 / torch.sqrt(1.0 + self.bin_cnt[state_ids])

            # def state_id_entropy(state_ids: torch.Tensor, S: int):
            #     # state_ids: (N,)\
            #     import math

            #     counts = torch.bincount(state_ids, minlength=S).float()
            #     probs = counts / counts.sum().clamp_min(1.0)

            #     entropy = -(probs * (probs + 1e-12).log()).sum()
            #     entropy_norm = entropy / math.log(S)

            #     return entropy_norm.item()

            # from termcolor import cprint
            # H = state_id_entropy(state_ids, self.S)
            # cprint(f"state_ids: {state_ids}")
            # cprint(f"counts: {counts}")
            # cprint(f"curiosity_reward: {curiosity_reward}")
            # cprint(f"[StateID] normalized entropy = {H:.3f}", "yellow")
            return curiosity_reward.detach()
        else:
            raise NotImplementedError(f"Curiosity model type {self.model_type} not implemented.")
        
    def update_normalization(self, obs, action, curi_obs):
        if self.obs_act_normalization:
            input_tensor = torch.cat([obs, action], dim=-1)
            self.obs_act_normalizer.update(input_tensor)
        
        if self.curi_obs_normalization:
            self.curi_obs_normalizer.update(curi_obs)

    @torch.no_grad()
    def _state_id_from_feats(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats
        self.model.eval()
        _, b = self.model(x, noise_scale=0.0)
        bits = (b >= 0.5).to(torch.long)
        v = (bits * 2 - 1).to(torch.float32)
        proj = v @ self.A.t()
        sim_bits = (proj >= 0).to(torch.long)
        code_u64 = _bits_to_u64(sim_bits)
        state_ids = code_u64.to(torch.long)
        return state_ids