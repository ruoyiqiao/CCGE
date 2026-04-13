import isaacgym
import torch
from torch import nn
from typing import Optional, Tuple

from algorithms.ppo.utils import RunningMeanStd
from algorithms.common.normalization import EmpiricalNormalization
import time
import os

class PushBox2DStateBank:
    def __init__(
            self,
            *,
            num_key_states: int,   # must be 12/24/60
            num_hand_keypoints: int,
            num_object_bins: int,
            device: torch.device,
            dtype: torch.dtype = torch.float32,
    ):
        self.S = int(num_key_states)
        self.L = int(num_hand_keypoints)
        self.K = int(num_object_bins)
        self.device = device
        self.dtype = dtype

        self.counts: Optional[torch.Tensor] = torch.zeros((self.S, self.L, self.K), device=self.device, dtype=torch.long)
        self.update_step = 0

    @torch.no_grad()
    def push(self, feats: torch.Tensor) -> None:
        # no buffer/rebuild needed for subgroup discretization
        return
    
    def reset(self, *, reset_counters: bool = False) -> None:
        if reset_counters and (self.counts is not None):
            self.counts.zero_()

    @torch.no_grad()
    def add_contacts(
        self,
        *,
        state_ids: torch.Tensor,     # (N,)
        contact_mask: torch.Tensor,  # (N,L) bool
        contact_bins: torch.Tensor,  # (N,L) long in [0,K-1]
    ) -> None:
        if self.counts is None:
            return
        state_ids = state_ids.to(self.device, dtype=torch.long)
        contact_mask = contact_mask.to(self.device, dtype=torch.bool)
        contact_bins = contact_bins.to(self.device, dtype=torch.long).clamp(0, self.K - 1)

        valid = contact_mask  # no -1 state in this bank
        if not valid.any():
            return

        ei, li = torch.nonzero(valid, as_tuple=True)
        si = state_ids[ei]
        bi = contact_bins[ei, li]

        # safe accumulation (handles duplicate indices)
        self.counts.index_put_((si, li, bi), torch.ones_like(ti, dtype=self.counts.dtype), accumulate=True)




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
    """
    bits01: (N, K) uint8/bool/long, K<=64
    returns: (N,) uint64 stored in int64 tensor
    """
    bits = bits01.to(torch.long)
    K = bits.shape[1]
    shifts = torch.arange(K, device=bits.device, dtype=torch.long)
    return (bits << shifts).sum(dim=1)

def state_id_entropy(state_ids: torch.Tensor, S: int):
    # state_ids: (N,)\
    import math

    counts = torch.bincount(state_ids, minlength=S).float()
    probs = counts / counts.sum().clamp_min(1.0)

    entropy = -(probs * (probs + 1e-12).log()).sum()
    entropy_norm = entropy / math.log(S)

    return entropy_norm.item()

class LearnedHashStateBank:

    def __init__(
        self,
        *,
        num_key_states: int,      # S
        feature_dim: int,         # F
        buffer_size: int,         # B
        num_hand_keypoints: int,  # L
        num_object_bins: int,     # K
        device: torch.device,
        code_dim: int = 256,
        simhash_dim: int = 64,
        hidden_dim: int = 512,
        noise_scale: float = 0.3,
        lambda_binary: float = 10.0,
        ae_lr: float = 1e-3,
        ae_update_steps: int = 10,
        ae_update_freq: int = 16,
        ae_num_minibatches: int = 8,
        seed: int = 0,
    ):
        self.S = int(num_key_states)
        self.F = int(feature_dim)
        self.B = int(buffer_size)
        self.L = int(num_hand_keypoints)
        self.K = int(num_object_bins)
        self.device = device

        self.code_dim = int(code_dim)
        self.simhash_dim = int(simhash_dim)
        assert 1 <= self.simhash_dim <= 64, "simhash_dim must be in [1,64] for u64 packing"

        self.noise_scale = float(noise_scale)
        self.lambda_binary = float(lambda_binary)
        self.ae_update_steps = int(ae_update_steps)
        self.ae_update_freq = int(ae_update_freq)
        self.ae_num_minibatches = int(ae_num_minibatches)
        self.step_count = 0

        self.counts = torch.zeros((self.S, self.L, self.K), device=self.device, dtype=torch.float32)

        # AE + optimizer
        self.ae = _HashAE(self.F, int(hidden_dim), self.code_dim).to(self.device)
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=float(ae_lr))

        # SimHash projection: code_dim -> simhash_dim
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        self.A = torch.randn((self.simhash_dim, self.code_dim), generator=g, device=self.device, dtype=torch.float32)

        self.buffer = torch.zeros((self.B, self.F), device=self.device, dtype=torch.float32)
        self.buf_n = torch.zeros((), device=self.device, dtype=torch.long)
        self.rebuild_count = 0 # unused

        self.normalizer = EmpiricalNormalization(self.F).to(self.device)

        self.last_recon = torch.tensor(0.0, device=self.device)
        self.last_reg = torch.tensor(0.0, device=self.device)
        self.last_entropy = torch.tensor(0.0, device=self.device)

        self.update_step = 0
        self.update_enabled = True

    def reset(self, *, reset_counters: bool = False) -> None:
        self.buf_n.zero_()
        if reset_counters:
            self.counts.zero_()

    def share_codebook_from(self, other: "LearnedHashStateBank", update_enabled: bool = False) -> None:
        """
        * Share the state-id mapping (AE + normalizer + SimHash projection) with other,
        * Keep this bank's own counts to avoid mixing per-part bin semantics.
        """
        self.ae = other.ae
        self.opt = other.opt
        self.A = other.A
        self.normalizer = other.normalizer
        self.update_enabled = bool(update_enabled)

    @torch.no_grad()
    def push(self, feats: torch.Tensor) -> None:
        if not self.update_enabled:
            return
        feats = feats.to(self.device, dtype=torch.float32)

        if self.normalizer.training:
            self.normalizer.update(feats)

        start = int(self.buf_n.item())
        end = min(self.B, start + int(feats.shape[0]))
        n_take = end - start
        if n_take > 0:
            self.buffer[start:end] = feats[:n_take]
            self.buf_n.fill_(end)


        k = int(self.ae_update_freq)
        if int(self.buf_n.item()) >= self.B:
            with torch.enable_grad():
                self._update_autoencoder()

        self.step_count += 1

        if int(self.buf_n.item()) >= self.B:
            self.buf_n.zero_()
        
        self.step_count += 1

    @torch.no_grad()
    def _state_id_from_feats(self, feats: torch.Tensor) -> torch.Tensor:

        x = feats.to(self.device, dtype=torch.float32)
        self.ae.eval()
        _, b = self.ae(x, noise_scale=0.0)
        bits = (b >= 0.5).to(torch.long)
        v = (bits * 2 - 1).to(torch.float32) # -1 or 1
        # v = (b - 0.5).to(torch.float32) * 2.0
        # v = b.to(torch.float32)
        proj = v @ self.A.t()
        sim_bits = (proj >= 0).to(torch.long)
        code_u64 = _bits_to_u64(sim_bits)
        assert (code_u64 < self.S).all()
        state_ids = code_u64.to(torch.long)


        from termcolor import cprint
        H = state_id_entropy(state_ids, self.S)
        # cprint(f"code_u64: {code_u64}")
        # cprint(f"self.S: {self.S}")
        # cprint(f"[StateID] normalized entropy = {H:.3f}", "yellow")
        self.last_entropy = torch.tensor(H, device=self.device)

        # # map to [0..S-1]
        # if (self.S & (self.S - 1)) == 0:
        #     h = (code_u64 * 11400714819323198485) & 0xFFFFFFFFFFFFFFFF
        #     state_ids = (h & (self.S - 1)).to(torch.long)
        # else:
        #     state_ids = (code_u64 % self.S).to(torch.long)
        return state_ids

    @torch.no_grad()
    def assign(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.to(self.device, dtype=torch.float32)
        feats_n = self.normalizer(feats)
        return self._state_id_from_feats(feats_n)

    def _update_autoencoder(self) -> None:
        self.ae.train()
        # X = self.buffer[:, :self.B].reshape(-1, self.F)  # (T*B,F)
        n = int(self.buf_n.item())
        X = self.buffer[:n]  # (n,F)

        # self.normalizer.train()
        # self.normalizer.update(X)
        Xn = self.normalizer(X)

        mb = max(1, int(self.ae_num_minibatches))
        chunks = torch.chunk(torch.randperm(n, device=self.device), mb)
        chunks = [c for c in chunks if c.numel() > 0]
        if not chunks:
            return

        steps = max(1, int(self.ae_update_steps)) * self.ae_num_minibatches
        for s in range(steps):
            idx = chunks[s % len(chunks)]
            x = Xn.index_select(0, idx)

            x_hat, b = self.ae(x, noise_scale=self.noise_scale)
            recon = torch.mean((x_hat - x) ** 2)
            reg = torch.minimum((1.0 - b) ** 2, b ** 2).mean()
            loss = recon + self.lambda_binary * reg

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

        self.last_recon = recon.detach()
        self.last_reg = reg.detach()

    def _rebuild_all(self) -> None:
        """ Keep for backward compatibility """
        # assert int(self.buf_n.item()) >= self.B
        # self.rebuild_count += 1
        # self.buf_n.zero_()

        # if (self.ae_update_freq > 0) and (self.rebuild_count % self.ae_update_freq == 0):
        #     with torch.enable_grad():
        #         self._update_autoencoder()
        return

    @torch.no_grad()
    def add_contacts(
        self,
        *,
        state_ids: torch.Tensor,     # (N,) in [0..S-1]
        contact_mask: torch.Tensor,  # (N,L) bool
        contact_bins: torch.Tensor,  # (N,L) long in [0,K-1]
    ) -> None:
        state_ids = state_ids.to(self.device, dtype=torch.long).clamp(0, self.S - 1)
        contact_mask = contact_mask.to(self.device, dtype=torch.bool)
        contact_bins = contact_bins.to(self.device, dtype=torch.long).clamp(0, self.K - 1)

        if not contact_mask.any():
            return

        ei, li = torch.nonzero(contact_mask, as_tuple=True)
        si = state_ids[ei]
        bi = contact_bins[ei, li]

        lin = ((si * self.L + li) * self.K + bi).to(torch.long)
        binc = torch.bincount(lin, minlength=self.S * self.L * self.K).view(self.S, self.L, self.K)
        self.counts.add_(binc.to(self.counts.dtype))

        # if True:
        #     if self.update_step == 0:
        #         time_now = time.strftime("%m-%d-%H-%M-%S", time.localtime(time.time()))
        #         self.save_dir = f"debug_counter_viz_hash/{time_now}"
        #         os.makedirs(self.save_dir, exist_ok=True)
        #     if self.update_step % 200 == 0:
        #         torch.save(self.counts.cpu(), f"{self.save_dir}/new_counts_{self.update_step}.pt")
        #     self.update_step += 1


    @torch.no_grad()
    def get_metrics(self):
        return {
            "hash_recon_loss": float(self.last_recon),
            "hash_binary_reg": float(self.last_reg),
            "stateid_entropy": float(self.last_entropy),
        }

