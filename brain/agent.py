"""Discrete SAC with factorized Bernoulli action heads for PolyTrack.

Action = 4 independent binary buttons (up, down, left, right) at the control
rate (50 Hz). The policy outputs 4 Bernoulli parameters; Q outputs 4×2 values
(one per button state). Factorized structure gives credit assignment across
gas/brake/steer without a 16-way combinatorial head. (A 16-way categorical is
a config switch away — see ACTOR_HEAD.)

REDQ-style: an ensemble of Q networks with a random-subset target (utd>1
friendly). Device-agnostic: mps/cuda/cpu.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTOR_HEAD = "bernoulli"  # or "categorical" (16-way)


def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class QEnsemble(nn.Module):
    def __init__(self, obs_dim: int, n: int = 4, hidden: int = 256):
        super().__init__()
        self.nets = nn.ModuleList([mlp([obs_dim, hidden, hidden, 8]) for _ in range(n)])  # 4 buttons × 2 states
        self.n = n

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # → (n_ensemble, B, 4, 2)
        return torch.stack([net(obs).view(obs.shape[0], 4, 2) for net in self.nets])


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden])
        self.head = nn.Linear(hidden, 4)  # logits per button

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.net(obs)))  # p(up), p(down), p(left), p(right)

    def logp(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log π(a|s) for factorized Bernoulli."""
        p = self.forward(obs).clamp(1e-6, 1 - 1e-6)
        a = actions.float()
        return (a * p.log() + (1 - a) * (1 - p).log()).sum(-1)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.forward(obs)
        dist = torch.distributions.Bernoulli(probs=p)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a, logp


@dataclass
class SacConfig:
    obs_dim: int
    gamma: float = 0.99
    alpha: float = 0.1          # initial alpha; auto-tuned if auto_alpha
    auto_alpha: bool = True     # learn log_alpha toward the target entropy
    target_entropy: float = 1.386  # 4 × ln(2) ≈ uniform-over-4-buttons entropy
    lr: float = 3e-4
    q_ensemble: int = 4
    redq_subset: int = 2
    hidden: int = 256
    target_update_tau: float = 0.005
    max_grad_norm: float = 1.0


class SAC:
    def __init__(self, cfg: SacConfig, device: str | None = None):
        self.cfg = cfg
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.actor = Actor(cfg.obs_dim, cfg.hidden).to(self.device)
        self.q = QEnsemble(cfg.obs_dim, cfg.q_ensemble, cfg.hidden).to(self.device)
        self.q_target = QEnsemble(cfg.obs_dim, cfg.q_ensemble, cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), cfg.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), cfg.lr)
        # learnable temperature
        self.log_alpha = torch.tensor(math.log(cfg.alpha), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ---- loss helpers --------------------------------------------------------

    def _q_of(self, q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Select Q(s, a) for factorized buttons.
        q_values: (..., 4, 2); actions: (..., 4) int → (..., 4)"""
        onehot = F.one_hot(actions.long(), num_classes=2).to(q_values.dtype)  # (...,4,2)
        return (q_values * onehot).sum(-1)  # (...,4)

    def q_loss(self, obs, actions, next_obs, rewards, dones):
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs)
            # REDQ: random Q subset for the target
            idx = random.sample(range(self.cfg.q_ensemble), self.cfg.redq_subset)
            qs = self.q_target(next_obs)[idx]  # (subset, B, 4, 2)
            q_next = self._q_of(qs, next_a.unsqueeze(0).expand(qs.shape[0], -1, -1))
            q_next = q_next.min(dim=0).values.sum(-1)  # min over subset, sum over buttons
            target = rewards + self.cfg.gamma * (1 - dones) * (q_next - self.alpha.detach() * next_logp)

        q_taken = self._q_of(self.q(obs), actions.unsqueeze(0).expand(self.cfg.q_ensemble, -1, -1))
        q_taken = q_taken.sum(-1)  # sum over buttons → (E, B)
        return F.mse_loss(q_taken, target.unsqueeze(0).expand(self.cfg.q_ensemble, -1))

    def actor_loss(self, obs):
        a, logp = self.actor.sample(obs)
        q_a = self._q_of(self.q(obs), a.unsqueeze(0).expand(self.cfg.q_ensemble, -1, -1))
        q_a = q_a.mean(dim=0).sum(-1)  # mean over ensemble, sum over buttons
        return (self.alpha.detach() * logp - q_a).mean(), logp.mean()

    def alpha_loss(self, logp: torch.Tensor) -> torch.Tensor:
        # drive entropy toward target: -log_alpha * (logp + target).detach()
        return (-self.log_alpha * (logp + self.cfg.target_entropy).detach()).mean()

    def update(self, batch) -> dict:
        obs, actions, next_obs, rewards, dones = [t.to(self.device) for t in batch]

        self.q_opt.zero_grad()
        ql = self.q_loss(obs, actions, next_obs, rewards, dones)
        ql.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.q_opt.step()

        self.actor_opt.zero_grad()
        al, logp = self.actor_loss(obs)
        al.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        metrics = {"q_loss": ql.item(), "actor_loss": al.item(), "logp": logp.item(), "alpha": self.alpha.item()}

        if self.cfg.auto_alpha:
            self.alpha_opt.zero_grad()
            al_alpha = self.alpha_loss(logp.detach())
            al_alpha.backward()
            self.alpha_opt.step()
            metrics["alpha_loss"] = al_alpha.item()

        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.mul_(1 - self.cfg.target_update_tau).add_(self.cfg.target_update_tau * p)
        return metrics

    def act(self, obs_tensor: torch.Tensor, deterministic: bool = False) -> tuple[int, int, int, int]:
        with torch.no_grad():
            p = self.actor(obs_tensor.to(self.device))
            if deterministic:
                a = (p > 0.5).float()
            else:
                a = torch.bernoulli(p)
        return tuple(int(v) for v in a[0].tolist())

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q": self.q.state_dict(),
                "q_target": self.q_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location=device or "cpu", weights_only=False)
        sac = cls(SacConfig(**ck["cfg"]), device=device)
        sac.actor.load_state_dict(ck["actor"])
        sac.q.load_state_dict(ck["q"])
        sac.q_target.load_state_dict(ck["q_target"])
        if "log_alpha" in ck:
            sac.log_alpha.data = ck["log_alpha"].to(sac.device)
        return sac
