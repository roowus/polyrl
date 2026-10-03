"""tmrl-faithful SAC for PolyTrack — continuous actions.

Directly modeled on tmrl's SpinupSacAgent (custom_algorithms.py) and
SquashedGaussianMLPActor (custom_models.py), adapted to PolyTrack's action
space: continuous [gas, brake, steer] in [-1,1]³, squashed-Gaussian policy,
twin Q critics, fixed-or-auto alpha, polyak target net.

The env takes 4 binary buttons; the bridge quantizes [gas,brake,steer] to
buttons (gas>0→up, brake>0→down, steer<0→left, steer>0→right) at the worker.
Continuous control is what tmrl's stability comes from — binary factorized
Bernoulli SAC is what kept collapsing in v1–v9.

Reference lines (tmrl v0.7.1):
  custom_algorithms.py:108-187  train()
  custom_models.py:52-128       SquashedGaussianMLPActor / MLPQFunction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2.0
LOG_STD_MIN = -10.0  # tmrl/Spinup values; not -20 (keeps exploration alive)


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """tmrl SquashedGaussianMLPActor: mean + log_std heads, tanh squash."""

    def __init__(self, obs_dim: int, act_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden])
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs, deterministic=False):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        raw = mu if deterministic else dist.rsample()
        action = torch.tanh(raw)
        # log prob with tanh correction (Spinup eq. 21)
        logp = dist.log_prob(raw).sum(-1)
        logp -= (2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))).sum(-1)
        return action, logp


class QFunction(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden)
        self.q1 = QFunction(obs_dim, act_dim, hidden)
        self.q2 = QFunction(obs_dim, act_dim, hidden)


@dataclass
class TmrlSacConfig:
    obs_dim: int
    act_dim: int = 3
    gamma: float = 0.997      # 50 Hz analog of tmrl's 0.995 @ 20 Hz
    polyak: float = 0.995
    alpha: float = 0.01       # fixed (tmrl ships LEARN_ENTROPY_COEF=false)
    lr_actor: float = 1e-5    # tmrl's notably-low actor LR
    lr_critic: float = 5e-5
    betas: tuple = (0.997, 0.997)  # tmrl's slow Adam second-moment
    hidden: int = 256


class TmrlSAC:
    def __init__(self, cfg: TmrlSacConfig, device: str | None = None):
        self.cfg = cfg
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = ActorCritic(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(self.device)
        import copy as _copy
        self.model_target = _copy.deepcopy(self.model)
        for p in self.model_target.parameters():
            p.requires_grad_(False)
        self.q_opt = torch.optim.Adam(
            list(self.model.q1.parameters()) + list(self.model.q2.parameters()),
            lr=cfg.lr_critic, betas=cfg.betas,
        )
        self.pi_opt = torch.optim.Adam(self.model.actor.parameters(), lr=cfg.lr_actor, betas=cfg.betas)
        self.alpha = torch.tensor(cfg.alpha, device=self.device)  # fixed

    def update(self, batch) -> dict:
        o, a, o2, r, d = [t.to(self.device) for t in batch]
        a = a.float()

        # Q update
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)
        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            q_pi_targ = torch.min(self.model_target.q1(o2, a2), self.model_target.q2(o2, a2))
            backup = r + self.cfg.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
        loss_q = (F.mse_loss(q1, backup) + F.mse_loss(q2, backup)) / 2
        self.q_opt.zero_grad()
        loss_q.backward()
        self.q_opt.step()

        # actor update (Q frozen)
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)
        pi, logp_pi = self.model.actor(o)
        q_pi = torch.min(self.model.q1(o, pi), self.model.q2(o, pi))
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        self.pi_opt.zero_grad()
        loss_pi.backward()
        self.pi_opt.step()
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        with torch.no_grad():
            for p, pt in zip(self.model.parameters(), self.model_target.parameters()):
                pt.mul_(self.cfg.polyak).add_((1 - self.cfg.polyak) * p)

        return {"q_loss": loss_q.item(), "actor_loss": loss_pi.item(), "logp": logp_pi.mean().item()}

    def act(self, obs_tensor: torch.Tensor, deterministic: bool = False) -> list[float]:
        with torch.no_grad():
            a, _ = self.model.actor(obs_tensor.to(self.device), deterministic=deterministic)
        return a[0].tolist()  # [gas, brake, steer] in [-1,1]

    def save(self, path):
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = ck["cfg"]
        cfg["betas"] = tuple(cfg["betas"])
        sac = cls(TmrlSacConfig(**cfg), device=device)
        sac.model.load_state_dict(ck["model"])
        sac.model_target.load_state_dict(ck["model"])
        for p in sac.model_target.parameters():
            p.requires_grad_(False)
        return sac


def quantize_to_buttons(gas: float, brake: float, steer: float) -> tuple[int, int, int, int]:
    """[gas, brake, steer] in [-1,1] → (up, down, left, right) 0/1."""
    return (
        1 if gas > 0.0 else 0,
        1 if brake > 0.0 else 0,
        1 if steer < -0.1 else 0,
        1 if steer > 0.1 else 0,
    )
