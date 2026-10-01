"""Behavior cloning pretrain from human demo ticks.

Trains the SAC actor's head on (featurized state → human action) pairs with
BCE. One demo lap = ~1175 ticks — tiny data, so we overfit mildly with
augmentation (state noise) and early stopping on held-out action accuracy.

Usage: uv run python -m brain.bc summer1
Saves: runs/bc_summer1.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .agent import Actor
from .demos import extract_demo, load_fixtures
from .features import OBS_DIM, featurize
from .tracklib import load_track_geom

REPO = Path(__file__).resolve().parent.parent


def train_bc(track: str, epochs: int = 200, lr: float = 1e-3, noise: float = 0.02, seed: int = 0):
    torch.manual_seed(seed)
    geom = load_track_geom(track)
    X, Y = [], []
    for fx in load_fixtures():
        if fx.get("track", track) != track:
            continue
        demo = extract_demo(track, fx["recording"])
        for st, a in zip(demo.states, demo.actions):
            X.append(featurize(st, geom))
            Y.append(a)
    if not X:
        raise SystemExit(f"no demos for {track}")

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    n = len(X)
    print(f"[bc] {track}: {n} ticks")

    # split
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    actor = Actor(OBS_DIM).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr)

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        actor.train()
        opt.zero_grad()
        xb = X[tr_idx].to(device)
        yb = Y[tr_idx].to(device)
        if noise > 0:
            xb = xb + noise * torch.randn_like(xb)
        p = actor(xb).clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(p, yb, reduction="none").sum(-1).mean()
        loss.backward()
        opt.step()

        # val
        actor.eval()
        with torch.no_grad():
            pv = actor(X[val_idx].to(device)).clamp(1e-6, 1 - 1e-6)
            vl = F.binary_cross_entropy(pv, Y[val_idx].to(device), reduction="none").sum(-1).mean().item()
            acc = ((pv > 0.5).float() == Y[val_idx].to(device)).float().mean().item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in actor.state_dict().items()}
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[bc] epoch {epoch}: train {loss.item():.4f} val {vl:.4f} val-acc {acc:.3f}")

    actor.load_state_dict(best_state)
    out = REPO / "runs" / f"bc_{track}.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"actor": actor.state_dict(), "obs_dim": OBS_DIM, "track": track}, out)
    print(f"[bc] saved {out} (best val {best_val:.4f})")
    return actor


if __name__ == "__main__":
    train_bc(sys.argv[1] if len(sys.argv) > 1 else "summer1")
