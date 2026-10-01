# PolyRL

**tmrl for [PolyTrack](https://www.kodub.com/apps/polytrack)** — a reinforcement-learning driver that learns to produce leaderboard-competitive recordings on arbitrary tracks, warm-startable from human example laps.

## How it works (TL;DR)

PolyTrack's physics is a Bullet engine compiled to WASM running in a dedicated simulation worker with a fixed 1 ms tick — and the game itself ships a **non-realtime mode** (used for ghosts, replays, and leaderboard recording verification) that steps the simulation as fast as the CPU allows. PolyRL hosts that unmodified worker headlessly in Node for training throughput, and drives the real game through a [PolyModLoader](https://polymodloader.com) mod for watching, demo capture, and recording export.

```
Python brain (PyTorch, SAC/REDQ)  ◀── WebSocket (JSON control + binary frames) ──▶  Node sim_host (K × stock sim workers)
        ▲
        └── PolyModLoader mod in the real game ──▶ watch policy drive · record demos · export recordings
```

Key facts this design rests on (reverse-engineered from the v0.6.3 desktop build):

- The worker protocol is a small message enum (`Init/Verify/CreateCar/StartCar/ControlCar/…`).
- `Verify` re-simulates a recording headless to check a finish — the same thing leaderboards do — so physics is deterministic across instances.
- A "recording" is 5 boolean channels (up/right/down/left/reset), each a list of 3-byte toggle frame numbers, zlib-deflated and base64url'd. 1 frame = 1 ms.
- The per-tick car state (position, quaternion, speed, wheel contacts, suspension, steering, …) is a 227-byte struct — ground truth, no computer vision needed.

## Layout

```
node/     Node side: shims.mjs (host shims), sim_host.mjs (headless workers),
          bridge.mjs (WS relay for the in-game mod), protocol.mjs, vendor/ (hash-pinned sim files)
brain/    Python side: codec, tracklib, env_server, worker_pool, agent (SAC/REDQ),
          replay+demo buffers, BC pretrain, train/eval/watch entrypoints
mod/      PolyModLoader mod (polyrl-mod): in-game panel, demo capture, watch mode, export
tests/    pytest: codec + recording round-trip (determinism proof)
scripts/  extract_sim.mjs (vendor the sim files from a local game copy)
configs/  reward weights, training configs
```

## Dev setup

```bash
pnpm install          # or npm install — node deps (ws)
uv sync               # python env (torch, websockets, pytest, …)
npm run extract-sim   # vendor simulation worker + wasm from ~/polytrack-dev/local-game-server
pytest                # codec + round-trip tests
```

## Status

Early development. See milestones in the plan: recording round-trip (determinism proof) → headless throughput → gym env → behavior cloning from demos → SAC/REDQ training → in-game integration → generalization.

## Ethics note

RL-generated recordings pass the game's own re-simulation validator — they are *valid* recordings by construction. Official leaderboards are nevertheless human competitions; PolyRL trains and compares locally and submits nothing without an explicit decision by the user.
