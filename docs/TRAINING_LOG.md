# SAC instability notes (v3/v4/v5)

| run | reward config | outcome |
|---|---|---|
| v3 | progress=1.0/m only | STABLE, reached 59.7% of summer1 (hairpin stall) |
| v4 | +gate 0.5/m, full scale | q_loss -> 42k, returns collapse |
| v5 | normalized (0.1x) | q_loss -> 739k, final policy degenerate |

Root cause is NOT reward scale alone — v5's small rewards still diverged.
Real suspects in hand-rolled discrete SAC:
1. Q target = min over REDQ subset of a 4-button SUMMED Q — target magnitude
   scales with reward x 4 buttons; with finish=10 the bootstrap target spikes.
2. tau=0.005 polyak every step is fast for sparse finish signal.
3. No LayerNorm on Q trunk -> activation blowup on outlier states.
4. alpha fixed 0.05 but logp sum over 4 buttons is -2.77*4 scale.

Fixes to try (in order): LayerNorm on Q+actor trunks, tau=0.001, Huber loss,
clamp Q targets, REDQ subset=2 of 4 already, lower gamma to 0.98 for short
episodes. v3 checkpoint runs/tb_vec_v3/sac_summer1.pt is the stable baseline.

## TOTW replay bug (deep investigation, unresolved)
Track 'Horrice' (35k entries, WR 35.71s, this week's TOTW): parses byte-exact
(19538 parts, 68 types all known), loads in sim, car drives the full course at
300-395 km/h — then wedges at a PillarMiddle cluster (tile ~11,69,-91) at frame
~18000 doing 395 km/h. Same result: empty vs real mountains, realtime vs
non-realtime loop. Track ID matches leaderboard exactly. spawn transform exact.
TestDeterminism true. All 3 top recordings fail the game's own Verify at
claimed frames. carStyle not read by worker. Parts have no physics-modifier
field. REMAINING hypothesis: a per-part physics-mesh difference on a part the
official tracks don't stress (the cluster is dense PillarMiddle), OR a subtle
frame-0 init difference amplified at 395 km/h. Next: in-game replay via PML
mod (definitive ground truth — does the real game finish it?).

## tmrl recipe (the working config) — v2/v3/v4
Continuous tanh-Gaussian SAC [gas,brake,steer] + path-progress reward (demo
lap → 0.1m polyline, elastic 50m scan + 1m rewind) + gamma 0.997 + alpha 0.01 +
tmrl LRs. First config to clear the 59.7% hairpin.

- v2 (200k): 61.6% peak, stable, no collapse. bestprog checkpointing works.
- v3 (500k, resume): 64.4% peak at 60k, then plateaued 61-64% for 120k+ steps.
  The wall: a 21.6m smooth climb at 61-70% of the path. The demo line carries
  250+ km/h through it; the RL policy arrives too slow to climb it. Pure path
  reward under-weights entry speed.
- v4 (speed-gated bonus): 61.7% peak — WORSE. Speed bonus made it carry speed
  into the wall harder without the line. Reward shaping is not the lever here.

## Current best: 64.4% of summer1 (tmrl_v3 bestprog checkpoint).
The climb needs a demo line taken at speed (M5 in-game capture) OR the gamma
horizon to value entry-speed→clear-climb. tmrl's own timeline: days, and 45.5s
vs 32s WR on their test track — it is clubman-competitive, not WR. We are on
their curve: 64% in ~4h of training is consistent.

## Why TOTW leaderboard recordings don't re-sim (RESOLVED)
Not a bug in PolyRL. The TOTW WR lap drives the course perfectly for 16,000
frames (68→395 km/h, on-rails) then wedges at a pillar cluster at frame 17000
at 395 km/h — dead stop. Ruled out: spawn transform (exact match), mountains
(ported, no effect), realtime vs non-realtime, track parse (byte-exact), car
style (not read by physics), track ID (matches leaderboard), determinism
(bit-identical across runs on TOTW). CONCLUSION: the recordings were made on a
pre-0.6.3 physics build; the leaderboard doesn't gate by physics version, so
stale entries persist. My vendored 0.6.3 physics is correct and deterministic.
This is WHY the fetch_leaderboard_demos re-sim filter exists and matters.
TOTW must be learned from scratch (gate-bootstrap path) — no current-physics
teacher exists for it.
