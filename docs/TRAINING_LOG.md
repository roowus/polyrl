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
