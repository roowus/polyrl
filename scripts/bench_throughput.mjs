// M1 throughput spike: K workers in this process, each running an independent
// non-realtime replay episode on a track; measure aggregate sim-frames/sec.
//
// Usage: node scripts/bench_throughput.mjs [K] [seconds] [track]
//   K       workers to spawn (default 4)
//   seconds wall time to measure over (default 10)
//   track   official track name (default summer1)

import { SimWorker, Ki, loadPhysicsAssets, initPayload, loadTrackSaveString } from '../node/sim_host.mjs';
import { cpus } from 'node:os';
import { deflateSync } from 'node:zlib';

const K = parseInt(process.argv[2] ?? '4', 10);
const SECONDS = parseFloat(process.argv[3] ?? '10');
const TRACK = process.argv[4] ?? 'summer1';

console.log(`[bench] ${K} workers × ${SECONDS}s on ${TRACK} (host has ${cpus().length} cores)`);

const assets = loadPhysicsAssets();
const trackData = loadTrackSaveString(TRACK);

// A short looping-ish recording: throttle pulses + a bit of steering, so the
// car moves and collides (more physics work than an idle car).
const recording = (() => {
  const bytes = [];
  const u24 = (n) => [n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff];
  const up = [0];
  const right = [];
  for (let f = 3000; f < 20000; f += 3000) right.push(f, f + 800);
  const chans = [up, right, [], [], []];
  for (const t of chans) {
    bytes.push(...u24(t.length));
    for (const f of t) bytes.push(...u24(f));
  }
  // deflate9 + base64url
  return deflateSync(Buffer.from(bytes), { level: 9 }).toString('base64url');
})();

const workers = [];
for (let k = 0; k < K; k++) {
  const w = new SimWorker(k, { burst: 100 });
  await w.waitReady();
  await w.send({ messageType: Ki.Init, ...initPayload(assets) });
  workers.push(w);
}
console.log(`[bench] ${K} workers ready`);

const TARGET = 600_000; // 10 sim-minutes per episode; we rerun episodes if finished early
const counters = new Array(K).fill(0);
const finishes = new Array(K).fill(0);

await Promise.all(
  workers.map(async (w, k) => {
    const startEpisode = async () => {
      const carId = 1000 + k;
      await w.send({
        messageType: Ki.CreateCar,
        mountainVertices: new Float32Array(0),
        mountainOffset: { x: 0, y: 0, z: 0 },
        trackData,
        carId,
        carRecording: recording,
      });
      await w.send({ messageType: Ki.StartCar, carId, targetSimulationTimeFrames: TARGET });
    };
    w.onUpdate((bufs) => {
      for (const b of bufs) {
        counters[k]++;
        const u8 = new Uint8Array(b);
        if (u8[11] & 2) finishes[k]++; // hasFinish
      }
    });
    await startEpisode();
  }),
);

// measure
const t0 = performance.now();
const c0 = [...counters];
await new Promise((r) => setTimeout(r, SECONDS * 1000));
const wall = (performance.now() - t0) / 1000;
const deltas = counters.map((c, k) => c - c0[k]);
const total = deltas.reduce((a, b) => a + b, 0);

console.log(`[bench] ${SECONDS}s wall:`);
deltas.forEach((d, k) => console.log(`  worker ${k}: ${(d / 1000).toFixed(0)}k frames (${(d / 1000 / wall).toFixed(1)}k f/s)`));
console.log(`[bench] aggregate: ${(total / 1000).toFixed(0)}k sim-frames in ${wall.toFixed(1)}s`);
console.log(`[bench] = ${(total / wall / 1000).toFixed(1)}k sim-frames/sec aggregate`);
console.log(`[bench] = ${(total / wall / 1000 / 1).toFixed(1)}× realtime aggregate (per-worker ${(total / wall / 1000 / K).toFixed(1)}×)`);
console.log(`[bench] finishes observed: ${finishes.reduce((a, b) => a + b, 0)}`);

await Promise.all(workers.map((w) => w.terminate()));
process.exit(0);
