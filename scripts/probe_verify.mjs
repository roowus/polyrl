// M0 core: build a recording programmatically, Verify it headless, and
// confirm the game's own evaluator agrees with the frame count.
//
// We synthesize a "hold throttle" recording with the JS port of the game's
// recording codec (toggle-frame lists, deflate9, base64url) and Verify it on
// summer1. This proves the recording → simulation → result loop end to end.

import { SimWorker, Ki, loadPhysicsAssets, initPayload, loadTrackSaveString } from '../node/sim_host.mjs';
import zlib from 'node:zlib';

// --- recording codec (JS mirror of brain/codec.py) -------------------------
const CH = ['up', 'right', 'down', 'left', 'reset'];
function u24(n) {
  return [n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff];
}
function serializeRecording(rec) {
  const bytes = [];
  for (const ch of CH) {
    const t = rec[ch] ?? [];
    bytes.push(...u24(t.length));
    for (const f of t) bytes.push(...u24(f));
  }
  const comp = zlib.deflateSync(Buffer.from(bytes), { level: 9 });
  return comp.toString('base64url');
}

// hold 'up' from frame 0 for the whole run
const recording = serializeRecording({ up: [0] });
console.log('recording string:', recording);

const w = new SimWorker(3);
await w.waitReady();
await w.send({ messageType: Ki.Init, ...initPayload(loadPhysicsAssets()) });
const trackData = loadTrackSaveString('summer1');

// Verify: does 'hold throttle forever' finish summer1 in N frames? Certainly
// not (it'll crash into the first corner). Verify returns finished && frames
// == targetFrames, so we expect FALSE with a sane target — the point is the
// evaluator RUNS and returns a verdict without error.
for (const target of [5000, 30000]) {
  const t0 = performance.now();
  const ok = await w.verify({
    trackData,
    carRecording: recording,
    carId: 1,
    targetFrames: target,
    mountainVertices: new Float32Array(0),
    mountainOffset: { x: 0, y: 0, z: 0 },
  });
  const wall = ((performance.now() - t0) / 1000).toFixed(2);
  console.log(`verify target=${target} → finished=${ok}  (${wall}s wall = ${(target / 1000 / (performance.now() - t0) * 1000 / 1000).toFixed(0)}× rt)`);
}
await w.terminate();
process.exit(0);
