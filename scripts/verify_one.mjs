// Replay a recording headless and report the outcome as JSON on stdout:
//   {"finished": bool, "frames": int, "wallMs": int}
// Usage: node scripts/verify_one.mjs <recording_b64url> <target_frames> [track]

import { SimWorker, Ki, loadPhysicsAssets, initPayload, loadTrackSaveString } from '../node/sim_host.mjs';

const [recording, targetFramesStr, trackName] = process.argv.slice(2);
const targetFrames = parseInt(targetFramesStr ?? '20000', 10);
const track = trackName ?? 'summer1';

const w = new SimWorker(99);
await w.waitReady();
await w.send({ messageType: Ki.Init, ...initPayload(loadPhysicsAssets()) });
const trackData = loadTrackSaveString(track);

// Replay path: CreateCar with the recording, StartCar non-realtime, watch
// UpdateResult states until finishFrames != null or we pass the target.
await w.send({
  messageType: Ki.CreateCar,
  mountainVertices: new Float32Array(0),
  mountainOffset: { x: 0, y: 0, z: 0 },
  trackData,
  carId: 1,
  carRecording: recording,
});

let last = null;
w.onUpdate((bufs) => {
  for (const b of bufs) last = new Uint8Array(b);
});

const t0 = performance.now();
await w.send({ messageType: Ki.StartCar, carId: 1, targetSimulationTimeFrames: targetFrames });

const result = await new Promise((resolve, reject) => {
  const iv = setInterval(() => {
    if (last) {
      const frames = last[4] | (last[5] << 8) | (last[6] << 16);
      const flags = last[11];
      const hasFinish = (flags & 2) !== 0;
      let finishFrames = null;
      if (hasFinish) finishFrames = last[12] | (last[13] << 8) | (last[14] << 16);
      if (hasFinish) {
        clearInterval(iv);
        resolve({ finished: true, frames: finishFrames });
        return;
      }
      if (frames >= targetFrames - 1) {
        clearInterval(iv);
        resolve({ finished: false, frames });
        return;
      }
    }
    if (performance.now() - t0 > 60000) {
      clearInterval(iv);
      reject(new Error('timeout'));
    }
  }, 5);
});

console.log(JSON.stringify({ ...result, wallMs: Math.round(performance.now() - t0) }));
await w.terminate();
process.exit(0);
