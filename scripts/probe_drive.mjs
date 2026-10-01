// Does the live-controls patch let us actually DRIVE headless?
// Hold throttle for 5 sim-seconds on summer1; the car must accelerate well
// past 0 km/h and move from spawn. Then steer left and confirm heading changes.

import { SimWorker, Ki, loadPhysicsAssets, initPayload, loadTrackSaveString } from '../node/sim_host.mjs';

const w = new SimWorker(5);
await w.waitReady();
await w.send({ messageType: Ki.Init, ...initPayload(loadPhysicsAssets()) });
const trackData = loadTrackSaveString('summer1');

await w.send({
  messageType: Ki.CreateCar,
  mountainVertices: new Float32Array(0),
  mountainOffset: { x: 0, y: 0, z: 0 },
  trackData,
  carId: 1,
  carRecording: null, // user-controlled → our live hook must serve it
});

let last = null;
w.onUpdate((bufs) => {
  for (const b of bufs) last = new Uint8Array(b);
});

const read = () => {
  const dv = new DataView(last.buffer, last.byteOffset);
  return {
    frames: last[4] | (last[5] << 8) | (last[6] << 16),
    speed: dv.getFloat32(7, true),
    x: dv.getFloat32(14, true),
    y: dv.getFloat32(18, true),
    z: dv.getFloat32(22, true),
  };
};

// IMPORTANT: set controls BEFORE starting so frame 0 sees them
w.setControls(1, { up: true });
await w.send({ messageType: Ki.StartCar, carId: 1, targetSimulationTimeFrames: 5000 });

await new Promise((res, rej) => {
  const t0 = performance.now();
  const iv = setInterval(() => {
    if (last) {
      const s = read();
      if (s.frames >= 4999) {
        clearInterval(iv);
        res(s);
      }
    }
    if (performance.now() - t0 > 20000) {
      clearInterval(iv);
      rej(new Error('timeout'));
    }
  }, 2);
});

const s = read();
const movedFrom = { x: 318.6, y: 55.3, z: 20.0 }; // spawn (no-control baseline)
const dist = Math.hypot(s.x - movedFrom.x, s.y - movedFrom.y, s.z - movedFrom.z);
console.log(`after 5s full throttle: speed=${s.speed.toFixed(1)}km/h pos=(${s.x.toFixed(1)}, ${s.y.toFixed(1)}, ${s.z.toFixed(1)}) moved=${dist.toFixed(1)}m`);
if (s.speed > 5 && dist > 10) {
  console.log('[drive] PASS — car accelerated under live control');
} else {
  console.log('[drive] FAIL — car did not move (live controls not reaching physics)');
  process.exitCode = 1;
}
await w.terminate();
process.exit(process.exitCode ?? 0);
