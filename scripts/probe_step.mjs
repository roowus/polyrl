import { SimWorker, Ki, loadPhysicsAssets, initPayload, loadTrackSaveString } from '../node/sim_host.mjs';

const w = new SimWorker(7);
await w.waitReady();
const assets = loadPhysicsAssets();
await w.send({ messageType: Ki.Init, ...initPayload(assets) });
const trackData = loadTrackSaveString('summer1');

// empty mountain like the game's >4500 fallback
await w.send({
  messageType: Ki.CreateCar,
  mountainVertices: new Float32Array(0),
  mountainOffset: { x: 0, y: 0, z: 0 },
  trackData,
  carId: 1,
  carRecording: null,
});

let last = null;
w.onUpdate((bufs) => {
  for (const b of bufs) last = new Uint8Array(b);
});

await w.send({ messageType: Ki.StartCar, carId: 1, targetSimulationTimeFrames: 200 });
await w.send({ messageType: Ki.ControlCar, carId: 1, up: true, right: false, down: false, left: false, reset: false });

await new Promise((res, rej) => {
  const t0 = performance.now();
  const iv = setInterval(() => {
    if (last) {
      const frames = last[4] | (last[5] << 8) | (last[6] << 16);
      if (frames >= 100) {
        clearInterval(iv);
        res();
      }
    }
    if (performance.now() - t0 > 20000) {
      clearInterval(iv);
      rej(new Error('timeout'));
    }
  }, 5);
});

console.log('state len', last.length, 'hex head:', Buffer.from(last.subarray(0, 40)).toString('hex'));
const dv = new DataView(last.buffer, last.byteOffset);
console.log('frames', last[4] | (last[5] << 8) | (last[6] << 16));
console.log('speedKmh', dv.getFloat32(7, true));
console.log('flags', last[11].toString(2));
console.log('pos', dv.getFloat32(12, true), dv.getFloat32(16, true), dv.getFloat32(20, true));
console.log('quat', dv.getFloat32(24, true), dv.getFloat32(28, true), dv.getFloat32(32, true), dv.getFloat32(36, true));
await w.terminate();
process.exit(0);
