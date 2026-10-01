// env_server_node — NDJSON stdio server driving headless sim workers for the
// Python env. One line in = one command, one line out = one response.
//
// Commands:
//   {cmd:"reset", env_id, track, max_frames}        → {env_id, state(hex)}
//   {cmd:"step",  env_id, frames, controls}         → {env_id, state(hex)}
//   {cmd:"close", env_id}                           → {ok:true}
//
// `state` is the 227-byte VO struct (carId header stripped) as hex.

import { SimWorker, Ki, loadPhysicsAssets, initPayload, loadTrackSaveString } from '../node/sim_host.mjs';
import readline from 'node:readline';

const assets = loadPhysicsAssets();
const trackCache = new Map();
const trackData = (name) => {
  if (!trackCache.has(name)) trackCache.set(name, loadTrackSaveString(name));
  return trackCache.get(name);
};

// env_id -> { worker, carId, lastState: Uint8Array|null, waiters: [] }
const envs = new Map();
let nextEnvId = 1;

async function cmdReset(msg) {
  const envId = msg.env_id ?? nextEnvId++;
  let env = envs.get(envId);
  if (env) {
    await env.worker.terminate();
  }
  const worker = new SimWorker(envId);
  await worker.waitReady();
  await worker.send({ messageType: Ki.Init, ...initPayload(assets) });

  const carId = 1;
  env = { worker, carId, lastState: null, cond: null };
  worker.onUpdate((bufs) => {
    for (const b of bufs) {
      const u8 = new Uint8Array(b);
      const id = u8[0] | (u8[1] << 8) | (u8[2] << 16) | (u8[3] << 24);
      if (id === carId) env.lastState = u8.slice(4); // strip carId header
    }
    env.cond?.();
  });
  envs.set(envId, env);

  await worker.send({
    messageType: Ki.CreateCar,
    mountainVertices: new Float32Array(0),
    mountainOffset: { x: 0, y: 0, z: 0 },
    trackData: trackData(msg.track),
    carId,
    carRecording: null,
  });
  env.maxFrames = msg.max_frames ?? 90000;

  // Track the frame target we last commanded, NOT the observed frame (which
  // lags). The h() loop stops exactly at target each time (burst=1), so the
  // commanded target IS the car's frame once waitFrames returns.
  env.target = 1;
  await worker.send({ messageType: Ki.StartCar, carId, targetSimulationTimeFrames: env.target });

  // wait for the first state so reset() returns a real observation
  await waitFrames(env, 1);
  return { env_id: envId, state: Buffer.from(env.lastState).toString('hex') };
}

function waitFrames(env, target) {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const check = () => {
      if (env.lastState) {
        const frames = env.lastState[0] | (env.lastState[1] << 8) | (env.lastState[2] << 16);
        if (frames >= target) {
          env.cond = null;
          resolve();
          return;
        }
      }
      if (performance.now() - t0 > 30000) {
        env.cond = null;
        reject(new Error(`waitFrames(${target}) timeout`));
        return;
      }
      env.cond = check;
    };
    env.cond = check;
    check();
  });
}

async function cmdStep(msg) {
  const env = envs.get(msg.env_id);
  if (!env) throw new Error(`no env ${msg.env_id}`);
  const c = msg.controls ?? {};
  env.worker.setControls(env.carId, {
    up: !!c.up, down: !!c.down, left: !!c.left, right: !!c.right, reset: !!c.reset,
  });
  const R = msg.frames ?? 20;
  // extend the commanded target by R; the worker stops exactly there (burst=1)
  env.target += R;
  await env.worker.send({ messageType: Ki.StartCar, carId: env.carId, targetSimulationTimeFrames: env.target });
  await waitFrames(env, env.target);
  return { env_id: msg.env_id, state: Buffer.from(env.lastState).toString('hex') };
}

async function cmdClose(msg) {
  const env = envs.get(msg.env_id);
  if (env) {
    await env.worker.terminate();
    envs.delete(msg.env_id);
  }
  return { ok: true };
}

// Run a recording to its end (or max_frames), collecting the 227-byte state
// every `sample_every` frames. Used for demo extraction: (state, action)
// pairs where action = the recording's buttons over the sampled window.
async function cmdRunRecording(msg) {
  const { track, recording, max_frames = 60000, sample_every = 20 } = msg;
  const worker = new SimWorker(9000 + (nextEnvId++), { burst: 100 });
  await worker.waitReady();
  await worker.send({ messageType: Ki.Init, ...initPayload(assets) });
  const carId = 1;
  await worker.send({
    messageType: Ki.CreateCar,
    mountainVertices: new Float32Array(0),
    mountainOffset: { x: 0, y: 0, z: 0 },
    trackData: trackData(track),
    carId,
    carRecording: recording,
  });

  const states = [];
  let finished = false;
  let finishFrames = null;
  let lastFrames = 0;
  // update buffers: [0..3]=carId u32 | then VO struct: frames u24 @4,
  // speedKmh f32 @7, flags u8 @11, [finishFrames u24 @12 if hasFinish], ...
  const parse = (u8) => ({
    frames: u8[4] | (u8[5] << 8) | (u8[6] << 16),
    flags: u8[11],
    hex: Buffer.from(u8.subarray(4)).toString('hex'),
  });
  worker.onUpdate((bufs) => {
    for (const b of bufs) {
      const u8 = new Uint8Array(b);
      const s = parse(u8);
      lastFrames = s.frames;
      const hasFinish = (s.flags & 2) !== 0;
      if (hasFinish) {
        finished = true;
        finishFrames = u8[12] | (u8[13] << 8) | (u8[14] << 16);
      }
      if (s.frames % sample_every === 0) states.push(s.hex);
    }
  });

  await worker.send({ messageType: Ki.StartCar, carId, targetSimulationTimeFrames: max_frames });

  const t0 = performance.now();
  await new Promise((resolve, reject) => {
    const iv = setInterval(() => {
      if (finished || lastFrames >= max_frames - 1) {
        clearInterval(iv);
        resolve();
      } else if (performance.now() - t0 > 120000) {
        clearInterval(iv);
        reject(new Error('run_recording timeout'));
      }
    }, 5);
  });

  await worker.terminate();
  return { finished, finishFrames, frames: lastFrames, samples: states };
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', async (line) => {
  const t = line.trim();
  if (!t) return;
  let msg;
  try {
    msg = JSON.parse(t);
  } catch {
    return;
  }
  try {
    let resp;
    if (msg.cmd === 'reset') resp = await cmdReset(msg);
    else if (msg.cmd === 'step') resp = await cmdStep(msg);
    else if (msg.cmd === 'close') resp = await cmdClose(msg);
    else if (msg.cmd === 'run_recording') resp = await cmdRunRecording(msg);
    else resp = { error: `unknown cmd ${msg.cmd}` };
    process.stdout.write(JSON.stringify(resp) + '\n');
  } catch (e) {
    process.stdout.write(JSON.stringify({ env_id: msg.env_id ?? null, error: String(e?.message ?? e) }) + '\n');
  }
});
