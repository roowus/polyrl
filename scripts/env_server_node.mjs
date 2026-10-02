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

// env_id -> { worker, carId, lastState: Uint8Array|null, cond }
const envs = new Map();
let nextEnvId = 1;

// Pre-spawned worker pool: booting a worker costs ~120 ms (wasm init), so we
// keep a warm pool sized POOL_SIZE and hand them out on reset.
const POOL_SIZE = parseInt(process.env.POLYRL_POOL ?? '8', 10);
const pool = [];
async function warmPool() {
  for (let k = 0; k < POOL_SIZE; k++) {
    const w = new SimWorker(10_000 + k);
    await w.waitReady();
    await w.send({ messageType: Ki.Init, ...initPayload(assets) });
    pool.push(w);
  }
  console.error(`[env_server] pool warm: ${pool.length} workers`);
}

async function cmdReset(msg) {
  const envId = msg.env_id ?? nextEnvId++;
  let env = envs.get(envId);
  if (env) {
    env.worker.clearControls(env.carId);
    env.worker.send({ messageType: Ki.DeleteCar, carId: env.carId });
    // keep the warm worker; just recreate the car on it
    const worker = env.worker;
    env.lastState = null;
    env.target = 0;
    env.maxFrames = msg.max_frames ?? 90000;
    await worker.send({
      messageType: Ki.CreateCar,
      mountainVertices: new Float32Array(0),
      mountainOffset: { x: 0, y: 0, z: 0 },
      trackData: trackData(msg.track),
      carId: env.carId,
      carRecording: null,
    });
    env.target = 1;
    await worker.send({ messageType: Ki.StartCar, carId: env.carId, targetSimulationTimeFrames: 1 });
    await waitFrames(env, 1);
    return { env_id: envId, state: Buffer.from(env.lastState).toString('hex') };
  }
  const worker = pool.length ? pool.pop() : await (async () => {
    const w = new SimWorker(envId);
    await w.waitReady();
    await w.send({ messageType: Ki.Init, ...initPayload(assets) });
    return w;
  })();
  const carId = 1;
  env = { worker, carId, lastState: null, cond: null, target: 0 };
  worker.onUpdate((bufs) => {
    for (const b of bufs) {
      const u8 = new Uint8Array(b);
      const id = u8[0] | (u8[1] << 8) | (u8[2] << 16) | (u8[3] << 24);
      if (id === carId) env.lastState = u8.slice(4);
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
  env.target = 1;
  await worker.send({ messageType: Ki.StartCar, carId, targetSimulationTimeFrames: 1 });
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

// step N envs at once: one command, one response; each env advances R frames.
// {cmd:"step_all", frames:R, actions:[{env_id, controls}, ...]}
async function cmdStepAll(msg) {
  const R = msg.frames ?? 20;
  const jobs = [];
  for (const { env_id, controls } of msg.actions) {
    const env = envs.get(env_id);
    if (!env) continue;
    const c = controls ?? {};
    env.worker.setControls(env.carId, {
      up: !!c.up, down: !!c.down, left: !!c.left, right: !!c.right, reset: !!c.reset,
    });
    env.target += R;
    jobs.push(
      env.worker
        .send({ messageType: Ki.StartCar, carId: env.carId, targetSimulationTimeFrames: env.target })
        .then(() => waitFrames(env, env.target))
        .then(() => ({ env_id, state: Buffer.from(env.lastState).toString('hex') })),
    );
  }
  const results = await Promise.all(jobs);
  return { results };
}

// reset N envs at once: {cmd:"reset_all", envs:[{env_id?, track, max_frames}, ...]}
async function cmdResetAll(msg) {
  const results = [];
  for (const e of msg.envs) {
    results.push(await cmdReset(e));
  }
  return { results };
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

// Score M recordings against a track, K workers in parallel. Each candidate
// gets its own worker; we replay it and return the per-sample states so the
// Python side can compute progress-through-gates + finish time.
// {cmd:"score_batch", track, sample_every, recordings: ["<b64>", ...]}
//   → { results: [{finished, finishFrames, frames, samples: [hex...]}] }
async function cmdScoreBatch(msg) {
  const { track, recordings, sample_every = 20 } = msg;
  const maxFrames = msg.max_frames ?? 60000;
  const results = await Promise.all(
    recordings.map(async (recording, k) => {
      const w = new SimWorker(30_000 + k, { burst: 100 });
      try {
        await w.waitReady();
        await w.send({ messageType: Ki.Init, ...initPayload(assets) });
        const carId = 1;
        await w.send({
          messageType: Ki.CreateCar,
          mountainVertices: new Float32Array(0),
          mountainOffset: { x: 0, y: 0, z: 0 },
          trackData: trackData(track),
          carId,
          carRecording: recording,
        });
        const samples = [];
        let finished = false;
        let finishFrames = null;
        let lastFrames = 0;
        w.onUpdate((bufs) => {
          for (const b of bufs) {
            const u8 = new Uint8Array(b);
            const frames = u8[4] | (u8[5] << 8) | (u8[6] << 16);
            lastFrames = frames;
            if (u8[11] & 2) {
              finished = true;
              finishFrames = u8[12] | (u8[13] << 8) | (u8[14] << 16);
            }
            if (frames % sample_every === 0) samples.push(Buffer.from(u8.subarray(4)).toString('hex'));
          }
        });
        await w.send({ messageType: Ki.StartCar, carId, targetSimulationTimeFrames: maxFrames });
        const t0 = performance.now();
        await new Promise((resolve) => {
          const iv = setInterval(() => {
            if (finished || lastFrames >= maxFrames - 1 || performance.now() - t0 > 60000) {
              clearInterval(iv);
              resolve();
            }
          }, 10);
        });
        return { finished, finishFrames, frames: lastFrames, samples };
      } finally {
        await w.terminate();
      }
    }),
  );
  return { results };
}

const rl = readline.createInterface({ input: process.stdin });
warmPool().then(() => rl.emit('ready'));
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
    else if (msg.cmd === 'step_all') resp = await cmdStepAll(msg);
    else if (msg.cmd === 'reset_all') resp = await cmdResetAll(msg);
    else if (msg.cmd === 'score_batch') resp = await cmdScoreBatch(msg);
    else resp = { error: `unknown cmd ${msg.cmd}` };
    process.stdout.write(JSON.stringify(resp) + '\n');
  } catch (e) {
    process.stdout.write(JSON.stringify({ env_id: msg.env_id ?? null, error: String(e?.message ?? e) }) + '\n');
  }
});
