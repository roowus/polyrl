#!/usr/bin/env node
// sim_host — boots K copies of the game's simulation worker in worker_threads
// and exposes the Ki protocol to controllers.
//
// CLI modes:
//   node sim_host.mjs selftest            S1 spike: boot 1 worker, Init,
//                                         TestDeterminism, Verify smoke
//   node sim_host.mjs serve [--port N]    WebSocket server (added in M2)
//
// The Ki protocol (from the game bundle):
//   in : Init=0 Verify=1 TestDeterminism=2 CreateCar=3 DeleteCar=4
//        StartCar=5 ControlCar=6 PauseCar=7
//   out: VerifyResult=8 DeterminismResult=9 UpdateResult=10

import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { readFileSync } from 'node:fs';
import { exportToSaveString } from './track_codec.mjs';

export const Ki = Object.freeze({
  Init: 0,
  Verify: 1,
  TestDeterminism: 2,
  CreateCar: 3,
  DeleteCar: 4,
  StartCar: 5,
  ControlCar: 6,
  PauseCar: 7,
  VerifyResult: 8,
  DeterminismResult: 9,
  UpdateResult: 10,
});

const HERE = dirname(fileURLToPath(import.meta.url));
const VENDOR_DIR = join(HERE, 'vendor');

/** Load vendored physics assets (built by scripts/build_physics_assets.mjs). */
export function loadPhysicsAssets() {
  return JSON.parse(readFileSync(join(VENDOR_DIR, 'physics_assets.json'), 'utf8'));
}

/** Load an official track as a SAVE string (what CreateCar/Verify expect).
 *  .track files are the EXPORT format (PolyTrack2 prefix + name/author
 *  header); convert to the save format via track_codec. */
export function loadTrackSaveString(name) {
  const raw = readFileSync(join(VENDOR_DIR, 'tracks', 'official', `${name}.track`), 'utf8').trim();
  return exportToSaveString(raw);
}

/** Build the Init payload from vendored physics assets. */
export function initPayload(assets, { isRealtime = false, version = '0.6.3' } = {}) {
  return {
    version,
    isRealtime,
    trackParts: assets.parts.map((p) => ({
      id: p.typeId,
      vertices: new Float32Array(p.vertices),
      detector: p.detector
        ? { type: p.detector.type === 'Checkpoint' ? 0 : 1, center: p.detector.center, size: p.detector.size }
        : null,
      startOffset: p.startOffset ? [p.startOffset.x, p.startOffset.y, p.startOffset.z] : null,
    })),
    carCollisionShapeVertices: new Float32Array(assets.car.collisionShapeVertices),
    carMassOffset: assets.car.massOffset,
  };
}

/** One hosted game-simulation worker. */
export class SimWorker {
  constructor(id, { burst = 1 } = {}) {
    this.id = id;
    this.worker = new Worker(join(HERE, 'sim_worker_entry.mjs'), {
      workerData: { vendorDir: VENDOR_DIR, burst },
    });
    this._pending = new Map(); // messageType -> [resolve,...] (FIFO)
    this._updateListeners = [];
    this.worker.on('message', (msg) => this._onMessage(msg));
    this.worker.on('error', (err) => {
      const e = err instanceof Error ? err : new Error(String(err));
      for (const waiters of this._pending.values()) waiters.forEach((r) => r.reject(e));
      this._pending.clear();
    });
  }

  _onMessage(msg) {
    if (msg?.type === 'host_ready') return;
    if (msg?.type === 'host_worker_error') {
      console.error(`[sim_worker ${this.id}] ${msg.message}\n${msg.stack ?? ''}`);
      return;
    }
    const t = msg?.messageType;
    if (t === Ki.UpdateResult) {
      for (const fn of this._updateListeners) fn(msg.carStateBuffers);
      return;
    }
    const waiters = this._pending.get(t);
    if (waiters?.length) waiters.shift().resolve(msg);
  }

  /** Wait until the game worker has finished its async physics init and
   *  drained its internal pre-init queue. We detect this by round-tripping a
   *  TestDeterminism and confirming a DeterminismResult comes back — that
   *  only happens after the real dispatcher is live. */
  async waitReady(timeoutMs = 120000) {
    const t0 = performance.now();
    for (;;) {
      try {
        const ok = await this.testDeterminism(5000);
        if (ok) return;
      } catch {
        /* retry until timeout */
      }
      if (performance.now() - t0 > timeoutMs) throw new Error('sim worker never became ready');
    }
  }

  /** Send a message; if replyType is given, await the next message of that type. */
  send(msg, replyType = null, timeoutMs = 30000) {
    this.worker.postMessage(msg);
    if (replyType == null) return Promise.resolve(null);
    return new Promise((resolve, reject) => {
      const list = this._pending.get(replyType) ?? [];
      const timer = setTimeout(() => {
        reject(new Error(`timeout waiting for messageType=${replyType}`));
      }, timeoutMs);
      list.push({
        resolve: (m) => {
          clearTimeout(timer);
          resolve(m);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
      this._pending.set(replyType, list);
    });
  }

  onUpdate(fn) {
    this._updateListeners.push(fn);
    return () => {
      const i = this._updateListeners.indexOf(fn);
      if (i >= 0) this._updateListeners.splice(i, 1);
    };
  }

  /** Set live controls for a car (headless non-realtime driving; requires the
   *  worker's live-controls patch, on by default in sim_worker_entry). */
  setControls(carId, { up = false, right = false, down = false, left = false, reset = false }) {
    this.worker.postMessage({ __polyrl: 'set_controls', carId, up, right, down, left, reset });
  }

  clearControls(carId) {
    this.worker.postMessage({ __polyrl: 'clear_controls', carId });
  }

  async init({ version, isRealtime, trackParts, carCollisionShapeVertices, carMassOffset }) {
    await this.send({
      messageType: Ki.Init,
      version,
      isRealtime,
      trackParts,
      carCollisionShapeVertices,
      carMassOffset,
    });
  }

  async testDeterminism(timeoutMs = 120000) {
    const res = await this.send({ messageType: Ki.TestDeterminism }, Ki.DeterminismResult, timeoutMs);
    return res.isDeterminstic === true;
  }

  async verify({ trackData, carRecording, targetFrames, mountainVertices, mountainOffset, carId = 900000 }, timeoutMs = 120000) {
    const res = await this.send(
      { messageType: Ki.Verify, trackData, carRecording, carId, targetFrames, mountainVertices, mountainOffset },
      Ki.VerifyResult,
      timeoutMs,
    );
    return res.result === true;
  }

  async terminate() {
    await this.worker.terminate();
  }
}

// ---------------------------------------------------------------------------
// S1 selftest

async function selftest() {
  console.log('[selftest] booting 1 worker…');
  const t0 = performance.now();
  const w = new SimWorker(0);
  await w.waitReady();
  console.log(`[selftest] worker ready (${(performance.now() - t0).toFixed(0)} ms)`);

  // The game's main thread sends Init with track parts + car collision shape
  // extracted from loaded assets. For the determinism smoke we don't need a
  // real track: Init with empty parts is enough to boot physics, and
  // TestDeterminism exercises the wasm itself.
  await w.init({
    version: '0.6.3',
    isRealtime: false,
    trackParts: [],
    carCollisionShapeVertices: new Float32Array(0),
    carMassOffset: { x: 0, y: 0, z: 0 },
  });
  console.log(`[selftest] Init sent (${(performance.now() - t0).toFixed(0)} ms)`);

  const det = await w.testDeterminism();
  console.log(`[selftest] TestDeterminism → ${det}`);
  if (!det) {
    console.error('[selftest] FAIL: physics not deterministic');
    process.exitCode = 1;
    await w.terminate();
    return;
  }
  console.log('[selftest] determinism PASS');
  await w.terminate();

  // ---- real-track drive: summer1, full throttle, 10 sim-seconds ----
  const assets = loadPhysicsAssets();
  console.log(`[track] physics assets: ${assets.parts.length} parts, car checksum ok=${assets.car.checksumOk}`);
  const trackData = loadTrackSaveString('summer1');

  const w2 = new SimWorker(1);
  w2.worker.on('error', (e) => console.error('[track] worker error:', e.message));
  await w2.waitReady();
  await w2.send({ messageType: Ki.Init, ...initPayload(assets) });

  const carId = 1;
  await w2.send({
    messageType: Ki.CreateCar,
    // minimal valid mountain mesh: single far-away degenerate triangle
    mountainVertices: new Float32Array([0, -1000, 0, 1, -1000, 0, 0, -1000, 1]),
    mountainOffset: { x: 0, y: 0, z: 0 },
    trackData,
    carId,
    carRecording: null,
  });
  console.log('[track] car created');

  let updates = 0;
  let lastState = null;
  w2.onUpdate((buffers) => {
    for (const buf of buffers) {
      updates++;
      lastState = new Uint8Array(buf);
    }
  });

  const target = 10_000; // 10 sim-seconds at 1 kHz
  await w2.send({ messageType: Ki.StartCar, carId, targetSimulationTimeFrames: target });

  // hold throttle the whole run
  await w2.send({ messageType: Ki.ControlCar, carId, up: true, right: false, down: false, left: false, reset: false });

  // wait until frames reach target (UpdateResult states carry frame count)
  const t1 = performance.now();
  await new Promise((resolve, reject) => {
    const timer = setInterval(() => {
      if (lastState) {
        const frames = lastState[4] | (lastState[5] << 8) | (lastState[6] << 16);
        if (frames >= target - 1) {
          clearInterval(timer);
          resolve();
        }
      }
      if (performance.now() - t1 > 60000) {
        clearInterval(timer);
        reject(new Error('timed out waiting for sim'));
      }
    }, 5);
  });
  const wall = (performance.now() - t1) / 1000;
  const frames = lastState[4] | (lastState[5] << 8) | (lastState[6] << 16);
  // state layout: [0..3]=carId u32, then the 227-byte VO struct
  // (frames u24 at 4, speedKmh f32 at 7, flags at 11, nextCp u16 at 12,
  //  pos 3×f32 at 14, quat 4×f32 at 26)
  const view = new DataView(lastState.buffer, lastState.byteOffset);
  const speed = view.getFloat32(7, true);
  const px = view.getFloat32(14, true),
    py = view.getFloat32(18, true),
    pz = view.getFloat32(22, true);
  console.log(
    `[track] PASS: simmed ${frames} frames in ${wall.toFixed(2)}s wall (${(frames / 1000 / wall).toFixed(1)}× realtime); ` +
      `updates=${updates} speed=${speed.toFixed(1)}km/h pos=(${px.toFixed(1)}, ${py.toFixed(1)}, ${pz.toFixed(1)})`,
  );
  await w2.terminate();
  console.log('[selftest] ALL PASS');
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const mode = process.argv[2] ?? 'selftest';
  if (mode === 'selftest') {
    selftest().catch((e) => {
      console.error('[selftest] error:', e);
      process.exitCode = 1;
    });
  } else {
    console.error(`unknown mode: ${mode}`);
    process.exit(2);
  }
}
