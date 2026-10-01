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

/** One hosted game-simulation worker. */
export class SimWorker {
  constructor(id) {
    this.id = id;
    this.worker = new Worker(join(HERE, 'sim_worker_entry.mjs'), {
      workerData: { vendorDir: VENDOR_DIR },
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
    const t = msg?.messageType;
    if (t === Ki.UpdateResult) {
      for (const fn of this._updateListeners) fn(msg.carStateBuffers);
      return;
    }
    const waiters = this._pending.get(t);
    if (waiters?.length) waiters.shift().resolve(msg);
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
  } else {
    console.log('[selftest] PASS');
  }
  await w.terminate();
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
