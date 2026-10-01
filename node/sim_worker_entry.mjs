// Entry point for a worker_thread hosting one copy of the game's
// simulation_worker.bundle.js. Receives config via workerData:
//   { vendorDir: string }
// Posts messages from the game worker up to the parent; takes parent
// messages and delivers them to the game worker.

import { parentPort, workerData } from 'node:worker_threads';
import { join } from 'node:path';
import { installShims, loadWorkerBundle } from './shims.mjs';

// The bundle's UMD libs (js-sha256) take their Node branch when they see
// `process.versions.node`, but the bundle was packed for a worker — its
// crypto/Buffer modules are stubs. Hide Node-ness so the pure-JS paths win.
// (This thread is a pure simulator; it needs no node: APIs at runtime.)
const nodeProcess = globalThis.process; // keep a private ref for error hooks
const stash = new Map();
for (const key of ['process', 'module', 'exports', 'require', 'global', 'Buffer']) {
  stash.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
  Object.defineProperty(globalThis, key, {
    value: undefined,
    writable: true,
    configurable: true,
  });
}

installShims({ baseDir: workerData.vendorDir, port: parentPort });

// surface full stacks from inside the sim worker (the game only reads .message)
const origOnerror = Object.getOwnPropertyDescriptor(globalThis, 'onerror');
Object.defineProperty(globalThis, 'onerror', {
  get: () => origOnerror?.get?.call(globalThis),
  set: (fn) =>
    origOnerror?.set?.call(globalThis, (msg, src, line, col, err) => {
      parentPort.postMessage({
        type: 'host_worker_error',
        message: String(msg),
        stack: err?.stack ?? null,
      });
      return fn ? fn(msg, src, line, col, err) : false;
    }),
  configurable: true,
});
nodeProcess.on('uncaughtException', (err) => {
  parentPort.postMessage({ type: 'host_worker_error', message: err.message, stack: err.stack });
});
nodeProcess.on('unhandledRejection', (err) => {
  parentPort.postMessage({
    type: 'host_worker_error',
    message: String(err?.message ?? err),
    stack: err?.stack ?? null,
  });
});

try {
  loadWorkerBundle(join(workerData.vendorDir, 'simulation_worker.bundle.js'));
} finally {
  for (const [key, desc] of stash) {
    if (desc) Object.defineProperty(globalThis, key, desc);
    else delete globalThis[key];
  }
}

// Signal readiness after the bundle's async physics init has had a chance to
// install its real onmessage handler. Callers may also just send messages —
// the bundle queues messages received before init completes.
queueMicrotask(() => parentPort.postMessage({ type: 'host_ready' }));
