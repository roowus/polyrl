// Host shims to run the game's simulation_worker.bundle.js inside a Node
// worker_thread. The bundle expects a Web Worker environment:
//   self, importScripts(path), XMLHttpRequest (sync GET), fetch,
//   atob/btoa, performance.now, postMessage/onmessage, URL, TextDecoder.
//
// Usage (inside a worker_thread):
//   import { installShims, loadWorkerBundle } from './shims.mjs';
//   installShims({ baseDir, port: parentPort });
//   loadWorkerBundle(join(baseDir, 'simulation_worker.bundle.js'));

import { readFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import vm from 'node:vm';

/**
 * Install the worker-environment shims onto globalThis.
 * @param {{ baseDir: string, port: import('node:worker_threads').MessagePort }} opts
 *   baseDir: directory the bundle's relative paths (e.g. "lib/polytrack_physics.js")
 *            resolve against — the vendor dir.
 *   port:    the worker_threads parentPort used for postMessage/onmessage.
 */
export function installShims({ baseDir, port }) {
  const g = globalThis;

  g.self = g;

  // ---- messaging -----------------------------------------------------------
  // Game worker uses both onmessage assignment and addEventListener("message"),
  // and posts via postMessage(msg, {transfer}). Bridge all of it onto the
  // worker_threads parentPort. Transfer hints are ignored (node
  // structured-clones ArrayBuffers; our protocol copies into fresh buffers).
  g.postMessage = (msg, _transfer) => port.postMessage(msg);
  const messageListeners = new Set();
  let onmessageHandler = null;
  Object.defineProperty(g, 'onmessage', {
    get: () => onmessageHandler,
    set: (fn) => {
      onmessageHandler = fn;
    },
    configurable: true,
  });
  g.addEventListener = (type, fn) => {
    if (type === 'message') messageListeners.add(fn);
  };
  g.removeEventListener = (type, fn) => {
    if (type === 'message') messageListeners.delete(fn);
  };
  g.dispatchEvent = (ev) => {
    if (ev?.type === 'message') {
      onmessageHandler?.(ev);
      for (const fn of messageListeners) fn(ev);
    }
    return true;
  };
  // Deliver on a macrotask (setImmediate), NOT queueMicrotask: the bundle's
  // module-scope `let a = performance.now()` (and friends) are initialized in
  // statements AFTER `onmessage = r` is installed. Real worker message events
  // are macrotasks and can only fire after module evaluation completes;
  // microtasks would race ahead of those initializers and hit TDZ errors.
  port.on('message', (data) => {
    setImmediate(() => g.dispatchEvent({ type: 'message', data }));
  });

  // ---- script loading ------------------------------------------------------
  const resolvePath = (p) => {
    // The glue resolves relative to self.location.href (the bundle URL).
    if (p.startsWith('file://')) return fileURLToPath(p);
    if (isAbsolute(p)) return p;
    return join(baseDir, p);
  };

  g.importScripts = (...paths) => {
    for (const p of paths) {
      const file = resolvePath(p);
      const code = readFileSync(file, 'utf8');
      vm.runInThisContext(code, { filename: file });
    }
  };

  // self.location: glue does new URL(".", self.location.href) to locate the wasm.
  if (!g.location) {
    g.location = { href: pathToFileURL(join(baseDir, 'simulation_worker.bundle.js')).href };
  }

  // ---- XHR (sync fallback the glue uses to fetch the wasm) -----------------
  g.XMLHttpRequest = class SyncFileXHR {
    open(method, url, async_ = true) {
      this._url = url;
      this._async = async_;
      this.responseType = '';
      this.status = 0;
      this.response = null;
    }
    send() {
      try {
        const file = resolvePath(String(this._url));
        const buf = readFileSync(file);
        this.status = 200;
        if (this.responseType === 'arraybuffer') {
          this.response = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        } else {
          this.response = buf.toString('utf8');
        }
        this.onload?.();
      } catch (err) {
        this.status = 404;
        this.onerror?.(err);
        if (this._async) return; // async failure surfaces via onerror
        throw err;
      }
    }
    setRequestHeader() {}
  };

  // ---- fetch (primary wasm path) -------------------------------------------
  // Node has fetch, but not for file:// URLs. Wrap it.
  const nodeFetch = g.fetch?.bind(g);
  g.fetch = async (url, init) => {
    const u = String(url);
    if (u.startsWith('file://') || !/^[a-z]+:\/\//i.test(u)) {
      const buf = readFileSync(resolvePath(u));
      return new Response(buf, { status: 200 });
    }
    return nodeFetch(u, init);
  };

  // atob/btoa exist in Node ≥16 globally; TextDecoder/performance/URL exist.
}

/**
 * Evaluate the worker bundle in this context. Returns a promise that resolves
 * once the bundle has run (its internal physics init completes asynchronously;
 * callers should wait for the first message or send Init after a tick).
 */
export function loadWorkerBundle(bundlePath) {
  const code = readFileSync(bundlePath, 'utf8');
  vm.runInThisContext(code, { filename: bundlePath });
}
