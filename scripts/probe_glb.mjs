import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import * as fs from 'node:fs';
import { Worker as NW } from 'node:worker_threads';

process.on('unhandledRejection', (e) => console.log('UNHANDLED:', e?.message ?? e));

const dracoDir = '/tmp/polytrack-game/lib/draco';

class Shim {
  constructor(url) {
    const u = String(url);
    console.log('Worker ctor url:', u.slice(0, 40));
    if (u.startsWith('blob:')) {
      this._q = [];
      fetch(u)
        .then((r) => r.text())
        .then((src) => {
          console.log('blob fetched, len', src.length);
          // browser-worker → node-worker adapter: the draco worker body uses
          // bare onmessage=/postMessage() globals.
          const wrapped = [
            "const { parentPort } = require('node:worker_threads');",
            'globalThis.self = globalThis;',
            'globalThis.postMessage = (m) => parentPort.postMessage(m);',
            'globalThis.onmessage = null;',
            "parentPort.on('message', (data) => globalThis.onmessage?.({ data }));",
            src,
          ].join('\n');
          fs.writeFileSync('/tmp/draco-worker.cjs', wrapped);
          this._w = new NW('/tmp/draco-worker.cjs');
          this._w.on('message', (m) => this.onmessage?.({ data: m }));
          this._w.on('error', (e) => console.log('worker err', e.message));
          for (const m of this._q) this._w.postMessage(m);
          this._q = [];
        })
        .catch((e) => console.log('blob fetch err', e.message));
    }
  }
  postMessage(m) {
    if (this._w) this._w.postMessage(m);
    else this._q.push(m);
  }
  terminate() {
    this._w?.terminate();
  }
}
globalThis.Worker = Shim;

THREE.FileLoader.prototype.load = function (url, onLoad, _p, onError) {
  let p = String(url);
  if (p.startsWith('file://')) p = p.slice(7);
  else if (!p.startsWith('/')) p = dracoDir + '/' + p;
  try {
    const b = fs.readFileSync(p);
    queueMicrotask(() =>
      onLoad(
        this.responseType === 'arraybuffer'
          ? b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)
          : b.toString('utf8'),
      ),
    );
  } catch (e) {
    queueMicrotask(() => onError?.(e));
  }
};

const draco = new DRACOLoader().setDecoderPath(dracoDir + '/');
draco.setDecoderConfig({ wasmBinary: fs.readFileSync(dracoDir + '/draco_decoder.wasm') });
const loader = new GLTFLoader().setDRACOLoader(draco);
const data = fs.readFileSync('/tmp/polytrack-game/models/car.glb');
console.log('parsing car.glb…');
loader.parse(
  data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
  '',
  (g) => {
    console.log('PARSED. nodes:', g.scene.children.map((c) => c.name).join(','));
    process.exit(0);
  },
  (e) => {
    console.log('PARSE ERR', e?.message ?? e);
    process.exit(1);
  },
);
setTimeout(() => {
  console.log('TIMEOUT 20s');
  process.exit(2);
}, 20000);
