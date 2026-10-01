#!/usr/bin/env node
// Build node/vendor/physics_assets.json:
//   - car collision shape vertices (car.glb "Collision" mesh, toNonIndexed)
//     — verified against the game's hardcoded SHA-256 checksum
//   - per-part-type physics shape vertices (merge part models per the game's
//     loader: apply transforms, mergeGeometries, toNonIndexed)
//     — verified against per-part checksums in part_configs.json
//   - car constants (massOffset etc.)
//
// Source of truth for the pipeline: main.bundle.js TrackPartManager.init
// (function `s(...)` / `o(...)` around "Physics geometry is missing").
//
// Usage: node scripts/build_physics_assets.mjs [gameDir]
//   gameDir defaults to /tmp/polytrack-game (or ~/polytrack-dev/local-game-server)

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { createHash } from 'node:crypto';

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const VENDOR = join(HERE, '..', 'node', 'vendor');

const gameDir = process.argv[2]
  ? resolve(process.argv[2])
  : existsSync('/tmp/polytrack-game/models')
    ? '/tmp/polytrack-game'
    : join(homedir(), 'polytrack-dev', 'local-game-server');

const MODEL_FILES = ['blocks', 'pillar', 'planes', 'road', 'road_wide', 'signs', 'wall_track'];
// config scene names ("Road", "Blocks", …) → glb file
const SCENE_TO_FILE = {
  Road: 'road',
  RoadWide: 'road_wide',
  Planes: 'planes',
  Blocks: 'blocks',
  Pillar: 'pillar',
  Signs: 'signs',
  WallTrack: 'wall_track',
};

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

// DRACOLoader spawns a browser-style Worker from a blob URL; Node's
// worker_threads rejects blob: URLs. We shim globalThis.Worker: on first
// construction with a blob: URL, fetch the blob's source (blob URLs created
// by URL.createObjectURL are readable via fetch in Node ≥18), write it to a
// temp file, and hand a real worker_threads.Worker that file. The draco
// worker protocol (postMessage {type:'init'|'decode', ...}) is compatible.
import { Worker as NodeWorker } from 'node:worker_threads';
import { mkdtempSync, writeFileSync as writeFs } from 'node:fs';
import { tmpdir } from 'node:os';

let draco = null;

class NodeWorkerShim {
  constructor(url) {
    const u = String(url);
    this._queue = [];
    if (u.startsWith('blob:')) {
      const dir = mkdtempSync(join(tmpdir(), 'polyrl-draco-'));
      this._path = join(dir, 'worker.cjs');
      fetch(u)
        .then((r) => r.text())
        .then((src) => {
          // browser-worker → node-worker adapter: the draco worker body uses
          // bare onmessage=/postMessage() globals and CJS require.
          const wrapped = [
            "const { parentPort } = require('node:worker_threads');",
            'globalThis.self = globalThis;',
            'globalThis.postMessage = (m) => parentPort.postMessage(m);',
            'globalThis.onmessage = null;',
            "parentPort.on('message', (data) => globalThis.onmessage?.({ data }));",
            src,
          ].join('\n');
          writeFs(this._path, wrapped);
          this._w = new NodeWorker(this._path);
          this._w.on('message', (m) => this.onmessage?.({ data: m }));
          this._w.on('error', (e) => this.onerror?.(e));
          for (const m of this._queue) this._w.postMessage(m);
          this._queue.length = 0;
        })
        .catch((e) => console.error('draco worker setup failed:', e));
    } else {
      this._w = new NodeWorker(u);
      this._w.on('message', (m) => this.onmessage?.({ data: m }));
    }
  }
  postMessage(m) {
    if (this._w) this._w.postMessage(m);
    else this._queue.push(m);
  }
  terminate() {
    this._w?.terminate();
  }
}
globalThis.Worker = NodeWorkerShim;

// three's FileLoader uses fetch, which Node won't do for file:// URLs or bare
// paths. Patch it to read from the filesystem; resolve relative paths against
// the current working directory of this script invocation (we only ever load
// the draco decoder files this way).
let fileLoaderBaseDir = null;
THREE.FileLoader.prototype.load = function (url, onLoad, _onProgress, onError) {
  try {
    let p = String(url);
    if (p.startsWith('file://')) p = fileURLToPath(p);
    else if (!p.startsWith('/')) p = join(fileLoaderBaseDir ?? process.cwd(), p);
    const buf = readFileSync(p);
    const response =
      this.responseType === 'arraybuffer'
        ? buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
        : buf.toString('utf8');
    queueMicrotask(() => onLoad(response));
  } catch (e) {
    if (onError) queueMicrotask(() => onError(e));
    else throw e;
  }
};

function loadGlb(path) {
  const data = readFileSync(path);
  const loader = new GLTFLoader().setDRACOLoader(draco);
  return new Promise((res, rej) =>
    loader.parse(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength), '', res, rej),
  );
}

// Replicates the game's per-mesh extraction `o(e, colorOverrides)`:
// clone geometry, attach a color attribute (we don't care about colors for
// physics, but the merge requires consistent attributes → keep them).
function meshToGeometry(mesh) {
  const geom = mesh.geometry.clone();
  const n = geom.attributes.position.array.length;
  const colors = new Float32Array(n);
  for (let i = 0; i < n; i += 3) {
    colors[i] = 1;
    colors[i + 1] = 1;
    colors[i + 2] = 1;
  }
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return geom;
}

// Replicates the game's `s(sceneName, meshName, flipX, flipY, flipZ, offset, scale, quaternion, colors)`:
// fetch mesh (or children) from the loaded scene, apply transforms.
function extractPartGeometries(scene, meshName, opts = {}) {
  const { flipX = false, flipY = false, flipZ = false, offset = null, scale = null, quaternion = null } = opts;
  const obj = scene.getObjectByName(meshName);
  if (!obj) throw new Error(`Mesh "${meshName}" missing`);
  let geoms;
  if (!obj.children || obj.children.length === 0) {
    const g = meshToGeometry(obj);
    obj.updateMatrixWorld(true);
    g.applyMatrix4(obj.matrix);
    geoms = [g];
  } else {
    geoms = obj.children.map((c) => meshToGeometry(c));
    obj.updateMatrixWorld(true);
    for (const g of geoms) g.applyMatrix4(obj.matrix);
  }

  // max Y over positions (used when flipY + offset)
  let maxY = -Infinity;
  if (flipY) {
    for (const g of geoms) {
      const a = g.attributes.position.array;
      for (let i = 0; i < a.length; i += 3) maxY = Math.max(maxY, a[i + 1]);
    }
  }

  for (const g of geoms) {
    if (scale) g.scale(scale.x, scale.y, scale.z);
    if (quaternion) g.applyQuaternion(quaternion);
    g.scale(flipX ? -1 : 1, flipY ? -1 : 1, flipZ ? -1 : 1);
    if (flipX || flipY || flipZ) {
      // flip triangle winding (the game does this to preserve normals' facing)
      const idx = g.index;
      if (idx) {
        for (let i = 0; i < idx.count; i += 3) {
          const a = idx.getX(i);
          const b = idx.getX(i + 1);
          const c = idx.getX(i + 2);
          idx.setXYZ(i, a, c, b);
        }
      } else {
        const pos = g.attributes.position;
        for (let i = 0; i < pos.count; i += 3) {
          const ax = pos.getX(i), ay = pos.getY(i), az = pos.getZ(i);
          const bx = pos.getX(i + 1), by = pos.getY(i + 1), bz = pos.getZ(i + 1);
          const cx = pos.getX(i + 2), cy = pos.getY(i + 2), cz = pos.getZ(i + 2);
          pos.setXYZ(i, ax, ay, az);
          pos.setXYZ(i + 1, cx, cy, cz);
          pos.setXYZ(i + 2, bx, by, bz);
        }
        // color attribute must follow the same swap to keep merge consistent
        const col = g.attributes.color;
        for (let i = 0; i < col.count; i += 3) {
          const bx = col.getX(i + 1), by = col.getY(i + 1), bz = col.getZ(i + 1);
          const cx = col.getX(i + 2), cy = col.getY(i + 2), cz = col.getZ(i + 2);
          col.setXYZ(i + 1, cx, cy, cz);
          col.setXYZ(i + 2, bx, by, bz);
        }
      }
    }
    if (offset) {
      if (flipY) g.translate(offset.x, offset.y + maxY, offset.z);
      else g.translate(offset.x, offset.y, offset.z);
    } else if (flipY) {
      g.translate(0, maxY, 0);
    }
  }
  return geoms;
}

async function main() {
  const configs = JSON.parse(readFileSync(join(VENDOR, 'part_configs.json'), 'utf8'));
  const dracoDir = join(gameDir, 'lib', 'draco');
  fileLoaderBaseDir = dracoDir;
  draco = new DRACOLoader().setDecoderPath(dracoDir + '/');
  // wasm decoder inline so the worker doesn't fetch it over http
  draco.setDecoderConfig({ wasmBinary: readFileSync(join(dracoDir, 'draco_decoder.wasm')) });

  // ---- load all GLB scenes -------------------------------------------------
  const scenes = new Map(); // sceneName -> THREE.Group (scene)
  for (const [sceneName, file] of Object.entries(SCENE_TO_FILE)) {
    const p = join(gameDir, 'models', `${file}.glb`);
    const gltf = await loadGlb(p);
    scenes.set(sceneName, gltf.scene);
  }
  const carGltf = await loadGlb(join(gameDir, 'models', 'car.glb'));

  // ---- car collision shape -------------------------------------------------
  const collisionMesh = carGltf.scene.getObjectByName('Collision');
  if (!collisionMesh) throw new Error('car.glb has no Collision mesh');
  const carGeom = collisionMesh.geometry.clone();
  collisionMesh.updateMatrixWorld(true);
  // game: e.geometry.toNonIndexed() then raw position array (no matrix apply
  // in nt() — the loader path for car models applies nothing; vertices are
  // local-space)
  const carNonIndexed = carGeom.index ? carGeom.toNonIndexed() : carGeom;
  const carVerts = new Float32Array(carNonIndexed.attributes.position.array);
  const carChecksum = sha256(Buffer.from(carVerts.buffer, carVerts.byteOffset, carVerts.byteLength));
  const EXPECTED_CAR = 'c12d4421883ae86b922550f98efea3cf5e6b9c168436f9f5c989ad33a41ce50b';
  console.log(`car collision: ${carVerts.length / 3} verts, sha256=${carChecksum}`);
  console.log(carChecksum === EXPECTED_CAR ? '  ✓ matches game checksum' : `  ✗ MISMATCH (expected ${EXPECTED_CAR})`);

  // ---- per-part physics shapes ----------------------------------------------
  const parts = [];
  let mismatches = 0;
  for (const part of configs.parts) {
    // game merges geometries of ALL themes' first color variant? No: it merges
    // per color-theme for visuals, and takes l ??= r from the FIRST theme's
    // merged geometry (l is set once: `l ??= r`). Themes share geometry, so
    // theme 0 suffices for physics.
    const geoms = [];
    for (const [sceneName, meshName, opts] of part.models) {
      const scene = scenes.get(sceneName);
      if (!scene) throw new Error(`no scene ${sceneName} for part ${part.type}`);
      for (const g of extractPartGeometries(scene, meshName, opts ?? {})) geoms.push(g);
    }
    const merged = BufferGeometryUtils.mergeGeometries(geoms, true).toNonIndexed();
    const verts = new Float32Array(merged.attributes.position.array);
    const sum = sha256(Buffer.from(verts.buffer, verts.byteOffset, verts.byteLength));
    const ok = sum === part.checksum;
    if (!ok) mismatches++;
    parts.push({
      typeId: part.typeId,
      type: part.type,
      detector: part.detector,
      startOffset: part.startOffset,
      checksum: part.checksum,
      checksumOk: ok,
      vertices: Array.from(verts, (v) => Math.round(v * 1e6) / 1e6),
    });
  }
  console.log(`parts: ${parts.length}, checksum mismatches: ${mismatches}`);
  if (mismatches) {
    for (const p of parts.filter((p) => !p.checksumOk).slice(0, 10)) console.log(`  ✗ ${p.type} (${p.typeId})`);
  }

  const out = {
    gameVersion: configs.gameVersion,
    car: {
      collisionShapeVertices: Array.from(carVerts, (v) => Math.round(v * 1e6) / 1e6),
      checksum: carChecksum,
      checksumOk: carChecksum === EXPECTED_CAR,
      massOffset: 0.6,
      suspensionResetLengthFront: null, // filled in from bundle consts later
      suspensionResetLengthRear: null,
    },
    parts,
  };
  writeFileSync(join(VENDOR, 'physics_assets.json'), JSON.stringify(out));
  console.log(`wrote ${join(VENDOR, 'physics_assets.json')} (${(JSON.stringify(out).length / 1e6).toFixed(1)} MB)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
