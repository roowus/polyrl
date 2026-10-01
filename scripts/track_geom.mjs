// Dump a track's placed parts + derived geometry as JSON on stdout.
// Usage: node scripts/track_geom.mjs <trackName>
//
// Parses the track payload (wo format, decoded via track_codec) into placed
// parts: {typeId, x, y, z, rotation, rotationAxis, checkpointOrder, startOrder}
// then resolves world-space gate centers using part_configs.json detector
// boxes, and emits start transform + gates + a centerline polyline.

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { parseExportString } from '../node/track_codec.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const VENDOR = join(HERE, '..', 'node', 'vendor');

const trackName = process.argv[2] ?? 'summer1';
const raw = readFileSync(join(VENDOR, 'tracks', 'official', `${trackName}.track`), 'utf8').trim();
const { name, payload } = parseExportString(raw);

const partConfigs = JSON.parse(readFileSync(join(VENDOR, 'part_configs.json'), 'utf8'));
const byTypeId = new Map(partConfigs.parts.map((p) => [p.typeId, p]));

// ---- parse payload (wo layout) ----------------------------------------------
const e = payload;
let i = 0;
const theme = e[i++];
const _sun = e[i++];
const offX = e[i] | (e[i + 1] << 8) | (e[i + 2] << 16) | (e[i + 3] << 24); i += 4;
const offY = e[i] | (e[i + 1] << 8) | (e[i + 2] << 16) | (e[i + 3] << 24); i += 4;
const offZ = e[i] | (e[i + 1] << 8) | (e[i + 2] << 16) | (e[i + 3] << 24); i += 4;
const h = 3 & e[i], c = (e[i] >> 2) & 3, A = (e[i] >> 4) & 3; i++;

const placed = [];
while (i < e.length) {
  const typeId = e[i++];
  const count = e[i] | (e[i + 1] << 8) | (e[i + 2] << 16) | (e[i + 3] << 24); i += 4;
  for (let k = 0; k < count; k++) {
    let x = 0; for (let t = 0; t < h; t++) x |= e[i + t] << (8 * t); x += offX; i += h;
    let y = 0; for (let t = 0; t < c; t++) y |= e[i + t] << (8 * t); y += offY; i += c;
    let z = 0; for (let t = 0; t < A; t++) z |= e[i + t] << (8 * t); z += offZ; i += A;
    const u = e[i++];
    const rotation = 3 & u;
    const rotationAxis = (u >> 2) & 7;
    const _color = e[i++];
    // checkpointOrder / startOrder presence is part-type dependent; the worker
    // checks uo/fo lists. We re-derive from configs: detector → checkpointOrder,
    // startOffset → startOrder.
    const cfg = byTypeId.get(typeId);
    let checkpointOrder = null;
    let startOrder = null;
    if (cfg?.detector && cfg.detector.type === 'Checkpoint') {
      checkpointOrder = e[i] | (e[i + 1] << 8); i += 2;
    }
    if (cfg?.startOffset) {
      startOrder = e[i] | (e[i + 1] << 8) | (e[i + 2] << 16) | (e[i + 3] << 24); i += 4;
    }
    placed.push({ typeId, x, y, z, rotation, rotationAxis, checkpointOrder, startOrder });
  }
}

// ---- world transforms --------------------------------------------------------
const PART_SIZE = 5.0; // main.bundle.js: partSize = 5
// rotationAxis: 0=Y+ 1=Y- 2=X+ 3=X- 4=Z+ 5=Z- (from the Yh table); rotation is
// 0..3 quarter-turns around that axis.
const AXIS_VECS = [
  [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
];
function quatFromAxisRotation(rotationAxis, rotation) {
  const [ax, ay, az] = AXIS_VECS[rotationAxis] ?? [0, 1, 0];
  const angle = (rotation * Math.PI) / 2;
  const s = Math.sin(angle / 2);
  return [ax * s, ay * s, az * s, Math.cos(angle / 2)];
}
function rotateVec(v, q) {
  const [x, y, z] = v;
  const [qx, qy, qz, qw] = q;
  // q * v * q^-1
  const uvx = qy * z - qz * y, uvy = qz * x - qx * z, uvz = qx * y - qy * x;
  const uuvx = qy * uvz - qz * uvy, uuvy = qz * uvx - qx * uvz, uuvz = qx * uvy - qy * uvx;
  return [
    x + 2 * (qw * uvx + uuvx),
    y + 2 * (qw * uvy + uuvy),
    z + 2 * (qw * uvz + uuvz),
  ];
}
function quatMul(a, b) {
  const [ax, ay, az, aw] = a, [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

const gates = [];
let start = null;
for (const p of placed) {
  const cfg = byTypeId.get(p.typeId);
  if (!cfg) continue;
  const q = quatFromAxisRotation(p.rotationAxis, p.rotation);
  const base = [p.x * PART_SIZE, p.y * PART_SIZE, p.z * PART_SIZE];

  if (cfg.detector) {
    const dc = rotateVec(cfg.detector.center, q);
    const ds = rotateVec(cfg.detector.size, q).map(Math.abs);
    gates.push({
      order: p.checkpointOrder ?? (cfg.detector.type === 'Finish' ? 9999 : -1),
      isFinish: cfg.detector.type === 'Finish',
      center: [base[0] + dc[0], base[1] + dc[1], base[2] + dc[2]],
      size: ds,
    });
  }
  if (cfg.startOffset && p.startOrder != null) {
    // Game's getStartTransform: spawn = partPos*partSize + startOffset applied
    // with quaternion = partRotation ∘ RotY(180°). The extra 180° flip is what
    // points the car down-track (and moves the spawn ahead of the line).
    const flip = [0, 1, 0, 0]; // RotY(180°): (x,y,z,w) = (0, sin90, 0, cos90)
    const q2 = quatMul(q, flip);
    const so = rotateVec([cfg.startOffset.x, cfg.startOffset.y, cfg.startOffset.z], q2);
    if (!start || p.startOrder >= start.startOrder) {
      start = {
        startOrder: p.startOrder,
        position: [base[0] + so[0], base[1] + so[1], base[2] + so[2]],
        quaternion: q2,
      };
    }
  }
}

gates.sort((a, b) => a.order - b.order);
// Centerline origin = the car spawn point (start part + rotated startOffset,
// matching the game's getStartTransform incl. the 180° flip). Progress=0 at
// spawn; the demo trajectory maps 0→100% across spawn…finish.
const centerline = [start?.position, ...gates.map((g) => g.center)].filter(Boolean);

const out = {
  trackName: name,
  trackHash: 'sha256:' + createHash('sha256').update(raw).digest('hex'),
  partCount: placed.length,
  start,
  gates,
  centerline,
};
console.log(JSON.stringify(out));
