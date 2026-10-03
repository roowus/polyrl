// Port of the game's createMountainVertices (main.bundle.js) — deterministic
// backdrop terrain ring around the track. The car physics CAN land on it, so
// for big/jumpy custom tracks (TOTW) it matters: an empty stub lets the car
// fall through where the real game has a floor.
//
// PRNG: fixed 128-float lookup table, index wraps (class `h` in the bundle).

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const PRNG_TABLE = JSON.parse(
  readFileSync(join(HERE, 'vendor', 'prng_table.json'), 'utf8'),
);

class TableRng {
  constructor(seed = 0) {
    this.i = seed % PRNG_TABLE.length;
  }
  next() {
    const v = PRNG_TABLE[this.i];
    this.i = (this.i + 1) % PRNG_TABLE.length;
    return v;
  }
}

const PART_SIZE = 5.0;

/**
 * bounds: {min:{x,y}, max:{x,y}} in TILE units (game uses x and z-as-"y" here).
 * Returns { vertices: Float32Array, offset: {x,y,z} } — matches the game's
 * {vertices, offset} shape for CreateCar/Verify.
 */
export function createMountainVertices(bounds) {
  const t = new TableRng(0);
  const n = Math.max(
    200,
    160 +
      Math.max(
        (Math.abs(bounds.max.x - bounds.min.x) * PART_SIZE) / 2 * Math.SQRT2,
        (Math.abs(bounds.max.y - bounds.min.y) * PART_SIZE) / 2 * Math.SQRT2,
      ),
  );
  const cx = (bounds.min.x + (bounds.max.x - bounds.min.x) / 2) * PART_SIZE;
  const cz = (bounds.min.y + (bounds.max.y - bounds.min.y) / 2) * PART_SIZE;
  if (n > 4500) return { vertices: new Float32Array(0), offset: { x: 0, y: 0, z: 0 } };

  const r = Math.floor(n / 10);
  const a = [];
  for (let i = 0; i < r; i++) {
    const row = [];
    for (let k = 0; k < 8; k++) {
      if (k === 0 || k === 7 || (k === 1 && t.next() < 0.5)) row.push(0);
      else row.push(t.next());
    }
    a.push(row);
  }
  const s = 100;
  const l = [];
  for (let i = 0; i < a.length; i++) {
    const t0 = (i / a.length) * Math.PI * 2;
    const t1 = ((i + 1) / a.length) * Math.PI * 2;
    const row = a[i];
    const nxt = i + 1 < a.length ? a[i + 1] : a[0];
    for (let k = 0; k < row.length - 1; k++) {
      const r0 = n + 100 * i, r1 = n + 100 * (i + 1);
      l.push(Math.cos(t0) * r0, row[k] * s, Math.sin(t0) * r0);
      l.push(Math.cos(t1) * r0, nxt[k] * s, Math.sin(t1) * r0);
      l.push(Math.cos(t1) * r1, nxt[k + 1] * s, Math.sin(t1) * r1);
      l.push(Math.cos(t0) * r0, row[k] * s, Math.sin(t0) * r0);
      l.push(Math.cos(t1) * r1, nxt[k + 1] * s, Math.sin(t1) * r1);
      l.push(Math.cos(t0) * r1, row[k + 1] * s, Math.sin(t0) * r1);
    }
  }
  return { vertices: new Float32Array(l), offset: { x: cx, y: 0, z: cz } };
}
