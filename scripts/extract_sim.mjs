#!/usr/bin/env node
// Vendor the simulation worker + physics wasm + tracks from a local game copy
// into node/vendor/, and record a manifest with per-file sha256 hashes.
//
// Usage: node scripts/extract_sim.mjs [sourceDir]
//   default sourceDir: ~/polytrack-dev/local-game-server

import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';

const src = resolve(process.argv[2] ?? join(homedir(), 'polytrack-dev', 'local-game-server'));
const dst = resolve(new URL('..', import.meta.url).pathname, 'node', 'vendor');

const FILES = [
  'simulation_worker.bundle.js',
  'lib/polytrack_physics.js',
  'polytrack_physics.wasm',
  'package.json', // carries the game version string
];

if (!existsSync(src)) {
  console.error(`source not found: ${src}`);
  console.error('pass a game directory: node scripts/extract_sim.mjs /path/to/extracted-polytrack');
  process.exit(1);
}

mkdirSync(dst, { recursive: true });

const hashes = {};
for (const f of FILES) {
  const s = join(src, f);
  if (!existsSync(s)) {
    console.error(`missing: ${s}`);
    process.exit(1);
  }
  const buf = readFileSync(s);
  hashes[f] = { sha256: createHash('sha256').update(buf).digest('hex'), bytes: buf.length };
  const out = join(dst, f);
  mkdirSync(resolve(out, '..'), { recursive: true });
  writeFileSync(out, buf);
}

// tracks directory (official + community) — needed for headless runs of built-in tracks
const tracksSrc = join(src, 'tracks');
if (existsSync(tracksSrc)) {
  const tracksDst = join(dst, 'tracks');
  rmSync(tracksDst, { recursive: true, force: true });
  cpSync(tracksSrc, tracksDst, { recursive: true });
}

let gameVersion = 'unknown';
try {
  gameVersion = JSON.parse(readFileSync(join(src, 'package.json'), 'utf8')).version ?? 'unknown';
} catch {}

const manifest = {
  source: src,
  gameVersion,
  extractedAt: new Date().toISOString(),
  files: hashes,
};
writeFileSync(join(dst, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');

console.log(`vendored ${FILES.length} files + tracks/ from ${src}`);
console.log(`game version: ${gameVersion}`);
for (const [f, h] of Object.entries(hashes)) {
  console.log(`  ${f}  ${h.bytes} B  sha256:${h.sha256.slice(0, 16)}…`);
}
