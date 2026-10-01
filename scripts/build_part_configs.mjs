#!/usr/bin/env node
// One-shot codegen: turn the game's minified part-configuration table
// (extracted from main.bundle.js) into clean JSON.
//
// Inputs (produced by extraction snippets):
//   /tmp/part_configs_raw.js   — the `[new d(...), ...]` array literal
//   /tmp/part_colors_raw.js    — the `c=[...]` color themes array literal
//   /tmp/part_type_enum.json   — full PartType name→id map (worker enum)
//
// Output: node/vendor/part_configs.json
//
// Config ctor args: (checksum, category, type, models, colors, tiles, detector?, startOffset?)
// with proxies standing in for the minified enums (r.A.*, s.A.*, a.A.*, o.A.*, i.Pq0).

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';
import { Vector3, Quaternion, Euler } from 'three';

const HERE = dirname(fileURLToPath(import.meta.url));

const typeEnum = JSON.parse(readFileSync('/tmp/part_type_enum.json', 'utf8'));

const raw = readFileSync('/tmp/part_configs_raw.js', 'utf8');
const colorsRaw = readFileSync('/tmp/part_colors_raw.js', 'utf8');

// name-keeping proxy: r.A.Special → 'Special'
const nameProxy = () => new Proxy({}, { get: () => new Proxy({}, { get: (_2, name) => name }) });

// Evaluate the colors array (c=[...]) first — configs reference it by name.
const colorsSandbox = { a: nameProxy() };
vm.createContext(colorsSandbox);
const themes = vm.runInContext(`(${colorsRaw.replace(/^c=/, '')})`, colorsSandbox);

const parts = [];
function d(checksum, category, type, models, colors, tiles, detector = null, startOffset = null) {
  if (!(type in typeEnum)) throw new Error(`unknown part type name: ${type}`);
  parts.push({
    checksum,
    category,
    type,
    typeId: typeEnum[type],
    models, // [[sceneFileBaseName, meshName, opts?], ...]
    colors, // [{id: themeName, colors: {...}}, ...]
    tiles,
    detector: detector ? { type: detector.type, center: detector.center, size: detector.size } : null,
    startOffset, // {x,y,z} | null
  });
}

const sandbox = {
  d,
  r: nameProxy(),
  s: nameProxy(),
  a: nameProxy(),
  o: { A: { Checkpoint: 'Checkpoint', Finish: 'Finish' } },
  i: {
    Pq0: Vector3,
    PTz: Quaternion,
    O9p: Euler,
  },
  c: themes,
  // h = c.concat([9 custom BlockSurface themes]) in the bundle
  h: themes.concat(
    ['#131313', '#501b1b', '#7f4d2b', '#93862d', '#2a5e30', '#236363', '#20244b', '#592759', '#302318'].map(
      (hex, k) => ({ id: `Custom${k}`, colors: { BlockSurface: hex } }),
    ),
  ),
};
vm.createContext(sandbox);
vm.runInContext(`(${raw})`, sandbox);

const out = { gameVersion: '0.6.3', themes, parts };
writeFileSync(join(HERE, '..', 'node', 'vendor', 'part_configs.json'), JSON.stringify(out, null, 2));
console.log(`parts: ${parts.length}, themes: ${themes.length}`);
const missing = parts.filter((p) => !(p.type in typeEnum));
if (missing.length) throw new Error('unresolved types: ' + missing.map((p) => p.type).join(','));
console.log('ok');
