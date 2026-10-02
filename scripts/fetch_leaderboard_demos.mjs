#!/usr/bin/env node
// Fetch leaderboard recordings for a track and keep only those that re-sim to
// a FINISH on the vendored physics (recordings from older physics builds won't
// — they decoded fine but the car dies mid-track; filter them out here).
//
// Usage: node scripts/fetch_leaderboard_demos.mjs <track> [amount] [out.json]
//   track   official track name (summer1)
//   amount  how many top entries to try (default 30)
//   out     output JSON (default fixtures/leaderboard_<track>.json)
//
// Writes: [{rank, frames, recording, verifiedState, simFinishFrames, lapSeconds}]

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { parseExportString } from '../node/track_codec.mjs';
import { SimWorker, Ki, loadPhysicsAssets, initPayload, loadTrackSaveString } from '../node/sim_host.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..');
const ORIGIN = 'https://app-polytrack-desktop.kodub.com';
const API = 'https://vps.kodub.com/v6';
const VERSION = '0.6.3';

const track = process.argv[2] ?? 'summer1';
const AMOUNT = parseInt(process.argv[3] ?? '30', 10);
const outPath = process.argv[4] ?? join(REPO, 'fixtures', `leaderboard_${track}.json`);

// trackId = sha256 of the raw track payload (the game's getId())
function trackId(name) {
  const raw = readFileSync(join(REPO, 'node', 'vendor', 'tracks', 'official', `${name}.track`), 'utf8').trim();
  const { payload } = parseExportString(raw);
  return createHash('sha256').update(Buffer.from(payload)).digest('hex');
}

async function api(path) {
  const res = await fetch(`https://vps.kodub.com/v6${path}`, { headers: { Origin: ORIGIN } });
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
  return res.json();
}

const tid = trackId(track);
console.log(`[lb] ${track} trackId=${tid.slice(0, 16)}…`);

const lb = await api(`/leaderboard?version=${VERSION}&trackId=${tid}&skip=0&amount=${AMOUNT}&onlyVerified=false`);
console.log(`[lb] total entries on leaderboard: ${lb.total}; fetched top ${lb.entries.length}`);
if (!lb.entries.length) {
  console.error('[lb] empty — wrong trackId?');
  process.exit(1);
}

// fetch recordings in chunks (the endpoint 400s on long id lists — the game
// itself only requests ≤10 ghost recordings at a time)
const CHUNK = 10;
const recs = [];
for (let c = 0; c < lb.entries.length; c += CHUNK) {
  const chunk = lb.entries.slice(c, c + CHUNK);
  const ids = chunk.map((e) => e.id).join(',');
  const part = await api(`/recordings?version=${VERSION}&ids=${ids}`);
  recs.push(...part);
}
console.log(`[lb] recordings fetched: ${recs.length}`);

// physics assets + one shared worker for re-sim validation
const assets = loadPhysicsAssets();
const trackData = loadTrackSaveString(track);

const kept = [];
for (let i = 0; i < recs.length; i++) {
  const entry = lb.entries[i];
  const rec = recs[i];
  if (!rec?.recording) continue;
  const claimed = rec.frames ?? entry.frames;
  // re-sim: allow coast margin past the last input (recordings end before the
  // line — the car coasts over). Generous window; verify_one showed +14 for a
  // real lap, physics-drifted ones die entirely.
  const target = claimed + 2000;
  const w = new SimWorker(20_000 + i);
  try {
    await w.waitReady();
    await w.send({ messageType: Ki.Init, ...initPayload(assets) });
    const carId = 1;
    await w.send({
      messageType: Ki.CreateCar,
      mountainVertices: new Float32Array(0),
      mountainOffset: { x: 0, y: 0, z: 0 },
      trackData,
      carId,
      carRecording: rec.recording,
    });
    let finishFrames = null;
    let lastFrames = 0;
    const done = new Promise((resolve) => {
      w.onUpdate((bufs) => {
        for (const b of bufs) {
          const u8 = new Uint8Array(b);
          lastFrames = u8[4] | (u8[5] << 8) | (u8[6] << 16);
          if (u8[11] & 2) {
            finishFrames = u8[12] | (u8[13] << 8) | (u8[14] << 16);
            resolve();
          }
        }
      });
      w.send({ messageType: Ki.StartCar, carId, targetSimulationTimeFrames: target });
      const t0 = performance.now();
      const iv = setInterval(() => {
        if (lastFrames >= target - 1 || performance.now() - t0 > 30000) {
          clearInterval(iv);
          resolve();
        }
      }, 10);
    });
    await done;
    if (finishFrames != null) {
      kept.push({
        rank: i + 1,
        nickname: entry.nickname,
        frames: claimed,
        simFinishFrames: finishFrames,
        lapSeconds: +(finishFrames / 1000).toFixed(3),
        verifiedState: rec.verifiedState,
        recording: rec.recording,
      });
      console.log(`  ✓ rank ${i + 1} ${entry.nickname ?? '?'}: finishes at ${finishFrames} (${(finishFrames / 1000).toFixed(2)}s) — claimed ${claimed}`);
    } else {
      console.log(`  ✗ rank ${i + 1} ${entry.nickname ?? '?'}: does not finish by ${target} (physics drift — skipped)`);
    }
  } finally {
    await w.terminate();
  }
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(kept, null, 2));
console.log(`[lb] kept ${kept.length}/${recs.length} recordings that re-sim → ${outPath}`);
process.exit(0);
