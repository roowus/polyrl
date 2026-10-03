// scan deeper into the summer1 leaderboard for re-simming laps; for each that
// finishes, report its speed through the climb region (path 60-70%)
import { SimWorker, Ki, loadPhysicsAssets, initPayload, loadTrackSaveString } from '/Users/rewis/projects/polyrl/node/sim_host.mjs';
import { parseExportString } from '/Users/rewis/projects/polyrl/node/track_codec.mjs';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const ORIGIN = 'https://app-polytrack-desktop.kodub.com';
const raw = readFileSync('/Users/rewis/projects/polyrl/node/vendor/tracks/official/summer1.track','utf8').trim();
const { payload } = parseExportString(raw);
const tid = createHash('sha256').update(Buffer.from(payload)).digest('hex');

async function api(p){ const r = await fetch(`https://vps.kodub.com/v6${p}`,{headers:{Origin:ORIGIN}}); if(!r.ok) throw new Error(p+' '+r.status); return r.json(); }

// fetch entries 30..80
const lb = await api(`/leaderboard?version=0.6.3&trackId=${tid}&skip=30&amount=50&onlyVerified=false`);
const recs = [];
for (let c = 0; c < lb.entries.length; c += 10) {
  const ids = lb.entries.slice(c, c + 10).map(e => e.id).join(',');
  recs.push(...(await api(`/recordings?version=0.6.3&ids=${ids}`)));
}
console.log('fetched', recs.length, 'recordings (ranks 31-80)');

const assets = loadPhysicsAssets();
const trackData = loadTrackSaveString('summer1');
let kept = 0;
for (let i = 0; i < recs.length; i++) {
  const rec = recs[i]; if (!rec?.recording) continue;
  const claimed = rec.frames ?? lb.entries[i].frames;
  const w = new SimWorker(40000+i);
  await w.waitReady();
  await w.send({ messageType: Ki.Init, ...initPayload(assets) });
  let fin = null, lastFrames = 0;
  w.onUpdate((bufs) => { for (const b of bufs) { const u8 = new Uint8Array(b); lastFrames = u8[4]|u8[5]<<8|u8[6]<<16; if (u8[11]&2) fin = u8[12]|u8[13]<<8|u8[14]<<16; } });
  await w.send({ messageType: Ki.CreateCar, mountainVertices: new Float32Array(0), mountainOffset:{x:0,y:0,z:0}, trackData, carId: 1, carRecording: rec.recording });
  await w.send({ messageType: Ki.StartCar, carId: 1, targetSimulationTimeFrames: claimed + 2000 });
  const t0 = performance.now();
  await new Promise(r => { const iv = setInterval(()=>{ if (fin!=null || lastFrames >= claimed+1999 || performance.now()-t0 > 30000) { clearInterval(iv); r(); } }, 10); });
  if (fin != null) { kept++; console.log(`  ✓ rank ${31+i} ${lb.entries[i].nickname ?? '?'}: finishes ${fin} (${(fin/1000).toFixed(2)}s)`); }
  await w.terminate();
}
console.log('kept', kept, '/', recs.length, 'that re-sim');
process.exit(0);
