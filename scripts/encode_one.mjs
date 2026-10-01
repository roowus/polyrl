// Encode a recording from JSON toggle lists on stdin → game-format string on
// stdout. Mirrors brain/codec.py (cross-language conformance test).
// Usage: echo '{"up":[0,250]}' | node scripts/encode_one.mjs

import zlib from 'node:zlib';

const CH = ['up', 'right', 'down', 'left', 'reset'];
function u24(n) {
  return [n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff];
}

let input = '';
process.stdin.on('data', (d) => (input += d));
process.stdin.on('end', () => {
  const rec = JSON.parse(input);
  const bytes = [];
  for (const ch of CH) {
    const t = rec[ch] ?? [];
    bytes.push(...u24(t.length));
    for (const f of t) bytes.push(...u24(f));
  }
  const comp = zlib.deflateSync(Buffer.from(bytes), { level: 9 });
  process.stdout.write(comp.toString('base64url'));
});
