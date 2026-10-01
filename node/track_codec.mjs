// PolyTrack track string codec — ports of the game's `Wa`/`za` (base-N),
// `Fa` (bit reader), `Ua` (bit writer), and the export/save wrappers.
//
// Alphabet (from the bundle's `Ra` table + `Da` charset):
//   'A'-'Z' → 0-25, 'a'-'z' → 26-51, '0'-'9' → 52-61, then 2 more chars
//   for 62/63 (see Da below). 5-bit groups for values 30/31 escape hatches,
//   6-bit otherwise (value < 30 → 6 bits; else 5 bits with r & 31).

import zlib from 'node:zlib';

// Da (value→char) is exactly 62 entries in the bundle: A-Z a-z 0-9.
// Values 30/31 are never emitted as 6-bit symbols — they switch to 5-bit
// mode, and 5-bit mode only emits values 0-31 (all < 62 representable).
// So 62/63 never need a character. Verified against the bundle's Da literal.
export const DA = [
  ...'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
  ...'abcdefghijklmnopqrstuvwxyz',
  ...'0123456789',
];
const RA = (() => {
  const t = new Array(128).fill(-1);
  for (let v = 0; v < DA.length; v++) t[DA[v].charCodeAt(0)] = v;
  return t;
})();

// ---- bit-level decode (exact port of Wa/Ua) --------------------------------
function Ua(t, e, i, r, s) {
  const n = Math.floor(e / 8);
  for (; n >= t.length; ) t.push(0);
  const a = e - 8 * n;
  t[n] |= (r << a) & 255;
  if (a > 8 - i && !s) {
    const e2 = n + 1;
    if (e2 >= t.length) t.push(0);
    t[e2] |= r >> (8 - a);
  }
}

export function decodeBits(str) {
  let e = 0;
  const i = [];
  const r = str.length;
  for (let s = 0; s < r; s++) {
    const n = str.charCodeAt(s);
    if (n >= RA.length) return null;
    const a = RA[n];
    if (a === -1) return null;
    if (30 & ~a) {
      Ua(i, e, 6, a, s === r - 1);
      e += 6;
    } else {
      Ua(i, e, 5, a, s === r - 1);
      e += 5;
    }
  }
  return new Uint8Array(i);
}

// ---- bit-level encode (exact port of za/Fa) --------------------------------
function Fa(t, e) {
  // read 6 bits at bit-offset e (with the game's boundary semantics)
  if (e >= 8 * t.length) throw new Error('Out of range');
  const i = Math.floor(e / 8);
  const r = t[i];
  const s = e - 8 * i;
  if (s <= 2 || i >= t.length - 1) return (r & (63 << s)) >>> s;
  return ((r & (63 << s)) >>> s) | ((t[i + 1] & (63 >>> (8 - s))) << (8 - s));
}

export function encodeBits(bytes) {
  let e = 0;
  let out = '';
  for (; e < 8 * bytes.length; ) {
    const r = Fa(bytes, e);
    let s;
    if (30 & ~r) {
      s = r;
      e += 6;
    } else {
      s = 31 & r;
      e += 5;
    }
    out += DA[s];
  }
  return out;
}

// ---- deflate wrappers --------------------------------------------------------
// The game uses pako Deflate with windowBits 9 (inner) then 15 (outer).
// zlib's deflate with windowBits=9 produces a header pako accepts; pako's
// Inflate (default) accepts any zlib stream. For DECODING we use inflateSync
// (auto-detects zlib header). For ENCODING we match windowBits.

function inflate(buf) {
  return zlib.inflateSync(Buffer.from(buf.buffer, buf.byteOffset, buf.length));
}
function deflate(buf, windowBits) {
  return zlib.deflateSync(Buffer.from(buf.buffer, buf.byteOffset, buf.length), {
    level: 9,
    windowBits,
    memLevel: 9,
  });
}

/**
 * Convert a `.track` file's export string ("PolyTrack24…") to the save string
 * the simulation worker's CreateCar/Verify expects (`fromSaveString` input).
 *
 * Export format:  'PolyTrack2' + za(deflate15( za(deflate9(header ‖ payload)) ))
 * Save format:    za(deflate15( za(deflate9(payload)) ))
 * where header = [nameLen u8, name utf8, authorLen u8, author utf8?,
 *                 lastModified flag u8, (epoch u32le)?]
 * and payload is the binary track body parsed by `wo`.
 *
 * So conversion = decode fully, then re-encode just the payload.
 */
export function exportToSaveString(exportStr) {
  const s = exportStr.replace(/\s+/g, '');
  if (!s.startsWith('PolyTrack2')) throw new Error('not a PolyTrack2 export string');
  const outer = decodeBits(s.substring(10));
  if (!outer) throw new Error('outer decode failed');
  const innerStr = inflate(outer).toString('utf8'); // pako to:'string'
  const inner = decodeBits(innerStr);
  if (!inner) throw new Error('inner decode failed');
  const full = inflate(inner);
  // strip the header: nameLen + name + authorLen(+author) + lastModified(1+4?)
  let o = 0;
  const nameLen = full[o++];
  o += nameLen;
  const authorLen = full[o++];
  o += authorLen;
  const lmFlag = full[o++];
  if (lmFlag === 1) o += 4;
  else if (lmFlag !== 0) throw new Error(`bad lastModified flag ${lmFlag}`);
  const payload = full.subarray(o);
  // re-encode as save string: za(deflate15( za(deflate9(payload)) ))
  const innerEnc = encodeBits(new Uint8Array(deflate(payload, 9)));
  return encodeBits(new Uint8Array(deflate(Buffer.from(innerEnc, 'utf8'), 15)));
}

/** Decode an export string into { metadata, payload } for inspection. */
export function parseExportString(exportStr) {
  const s = exportStr.replace(/\s+/g, '');
  if (!s.startsWith('PolyTrack2')) throw new Error('not a PolyTrack2 export string');
  const outer = decodeBits(s.substring(10));
  const innerStr = inflate(outer).toString('utf8');
  const inner = decodeBits(innerStr);
  const full = inflate(inner);
  let o = 0;
  const nameLen = full[o++];
  const name = full.subarray(o, o + nameLen).toString('utf8');
  o += nameLen;
  const authorLen = full[o++];
  const author = authorLen > 0 ? full.subarray(o, o + authorLen).toString('utf8') : null;
  o += authorLen;
  const lmFlag = full[o++];
  const lastModified = lmFlag === 1 ? new Date(1000 * full.readUInt32LE(o)) : null;
  if (lmFlag === 1) o += 4;
  return { name, author, lastModified, payload: full.subarray(o) };
}
