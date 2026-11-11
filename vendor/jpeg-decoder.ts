// Minimal baseline JPEG decoder (sequential DCT, 8-bit samples), standard library only.
//
// What it covers, and why that is enough here: the photos in this competition are
// ordinary libjpeg-style files (SOF0, 8-bit, 1 or 3 components, sampling factors 1 or 2
// per axis, no restart markers in practice). This decoder handles exactly that family:
//   - baseline sequential DCT (SOF0), 8-bit samples
//   - 1 component (grayscale) or 3 components (YCbCr)
//   - sampling factors 1x1, 2x1, 1x2, 2x2 per component
//   - standard and custom quantisation and Huffman tables
//   - restart markers (DRI), if present, by resetting the DC predictors at each interval
// It refuses progressive (SOF2), 12-bit and 4-component files instead of guessing: the
// caller falls back to the default guess for a photo it cannot decode.
//
// It is written from the JPEG specification (ITU T.81), not copied from a library, so the
// repository carries no third-party code here beyond the kit's own agent SDK.

export interface DecodedImage {
  width: number;
  height: number;
  /** RGB bytes, row-major, width * height * 3. */
  data: Uint8Array;
}

/** The zigzag scan order: index k of the zigzag sequence sits at natural position ZIGZAG[k]. */
const ZIGZAG: readonly number[] = [
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
  12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/** cos((2x + 1) * u * PI / 16), x and u in 0..7. */
const COS: number[][] = Array.from({ length: 8 }, (_, x) =>
  Array.from({ length: 8 }, (_, u) => Math.cos(((2 * x + 1) * u * Math.PI) / 16)),
);
/** C(u) in the IDCT sum: 1 / sqrt(2) for u = 0, else 1. */
const CU: readonly number[] = [1 / Math.SQRT2, 1, 1, 1, 1, 1, 1, 1];

/** One Huffman table, as the DHT segment stores it: 16 counts, then the symbols. */
interface HuffTable {
  counts: number[]; // counts[len - 1] = number of codes of length len, len 1..16
  symbols: number[];
  firstCode: number[]; // firstCode[len] = first canonical code of that length
  firstIndex: number[]; // firstIndex[len] = index into symbols of that code
}

function buildHuff(counts: number[], symbols: number[]): HuffTable {
  const firstCode = new Array<number>(17).fill(0);
  const firstIndex = new Array<number>(17).fill(0);
  let code = 0;
  let index = 0;
  for (let len = 1; len <= 16; len++) {
    firstCode[len] = code;
    firstIndex[len] = index;
    index += counts[len - 1]!;
    code = (code + counts[len - 1]!) << 1;
  }
  return { counts, symbols, firstCode, firstIndex };
}

/** Bit reader over entropy-coded data: 0xFF00 unstuffs to a 0xFF data byte. */
class BitReader {
  private readonly bytes: Uint8Array;
  private pos: number;
  private bitBuf = 0;
  private bitCnt = 0;

  constructor(bytes: Uint8Array, start: number) {
    this.bytes = bytes;
    this.pos = start;
  }

  private nextByte(): number {
    if (this.pos >= this.bytes.length) return 0xff; // past the end: pad with 1 bits
    const b = this.bytes[this.pos]!;
    if (b === 0xff) {
      const n = this.bytes[this.pos + 1];
      if (n === 0x00) {
        this.pos += 2;
        return 0xff;
      }
      // A marker (restart or EOI): the caller handles it after aligning.
      return 0xff;
    }
    this.pos += 1;
    return b;
  }

  bit(): number {
    if (this.bitCnt === 0) {
      this.bitBuf = this.nextByte();
      this.bitCnt = 8;
    }
    this.bitCnt -= 1;
    return (this.bitBuf >> this.bitCnt) & 1;
  }

  bits(n: number): number {
    let v = 0;
    for (let i = 0; i < n; i++) v = (v << 1) | this.bit();
    return v;
  }

  /** Drop the rest of the current byte, so the next read starts at a byte boundary. */
  align(): void {
    this.bitCnt = 0;
  }

  /** The byte position after the last byte fully consumed. */
  get bytePos(): number {
    return this.bitCnt === 0 ? this.pos : this.pos - 1;
  }

  /** Skip the two-byte marker at the current byte position (after align()). */
  skipMarker(): void {
    this.pos += 2;
  }
}

function huffDecode(br: BitReader, table: HuffTable): number {
  let code = 0;
  for (let len = 1; len <= 16; len++) {
    code = (code << 1) | br.bit();
    const count = table.counts[len - 1]!;
    if (code - table.firstCode[len]! < count) {
      return table.symbols[table.firstIndex[len]! + (code - table.firstCode[len]!)]!;
    }
  }
  throw new Error('invalid Huffman code');
}

/** JPEG's receive-and-extend: the magnitude bits of a DC difference or AC coefficient. */
function receiveExtend(br: BitReader, n: number): number {
  if (n === 0) return 0;
  const v = br.bits(n);
  return v < 1 << (n - 1) ? v - ((1 << n) - 1) : v;
}

/** The 2-D inverse DCT of one 8x8 block, coefficients in natural (row-major) order. */
function idctBlock(input: Float64Array): Float64Array {
  const tmp = new Float64Array(64);
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      let s = 0;
      for (let u = 0; u < 8; u++) s += CU[u]! * input[y * 8 + u]! * COS[x]![u]!;
      tmp[y * 8 + x] = s;
    }
  }
  const out = new Float64Array(64);
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      let s = 0;
      for (let v = 0; v < 8; v++) s += CU[v]! * tmp[v * 8 + x]! * COS[y]![v]!;
      out[y * 8 + x] = 0.25 * s;
    }
  }
  return out;
}

const clamp = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));

interface Component {
  id: number;
  h: number; // horizontal sampling factor
  v: number; // vertical sampling factor
  tq: number; // quantisation table id
}

/**
 * Decode a baseline JPEG (see the header) to RGB. Throws on anything outside that family
 * or on malformed data; the caller treats a throw as "no pixels for this photo".
 */
export function decodeJpeg(bytes: Uint8Array): DecodedImage {
  if (bytes.length < 2 || bytes[0] !== 0xff || bytes[1] !== 0xd8) {
    throw new Error('not a JPEG (no SOI)');
  }
  const quantTables = new Map<number, number[]>();
  const dcTables = new Map<number, HuffTable>();
  const acTables = new Map<number, HuffTable>();
  let width = 0;
  let height = 0;
  let components: Component[] = [];
  let restartInterval = 0;
  let scanComponents: { cs: number; td: number; ta: number }[] = [];
  let entropyStart = -1;

  let i = 2;
  while (i + 1 < bytes.length) {
    if (bytes[i] !== 0xff) {
      i += 1;
      continue;
    }
    const marker = bytes[i + 1]!;
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      i += 2;
      continue;
    }
    if (marker === 0xd9) break; // EOI
    if (i + 4 > bytes.length) throw new Error('truncated segment header');
    const len = (bytes[i + 2]! << 8) | bytes[i + 3]!;
    if (len < 2 || i + 2 + len > bytes.length) throw new Error('truncated segment');
    const seg = bytes.subarray(i + 4, i + 2 + len);
    if (marker === 0xdb) {
      // DQT: one or more tables; 8-bit (Pq 0) only.
      let p = 0;
      while (p < seg.length) {
        const pqTq = seg[p]!;
        const pq = pqTq >> 4;
        const tq = pqTq & 15;
        if (pq !== 0) throw new Error('16-bit quantisation table (unsupported)');
        const table = Array.from(seg.subarray(p + 1, p + 65));
        if (table.length !== 64) throw new Error('short quantisation table');
        quantTables.set(tq, table);
        p += 65;
      }
    } else if (marker === 0xc0) {
      // SOF0: baseline DCT.
      const precision = seg[0]!;
      if (precision !== 8) throw new Error(`sample precision ${precision} (only 8-bit supported)`);
      height = (seg[1]! << 8) | seg[2]!;
      width = (seg[3]! << 8) | seg[4]!;
      const ncomp = seg[5]!;
      if (ncomp !== 1 && ncomp !== 3) throw new Error(`${ncomp} components (only 1 or 3 supported)`);
      components = [];
      for (let c = 0; c < ncomp; c++) {
        const id = seg[6 + 3 * c]!;
        const hv = seg[7 + 3 * c]!;
        const tq = seg[8 + 3 * c]!;
        components.push({ id, h: hv >> 4, v: hv & 15, tq });
      }
    } else if (marker === 0xc2) {
      throw new Error('progressive JPEG (unsupported)');
    } else if (marker === 0xc4) {
      // DHT: one or more tables.
      let p = 0;
      while (p < seg.length) {
        const tcTh = seg[p]!;
        const tc = tcTh >> 4;
        const th = tcTh & 15;
        const counts = Array.from(seg.subarray(p + 1, p + 17));
        const total = counts.reduce((a, b) => a + b, 0);
        const symbols = Array.from(seg.subarray(p + 17, p + 17 + total));
        if (symbols.length !== total) throw new Error('short Huffman table');
        const table = buildHuff(counts, symbols);
        if (tc === 0) dcTables.set(th, table);
        else acTables.set(th, table);
        p += 17 + total;
      }
    } else if (marker === 0xdd) {
      // DRI: restart interval, in MCUs.
      restartInterval = (seg[0]! << 8) | seg[1]!;
    } else if (marker === 0xda) {
      // SOS: the scan header, then the entropy-coded data.
      const ns = seg[0]!;
      scanComponents = [];
      for (let c = 0; c < ns; c++) {
        const cs = seg[1 + 2 * c]!;
        const tdTa = seg[2 + 2 * c]!;
        scanComponents.push({ cs, td: tdTa >> 4, ta: tdTa & 15 });
      }
      entropyStart = i + 2 + len;
      break;
    }
    i += 2 + len;
  }
  if (entropyStart < 0) throw new Error('no scan (SOS) found');
  if (width === 0 || height === 0 || components.length === 0) throw new Error('no frame (SOF0) found');

  const maxH = Math.max(...components.map((c) => c.h));
  const maxV = Math.max(...components.map((c) => c.v));
  const mcuCols = Math.ceil(width / (8 * maxH));
  const mcuRows = Math.ceil(height / (8 * maxV));

  // One plane per component, at its own sampled resolution.
  const planes = components.map((c) => ({
    stride: mcuCols * 8 * c.h,
    rows: mcuRows * 8 * c.v,
    data: new Float64Array(mcuCols * 8 * c.h * mcuRows * 8 * c.v),
  }));

  const br = new BitReader(bytes, entropyStart);
  const dcPred = new Array<number>(components.length).fill(0);
  const coeffs = new Float64Array(64);
  let mcuCount = 0;

  const decodeBlock = (compIndex: number, td: number, ta: number): Float64Array => {
    const comp = components[compIndex]!;
    const dcTable = dcTables.get(td);
    const acTable = acTables.get(ta);
    const quant = quantTables.get(comp.tq);
    if (dcTable === undefined || acTable === undefined || quant === undefined) {
      throw new Error('missing Huffman or quantisation table for the scan');
    }
    coeffs.fill(0);
    const dcCat = huffDecode(br, dcTable);
    const diff = receiveExtend(br, dcCat);
    dcPred[compIndex] = dcPred[compIndex]! + diff;
    coeffs[0] = dcPred[compIndex]! * quant[0]!;
    let k = 1;
    while (k < 64) {
      const rs = huffDecode(br, acTable);
      const run = rs >> 4;
      const size = rs & 15;
      if (size === 0) {
        if (run === 15) {
          k += 16; // ZRL
          continue;
        }
        break; // EOB
      }
      k += run;
      if (k >= 64) break;
      coeffs[ZIGZAG[k]!] = receiveExtend(br, size) * quant[ZIGZAG[k]!]!;
      k += 1;
    }
    return idctBlock(coeffs);
  };

  for (let my = 0; my < mcuRows; my++) {
    for (let mx = 0; mx < mcuCols; mx++) {
      if (restartInterval > 0 && mcuCount > 0 && mcuCount % restartInterval === 0) {
        br.align();
        br.skipMarker(); // the RSTm marker
        dcPred.fill(0);
      }
      for (let ci = 0; ci < scanComponents.length; ci++) {
        const sc = scanComponents[ci]!;
        const compIndex = components.findIndex((c) => c.id === sc.cs);
        if (compIndex < 0) throw new Error('scan component not in the frame');
        const comp = components[compIndex]!;
        const plane = planes[compIndex]!;
        for (let by = 0; by < comp.v; by++) {
          for (let bx = 0; bx < comp.h; bx++) {
            const block = decodeBlock(compIndex, sc.td, sc.ta);
            const ox = (mx * comp.h + bx) * 8;
            const oy = (my * comp.v + by) * 8;
            for (let y = 0; y < 8; y++) {
              for (let x = 0; x < 8; x++) {
                plane.data[(oy + y) * plane.stride + (ox + x)] = block[y * 8 + x]! + 128;
              }
            }
          }
        }
      }
      mcuCount += 1;
    }
  }

  // Assemble RGB, upsampling chroma by nearest neighbour, and crop to the frame.
  const rgb = new Uint8Array(width * height * 3);
  const yPlane = planes[0]!;
  const cbPlane = planes.length > 1 ? planes[1]! : null;
  const crPlane = planes.length > 2 ? planes[2]! : null;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const yv = yPlane.data[y * yPlane.stride + x]!;
      let r: number;
      let g: number;
      let b: number;
      if (cbPlane === null || crPlane === null) {
        r = yv;
        g = yv;
        b = yv;
      } else {
        const cbComp = components[1]!;
        const crComp = components[2]!;
        const cbx = Math.floor((x * cbComp.h) / maxH);
        const cby = Math.floor((y * cbComp.v) / maxV);
        const crx = Math.floor((x * crComp.h) / maxH);
        const cry = Math.floor((y * crComp.v) / maxV);
        const cb = cbPlane.data[cby * cbPlane.stride + cbx]!;
        const cr = crPlane.data[cry * crPlane.stride + crx]!;
        r = yv + 1.402 * (cr - 128);
        g = yv - 0.34414 * (cb - 128) - 0.71414 * (cr - 128);
        b = yv + 1.772 * (cb - 128);
      }
      const o = (y * width + x) * 3;
      rgb[o] = clamp(r);
      rgb[o + 1] = clamp(g);
      rgb[o + 2] = clamp(b);
    }
  }
  return { width, height, data: rgb };
}
