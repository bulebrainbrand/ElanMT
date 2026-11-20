import type { WeightCursor } from "./byteCursor";
import { spend } from "./eventLoop";

/**
 * Weights are packed two bytes per UTF-16 code unit: (hi << 8) | lo, where the
 * even byte offset is hi and the odd one is lo. On an engine that stores
 * strings as two bytes per character this halves memory compared to one byte
 * per character; on V8 (which would have used one-byte strings) it costs the
 * same. Chunking keeps every string small and the byte offset math to a shift.
 * Each chunk holds STRING_CHUNK_SIZE bytes, i.e. STRING_CHUNK_SIZE / 2 chars.
 */
export const STRING_CHUNK_SHIFT = 18;
export const STRING_CHUNK_SIZE = 1 << STRING_CHUNK_SHIFT;
const STRING_CHUNK_MASK = STRING_CHUNK_SIZE - 1;
const BUILD_PART = 8192;

/** Pack 0..255 bytes two per char. An odd trailing byte is padded with lo = 0. */
export function bytesToPacked(bytes: ArrayLike<number>, length = bytes.length, start = 0): string {
  const parts: string[] = [];
  const step = BUILD_PART * 2;
  for (let i = 0; i < length; i += step) {
    const end = Math.min(i + step, length);
    const sub: number[] = [];
    for (let j = i; j < end; j += 2) {
      const hi = (bytes[start + j] ?? 0) & 0xff;
      const lo = j + 1 < end ? (bytes[start + j + 1] ?? 0) & 0xff : 0;
      sub.push((hi << 8) | lo);
    }
    parts.push(String.fromCharCode.apply(null, sub));
  }
  return parts.join("");
}

/** Convert a raw byte array into bounded packed strings. */
export function bytesToStrings(bytes: ArrayLike<number>): string[] {
  const strings: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += STRING_CHUNK_SIZE) {
    strings.push(bytesToPacked(bytes, Math.min(STRING_CHUNK_SIZE, bytes.length - offset), offset));
  }
  return strings;
}

function float32FromBits(bits: number): number {
  const sign = bits & 0x80000000 ? -1 : 1;
  const exponent = (bits >>> 23) & 0xff;
  const fraction = bits & 0x7fffff;
  if (exponent === 0) {
    if (fraction === 0) return sign === 1 ? 0 : -0;
    return sign * fraction * 2 ** -149;
  }
  if (exponent === 255) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + fraction * 2 ** -23) * 2 ** (exponent - 127);
}

/** Weight source backed by strings packing two bytes per char. */
export class StringCursor implements WeightCursor {
  readonly strings: string[];
  readonly byteLength: number;

  constructor(strings: string[], byteLength: number) {
    this.strings = strings;
    this.byteLength = byteLength;
  }

  private byteAt(offset: number): number {
    const chunk = this.strings[offset >>> STRING_CHUNK_SHIFT];
    if (chunk === undefined) return 0;
    const c = chunk.charCodeAt((offset & STRING_CHUNK_MASK) >>> 1);
    return offset & 1 ? c & 0xff : c >>> 8;
  }

  readBytesInto(
    out: number[],
    tensorOffset: number,
    length: number,
    callback: () => void,
    outBase = 0,
    unsigned = false,
  ): void {
    if (length < 0 || outBase < 0 || length > out.length - outBase) {
      throw new RangeError(
        `readBytesInto range [${outBase}, ${outBase + length}) exceeds buffer ${out.length}`,
      );
    }
    if (tensorOffset < 0 || tensorOffset + length > this.byteLength) {
      throw new RangeError(
        `readBytesInto range [${tensorOffset}, ${tensorOffset + length}) exceeds weights`,
      );
    }
    let i = 0;
    while (i < length) {
      const absolute = tensorOffset + i;
      const chunkIndex = absolute >>> STRING_CHUNK_SHIFT;
      const chunk = this.strings[chunkIndex];
      if (chunk === undefined) {
        for (; i < length; i++) out[outBase + i] = 0;
        break;
      }
      const local = absolute & STRING_CHUNK_MASK;
      const run = Math.min(length - i, STRING_CHUNK_SIZE - local);
      for (let j = 0; j < run; j++) {
        const p = local + j;
        const c = chunk.charCodeAt(p >>> 1);
        const byte = p & 1 ? c & 0xff : c >>> 8;
        out[outBase + i + j] = unsigned ? byte : (byte << 24) >> 24;
      }
      i += run;
    }
    spend(length);
    callback();
  }

  readBytes(tensorOffset: number, length: number, callback: (bytes: number[]) => void): void {
    const out = new Array<number>(length).fill(0);
    this.readBytesInto(out, tensorOffset, length, () => callback(out));
  }

  readRawBytes(tensorOffset: number, length: number, callback: (bytes: number[]) => void): void {
    const out = new Array<number>(length).fill(0);
    this.readBytesInto(out, tensorOffset, length, () => callback(out), 0, true);
  }

  readFloats(tensorOffset: number, count: number, callback: (values: number[]) => void): void {
    const values = new Array<number>(count).fill(0);
    for (let i = 0; i < count; i++) {
      const o = tensorOffset + i * 4;
      const bits =
        (this.byteAt(o) |
          (this.byteAt(o + 1) << 8) |
          (this.byteAt(o + 2) << 16) |
          (this.byteAt(o + 3) << 24)) >>>
        0;
      values[i] = float32FromBits(bits);
    }
    spend(count * 4);
    callback(values);
  }
}
