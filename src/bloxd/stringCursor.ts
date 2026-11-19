import type { WeightCursor } from "./byteCursor";
import { spend } from "./eventLoop";

/**
 * One-byte strings are a compact weight source on V8: char codes 0..255 use
 * one byte per character, and random access is charCodeAt(). Chunking keeps
 * every string small and the byte offset math to a shift.
 */
export const STRING_CHUNK_SHIFT = 19;
export const STRING_CHUNK_SIZE = 1 << STRING_CHUNK_SHIFT;
const STRING_CHUNK_MASK = STRING_CHUNK_SIZE - 1;
const BUILD_PART = 8192;

/** Decode 0..255 bytes into a flat Latin-1 string. */
export function bytesToLatin1(bytes: ArrayLike<number>, length = bytes.length, start = 0): string {
  const parts: string[] = [];
  for (let i = 0; i < length; i += BUILD_PART) {
    const end = Math.min(i + BUILD_PART, length);
    const sub: number[] = [];
    for (let j = i; j < end; j++) sub.push(bytes[start + j] ?? 0);
    parts.push(String.fromCharCode.apply(null, sub));
  }
  return parts.join("");
}

/** Convert a raw byte array into bounded Latin-1 strings. */
export function bytesToStrings(bytes: ArrayLike<number>): string[] {
  const strings: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += STRING_CHUNK_SIZE) {
    strings.push(bytesToLatin1(bytes, Math.min(STRING_CHUNK_SIZE, bytes.length - offset), offset));
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

/** Weight source backed by Latin-1 strings. */
export class StringCursor implements WeightCursor {
  readonly strings: string[];
  readonly byteLength: number;

  constructor(strings: string[], byteLength: number) {
    this.strings = strings;
    this.byteLength = byteLength;
  }

  private byteAt(offset: number): number {
    const chunk = this.strings[offset >>> STRING_CHUNK_SHIFT];
    return chunk === undefined ? 0 : chunk.charCodeAt(offset & STRING_CHUNK_MASK);
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
        const byte = chunk.charCodeAt(local + j);
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
