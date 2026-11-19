import type { WeightCursor } from "./byteCursor";
import { spend } from "./eventLoop";

/** 524,288 32-bit words = 2 MiB per number[] chunk. */
export const WORD_CHUNK_SHIFT = 19;
export const WORD_CHUNK_SIZE = 1 << WORD_CHUNK_SHIFT;
export const WORD_CHUNK_MASK = WORD_CHUNK_SIZE - 1;

/** Pack a byte-like sequence into 32-bit word chunks. */
export function packBytesToWords(bytes: ArrayLike<number>): number[][] {
  const words: number[][] = [];
  for (let i = 0; i < bytes.length; i += 4) {
    const wordIndex = i >>> 2;
    const chunkIndex = wordIndex >>> WORD_CHUNK_SHIFT;
    let chunk = words[chunkIndex];
    if (chunk === undefined) {
      chunk = [];
      words[chunkIndex] = chunk;
    }
    chunk[wordIndex & WORD_CHUNK_MASK] =
      (bytes[i] ?? 0) |
      ((bytes[i + 1] ?? 0) << 8) |
      ((bytes[i + 2] ?? 0) << 16) |
      ((bytes[i + 3] ?? 0) << 24);
  }
  return words;
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

/** Weight source backed by 32-bit words split into bounded number[] chunks. */
export class MemoryCursor implements WeightCursor {
  readonly words: number[][];
  readonly byteLength: number;

  constructor(words: number[][], byteLength: number) {
    this.words = words;
    this.byteLength = byteLength;
  }

  private wordAt(wordIndex: number): number {
    const chunk = this.words[wordIndex >>> WORD_CHUNK_SHIFT];
    return chunk === undefined ? 0 : chunk[wordIndex & WORD_CHUNK_MASK];
  }

  private byteAt(index: number): number {
    return (this.wordAt(index >>> 2) >>> ((index & 3) * 8)) & 0xff;
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
    if ((tensorOffset & 3) === 0) {
      for (; i + 4 <= length; i += 4) {
        const word = this.wordAt((tensorOffset + i) >>> 2);
        const b0 = word & 0xff;
        const b1 = (word >>> 8) & 0xff;
        const b2 = (word >>> 16) & 0xff;
        const b3 = (word >>> 24) & 0xff;
        if (unsigned) {
          out[outBase + i] = b0;
          out[outBase + i + 1] = b1;
          out[outBase + i + 2] = b2;
          out[outBase + i + 3] = b3;
        } else {
          out[outBase + i] = (b0 << 24) >> 24;
          out[outBase + i + 1] = (b1 << 24) >> 24;
          out[outBase + i + 2] = (b2 << 24) >> 24;
          out[outBase + i + 3] = (b3 << 24) >> 24;
        }
      }
    }
    for (; i < length; i++) {
      const byte = this.byteAt(tensorOffset + i);
      out[outBase + i] = unsigned ? byte : (byte << 24) >> 24;
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
      if ((o & 3) === 0) {
        values[i] = float32FromBits(this.wordAt(o >>> 2) >>> 0);
      } else {
        const bits =
          (this.byteAt(o) |
            (this.byteAt(o + 1) << 8) |
            (this.byteAt(o + 2) << 16) |
            (this.byteAt(o + 3) << 24)) >>>
          0;
        values[i] = float32FromBits(bits);
      }
    }
    spend(count * 4);
    callback(values);
  }
}
