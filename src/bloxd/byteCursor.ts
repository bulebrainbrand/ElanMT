import { readData } from "./blockdata";
import { spend } from "./eventLoop";

/**
 * Random-access reader over safe8 text stored in block data.
 *
 * The first read of a chunk may have to wait for Bloxd to load it; in that case
 * readData defers the callback and the event loop resumes on a later tick.
 * Every completed read reports its decoded byte count to the tick budget.
 */
export class ByteCursor {
  readonly chunkSize: number;
  readonly coordFn: (index: number) => [number, number, number];
  chunkIndex = -1;
  chunkText = "";

  constructor(chunkSize: number, coordFn: (index: number) => [number, number, number]) {
    this.chunkSize = chunkSize;
    this.coordFn = coordFn;
  }

  _ensureChunkLoaded(charOffset: number, callback: (localOffset: number) => void): void {
    const neededChunk = Math.floor(charOffset / this.chunkSize);
    if (neededChunk === this.chunkIndex) {
      callback(charOffset - neededChunk * this.chunkSize);
      return;
    }
    const pos = this.coordFn(neededChunk);
    readData(pos, (text = "") => {
      this.chunkText = text;
      this.chunkIndex = neededChunk;
      callback(charOffset - neededChunk * this.chunkSize);
    });
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

    let encodedCharOffset = Math.floor((tensorOffset * 8) / 3);
    const skipBits = (tensorOffset * 8) % 3;
    let firstChar = true;
    let bitBuffer = 0;
    let bitCount = 0;
    let outIdx = 0;
    let lastChunkIndex = -1;

    const readNext = (): void => {
      if (outIdx >= length) {
        spend(length);
        callback();
        return;
      }

      this._ensureChunkLoaded(encodedCharOffset, (localPos) => {
        if (this.chunkIndex === lastChunkIndex && localPos >= this.chunkText.length) {
          throw new Error(`stalled safe8 read at chunk ${this.chunkIndex}`);
        }
        lastChunkIndex = this.chunkIndex;
        const text = this.chunkText;
        let i = localPos;

        while (i < text.length && outIdx < length) {
          const digit = text.charCodeAt(i) - 48;
          if (digit < 0 || digit > 7) {
            throw new Error(`Invalid safe8 character code: ${text.charCodeAt(i)}`);
          }
          i++;

          if (firstChar) {
            firstChar = false;
            if (skipBits > 0) {
              bitBuffer = digit & ((1 << (3 - skipBits)) - 1);
              bitCount = 3 - skipBits;
            } else {
              bitBuffer = digit;
              bitCount = 3;
            }
          } else {
            bitBuffer = (bitBuffer << 3) | digit;
            bitCount += 3;
          }

          while (bitCount >= 8 && outIdx < length) {
            bitCount -= 8;
            const byte = (bitBuffer >> bitCount) & 0xff;
            out[outBase + outIdx++] = unsigned ? byte : (byte << 24) >> 24;
          }
          bitBuffer &= (1 << bitCount) - 1;
        }

        encodedCharOffset += i - localPos;
        readNext();
      });
    };

    readNext();
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
    this.readRawBytes(tensorOffset, count * 4, (bytes) => {
      const values = new Array<number>(count).fill(0);
      const buffer = new ArrayBuffer(4);
      const view = new DataView(buffer);
      for (let i = 0; i < count; i++) {
        const o = i * 4;
        view.setUint8(0, bytes[o]);
        view.setUint8(1, bytes[o + 1]);
        view.setUint8(2, bytes[o + 2]);
        view.setUint8(3, bytes[o + 3]);
        values[i] = view.getFloat32(0, true);
      }
      callback(values);
    });
  }
}
