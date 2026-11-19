import type { WeightCursor } from "./byteCursor";
import { queueTask, update as updateEventLoop } from "./eventLoop";
import { translateBlock } from "./generate";
import { BlockMarian } from "./marian";
import { MemoryCursor, WORD_CHUNK_MASK, WORD_CHUNK_SHIFT, packBytesToWords } from "./memoryCursor";
import { loadTokenizer } from "./tokenizerBlock";
import type { ModelConfig, ModelWeights } from "./types";

export interface MemoryCallbacks {
  onToken?: (tokens: number[], text: string) => void;
  onComplete?: (tokens: number[], promptLength: number, text: string) => void;
}

/** Read the whole stream into 512 KiB tasks and 2 MiB word chunks. */
export function loadWeightsIntoMemory(
  cursor: WeightCursor,
  totalLengthOrWeights: number | ModelWeights,
  callback: (cursor: MemoryCursor) => void,
): void {
  const totalLength =
    typeof totalLengthOrWeights === "number"
      ? totalLengthOrWeights
      : (totalLengthOrWeights.totalLength ?? Number.NaN);
  if (!Number.isFinite(totalLength) || totalLength <= 0) {
    throw new RangeError(`invalid weight length: ${String(totalLength)}`);
  }
  const words: number[][] = [];
  const chunkSize = 512 * 1024;
  const temp = new Array<number>(chunkSize).fill(0);
  let offset = 0;
  const next = (): void => {
    if (offset >= totalLength) {
      callback(new MemoryCursor(words, totalLength));
      return;
    }
    const length = Math.min(chunkSize, totalLength - offset);
    const start = offset;
    offset += length;
    cursor.readBytesInto(
      temp,
      start,
      length,
      () => {
        const wordStart = start >>> 2;
        for (let i = 0; i < length; i += 4) {
          const wordIndex = wordStart + (i >>> 2);
          const chunkIndex = wordIndex >>> WORD_CHUNK_SHIFT;
          let chunk = words[chunkIndex];
          if (chunk === undefined) {
            chunk = [];
            words[chunkIndex] = chunk;
          }
          const b1 = i + 1 < length ? temp[i + 1] : 0;
          const b2 = i + 2 < length ? temp[i + 2] : 0;
          const b3 = i + 3 < length ? temp[i + 3] : 0;
          chunk[wordIndex & WORD_CHUNK_MASK] = (temp[i] ?? 0) | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        queueTask(next);
      },
      0,
      true,
    );
  };
  next();
}

/**
 * Run one translation against an in-memory weight cursor. The event-loop
 * queue is drained synchronously; Bloxd instead calls tick() so the same
 * generator can spread its work across game ticks.
 */
export function translateInMemory(
  cursor: MemoryCursor,
  config: ModelConfig,
  weights: ModelWeights,
  text: string,
  callbacks: MemoryCallbacks = {},
  maxNewTokens = 32,
): void {
  const model = new BlockMarian(cursor, config, weights);
  let finished = false;
  model.prepare(() => {
    loadTokenizer(cursor, weights, config.eosId, (tokenizer) => {
      translateBlock(
        model,
        tokenizer,
        text,
        {
          onToken: (tokens, decoded) => callbacks.onToken?.(tokens, decoded),
          onComplete: (tokens, sourceTokens) => {
            const decoded = tokenizer.decode(tokens);
            callbacks.onComplete?.(tokens, sourceTokens.length, decoded);
            finished = true;
          },
        },
        { maxNewTokens },
      );
      while (!finished) updateEventLoop();
    });
  });
}

/** Node convenience wrapper: bytes are the raw weights_*.bin file. */
export function translateInt4Bytes(
  bytes: number[],
  config: ModelConfig,
  weights: ModelWeights,
  text: string,
  callbacks: MemoryCallbacks = {},
  maxNewTokens = 32,
): void {
  translateInMemory(
    new MemoryCursor(packBytesToWords(bytes), bytes.length),
    config,
    weights,
    text,
    callbacks,
    maxNewTokens,
  );
}
