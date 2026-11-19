import type { WeightCursor } from "./byteCursor";
import { queueTask, spendLoad, update as updateEventLoop } from "./eventLoop";
import { translateBlock } from "./generate";
import { BlockMarian } from "./marian";
import { MemoryCursor, WORD_CHUNK_MASK, WORD_CHUNK_SHIFT } from "./memoryCursor";
import { StringCursor, bytesToLatin1, bytesToStrings } from "./stringCursor";
import { loadTokenizer } from "./tokenizerBlock";
import type { ModelConfig, ModelWeights } from "./types";

export interface MemoryCallbacks {
  onToken?: (tokens: number[], text: string) => void;
  onComplete?: (tokens: number[], promptLength: number, text: string) => void;
}

/** Read the whole stream into Latin-1 string chunks, one buffer per task. */
export function loadWeightsAsStrings(
  cursor: WeightCursor,
  totalLengthOrWeights: number | ModelWeights,
  callback: (cursor: StringCursor) => void,
): void {
  const totalLength =
    typeof totalLengthOrWeights === "number"
      ? totalLengthOrWeights
      : (totalLengthOrWeights.totalLength ?? Number.NaN);
  if (!Number.isFinite(totalLength) || totalLength <= 0) {
    throw new RangeError(`invalid weight length: ${String(totalLength)}`);
  }
  const strings: string[] = [];
  const chunkSize = 256 * 1024;
  const temp = new Array<number>(chunkSize).fill(0);
  let offset = 0;
  const next = (): void => {
    if (offset >= totalLength) {
      callback(new StringCursor(strings, totalLength));
      return;
    }
    console.log(offset);
    const length = Math.min(chunkSize, totalLength - offset);
    const start = offset;
    offset += length;
    cursor.readBytesInto(
      temp,
      start,
      length,
      () => {
        strings.push(bytesToLatin1(temp, length));
        spendLoad(length);
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
  cursor: WeightCursor,
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
    new StringCursor(bytesToStrings(bytes), bytes.length),
    config,
    weights,
    text,
    callbacks,
    maxNewTokens,
  );
}
