/**
 * Bloxd world-code entry point for the in-memory int4 model.
 *
 * The 41 MB NF4 stream lives in block data. On the first think() call it is
 * read once into a number[], then every layer runs from memory; tick() only
 * has to advance the generator queue, not fetch weights again.
 */

import { ByteCursor } from "./byteCursor";
import { CHUNK_SIZE } from "./constants";
import { queueTask, setTickBudget, update as updateEventLoop } from "./eventLoop";
import { translateBlock } from "./generate";
import { update as updateChunkLoads } from "./loadChunk";
import { BlockMarian } from "./marian";
import { loadWeightsIntoMemory, translateInt4Bytes as translateBytes } from "./memory";
import type { MemoryCallbacks } from "./memory";
import { MemoryCursor } from "./memoryCursor";
import { MODEL_CONFIG as INT4_CONFIG, MODEL_WEIGHTS as INT4_WEIGHTS } from "./manifestInt4";
import { loadTokenizer } from "./tokenizerBlock";
import { createWeightCoordFn } from "./weightCoordFn";

/** int4 weights live on the z=64 layer, away from the int8 copy at z=32. */
const INT4_ORIGIN = [0, 1, 64] as const;

const sourceCursor = new ByteCursor(CHUNK_SIZE, createWeightCoordFn(INT4_ORIGIN));

// In-memory reads are cheap, so allow more work per tick than the streaming
// int8 path. The caller can still override this with setTickBudget().
setTickBudget(8 * 1024 * 1024);
let memoryCursor: MemoryCursor | undefined;
let model: BlockMarian | undefined;
let loading = false;
const waiting: ((model: BlockMarian, cursor: MemoryCursor) => void)[] = [];

globalThis.tick = () => {
  updateChunkLoads();
  updateEventLoop();
};

function ensureModel(callback: (model: BlockMarian, cursor: MemoryCursor) => void): void {
  if (model !== undefined && memoryCursor !== undefined) {
    callback(model, memoryCursor);
    return;
  }
  waiting.push(callback);
  if (loading) return;
  loading = true;
  const totalLength = INT4_WEIGHTS.totalLength;
  if (totalLength === undefined || totalLength <= 0) {
    throw new Error("int4 manifest is missing totalLength");
  }
  loadWeightsIntoMemory(sourceCursor, totalLength, (cursor) => {
    memoryCursor = cursor;
    const loaded = new BlockMarian(cursor, INT4_CONFIG, INT4_WEIGHTS);
    loaded.prepare(() => {
      model = loaded;
      const callbacks = waiting.splice(0);
      for (let i = 0; i < callbacks.length; i++) callbacks[i](loaded, cursor);
    });
  });
}

interface ThinkCallbacks {
  onToken?: (tokens: number[], text: string) => void;
  onComplete?: (tokens: number[], promptLength: number, text: string) => void;
}

function think(text: string, callbacks: ThinkCallbacks = {}, maxNewTokens = 32): void {
  queueTask(() => {
    ensureModel((loaded, cursor) => {
      console.log("model loaded");
      loadTokenizer(cursor, INT4_WEIGHTS, INT4_CONFIG.eosId, (tokenizer) => {
        console.log("tokenizer loaded");
        translateBlock(
          loaded,
          tokenizer,
          text,
          {
            onToken: (tokens, decoded) => callbacks.onToken?.(tokens, decoded),
            onComplete: (tokens, sourceTokens) => {
              const decoded = tokenizer.decode(tokens);
              callbacks.onComplete?.(tokens, sourceTokens.length, decoded);
              if (!callbacks.onComplete) console.log(decoded);
            },
          },
          { maxNewTokens },
        );
      });
    });
  });
}

const enqueueTranslation = think;

globalThis.think = ((
  text: string,
  callbacks?: ((tokens: number[], promptLength: number) => void) | ThinkCallbacks,
  _stream = true,
  maxNewTokens = 32,
) => {
  if (typeof callbacks === "object" && callbacks !== null) {
    enqueueTranslation(text, callbacks, maxNewTokens);
    return;
  }
  enqueueTranslation(
    text,
    {
      onComplete: (tokens, promptLength, decoded) => {
        if (callbacks) callbacks(tokens, promptLength);
        else console.log(decoded);
      },
    },
    maxNewTokens,
  );
}) as typeof globalThis.think;

globalThis.setTickBudget = setTickBudget;

function translateInt4Bytes(
  bytes: number[],
  text: string,
  callbacks: MemoryCallbacks = {},
  maxNewTokens = 32,
): void {
  translateBytes(bytes, INT4_CONFIG, INT4_WEIGHTS, text, callbacks, maxNewTokens);
}
