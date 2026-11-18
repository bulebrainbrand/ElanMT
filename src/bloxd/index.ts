/**
 * Bloxd world-code entry point for ElanMT.
 *
 * The weights and the raw source.spm live in block data. Every forward step
 * streams int8 rows through ByteCursor and charges the decoded byte count to
 * the per-tick budget. Call think() to enqueue a translation; the normal tick
 * hook drains the work queue.
 */

import { update as updateChunkLoads } from "./loadChunk";
import { queueTask, setTickBudget, update as updateEventLoop } from "./eventLoop";
import { ByteCursor } from "./byteCursor";
import { CHUNK_SIZE } from "./constants";
import { createWeightCoordFn } from "./weightCoordFn";
import { MODEL_CONFIG, MODEL_WEIGHTS } from "./manifest";
import { BlockMarian } from "./marian";
import { loadTokenizer } from "./tokenizerBlock";
import { translateBlock } from "./generate";

const cursor = new ByteCursor(CHUNK_SIZE, createWeightCoordFn());
const model = new BlockMarian(cursor, MODEL_CONFIG, MODEL_WEIGHTS);

globalThis.tick = () => {
  updateChunkLoads();
  updateEventLoop();
};

export interface ThinkCallbacks {
  onToken?: (tokens: number[], text: string) => void;
  onComplete?: (tokens: number[], promptLength: number, text: string) => void;
}

export function think(text: string, callbacks: ThinkCallbacks = {}, maxNewTokens = 32): void {
  queueTask(() => {
    loadTokenizer(cursor, MODEL_WEIGHTS, MODEL_CONFIG.eosId, (tokenizer) => {
      translateBlock(
        model,
        tokenizer,
        text,
        {
          onToken: (tokens, decoded) => {
            callbacks.onToken?.(tokens, decoded);
            if (streamToPlayers) {
              const client = api.getPlayerIds?.();
              if (client !== undefined && api.setClientOption !== undefined) {
                for (let i = 0; i < client.length; i++) {
                  api.setClientOption(client[i], "middleTextLower", decoded);
                }
              }
            }
          },
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
}

let streamToPlayers = true;

export function setStreamToPlayers(enabled: boolean): void {
  streamToPlayers = enabled;
}

export function reset(): void {
  streamToPlayers = true;
}

// Capture the module-level function before assigning the same global name.
// In a plain world-code script `globalThis.think` would otherwise shadow it.
const enqueueTranslation = think;

// Bloxd reads these globals after evaluating world code. The second argument
// can be either the classic callback or a ThinkCallbacks object.
globalThis.think = ((
  text: string,
  callbacks?: ((tokens: number[], promptLength: number) => void) | ThinkCallbacks,
  stream = true,
  maxNewTokens = 32,
) => {
  setStreamToPlayers(stream);
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

// Exported for the Node/QuickJS harnesses. World code only needs tick/think.
export { ByteCursor } from "./byteCursor";
export { encode as encodeSafe8, decode as decodeSafe8 } from "./safe8";
export { MODEL_CONFIG, MODEL_WEIGHTS } from "./manifest";
export { BlockMarian } from "./marian";
export { translateBlock } from "./generate";
export { loadTokenizer } from "./tokenizerBlock";
export { linearInto, computeLogits } from "./linear";
export { update, setTickBudget, getTickBudget } from "./eventLoop";
