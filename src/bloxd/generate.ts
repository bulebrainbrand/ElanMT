import type { SentencePieceTokenizer } from "../sentencepiece";
import { queueTask } from "./eventLoop";
import type { BlockMarian, CrossCache, DecoderState } from "./marian";
import { argmax } from "./ops";

export interface TranslateCallbacks {
  onToken?: (tokens: number[], text: string) => void;
  onComplete?: (tokens: number[], sourceTokens: number[]) => void;
}

export interface TranslateOptions {
  maxNewTokens?: number;
  maxSourceTokens?: number;
}

function applyBadWords(logits: number[], badWordIds: number[][]): void {
  for (let i = 0; i < badWordIds.length; i++) {
    const ids = badWordIds[i];
    if (ids.length === 1) logits[ids[0]] = -Infinity;
  }
}

function forceEos(logits: number[], eosId: number): void {
  for (let i = 0; i < logits.length; i++) logits[i] = -Infinity;
  logits[eosId] = 0;
}

/** Greedy Marian translation; weights stream from block data every step. */
export function translateBlock(
  model: BlockMarian,
  tokenizer: SentencePieceTokenizer,
  text: string,
  callbacks: TranslateCallbacks = {},
  options: TranslateOptions = {},
): void {
  const maxNewTokens = Math.max(1, options.maxNewTokens ?? 32);
  const maxSourceTokens = Math.max(1, options.maxSourceTokens ?? 512);
  const { ids: sourceTokens } = tokenizer.encode(text, maxSourceTokens);

  if (sourceTokens.length === 1 && sourceTokens[0] === model.config.eosId) {
    callbacks.onComplete?.([], sourceTokens);
    return;
  }

  model.prepare(() => {
    model.encode(sourceTokens, (memory) => {
      model.createCrossCache(memory, (cross: CrossCache) => {
        const state: DecoderState = model.createState();
        const output: number[] = [];
        let input = model.config.decoderStartId;

        const step = (): void => {
          if (output.length >= maxNewTokens) {
            callbacks.onComplete?.(output, sourceTokens);
            return;
          }
          model.decode(input, state, cross, (logits) => {
            applyBadWords(logits, model.config.badWordIds);
            if (output.length === maxNewTokens - 1) forceEos(logits, model.config.eosId);
            const token = argmax(logits);
            output.push(token);
            callbacks.onToken?.(output, tokenizer.decode(output));
            if (token === model.config.eosId) {
              callbacks.onComplete?.(output, sourceTokens);
              return;
            }
            input = token;
            queueTask(step);
          });
        };

        queueTask(step);
      });
    });
  });
}
