import { SentencePieceTokenizer } from "../sentencepiece";
import type { WeightCursor } from "./byteCursor";
import type { ModelWeights } from "./types";

let tokenizer: SentencePieceTokenizer | undefined;

/** Load and parse the raw source.spm blob stored in block data. */
export function loadTokenizer(
  cursor: WeightCursor,
  weights: ModelWeights,
  eosId: number,
  callback: (tokenizer: SentencePieceTokenizer) => void,
): void {
  if (tokenizer !== undefined) {
    callback(tokenizer);
    return;
  }
  cursor.readRawBytes(weights.tokenizer.offset, weights.tokenizer.length, (bytes) => {
    tokenizer = new SentencePieceTokenizer(bytes, eosId, 1);
    callback(tokenizer);
  });
}

export function resetTokenizerForTests(): void {
  tokenizer = undefined;
}
