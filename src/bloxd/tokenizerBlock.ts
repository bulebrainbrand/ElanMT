import { SentencePieceTokenizer } from "../sentencepiece";
import type { ByteCursor } from "./byteCursor";
import type { ModelWeights } from "./types";

let tokenizer: SentencePieceTokenizer | undefined;

/** Load and parse the raw source.spm blob stored in block data. */
export function loadTokenizer(
  cursor: ByteCursor,
  weights: ModelWeights,
  eosId: number,
  callback: (tokenizer: SentencePieceTokenizer) => void,
): void {
  if (tokenizer !== undefined) {
    callback(tokenizer);
    return;
  }
  cursor.readRawBytes(weights.tokenizer.offset, weights.tokenizer.length, (bytes) => {
    const data = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) data[i] = bytes[i];
    tokenizer = new SentencePieceTokenizer(data, eosId, 1);
    callback(tokenizer);
  });
}

export function resetTokenizerForTests(): void {
  tokenizer = undefined;
}
