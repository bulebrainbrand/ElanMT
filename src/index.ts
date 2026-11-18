/**
 * ElanMT (Marian ja-en) inference for browsers and Node.
 *
 *   import { loadElanMT } from "./remote/index.js";
 *   const translator = await loadElanMT("./model");
 *   console.log(translator.translate("こんにちは。私はAIです。").translation);
 *
 * The model directory must contain config.json, source.spm and
 * model.safetensors from Mitsua/elan-mt-bt-ja-en.
 */

import { parseSafetensors } from "./safetensors";
import { SentencePieceTokenizer } from "./sentencepiece";
import { MarianModel, type CrossCache, type DecoderState, type MarianConfig } from "./marian";

export type { MarianConfig } from "./marian";
export { MarianModel } from "./marian";
export { SentencePieceTokenizer } from "./sentencepiece";
export { parseSafetensors } from "./safetensors";

export interface ElanMTAssets {
  config: MarianConfig | string | ArrayBuffer | Uint8Array;
  /** source.spm (the target spm is identical in the ElanMT checkpoint). */
  spm: ArrayBuffer | Uint8Array;
  /** model.safetensors */
  weights: ArrayBuffer | Uint8Array;
}

export interface LoadOptions {
  fetch?: typeof fetch;
  /** Default beam size; falls back to config.num_beams (4). */
  numBeams?: number;
  /** Default maximum generated tokens (same default as scripts/infer_dump.py). */
  maxNewTokens?: number;
  /** Default maximum source tokens (including EOS). */
  maxSourceTokens?: number;
  /** Called while downloading model.safetensors. */
  onProgress?: (loadedBytes: number, totalBytes: number) => void;
}

export interface TranslateOptions {
  numBeams?: number;
  maxNewTokens?: number;
  maxSourceTokens?: number;
}

export interface TranslationResult {
  source: string;
  translation: string;
  sourceTokenIds: number[];
  sourceTruncated: boolean;
  /** Generated ids without decoder_start_token_id, including EOS when finished. */
  outputTokenIds: number[];
  outputTokens: string[];
}

interface Beam {
  tokens: number[];
  score: number;
  state: DecoderState;
}

interface FinishedHypothesis {
  tokens: number[];
  score: number;
  endedWithEos: boolean;
}

interface Candidate {
  beamIndex: number;
  token: number;
  score: number;
}

function decodeConfig(config: MarianConfig | string | ArrayBuffer | Uint8Array): MarianConfig {
  if (typeof config === "string") return JSON.parse(config) as MarianConfig;
  if (config instanceof Uint8Array || config instanceof ArrayBuffer) {
    return JSON.parse(new TextDecoder().decode(config)) as MarianConfig;
  }
  return config;
}

function joinUrl(base: string, file: string): string {
  return `${base.replace(/\/+$/, "")}/${file}`;
}

async function fetchBuffer(
  url: string,
  fetchImpl: typeof fetch,
  onProgress?: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  const response = await fetchImpl(url);
  if (!response.ok) {
    throw new Error(`ElanMT: failed to fetch ${url} (${response.status})`);
  }
  const total = Number(response.headers.get("content-length") ?? 0);
  if (!response.body || !onProgress) return response.arrayBuffer();

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const merged = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return merged.buffer;
}

function applyBadWords(logits: Float32Array, badWords: number[][]): void {
  for (const ids of badWords) {
    if (ids.length === 1) {
      const id = ids[0];
      if (id >= 0 && id < logits.length) logits[id] = -Infinity;
    }
  }
}

function topKLogits(logits: Float32Array, k: number): { ids: number[]; scores: number[] } {
  const ids: number[] = [];
  const scores: number[] = [];
  for (let id = 0; id < logits.length; id++) {
    const score = logits[id];
    if (!Number.isFinite(score)) continue;
    if (ids.length < k) {
      let pos = ids.length;
      while (pos > 0 && scores[pos - 1] < score) pos--;
      ids.splice(pos, 0, id);
      scores.splice(pos, 0, score);
    } else if (score > scores[scores.length - 1]) {
      let pos = ids.length - 1;
      while (pos > 0 && scores[pos - 1] < score) pos--;
      ids.splice(pos, 0, id);
      scores.splice(pos, 0, score);
    }
  }
  return { ids, scores };
}

export class ElanMT {
  readonly config: MarianConfig;
  readonly model: MarianModel;
  readonly tokenizer: SentencePieceTokenizer;
  private readonly numBeams: number;
  private readonly maxNewTokens: number;
  private readonly maxSourceTokens: number;

  constructor(
    config: MarianConfig,
    tokenizer: SentencePieceTokenizer,
    model: MarianModel,
    options: LoadOptions = {},
  ) {
    this.config = config;
    this.tokenizer = tokenizer;
    this.model = model;
    this.numBeams = Math.max(1, options.numBeams ?? config.num_beams ?? 1);
    this.maxNewTokens = Math.max(1, options.maxNewTokens ?? 128);
    this.maxSourceTokens = Math.max(1, options.maxSourceTokens ?? 512);
  }

  translate(text: string, options: TranslateOptions = {}): TranslationResult {
    const maxSourceTokens = Math.max(1, options.maxSourceTokens ?? this.maxSourceTokens);
    const maxNewTokens = Math.max(1, options.maxNewTokens ?? this.maxNewTokens);
    const numBeams = Math.max(1, options.numBeams ?? this.numBeams);

    const { ids, truncated } = this.tokenizer.encode(text, maxSourceTokens);
    if (ids.length === 1 && ids[0] === this.model.eosId) {
      return {
        source: text,
        translation: "",
        sourceTokenIds: ids,
        sourceTruncated: truncated,
        outputTokenIds: [],
        outputTokens: [],
      };
    }
    const memory = this.model.encode(ids);
    const cross = this.model.createCrossCache(memory, ids.length);

    const output =
      numBeams <= 1
        ? this.greedy(cross, maxNewTokens)
        : this.beamSearch(cross, maxNewTokens, numBeams);

    return {
      source: text,
      translation: this.tokenizer.decode(output),
      sourceTokenIds: ids,
      sourceTruncated: truncated,
      outputTokenIds: output,
      outputTokens: output.map((id) =>
        id >= 0 && id < this.tokenizer.pieces.length ? this.tokenizer.pieces[id] : "",
      ),
    };
  }

  private prepareLogits(logits: Float32Array, step: number, maxNewTokens: number): void {
    applyBadWords(logits, this.model.badWordIds);
    if (step === maxNewTokens - 1) {
      // ForcedEOSTokenLogitsProcessor at max_length.
      for (let i = 0; i < logits.length; i++) logits[i] = -Infinity;
      logits[this.model.eosId] = 0;
    }
  }

  private greedy(cross: CrossCache, maxNewTokens: number): number[] {
    const state = this.model.createDecoderState(this.model.maxLength, maxNewTokens);
    const output: number[] = [];
    let input = this.model.decoderStartId;

    for (let step = 0; step < maxNewTokens; step++) {
      const logits = this.model.step(state, input, cross);
      this.prepareLogits(logits, step, maxNewTokens);

      let bestId = 0;
      let bestScore = -Infinity;
      for (let id = 0; id < logits.length; id++) {
        if (logits[id] > bestScore) {
          bestScore = logits[id];
          bestId = id;
        }
      }
      if (bestId === this.model.eosId) {
        output.push(bestId);
        break;
      }
      output.push(bestId);
      input = bestId;
    }
    return output;
  }

  private beamSearch(cross: CrossCache, maxNewTokens: number, numBeams: number): number[] {
    const lengthPenalty = 1;
    let active: Beam[] = [
      {
        tokens: [],
        score: 0,
        state: this.model.createDecoderState(this.model.maxLength, maxNewTokens),
      },
    ];
    const finished: FinishedHypothesis[] = [];

    for (let step = 0; step < maxNewTokens; step++) {
      if (active.length === 0) break;
      const candidates: Candidate[] = [];

      for (let beamIndex = 0; beamIndex < active.length; beamIndex++) {
        const beam = active[beamIndex];
        const input =
          beam.tokens.length === 0
            ? this.model.decoderStartId
            : beam.tokens[beam.tokens.length - 1];
        const logits = this.model.step(beam.state, input, cross);
        this.prepareLogits(logits, step, maxNewTokens);

        let max = -Infinity;
        for (let i = 0; i < logits.length; i++) {
          if (logits[i] > max) max = logits[i];
        }
        let sum = 0;
        for (let i = 0; i < logits.length; i++) {
          sum += Math.exp(logits[i] - max);
        }
        const logSum = Math.log(sum) + max;

        const top = topKLogits(logits, 2 * numBeams);
        for (let i = 0; i < top.ids.length; i++) {
          candidates.push({
            beamIndex,
            token: top.ids[i],
            score: beam.score + (top.scores[i] - logSum),
          });
        }
      }

      candidates.sort((a, b) => b.score - a.score);
      if (candidates.length > 2 * numBeams) candidates.length = 2 * numBeams;
      const nextActive: Beam[] = [];

      for (const candidate of candidates) {
        if (candidate.token === this.model.eosId) {
          finished.push({
            tokens: active[candidate.beamIndex].tokens.slice(),
            score: candidate.score,
            endedWithEos: true,
          });
        } else if (nextActive.length < numBeams) {
          const parent = active[candidate.beamIndex];
          nextActive.push({
            tokens: [...parent.tokens, candidate.token],
            score: candidate.score,
            state: parent.state.clone(),
          });
        }
      }

      active = nextActive;

      if (finished.length >= numBeams && active.length > 0) {
        const worstFinished = finished
          .map((h) => h.score / (h.tokens.length + 1) ** lengthPenalty)
          .sort((a, b) => a - b)[0];
        const bestActive = Math.max(
          ...active.map((b) => b.score / (b.tokens.length + 1) ** lengthPenalty),
        );
        if (worstFinished >= bestActive) break;
      }
    }

    for (const beam of active) {
      finished.push({
        tokens: beam.tokens.slice(),
        score: beam.score,
        endedWithEos: false,
      });
    }
    if (finished.length === 0) return [];

    finished.sort(
      (a, b) =>
        b.score / (b.tokens.length + 1) ** lengthPenalty -
        a.score / (a.tokens.length + 1) ** lengthPenalty,
    );
    const best = finished[0];
    const result = best.tokens.slice();
    if (best.endedWithEos) result.push(this.model.eosId);
    return result;
  }
}

export async function loadElanMT(
  source: string | ElanMTAssets,
  options: LoadOptions = {},
): Promise<ElanMT> {
  let config: MarianConfig;
  let spm: ArrayBuffer | Uint8Array;
  let weights: ArrayBuffer | Uint8Array;

  if (typeof source === "string") {
    const fetchImpl = options.fetch ?? fetch;
    const configResponse = await fetchImpl(joinUrl(source, "config.json"));
    if (!configResponse.ok) {
      throw new Error(`ElanMT: failed to fetch config.json (${configResponse.status})`);
    }
    config = (await configResponse.json()) as MarianConfig;
    const spmResponse = await fetchImpl(joinUrl(source, "source.spm"));
    if (!spmResponse.ok) {
      throw new Error(`ElanMT: failed to fetch source.spm (${spmResponse.status})`);
    }
    spm = await spmResponse.arrayBuffer();
    weights = await fetchBuffer(
      joinUrl(source, "model.safetensors"),
      fetchImpl,
      options.onProgress,
    );
  } else {
    config = decodeConfig(source.config);
    spm = source.spm;
    weights = source.weights;
  }

  const tokenizer = new SentencePieceTokenizer(spm, config.eos_token_id ?? 0, 1);
  const tensors = parseSafetensors(weights);
  const model = new MarianModel(config, tensors);
  return new ElanMT(config, tokenizer, model, options);
}

export default loadElanMT;
