/**
 * MarianMT (encoder-decoder transformer) forward pass and incremental decoder.
 *
 * This mirrors Hugging Face's Marian implementation used by
 * Mitsua/elan-mt-bt-ja-en: post-norm layers, sinusoidal positions, scaled
 * embeddings, tied output projection, and a final logits bias.
 */

import type { Tensor, TensorMap } from "./safetensors";

export interface MarianConfig {
  d_model: number;
  encoder_layers: number;
  decoder_layers: number;
  encoder_attention_heads: number;
  decoder_attention_heads: number;
  encoder_ffn_dim: number;
  decoder_ffn_dim: number;
  activation_function?: string;
  scale_embedding?: boolean;
  max_position_embeddings?: number;
  vocab_size: number;
  decoder_vocab_size?: number;
  eos_token_id?: number;
  pad_token_id?: number;
  decoder_start_token_id?: number;
  bos_token_id?: number;
  num_beams?: number;
  max_length?: number;
  bad_words_ids?: number[][];
  static_position_embeddings?: boolean;
  normalize_embedding?: boolean;
}

interface LinearWeights {
  w: Float32Array;
  b: Float32Array;
  inDim: number;
  outDim: number;
}

interface AttentionWeights {
  q: LinearWeights;
  k: LinearWeights;
  v: LinearWeights;
  out: LinearWeights;
}

interface EncoderLayerWeights {
  selfAttn: AttentionWeights;
  selfAttnNorm: LinearWeights;
  finalNorm: LinearWeights;
  fc1: LinearWeights;
  fc2: LinearWeights;
}

interface DecoderLayerWeights {
  selfAttn: AttentionWeights;
  selfAttnNorm: LinearWeights;
  encoderAttn: AttentionWeights;
  encoderAttnNorm: LinearWeights;
  finalNorm: LinearWeights;
  fc1: LinearWeights;
  fc2: LinearWeights;
}

export interface CrossCache {
  k: Float32Array[];
  v: Float32Array[];
  length: number;
}

function sinusoidalPositions(maxPositions: number, dModel: number): Float32Array {
  const table = new Float32Array(maxPositions * dModel);
  const half = Math.floor(dModel / 2);
  // Torch's MarianSinusoidalPositionalEmbedding uses log(10000) / half_dim.
  const rate = Math.log(10000) / half;
  for (let pos = 0; pos < maxPositions; pos++) {
    for (let i = 0; i < half; i++) {
      const angle = pos * Math.exp(-i * rate);
      table[pos * dModel + i] = Math.sin(angle);
      table[pos * dModel + half + i] = Math.cos(angle);
    }
    if (dModel % 2 === 1) table[pos * dModel + dModel - 1] = 0;
  }
  return table;
}

function activation(name: string, x: number): number {
  if (name === "relu") return x > 0 ? x : 0;
  if (name === "gelu" || name === "gelu_new") {
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
  }
  if (name === "tanh") return Math.tanh(x);
  // Marian's default "swish" == SiLU.
  return x / (1 + Math.exp(-x));
}

/** out = x @ W^T + b, with W stored as [outDim, inDim]. */
function linear(out: Float32Array, x: Float32Array, xOffset: number, weights: LinearWeights): void {
  const { w, b, inDim, outDim } = weights;
  for (let o = 0; o < outDim; o++) {
    const row = o * inDim;
    let sum = b[o];
    for (let i = 0; i < inDim; i++) sum += x[xOffset + i] * w[row + i];
    out[o] = sum;
  }
}

function layerNorm(
  x: Float32Array,
  offset: number,
  dim: number,
  weight: Float32Array,
  bias: Float32Array,
): void {
  let mean = 0;
  for (let i = 0; i < dim; i++) mean += x[offset + i];
  mean /= dim;
  let variance = 0;
  for (let i = 0; i < dim; i++) {
    const diff = x[offset + i] - mean;
    variance += diff * diff;
  }
  variance /= dim;
  const inv = 1 / Math.sqrt(variance + 1e-5);
  for (let i = 0; i < dim; i++) {
    x[offset + i] = (x[offset + i] - mean) * inv * weight[i] + bias[i];
  }
}

export class MarianModel {
  readonly config: MarianConfig;
  readonly dModel: number;
  readonly vocabSize: number;
  readonly maxPositions: number;
  readonly numBeams: number;
  readonly maxLength: number;
  readonly eosId: number;
  readonly padId: number;
  readonly decoderStartId: number;
  readonly badWordIds: number[][];

  private readonly encoderEmbed: Float32Array;
  private readonly decoderEmbed: Float32Array;
  private readonly lmHead: Float32Array;
  private readonly finalLogitsBias: Float32Array;
  private readonly encoderPositions: Float32Array;
  private readonly decoderPositions: Float32Array;
  private readonly encoderLayers: EncoderLayerWeights[] = [];
  private readonly decoderLayers: DecoderLayerWeights[] = [];
  private readonly embedScale: number;
  private readonly activationName: string;
  private readonly encoderHeads: number;
  private readonly decoderHeads: number;

  constructor(config: MarianConfig, tensors: TensorMap) {
    this.config = config;
    this.dModel = config.d_model;
    this.vocabSize = config.decoder_vocab_size ?? config.vocab_size;
    this.maxPositions = config.max_position_embeddings ?? Math.max(512, config.max_length ?? 512);
    this.numBeams = config.num_beams ?? 4;
    this.maxLength = config.max_length ?? 512;
    this.eosId = config.eos_token_id ?? 0;
    this.padId = config.pad_token_id ?? this.vocabSize - 1;
    this.decoderStartId = config.decoder_start_token_id ?? this.padId;
    this.badWordIds = config.bad_words_ids ?? [];
    this.embedScale = config.scale_embedding ? Math.sqrt(this.dModel) : 1;
    this.activationName = config.activation_function ?? "swish";
    this.encoderHeads = config.encoder_attention_heads;
    this.decoderHeads = config.decoder_attention_heads;

    if (config.normalize_embedding) {
      throw new Error("marian: normalize_embedding is not supported");
    }

    const get = (name: string): Tensor => {
      const tensor = tensors.get(name);
      if (!tensor) throw new Error(`marian: missing tensor ${name}`);
      return tensor;
    };
    const optional = (name: string): Tensor | undefined => tensors.get(name);

    const shared = get("model.shared.weight");
    this.encoderEmbed = (optional("model.encoder.embed_tokens.weight") ?? shared).data;
    this.decoderEmbed = (optional("model.decoder.embed_tokens.weight") ?? shared).data;
    this.lmHead = (optional("lm_head.weight") ?? shared).data;
    const bias = optional("final_logits_bias");
    this.finalLogitsBias = bias ? bias.data : new Float32Array(this.vocabSize);

    const encPos = optional("model.encoder.embed_positions.weight");
    const decPos = optional("model.decoder.embed_positions.weight");
    this.encoderPositions = encPos
      ? encPos.data
      : sinusoidalPositions(this.maxPositions, this.dModel);
    this.decoderPositions = decPos
      ? decPos.data
      : sinusoidalPositions(this.maxPositions, this.dModel);

    const readLinear = (prefix: string): LinearWeights => {
      const w = get(`${prefix}.weight`);
      const b = get(`${prefix}.bias`);
      const outDim = w.shape[0];
      const inDim = w.shape[1] ?? 1;
      if (b.data.length !== outDim) {
        throw new Error(`marian: bad bias shape for ${prefix}`);
      }
      return { w: w.data, b: b.data, inDim, outDim };
    };

    const readAttention = (prefix: string): AttentionWeights => ({
      q: readLinear(`${prefix}.q_proj`),
      k: readLinear(`${prefix}.k_proj`),
      v: readLinear(`${prefix}.v_proj`),
      out: readLinear(`${prefix}.out_proj`),
    });

    for (let i = 0; i < config.encoder_layers; i++) {
      const prefix = `model.encoder.layers.${i}`;
      this.encoderLayers.push({
        selfAttn: readAttention(`${prefix}.self_attn`),
        selfAttnNorm: readLinear(`${prefix}.self_attn_layer_norm`),
        finalNorm: readLinear(`${prefix}.final_layer_norm`),
        fc1: readLinear(`${prefix}.fc1`),
        fc2: readLinear(`${prefix}.fc2`),
      });
    }

    for (let i = 0; i < config.decoder_layers; i++) {
      const prefix = `model.decoder.layers.${i}`;
      this.decoderLayers.push({
        selfAttn: readAttention(`${prefix}.self_attn`),
        selfAttnNorm: readLinear(`${prefix}.self_attn_layer_norm`),
        encoderAttn: readAttention(`${prefix}.encoder_attn`),
        encoderAttnNorm: readLinear(`${prefix}.encoder_attn_layer_norm`),
        finalNorm: readLinear(`${prefix}.final_layer_norm`),
        fc1: readLinear(`${prefix}.fc1`),
        fc2: readLinear(`${prefix}.fc2`),
      });
    }
  }

  private embed(
    out: Float32Array,
    ids: readonly number[],
    embed: Float32Array,
    positions: Float32Array,
    length: number,
  ): void {
    const d = this.dModel;
    const scale = this.embedScale;
    for (let t = 0; t < length; t++) {
      const id = ids[t];
      if (id < 0 || id * d >= embed.length) {
        throw new Error(`marian: token id ${id} is out of range`);
      }
      if (t >= this.maxPositions) {
        throw new Error("marian: input is longer than the position table");
      }
      const src = id * d;
      const dst = t * d;
      const pos = t * d;
      for (let i = 0; i < d; i++) {
        out[dst + i] = embed[src + i] * scale + positions[pos + i];
      }
    }
  }

  private projectAll(input: Float32Array, length: number, weights: LinearWeights): Float32Array {
    const d = this.dModel;
    const out = new Float32Array(length * d);
    const row = new Float32Array(d);
    for (let t = 0; t < length; t++) {
      linear(row, input, t * d, weights);
      out.set(row, t * d);
    }
    return out;
  }

  private attention(
    input: Float32Array,
    length: number,
    weights: AttentionWeights,
    heads: number,
  ): Float32Array {
    const d = this.dModel;
    const headDim = d / heads;
    const q = this.projectAll(input, length, weights.q);
    const k = this.projectAll(input, length, weights.k);
    const v = this.projectAll(input, length, weights.v);
    const context = new Float32Array(length * d);
    const scores = new Float32Array(length);
    const scale = 1 / Math.sqrt(headDim);

    for (let h = 0; h < heads; h++) {
      const base = h * headDim;
      for (let i = 0; i < length; i++) {
        let max = -Infinity;
        for (let j = 0; j < length; j++) {
          let dot = 0;
          for (let e = 0; e < headDim; e++) {
            dot += q[i * d + base + e] * k[j * d + base + e];
          }
          const score = dot * scale;
          scores[j] = score;
          if (score > max) max = score;
        }
        let sum = 0;
        for (let j = 0; j < length; j++) {
          const e = Math.exp(scores[j] - max);
          scores[j] = e;
          sum += e;
        }
        const inv = 1 / sum;
        for (let j = 0; j < length; j++) {
          const p = scores[j] * inv;
          const vBase = j * d + base;
          const cBase = i * d + base;
          for (let e = 0; e < headDim; e++) {
            context[cBase + e] += p * v[vBase + e];
          }
        }
      }
    }

    const out = new Float32Array(length * d);
    const row = new Float32Array(d);
    for (let t = 0; t < length; t++) {
      linear(row, context, t * d, weights.out);
      out.set(row, t * d);
    }
    return out;
  }

  private feedForward(
    input: Float32Array,
    length: number,
    fc1: LinearWeights,
    fc2: LinearWeights,
  ): Float32Array {
    const d = this.dModel;
    const hidden = new Float32Array(fc1.outDim);
    const out = new Float32Array(length * d);
    const row = new Float32Array(d);
    for (let t = 0; t < length; t++) {
      linear(hidden, input, t * d, fc1);
      for (let i = 0; i < hidden.length; i++) {
        hidden[i] = activation(this.activationName, hidden[i]);
      }
      linear(row, hidden, 0, fc2);
      out.set(row, t * d);
    }
    return out;
  }

  private normalizeRows(x: Float32Array, length: number, norm: LinearWeights): void {
    for (let t = 0; t < length; t++) {
      layerNorm(x, t * this.dModel, this.dModel, norm.w, norm.b);
    }
  }

  /** Run the encoder and return memory [sourceLength, d_model]. */
  encode(ids: readonly number[]): Float32Array {
    const length = ids.length;
    const d = this.dModel;
    if (length === 0) return new Float32Array(0);

    const x = new Float32Array(length * d);
    this.embed(x, ids, this.encoderEmbed, this.encoderPositions, length);

    for (const layer of this.encoderLayers) {
      const attn = this.attention(x, length, layer.selfAttn, this.encoderHeads);
      for (let i = 0; i < x.length; i++) x[i] += attn[i];
      this.normalizeRows(x, length, layer.selfAttnNorm);

      const ffn = this.feedForward(x, length, layer.fc1, layer.fc2);
      for (let i = 0; i < x.length; i++) x[i] += ffn[i];
      this.normalizeRows(x, length, layer.finalNorm);
    }
    return x;
  }

  /** Precompute cross-attention keys/values for one encoded source. */
  createCrossCache(memory: Float32Array, sourceLength: number): CrossCache {
    const k: Float32Array[] = [];
    const v: Float32Array[] = [];
    for (const layer of this.decoderLayers) {
      k.push(this.projectAll(memory, sourceLength, layer.encoderAttn.k));
      v.push(this.projectAll(memory, sourceLength, layer.encoderAttn.v));
    }
    return { k, v, length: sourceLength };
  }

  createDecoderState(maxLength: number, maxNewTokens: number): DecoderState {
    return new DecoderState(
      this.dModel,
      this.decoderLayers.length,
      Math.min(this.maxPositions, maxLength + maxNewTokens + 1),
    );
  }

  private crossAttention(
    queryInput: Float32Array,
    queryLength: number,
    weights: AttentionWeights,
    heads: number,
    keys: Float32Array,
    values: Float32Array,
    keyLength: number,
  ): Float32Array {
    const d = this.dModel;
    const headDim = d / heads;
    const q = this.projectAll(queryInput, queryLength, weights.q);
    const context = new Float32Array(queryLength * d);
    const scores = new Float32Array(keyLength);
    const scale = 1 / Math.sqrt(headDim);

    for (let h = 0; h < heads; h++) {
      const base = h * headDim;
      for (let i = 0; i < queryLength; i++) {
        let max = -Infinity;
        for (let j = 0; j < keyLength; j++) {
          let dot = 0;
          for (let e = 0; e < headDim; e++) {
            dot += q[i * d + base + e] * keys[j * d + base + e];
          }
          const score = dot * scale;
          scores[j] = score;
          if (score > max) max = score;
        }
        let sum = 0;
        for (let j = 0; j < keyLength; j++) {
          const e = Math.exp(scores[j] - max);
          scores[j] = e;
          sum += e;
        }
        const inv = 1 / sum;
        for (let j = 0; j < keyLength; j++) {
          const p = scores[j] * inv;
          const vBase = j * d + base;
          const cBase = i * d + base;
          for (let e = 0; e < headDim; e++) {
            context[cBase + e] += p * values[vBase + e];
          }
        }
      }
    }

    const out = new Float32Array(queryLength * d);
    const row = new Float32Array(d);
    for (let t = 0; t < queryLength; t++) {
      linear(row, context, t * d, weights.out);
      out.set(row, t * d);
    }
    return out;
  }

  /**
   * Decode one token, updating the KV cache. Returns logits for the vocab.
   * Positions are 0-based and match Hugging Face's past_key_values_length.
   */
  step(state: DecoderState, token: number, cross: CrossCache): Float32Array {
    const d = this.dModel;
    const pos = state.length;
    if (pos >= this.maxPositions) {
      throw new Error("marian: decoder exceeded the position table");
    }
    state.ensureCapacity(pos + 1);

    const x = new Float32Array(d);
    const src = token * d;
    for (let i = 0; i < d; i++) {
      x[i] = this.decoderEmbed[src + i] * this.embedScale + this.decoderPositions[pos * d + i];
    }

    const q = new Float32Array(d);
    const k = new Float32Array(d);
    const v = new Float32Array(d);
    const context = new Float32Array(d);
    const scores = new Float32Array(pos + 1);
    const row = new Float32Array(d);

    for (let layerIndex = 0; layerIndex < this.decoderLayers.length; layerIndex++) {
      const layer = this.decoderLayers[layerIndex];
      const headDim = d / this.decoderHeads;
      const scale = 1 / Math.sqrt(headDim);

      // Masked self-attention over the cached prefix.
      linear(q, x, 0, layer.selfAttn.q);
      linear(k, x, 0, layer.selfAttn.k);
      linear(v, x, 0, layer.selfAttn.v);
      state.k[layerIndex].set(k, pos * d);
      state.v[layerIndex].set(v, pos * d);
      context.fill(0);

      for (let h = 0; h < this.decoderHeads; h++) {
        const base = h * headDim;
        let max = -Infinity;
        for (let j = 0; j <= pos; j++) {
          let dot = 0;
          const kBase = j * d + base;
          for (let e = 0; e < headDim; e++) dot += q[base + e] * state.k[layerIndex][kBase + e];
          const score = dot * scale;
          scores[j] = score;
          if (score > max) max = score;
        }
        let sum = 0;
        for (let j = 0; j <= pos; j++) {
          const e = Math.exp(scores[j] - max);
          scores[j] = e;
          sum += e;
        }
        const inv = 1 / sum;
        for (let j = 0; j <= pos; j++) {
          const p = scores[j] * inv;
          const vBase = j * d + base;
          for (let e = 0; e < headDim; e++) {
            context[base + e] += p * state.v[layerIndex][vBase + e];
          }
        }
      }

      linear(row, context, 0, layer.selfAttn.out);
      for (let i = 0; i < d; i++) x[i] += row[i];
      layerNorm(x, 0, d, layer.selfAttnNorm.w, layer.selfAttnNorm.b);

      // Cross-attention over the fixed encoder memory.
      const crossOut = this.crossAttention(
        x,
        1,
        layer.encoderAttn,
        this.decoderHeads,
        cross.k[layerIndex],
        cross.v[layerIndex],
        cross.length,
      );
      for (let i = 0; i < d; i++) x[i] += crossOut[i];
      layerNorm(x, 0, d, layer.encoderAttnNorm.w, layer.encoderAttnNorm.b);

      // Feed-forward.
      const hidden = new Float32Array(layer.fc1.outDim);
      linear(hidden, x, 0, layer.fc1);
      for (let i = 0; i < hidden.length; i++) {
        hidden[i] = activation(this.activationName, hidden[i]);
      }
      linear(row, hidden, 0, layer.fc2);
      for (let i = 0; i < d; i++) x[i] += row[i];
      layerNorm(x, 0, d, layer.finalNorm.w, layer.finalNorm.b);
    }

    state.length = pos + 1;

    const logits = new Float32Array(this.vocabSize);
    for (let id = 0; id < this.vocabSize; id++) {
      const base = id * d;
      let sum = this.finalLogitsBias[id] ?? 0;
      for (let i = 0; i < d; i++) sum += x[i] * this.lmHead[base + i];
      logits[id] = sum;
    }
    return logits;
  }
}

export class DecoderState {
  readonly k: Float32Array[];
  readonly v: Float32Array[];
  length = 0;
  private capacity: number;

  constructor(
    private readonly dModel: number,
    layers: number,
    capacity: number,
  ) {
    this.capacity = Math.max(1, capacity);
    this.k = Array.from({ length: layers }, () => new Float32Array(this.capacity * dModel));
    this.v = Array.from({ length: layers }, () => new Float32Array(this.capacity * dModel));
  }

  ensureCapacity(length: number): void {
    if (length <= this.capacity) return;
    let next = this.capacity;
    while (next < length) next *= 2;
    for (let i = 0; i < this.k.length; i++) {
      const k = new Float32Array(next * this.dModel);
      const v = new Float32Array(next * this.dModel);
      k.set(this.k[i].subarray(0, this.length * this.dModel));
      v.set(this.v[i].subarray(0, this.length * this.dModel));
      this.k[i] = k;
      this.v[i] = v;
    }
    this.capacity = next;
  }

  clone(): DecoderState {
    const copy = new DecoderState(
      this.dModel,
      this.k.length,
      Math.max(this.capacity, this.length + 1),
    );
    copy.length = this.length;
    const used = this.length * this.dModel;
    for (let i = 0; i < this.k.length; i++) {
      copy.k[i].set(this.k[i].subarray(0, used));
      copy.v[i].set(this.v[i].subarray(0, used));
    }
    return copy;
  }
}
