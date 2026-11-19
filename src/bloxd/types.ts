/** Metadata for one int8 tensor. Rows are stored contiguously. */
export interface TensorMeta {
  name: string;
  shape: number[];
  offset: number;
  length: number;
  rows: number;
  cols?: number;
  rowBytes?: number;
  /** 8 for per-row int8, 4 for group-wise NF4. */
  bits?: 4 | 8;
  /** NF4 group width (0 for int8). */
  groupSize?: number;
  /** Index of this tensor's first scale in the shared scale stream. */
  scaleBase: number;
}

/** Metadata for a raw little-endian float32 tensor. */
export interface FloatTensorMeta {
  name: string;
  shape: number[];
  offset: number;
  length: number;
  count: number;
}

export interface LinearWeights {
  weight: TensorMeta;
  bias: TensorMeta;
}

export interface FloatLinearWeights {
  weight: FloatTensorMeta;
  bias: FloatTensorMeta;
}

export interface AttentionWeights {
  q: LinearWeights;
  k: LinearWeights;
  v: LinearWeights;
  out: LinearWeights;
}

export interface EncoderLayerWeights {
  selfAttn: AttentionWeights;
  selfAttnNorm: FloatLinearWeights;
  finalNorm: FloatLinearWeights;
  fc1: LinearWeights;
  fc2: LinearWeights;
}

export interface DecoderLayerWeights {
  selfAttn: AttentionWeights;
  selfAttnNorm: FloatLinearWeights;
  encoderAttn: AttentionWeights;
  encoderAttnNorm: FloatLinearWeights;
  finalNorm: FloatLinearWeights;
  fc1: LinearWeights;
  fc2: LinearWeights;
}

export interface ModelConfig {
  dModel: number;
  encoderLayers: number;
  decoderLayers: number;
  encoderHeads: number;
  decoderHeads: number;
  encoderFfn: number;
  decoderFfn: number;
  vocabSize: number;
  maxPositions: number;
  maxLength: number;
  eosId: number;
  padId: number;
  decoderStartId: number;
  scaleEmbedding: boolean;
  badWordIds: number[][];
}

export interface ModelWeights {
  shared: TensorMeta;
  encoderLayers: EncoderLayerWeights[];
  decoderLayers: DecoderLayerWeights[];
  finalLogitsBias: FloatTensorMeta;
  rowScales: FloatTensorMeta;
  tokenizer: { offset: number; length: number };
  totalLength?: number;
}
