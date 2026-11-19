/** Generator-friendly Marian encoder/decoder over int8 block-data weights. */

import type { WeightCursor } from "./byteCursor";
import {
  computeLogits,
  linearInto,
  readFloatVector,
  readRowDequantized,
  readVectorDequantized,
} from "./linear";
import {
  crossAttentionStep,
  encoderAttention,
  layerNormInto,
  positionVector,
  selfAttentionStep,
  silu,
} from "./ops";
import type { FloatLinearWeights, LinearWeights, ModelConfig, ModelWeights } from "./types";

interface NormVectors {
  weight: number[];
  bias: number[];
}

interface EncoderNormVectors {
  selfAttn: NormVectors;
  final: NormVectors;
}

interface DecoderNormVectors {
  selfAttn: NormVectors;
  encoderAttn: NormVectors;
  final: NormVectors;
}

export interface DecoderState {
  selfKeys: number[][][];
  selfValues: number[][][];
  length: number;
}

export interface CrossCache {
  keys: number[][][];
  values: number[][][];
}

type Done = () => void;
type Task = (done: Done) => void;

function series(tasks: Task[], done: Done): void {
  let index = 0;
  const next = (): void => {
    if (index >= tasks.length) {
      done();
      return;
    }
    const task = tasks[index++];
    task(next);
  };
  next();
}

export class BlockMarian {
  readonly config: ModelConfig;
  readonly weights: ModelWeights;

  private rowScales: number[] = [];
  private readonly biasCache = new Map<number, number[]>();
  private embedScale = 1;
  private finalLogitsBias: number[] = [];
  private prepared = false;
  private prepareCallbacks: Done[] = [];

  private encoderNorms: EncoderNormVectors[] = [];
  private decoderNorms: DecoderNormVectors[] = [];

  private readonly encoderNormCache: Partial<EncoderNormVectors>[] = [];
  private readonly decoderNormCache: Partial<DecoderNormVectors>[] = [];

  constructor(
    readonly cursor: WeightCursor,
    config: ModelConfig,
    weights: ModelWeights,
  ) {
    this.config = config;
    this.weights = weights;
  }

  prepare(callback: Done): void {
    this.embedScale = this.config.scaleEmbedding ? Math.sqrt(this.config.dModel) : 1;
    if (this.prepared) {
      callback();
      return;
    }
    this.prepareCallbacks.push(callback);
    if (this.prepareCallbacks.length > 1) return;

    const tasks: Task[] = [];
    tasks.push((done) => {
      this.cursor.readFloats(
        this.weights.rowScales.offset,
        this.weights.rowScales.count,
        (values) => {
          this.rowScales = values;
          done();
        },
      );
    });
    tasks.push((done) => {
      this.cursor.readFloats(
        this.weights.finalLogitsBias.offset,
        this.weights.finalLogitsBias.count,
        (values) => {
          this.finalLogitsBias = values;
          done();
        },
      );
    });

    for (let layer = 0; layer < this.config.encoderLayers; layer++) {
      const weights = this.weights.encoderLayers[layer];
      for (const linear of [
        weights.selfAttn.q,
        weights.selfAttn.k,
        weights.selfAttn.v,
        weights.selfAttn.out,
        weights.fc1,
        weights.fc2,
      ]) {
        tasks.push((done) => this.loadBias(linear, done));
      }
      tasks.push((done) =>
        this.loadNormPair(weights.selfAttnNorm, (norm) => {
          this.encoderNormCache[layer] = { selfAttn: norm };
          done();
        }),
      );
      tasks.push((done) =>
        this.loadNormPair(weights.finalNorm, (norm) => {
          const entry = this.encoderNormCache[layer]!;
          entry.final = norm;
          done();
        }),
      );
    }

    for (let layer = 0; layer < this.config.decoderLayers; layer++) {
      const weights = this.weights.decoderLayers[layer];
      for (const linear of [
        weights.selfAttn.q,
        weights.selfAttn.k,
        weights.selfAttn.v,
        weights.selfAttn.out,
        weights.encoderAttn.q,
        weights.encoderAttn.k,
        weights.encoderAttn.v,
        weights.encoderAttn.out,
        weights.fc1,
        weights.fc2,
      ]) {
        tasks.push((done) => this.loadBias(linear, done));
      }
      tasks.push((done) =>
        this.loadNormPair(weights.selfAttnNorm, (norm) => {
          this.decoderNormCache[layer] = { selfAttn: norm };
          done();
        }),
      );
      tasks.push((done) =>
        this.loadNormPair(weights.encoderAttnNorm, (norm) => {
          const entry = this.decoderNormCache[layer]!;
          entry.encoderAttn = norm;
          done();
        }),
      );
      tasks.push((done) =>
        this.loadNormPair(weights.finalNorm, (norm) => {
          const entry = this.decoderNormCache[layer]!;
          entry.final = norm;
          done();
        }),
      );
    }

    series(tasks, () => {
      this.encoderNorms = new Array(this.config.encoderLayers);
      this.decoderNorms = new Array(this.config.decoderLayers);
      for (let i = 0; i < this.config.encoderLayers; i++) {
        const cached = this.encoderNormCache[i]!;
        this.encoderNorms[i] = {
          selfAttn: cached.selfAttn!,
          final: cached.final!,
        };
      }
      for (let i = 0; i < this.config.decoderLayers; i++) {
        const cached = this.decoderNormCache[i]!;
        this.decoderNorms[i] = {
          selfAttn: cached.selfAttn!,
          encoderAttn: cached.encoderAttn!,
          final: cached.final!,
        };
      }
      this.prepared = true;
      const callbacks = this.prepareCallbacks;
      this.prepareCallbacks = [];
      for (let i = 0; i < callbacks.length; i++) callbacks[i]();
    });
  }

  private loadNormPair(weights: FloatLinearWeights, callback: (norm: NormVectors) => void): void {
    let weight: number[] | undefined;
    readFloatVector(this.cursor, weights.weight, (w) => {
      weight = w;
      readFloatVector(this.cursor, weights.bias, (b) => {
        callback({ weight: weight as number[], bias: b });
      });
    });
  }

  private loadBias(weights: LinearWeights, callback: Done): void {
    if (this.biasCache.has(weights.bias.offset)) {
      callback();
      return;
    }
    readVectorDequantized(this.cursor, this.rowScales, weights.bias, (bias) => {
      this.biasCache.set(weights.bias.offset, bias);
      callback();
    });
  }

  private linearWithBias(
    weights: LinearWeights,
    inputs: number[][],
    outs: number[][],
    callback: Done,
  ): void {
    linearInto(this.cursor, this.rowScales, weights.weight, inputs, outs, () => {
      const bias = this.biasCache.get(weights.bias.offset);
      if (bias === undefined) {
        throw new Error(`Bias ${weights.bias.name} was not loaded`);
      }
      for (let t = 0; t < outs.length; t++) {
        const out = outs[t];
        for (let i = 0; i < out.length; i++) {
          out[i] = Math.fround(out[i] - -bias[i]);
        }
      }
      callback();
    });
  }

  private assertPrepared(): void {
    if (!this.prepared) throw new Error("BlockMarian.prepare() has not finished");
  }

  private embedding(token: number, pos: number, callback: (vec: number[]) => void): void {
    readRowDequantized(this.cursor, this.rowScales, this.weights.shared, token, (vec) => {
      const position = positionVector(pos, this.config.dModel);
      for (let i = 0; i < vec.length; i++) {
        vec[i] = Math.fround(vec[i] * this.embedScale - -position[i]);
      }
      callback(vec);
    });
  }

  private encoderLayer(
    layerIndex: number,
    x: number[][],
    callback: (out: number[][]) => void,
  ): void {
    const layer = this.weights.encoderLayers[layerIndex];
    const norms = this.encoderNorms[layerIndex];
    const length = x.length;
    const d = this.config.dModel;
    const heads = this.config.encoderHeads;

    const q = Array.from({ length }, () => new Array<number>(d).fill(0));
    const k = Array.from({ length }, () => new Array<number>(d).fill(0));
    const v = Array.from({ length }, () => new Array<number>(d).fill(0));
    const attn = Array.from({ length }, () => new Array<number>(d).fill(0));
    const normed = Array.from({ length }, () => new Array<number>(d).fill(0));
    const hidden = Array.from({ length }, () => new Array<number>(this.config.encoderFfn).fill(0));
    const ffn = Array.from({ length }, () => new Array<number>(d).fill(0));

    this.linearWithBias(layer.selfAttn.q, x, q, () => {
      this.linearWithBias(layer.selfAttn.k, x, k, () => {
        this.linearWithBias(layer.selfAttn.v, x, v, () => {
          encoderAttention(q, k, v, heads, attn);
          const projected = Array.from({ length }, () => new Array<number>(d).fill(0));
          this.linearWithBias(layer.selfAttn.out, attn, projected, () => {
            for (let t = 0; t < length; t++) {
              layerNormInto(
                addVectors(x[t], projected[t]),
                norms.selfAttn.weight,
                norms.selfAttn.bias,
                normed[t],
              );
            }

            this.linearWithBias(layer.fc1, normed, hidden, () => {
              for (let t = 0; t < length; t++) {
                for (let i = 0; i < hidden[t].length; i++) hidden[t][i] = silu(hidden[t][i]);
              }
              this.linearWithBias(layer.fc2, hidden, ffn, () => {
                const out = new Array<number[]>(length);
                for (let t = 0; t < length; t++) {
                  const next = new Array<number>(d).fill(0);
                  layerNormInto(
                    addVectors(normed[t], ffn[t]),
                    norms.final.weight,
                    norms.final.bias,
                    next,
                  );
                  out[t] = next;
                }
                callback(out);
              });
            });
          });
        });
      });
    });
  }

  encode(ids: number[], callback: (memory: number[][]) => void): void {
    this.assertPrepared();
    const tasks: Task[] = [];
    const embeddings: number[][] = new Array(ids.length);
    for (let i = 0; i < ids.length; i++) {
      tasks.push((done) => {
        this.embedding(ids[i], i, (vec) => {
          embeddings[i] = vec;
          done();
        });
      });
    }
    series(tasks, () => {
      let layer = 0;
      const run = (x: number[][]): void => {
        if (layer >= this.config.encoderLayers) {
          callback(x);
          return;
        }
        this.encoderLayer(layer++, x, run);
      };
      run(embeddings);
    });
  }

  createCrossCache(memory: number[][], callback: (cross: CrossCache) => void): void {
    this.assertPrepared();
    const keys: number[][][] = new Array(this.config.decoderLayers);
    const values: number[][][] = new Array(this.config.decoderLayers);
    const tasks: Task[] = [];
    for (let layer = 0; layer < this.config.decoderLayers; layer++) {
      const weights = this.weights.decoderLayers[layer].encoderAttn;
      keys[layer] = Array.from({ length: memory.length }, () =>
        new Array<number>(this.config.dModel).fill(0),
      );
      values[layer] = Array.from({ length: memory.length }, () =>
        new Array<number>(this.config.dModel).fill(0),
      );
      tasks.push((done) => {
        this.linearWithBias(weights.k, memory, keys[layer], () => {
          this.linearWithBias(weights.v, memory, values[layer], done);
        });
      });
    }
    series(tasks, () => callback({ keys, values }));
  }

  createState(): DecoderState {
    return {
      selfKeys: Array.from({ length: this.config.decoderLayers }, () => []),
      selfValues: Array.from({ length: this.config.decoderLayers }, () => []),
      length: 0,
    };
  }

  decode(
    token: number,
    state: DecoderState,
    cross: CrossCache,
    callback: (logits: number[]) => void,
  ): void {
    this.assertPrepared();
    const pos = state.length;
    if (pos >= this.config.maxPositions) {
      throw new RangeError("Decoder exceeded the position table");
    }

    this.embedding(token, pos, (embedding) => {
      const x = embedding;
      const d = this.config.dModel;
      const heads = this.config.decoderHeads;

      const runLayer = (layerIndex: number): void => {
        if (layerIndex >= this.config.decoderLayers) {
          state.length = pos + 1;
          computeLogits(this.cursor, this.rowScales, this.weights.shared, x, (logits) => {
            for (let i = 0; i < logits.length; i++)
              logits[i] = Math.fround(logits[i] - -this.finalLogitsBias[i]);
            callback(logits);
          });
          return;
        }

        const layer = this.weights.decoderLayers[layerIndex];
        const norms = this.decoderNorms[layerIndex];
        const q = new Array<number>(d).fill(0);
        const k = new Array<number>(d).fill(0);
        const v = new Array<number>(d).fill(0);
        const attn = new Array<number>(d).fill(0);

        this.linearWithBias(layer.selfAttn.q, [x], [q], () => {
          this.linearWithBias(layer.selfAttn.k, [x], [k], () => {
            this.linearWithBias(layer.selfAttn.v, [x], [v], () => {
              state.selfKeys[layerIndex][pos] = k;
              state.selfValues[layerIndex][pos] = v;
              selfAttentionStep(
                q,
                state.selfKeys[layerIndex],
                state.selfValues[layerIndex],
                pos,
                heads,
                attn,
              );
              const selfProjected = new Array<number>(d).fill(0);
              this.linearWithBias(layer.selfAttn.out, [attn], [selfProjected], () => {
                const afterSelf = new Array<number>(d).fill(0);
                layerNormInto(
                  addVectors(x, selfProjected),
                  norms.selfAttn.weight,
                  norms.selfAttn.bias,
                  afterSelf,
                );

                const crossQ = new Array<number>(d).fill(0);
                this.linearWithBias(layer.encoderAttn.q, [afterSelf], [crossQ], () => {
                  const crossOut = new Array<number>(d).fill(0);
                  crossAttentionStep(
                    crossQ,
                    cross.keys[layerIndex],
                    cross.values[layerIndex],
                    heads,
                    crossOut,
                  );
                  const crossProjected = new Array<number>(d).fill(0);
                  this.linearWithBias(layer.encoderAttn.out, [crossOut], [crossProjected], () => {
                    const afterCross = new Array<number>(d).fill(0);
                    layerNormInto(
                      addVectors(afterSelf, crossProjected),
                      norms.encoderAttn.weight,
                      norms.encoderAttn.bias,
                      afterCross,
                    );

                    const hidden = new Array<number>(this.config.decoderFfn).fill(0);
                    this.linearWithBias(layer.fc1, [afterCross], [hidden], () => {
                      for (let i = 0; i < hidden.length; i++) hidden[i] = silu(hidden[i]);
                      const ffn = new Array<number>(d).fill(0);
                      this.linearWithBias(layer.fc2, [hidden], [ffn], () => {
                        const next = new Array<number>(d).fill(0);
                        layerNormInto(
                          addVectors(afterCross, ffn),
                          norms.final.weight,
                          norms.final.bias,
                          next,
                        );
                        for (let i = 0; i < d; i++) x[i] = next[i];
                        runLayer(layerIndex + 1);
                      });
                    });
                  });
                });
              });
            });
          });
        });
      };

      runLayer(0);
    });
  }
}

function addVectors(a: number[], b: number[]): number[] {
  const out = new Array<number>(a.length).fill(0);
  for (let i = 0; i < a.length; i++) out[i] = Math.fround(a[i] - -b[i]);
  return out;
}
