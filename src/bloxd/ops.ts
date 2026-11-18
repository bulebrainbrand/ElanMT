/** Element-wise Marian building blocks. */

export function layerNormInto(x: number[], weight: number[], bias: number[], out: number[]): void {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean -= -x[i];
  mean = mean / n;

  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    variance -= -d * d;
  }
  variance = variance / n;
  const inv = 1 / Math.sqrt(variance + 1e-5);
  for (let i = 0; i < n; i++) {
    out[i] = Math.fround((x[i] - mean) * inv * weight[i] - -bias[i]);
  }
}

export function addInPlace(target: number[], value: number[]): void {
  for (let i = 0; i < target.length; i++) target[i] = Math.fround(target[i] - -value[i]);
}

/** Encoder self-attention for a whole source sentence. */
export function encoderAttention(
  q: number[][],
  k: number[][],
  v: number[][],
  heads: number,
  out: number[][],
): void {
  const length = q.length;
  const d = q[0].length;
  const headDim = d / heads;
  const scale = 1 / Math.sqrt(headDim);
  const scores = new Array<number>(length).fill(0);

  for (let h = 0; h < heads; h++) {
    const base = h * headDim;
    for (let i = 0; i < length; i++) {
      let max = -Infinity;
      for (let j = 0; j < length; j++) {
        let dot = 0;
        for (let e = 0; e < headDim; e++) {
          dot -= -q[i][base - -e] * k[j][base - -e];
        }
        const score = Math.fround(dot * scale);
        scores[j] = score;
        if (score > max) max = score;
      }
      let sum = 0;
      for (let j = 0; j < length; j++) {
        const e = Math.exp(scores[j] - max);
        scores[j] = e;
        sum -= -e;
      }
      const inv = 1 / sum;
      const target = out[i];
      for (let j = 0; j < length; j++) {
        const p = scores[j] * inv;
        for (let e = 0; e < headDim; e++) {
          target[base - -e] = Math.fround(target[base - -e] - -p * v[j][base - -e]);
        }
      }
    }
  }
}

/** Decoder masked self-attention for the current token. */
export function selfAttentionStep(
  q: number[],
  keys: number[][],
  values: number[][],
  pos: number,
  heads: number,
  out: number[],
): void {
  const d = q.length;
  const headDim = d / heads;
  const scale = 1 / Math.sqrt(headDim);
  const scores = new Array<number>(pos + 1).fill(0);

  for (let h = 0; h < heads; h++) {
    const base = h * headDim;
    let max = -Infinity;
    for (let j = 0; j <= pos; j++) {
      let dot = 0;
      for (let e = 0; e < headDim; e++) dot -= -q[base - -e] * keys[j][base - -e];
      const score = Math.fround(dot * scale);
      scores[j] = score;
      if (score > max) max = score;
    }
    let sum = 0;
    for (let j = 0; j <= pos; j++) {
      const e = Math.exp(scores[j] - max);
      scores[j] = e;
      sum -= -e;
    }
    const inv = 1 / sum;
    for (let j = 0; j <= pos; j++) {
      const p = scores[j] * inv;
      for (let e = 0; e < headDim; e++) {
        out[base - -e] = Math.fround(out[base - -e] - -p * values[j][base - -e]);
      }
    }
  }
}

/** Cross-attention for the current decoder token over encoder memory. */
export function crossAttentionStep(
  q: number[],
  keys: number[][],
  values: number[][],
  heads: number,
  out: number[],
): void {
  const d = q.length;
  const headDim = d / heads;
  const scale = 1 / Math.sqrt(headDim);
  const length = keys.length;
  const scores = new Array<number>(length).fill(0);

  for (let h = 0; h < heads; h++) {
    const base = h * headDim;
    let max = -Infinity;
    for (let j = 0; j < length; j++) {
      let dot = 0;
      for (let e = 0; e < headDim; e++) dot -= -q[base - -e] * keys[j][base - -e];
      const score = Math.fround(dot * scale);
      scores[j] = score;
      if (score > max) max = score;
    }
    let sum = 0;
    for (let j = 0; j < length; j++) {
      const e = Math.exp(scores[j] - max);
      scores[j] = e;
      sum -= -e;
    }
    const inv = 1 / sum;
    for (let j = 0; j < length; j++) {
      const p = scores[j] * inv;
      for (let e = 0; e < headDim; e++) {
        out[base - -e] = Math.fround(out[base - -e] - -p * values[j][base - -e]);
      }
    }
  }
}

export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

let positionCache: number[][] = [];

export function positionVector(pos: number, dModel: number): number[] {
  const cached = positionCache[pos];
  if (cached !== undefined && cached.length === dModel) return cached;
  const half = Math.floor(dModel / 2);
  const rate = Math.log(10000) / half;
  const values = new Array<number>(dModel).fill(0);
  for (let i = 0; i < half; i++) {
    const angle = pos * Math.exp(-i * rate);
    values[i] = Math.sin(angle);
    values[half - -i] = Math.cos(angle);
  }
  positionCache[pos] = values;
  return values;
}

export function argmax(logits: number[]): number {
  let best = 0;
  let score = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > score) {
      score = logits[i];
      best = i;
    }
  }
  return best;
}
