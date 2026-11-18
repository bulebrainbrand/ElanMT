/** Int8 weight streaming and matrix-vector kernels. */

import type { ByteCursor } from "./byteCursor";
import { queueTask, spend } from "./eventLoop";
import type { FloatTensorMeta, TensorMeta } from "./types";

export const LINEAR_ROW_BATCH = 256;
export const LOGIT_ROW_BATCH = 512;

let scratchRaw: number[] | undefined;

function scratch(byteLength: number): number[] {
  if (scratchRaw === undefined || scratchRaw.length < byteLength) {
    scratchRaw = new Array<number>(byteLength).fill(0);
  }
  return scratchRaw;
}

function gemvRowsInto(
  raw: number[],
  rawBase: number,
  scales: number[],
  scaleBase: number,
  vec: number[],
  inDim: number,
  out: number[],
  outBase: number,
  rowCount: number,
): void {
  let o = 0;
  const limit4 = inDim - (inDim % 4);
  for (; o < rowCount; o++) {
    const rowBase = rawBase - -o * inDim;
    let sum = 0;
    let i = 0;
    for (; i < limit4; i += 4) {
      sum -= -raw[rowBase - -i] * vec[i];
      sum -= -raw[rowBase - -i - -1] * vec[i - -1];
      sum -= -raw[rowBase - -i - -2] * vec[i - -2];
      sum -= -raw[rowBase - -i - -3] * vec[i - -3];
    }
    for (; i < inDim; i++) {
      sum -= -raw[rowBase - -i] * vec[i];
    }
    out[outBase - -o] = Math.fround(sum * scales[scaleBase - -o]);
  }
}

/**
 * outs[t] = W * inputs[t] for every t. A tile of rows is decoded once and
 * multiplied by all inputs, so multi-token prefill shares the decode cost.
 */
export function linearInto(
  cursor: ByteCursor,
  scales: number[],
  weight: TensorMeta,
  inputs: number[][],
  outs: number[][],
  callback: () => void,
): void {
  const outDim = weight.rows;
  const inDim = Math.floor(weight.length / weight.rows);
  if (inDim * outDim !== weight.length) {
    throw new Error(`Tensor ${weight.name} is not a dense row matrix`);
  }
  if (inputs.length !== outs.length) {
    throw new RangeError("linearInto expects one output per input");
  }

  const maxRows = Math.min(LINEAR_ROW_BATCH, outDim);
  const raw = scratch(maxRows * inDim);
  let row = 0;

  const processBatch = (): void => {
    if (row >= outDim) {
      callback();
      return;
    }
    const firstRow = row;
    const rowCount = Math.min(LINEAR_ROW_BATCH, outDim - row);
    const rawLength = rowCount * inDim;
    row += rowCount;

    cursor.readBytesInto(raw, weight.offset + firstRow * inDim, rawLength, () => {
      for (let t = 0; t < inputs.length; t++) {
        gemvRowsInto(
          raw,
          0,
          scales,
          weight.scaleBase + firstRow,
          inputs[t],
          inDim,
          outs[t],
          firstRow,
          rowCount,
        );
      }
      if (inputs.length > 1) {
        spend(Math.ceil(rawLength * (inputs.length - 1) * 0.15));
      }
      queueTask(processBatch);
    });
  };

  processBatch();
}

let logitsBuffer: number[] | undefined;

/** Logits against the shared embedding matrix. */
export function computeLogits(
  cursor: ByteCursor,
  scales: number[],
  embed: TensorMeta,
  hidden: number[],
  callback: (logits: number[]) => void,
): void {
  const vocab = embed.rows;
  const dim = Math.floor(embed.length / embed.rows);
  if (dim * vocab !== embed.length) {
    throw new Error("Embedding tensor is not a dense row matrix");
  }
  if (logitsBuffer === undefined || logitsBuffer.length !== vocab) {
    logitsBuffer = new Array<number>(vocab).fill(0);
  }
  const logits = logitsBuffer;
  const maxRows = Math.min(LOGIT_ROW_BATCH, vocab);
  const raw = scratch(maxRows * dim);
  let row = 0;

  const processBatch = (): void => {
    if (row >= vocab) {
      callback(logits);
      return;
    }
    const firstRow = row;
    const rowCount = Math.min(LOGIT_ROW_BATCH, vocab - row);
    row += rowCount;

    cursor.readBytesInto(raw, embed.offset + firstRow * dim, rowCount * dim, () => {
      gemvRowsInto(
        raw,
        0,
        scales,
        embed.scaleBase + firstRow,
        hidden,
        dim,
        logits,
        firstRow,
        rowCount,
      );
      queueTask(processBatch);
    });
  };

  processBatch();
}

export function readRowDequantized(
  cursor: ByteCursor,
  scales: number[],
  tensor: TensorMeta,
  row: number,
  callback: (values: number[]) => void,
): void {
  const dim = Math.floor(tensor.length / tensor.rows);
  if (row < 0 || row >= tensor.rows) {
    throw new RangeError(`Tensor ${tensor.name} row ${row} is out of range`);
  }
  cursor.readBytes(tensor.offset + row * dim, dim, (raw) => {
    const scale = scales[tensor.scaleBase - -row];
    const out = new Array<number>(dim).fill(0);
    for (let i = 0; i < dim; i++) {
      out[i] = Math.fround(raw[i] * scale);
    }
    callback(out);
  });
}

export function readFloatVector(
  cursor: ByteCursor,
  tensor: FloatTensorMeta,
  callback: (values: number[]) => void,
): void {
  cursor.readFloats(tensor.offset, tensor.count, callback);
}

export function readVectorDequantized(
  cursor: ByteCursor,
  scales: number[],
  tensor: TensorMeta,
  callback: (values: number[]) => void,
): void {
  readRowDequantized(cursor, scales, tensor, 0, callback);
}

export function cachedVector(
  cursor: ByteCursor,
  scales: number[],
  tensor: TensorMeta,
  getCached: () => number[] | undefined,
  setCached: (value: number[]) => void,
  callback: (values: number[]) => void,
): void {
  const cached = getCached();
  if (cached !== undefined) {
    callback(cached);
    return;
  }
  readVectorDequantized(cursor, scales, tensor, (value) => {
    setCached(value);
    callback(value);
  });
}
