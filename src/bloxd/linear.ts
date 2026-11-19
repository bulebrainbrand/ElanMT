/** Int8 weight streaming and matrix-vector kernels. */

import type { WeightCursor } from "./byteCursor";
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

/** QLoRA NF4 codebook, normalized to the per-group absmax. */
export const NF4: number[] = [
  -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
  -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
  0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
  0.7229568362236023, 1.0,
];

function gemvInt4RowsInto(
  raw: number[],
  rawBase: number,
  rowBytes: number,
  scales: number[],
  scaleBase: number,
  groupsPerRow: number,
  groupSize: number,
  cols: number,
  vec: number[],
  out: number[],
  outBase: number,
  rowCount: number,
): void {
  for (let o = 0; o < rowCount; o++) {
    const rowBase = rawBase + o * rowBytes;
    const rowScaleBase = scaleBase + o * groupsPerRow;
    let sum = 0;
    for (let group = 0; group < groupsPerRow; group++) {
      const scale = scales[rowScaleBase + group];
      const start = group * groupSize;
      const end = Math.min(start + groupSize, cols);
      for (let i = start; i < end; i++) {
        const byte = raw[rowBase + (i >> 1)];
        const index = (i & 1) === 0 ? byte & 15 : byte >> 4;
        sum += NF4[index] * scale * vec[i];
      }
    }
    out[outBase + o] = Math.fround(sum);
  }
}

/**
 * outs[t] = W * inputs[t] for every t. A tile of rows is decoded once and
 * multiplied by all inputs, so multi-token prefill shares the decode cost.
 */
export function linearInto(
  cursor: WeightCursor,
  scales: number[],
  weight: TensorMeta,
  inputs: number[][],
  outs: number[][],
  callback: () => void,
): void {
  const outDim = weight.rows;
  const bits = weight.bits ?? 8;
  const inDim = weight.cols ?? Math.floor(weight.length / weight.rows);
  const rowBytes = weight.rowBytes ?? (bits === 4 ? Math.ceil(inDim / 2) : inDim);
  const groupSize = weight.groupSize ?? 128;
  const groupsPerRow = bits === 4 ? Math.ceil(inDim / groupSize) : 1;
  if (outDim * rowBytes !== weight.length) {
    throw new Error(`Tensor ${weight.name} is not a dense row matrix`);
  }
  if (inputs.length !== outs.length) {
    throw new RangeError("linearInto expects one output per input");
  }

  const maxRows = Math.min(LINEAR_ROW_BATCH, outDim);
  const raw = scratch(maxRows * rowBytes);
  let row = 0;

  const processBatch = (): void => {
    if (row >= outDim) {
      callback();
      return;
    }
    const firstRow = row;
    const rowCount = Math.min(LINEAR_ROW_BATCH, outDim - row);
    const rawLength = rowCount * rowBytes;
    row += rowCount;

    cursor.readBytesInto(
      raw,
      weight.offset + firstRow * rowBytes,
      rawLength,
      () => {
        for (let t = 0; t < inputs.length; t++) {
          if (bits === 4) {
            gemvInt4RowsInto(
              raw,
              0,
              rowBytes,
              scales,
              weight.scaleBase + firstRow * groupsPerRow,
              groupsPerRow,
              groupSize,
              inDim,
              inputs[t],
              outs[t],
              firstRow,
              rowCount,
            );
          } else {
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
        }
        if (inputs.length > 1) {
          spend(Math.ceil(rawLength * (inputs.length - 1) * 0.15));
        }
        queueTask(processBatch);
      },
      0,
      bits === 4,
    );
  };

  processBatch();
}

let logitsBuffer: number[] | undefined;

/** Logits against the shared embedding matrix. */
export function computeLogits(
  cursor: WeightCursor,
  scales: number[],
  embed: TensorMeta,
  hidden: number[],
  callback: (logits: number[]) => void,
): void {
  const vocab = embed.rows;
  const bits = embed.bits ?? 8;
  const dim = embed.cols ?? Math.floor(embed.length / embed.rows);
  const rowBytes = embed.rowBytes ?? (bits === 4 ? Math.ceil(dim / 2) : dim);
  const groupSize = embed.groupSize ?? 128;
  const groupsPerRow = bits === 4 ? Math.ceil(dim / groupSize) : 1;
  if (vocab * rowBytes !== embed.length) {
    throw new Error("Embedding tensor is not a dense row matrix");
  }
  if (logitsBuffer === undefined || logitsBuffer.length !== vocab) {
    logitsBuffer = new Array<number>(vocab).fill(0);
  }
  const logits = logitsBuffer;
  const maxRows = Math.min(LOGIT_ROW_BATCH, vocab);
  const raw = scratch(maxRows * rowBytes);
  let row = 0;

  const processBatch = (): void => {
    if (row >= vocab) {
      callback(logits);
      return;
    }
    const firstRow = row;
    const rowCount = Math.min(LOGIT_ROW_BATCH, vocab - row);
    row += rowCount;

    cursor.readBytesInto(
      raw,
      embed.offset + firstRow * rowBytes,
      rowCount * rowBytes,
      () => {
        if (bits === 4) {
          gemvInt4RowsInto(
            raw,
            0,
            rowBytes,
            scales,
            embed.scaleBase + firstRow * groupsPerRow,
            groupsPerRow,
            groupSize,
            dim,
            hidden,
            logits,
            firstRow,
            rowCount,
          );
        } else {
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
        }
        queueTask(processBatch);
      },
      0,
      bits === 4,
    );
  };

  processBatch();
}

export function readRowDequantized(
  cursor: WeightCursor,
  scales: number[],
  tensor: TensorMeta,
  row: number,
  callback: (values: number[]) => void,
): void {
  const bits = tensor.bits ?? 8;
  const dim = tensor.cols ?? Math.floor(tensor.length / tensor.rows);
  const rowBytes = tensor.rowBytes ?? (bits === 4 ? Math.ceil(dim / 2) : dim);
  if (row < 0 || row >= tensor.rows) {
    throw new RangeError(`Tensor ${tensor.name} row ${row} is out of range`);
  }
  const out = new Array<number>(dim).fill(0);
  if (bits !== 4) {
    cursor.readBytes(tensor.offset + row * rowBytes, dim, (raw) => {
      const scale = scales[tensor.scaleBase - -row];
      for (let i = 0; i < dim; i++) out[i] = Math.fround(raw[i] * scale);
      callback(out);
    });
    return;
  }
  const groupSize = tensor.groupSize ?? 128;
  const groupsPerRow = Math.ceil(dim / groupSize);
  cursor.readRawBytes(tensor.offset + row * rowBytes, rowBytes, (raw) => {
    const scaleBase = tensor.scaleBase + row * groupsPerRow;
    for (let i = 0; i < dim; i++) {
      const byte = raw[i >> 1];
      const index = (i & 1) === 0 ? byte & 15 : byte >> 4;
      out[i] = Math.fround(NF4[index] * scales[scaleBase + Math.floor(i / groupSize)]);
    }
    callback(out);
  });
}

export function readFloatVector(
  cursor: WeightCursor,
  tensor: FloatTensorMeta,
  callback: (values: number[]) => void,
): void {
  cursor.readFloats(tensor.offset, tensor.count, callback);
}

export function readVectorDequantized(
  cursor: WeightCursor,
  scales: number[],
  tensor: TensorMeta,
  callback: (values: number[]) => void,
): void {
  readRowDequantized(cursor, scales, tensor, 0, callback);
}

export function cachedVector(
  cursor: WeightCursor,
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
