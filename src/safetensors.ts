/**
 * Minimal safetensors reader with FP16/BF16 -> FP32 conversion.
 *
 * The ElanMT checkpoint is a 121 MB F16 safetensors file, so we deliberately
 * keep this dependency-free (it runs both in the browser and in Node).
 */

export interface Tensor {
  name: string;
  shape: number[];
  /** Row-major float32 data. */
  data: Float32Array;
}

export type TensorMap = Map<string, Tensor>;

const F16_TABLE = (() => {
  const table = new Float32Array(1 << 16);
  for (let bits = 0; bits < 1 << 16; bits++) {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x3ff;
    let value: number;
    if (exponent === 0) {
      value = fraction === 0 ? 0 : fraction * 2 ** -24;
    } else if (exponent === 0x1f) {
      value = fraction === 0 ? Infinity : NaN;
    } else {
      value = (fraction + 1024) * 2 ** (exponent - 25);
    }
    table[bits] = sign * value;
  }
  return table;
})();

function bf16ToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 7) & 0xff;
  const fraction = bits & 0x7f;
  if (exponent === 0) return sign * (fraction / 128) * 2 ** -126;
  if (exponent === 0xff) return fraction === 0 ? sign * Infinity : NaN;
  return sign * (1 + fraction / 128) * 2 ** (exponent - 127);
}

function parseHeader(bytes: Uint8Array): {
  header: Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  dataStart: number;
} {
  if (bytes.byteLength < 8) throw new Error("safetensors: file is too small");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  const dataStart = 8 + headerLength;
  if (dataStart > bytes.byteLength) throw new Error("safetensors: invalid header length");
  const text = new TextDecoder().decode(bytes.subarray(8, dataStart));
  const header = JSON.parse(text) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] }
  >;
  return { header, dataStart };
}

/**
 * Parse a safetensors buffer. Tensor data is copied into Float32Array values so
 * the (large) source buffer can be released afterwards.
 */
export function parseSafetensors(input: ArrayBuffer | Uint8Array): TensorMap {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const { header, dataStart } = parseHeader(bytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tensors: TensorMap = new Map();

  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [begin, end] = info.data_offsets;
    const offset = dataStart + begin;
    const count = info.shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(count);
    const byteLength = end - begin;

    if (info.dtype === "F16") {
      if (byteLength !== count * 2) {
        throw new Error(`safetensors: bad F16 byte length for ${name}`);
      }
      for (let i = 0; i < count; i++) {
        data[i] = F16_TABLE[view.getUint16(offset + i * 2, true)];
      }
    } else if (info.dtype === "BF16") {
      if (byteLength !== count * 2) {
        throw new Error(`safetensors: bad BF16 byte length for ${name}`);
      }
      for (let i = 0; i < count; i++) {
        data[i] = bf16ToFloat(view.getUint16(offset + i * 2, true));
      }
    } else if (info.dtype === "F32") {
      if (byteLength !== count * 4) {
        throw new Error(`safetensors: bad F32 byte length for ${name}`);
      }
      for (let i = 0; i < count; i++) {
        data[i] = view.getFloat32(offset + i * 4, true);
      }
    } else {
      throw new Error(`safetensors: unsupported dtype ${info.dtype} for ${name}`);
    }

    tensors.set(name, { name, shape: info.shape, data });
  }

  return tensors;
}
