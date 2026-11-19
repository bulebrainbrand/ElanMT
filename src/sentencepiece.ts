/**
 * Dependency-free SentencePiece reader for the unigram models used by Marian.
 *
 * The .spm file is a protobuf containing the vocabulary, unigram scores, and
 * the precompiled normalization charsmap. Everything needed for inference is
 * parsed at runtime; there is no generated vocabulary blob in this repository.
 *
 * The implementation avoids TextEncoder/TextDecoder because Bloxd's world code
 * runtime does not provide them.
 */

import { decodeUtf8, encodeUtf8 } from "./utf8";

const WHITESPACE_RE =
  /[\t\n\v\f\r \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/g;

function readUint32LE(bytes: number[], offset: number): number {
  return (
    (bytes[offset] |
      (bytes[offset + 1] << 8) |
      (bytes[offset + 2] << 16) |
      (bytes[offset + 3] << 24)) >>>
    0
  );
}

function readFloat32LE(bytes: number[], offset: number): number {
  const bits = readUint32LE(bytes, offset);
  const sign = bits & 0x80000000 ? -1 : 1;
  const exponent = (bits >>> 23) & 0xff;
  const fraction = bits & 0x7fffff;
  if (exponent === 0) {
    if (fraction === 0) return sign === 1 ? 0 : -0;
    return sign * fraction * 2 ** -149;
  }
  if (exponent === 255) {
    return fraction === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * (1 + fraction * 2 ** -23) * 2 ** (exponent - 127);
}

class ProtobufReader {
  offset = 0;

  constructor(private readonly bytes: number[]) {}

  get done(): boolean {
    return this.offset >= this.bytes.length;
  }

  varint(): number {
    let result = 0;
    let shift = 0;
    while (this.offset < this.bytes.length) {
      const byte = this.bytes[this.offset++];
      result += (byte & 0x7f) * 2 ** shift;
      if ((byte & 0x80) === 0) return result;
      shift += 7;
      if (shift > 63) throw new Error("sentencepiece: varint is too long");
    }
    throw new Error("sentencepiece: truncated varint");
  }

  tag(): { field: number; wire: number } {
    const value = this.varint();
    return { field: Math.floor(value / 8), wire: value & 7 };
  }

  float32(): number {
    const value = readFloat32LE(this.bytes, this.offset);
    this.offset += 4;
    return value;
  }

  bytesField(length: number): number[] {
    const value = this.bytes.slice(this.offset, this.offset + length);
    this.offset += length;
    return value;
  }

  skip(wire: number): void {
    if (wire === 0) {
      this.varint();
    } else if (wire === 1) {
      this.offset += 8;
    } else if (wire === 2) {
      // Capture the length before mutating the offset: `this.offset += this.varint()`
      // would read the old offset before the varint advances it.
      const length = this.varint();
      this.offset += length;
    } else if (wire === 5) {
      this.offset += 4;
    } else {
      throw new Error(`sentencepiece: unsupported protobuf wire type ${wire}`);
    }
  }
}

export const enum PieceType {
  Normal = 1,
  Unknown = 2,
  Control = 3,
  UserDefined = 4,
  Unused = 5,
  Byte = 6,
}

interface RawNormalizerSpec {
  name: string;
  charsmap: number[];
  addDummyPrefix: boolean;
  removeExtraWhitespaces: boolean;
  escapeWhitespaces: boolean;
}

function parseNormalizerSpec(bytes: number[]): RawNormalizerSpec {
  const reader = new ProtobufReader(bytes);
  const spec: RawNormalizerSpec = {
    name: "nmt_nfkc",
    charsmap: [],
    addDummyPrefix: true,
    removeExtraWhitespaces: true,
    escapeWhitespaces: true,
  };
  while (!reader.done) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) {
      spec.name = decodeUtf8(reader.bytesField(reader.varint()));
    } else if (field === 2 && wire === 2) {
      spec.charsmap = reader.bytesField(reader.varint());
    } else if (field === 3 && wire === 0) {
      spec.addDummyPrefix = reader.varint() !== 0;
    } else if (field === 4 && wire === 0) {
      spec.removeExtraWhitespaces = reader.varint() !== 0;
    } else if (field === 5 && wire === 0) {
      spec.escapeWhitespaces = reader.varint() !== 0;
    } else {
      reader.skip(wire);
    }
  }
  return spec;
}

interface RawPiece {
  piece: string;
  score: number;
  type: PieceType;
}

function parsePiece(bytes: number[]): RawPiece {
  const reader = new ProtobufReader(bytes);
  let piece = "";
  let score = 0;
  let type: PieceType = PieceType.Normal;
  while (!reader.done) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) {
      piece = decodeUtf8(reader.bytesField(reader.varint()));
    } else if (field === 2 && wire === 5) {
      score = reader.float32();
    } else if (field === 3 && wire === 0) {
      type = reader.varint() as PieceType;
    } else {
      reader.skip(wire);
    }
  }
  return { piece, score, type };
}

interface ParsedModel {
  pieces: RawPiece[];
  normalizer: RawNormalizerSpec;
  modelType: number;
}

function parseModel(bytes: number[]): ParsedModel {
  const reader = new ProtobufReader(bytes);
  const pieces: RawPiece[] = [];
  let normalizer: RawNormalizerSpec | null = null;
  let modelType = 1;

  while (!reader.done) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) {
      pieces.push(parsePiece(reader.bytesField(reader.varint())));
    } else if (field === 3 && wire === 2) {
      normalizer = parseNormalizerSpec(reader.bytesField(reader.varint()));
    } else if (field === 2 && wire === 2) {
      const trainer = new ProtobufReader(reader.bytesField(reader.varint()));
      while (!trainer.done) {
        const t = trainer.tag();
        if (t.field === 3 && t.wire === 0) modelType = trainer.varint();
        else trainer.skip(t.wire);
      }
    } else {
      reader.skip(wire);
    }
  }

  return {
    pieces,
    normalizer: normalizer ?? {
      name: "nmt_nfkc",
      charsmap: [],
      addDummyPrefix: true,
      removeExtraWhitespaces: true,
      escapeWhitespaces: true,
    },
    modelType,
  };
}

/**
 * Darts-clone double-array lookup used by SentencePiece's precompiled
 * normalization charsmap.
 */
class CharsMap {
  private readonly units: number[];
  private readonly blob: number[];
  private readonly cache = new Map<number, string>();

  constructor(charsmap: number[]) {
    if (charsmap.length < 4) {
      this.units = [];
      this.blob = [];
      return;
    }
    const trieBytes = readUint32LE(charsmap, 0);
    if (trieBytes % 4 !== 0 || 4 + trieBytes > charsmap.length) {
      throw new Error("sentencepiece: invalid normalization charsmap");
    }
    const unitCount = trieBytes / 4;
    this.units = new Array<number>(unitCount);
    for (let i = 0; i < unitCount; i++) {
      this.units[i] = readUint32LE(charsmap, 4 + i * 4);
    }
    this.blob = charsmap.slice(4 + trieBytes);
  }

  private static offset(unit: number): number {
    return ((unit >>> 10) << ((unit & 0x200) >>> 6)) >>> 0;
  }

  private valueString(valueOffset: number): string {
    const cached = this.cache.get(valueOffset);
    if (cached !== undefined) return cached;
    let end = valueOffset;
    while (end < this.blob.length && this.blob[end] !== 0) end++;
    const text = decodeUtf8(this.blob.slice(valueOffset, end));
    this.cache.set(valueOffset, text);
    return text;
  }

  /**
   * Normalize every UTF-8 character (or longest matching byte sequence) using
   * the compiled charsmap. Unmapped characters are copied verbatim.
   */
  normalizeBytes(input: number[]): number[] {
    if (this.units.length === 0) return input;
    const output: number[] = [];
    const { units } = this;

    let i = 0;
    while (i < input.length) {
      let pos = 0;
      let matchLength = 0;
      let value = -1;

      for (let j = i; j < input.length; j++) {
        const byte = input[j];
        const unit = units[pos];
        if (unit === undefined) break;
        pos = (pos ^ CharsMap.offset(unit) ^ byte) >>> 0;
        if (pos >= units.length) break;
        const next = units[pos];
        if ((next & 0x800000ff) !== byte) break;
        if ((next & 0x100) !== 0) {
          const leaf = units[(pos ^ CharsMap.offset(next)) >>> 0];
          if (leaf !== undefined) {
            matchLength = j - i + 1;
            value = leaf & 0x7fffffff;
          }
        }
      }

      if (matchLength > 0) {
        const mapped = encodeUtf8(this.valueString(value));
        for (let k = 0; k < mapped.length; k++) output.push(mapped[k]);
        i += matchLength;
      } else {
        const lead = input[i];
        const length = lead < 0x80 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
        for (let k = 0; k < length && i + k < input.length; k++) {
          output.push(input[i + k]);
        }
        i += length;
      }
    }

    return output;
  }
}

export class SentencePieceTokenizer {
  readonly pieces: string[];
  readonly scores: number[];
  readonly types: number[];

  private readonly eosId: number;
  private readonly unkId: number;
  private readonly minScore: number;
  private readonly unkScore: number;
  private readonly pieceIds = new Map<string, number>();
  private readonly maxPieceLength: number;
  private readonly charsMap: CharsMap;
  private readonly normalizer: RawNormalizerSpec;

  constructor(bytes: number[], eosId = 0, unkId = 1) {
    const model = parseModel(bytes);
    // 1 = unigram. BPE/word/char models would need a different encoder.
    if (model.modelType !== 1) {
      throw new Error(
        `sentencepiece: model type ${model.modelType} is not supported (only unigram)`,
      );
    }

    this.eosId = eosId;
    this.unkId = unkId;
    this.normalizer = model.normalizer;
    this.charsMap = new CharsMap(model.normalizer.charsmap);
    this.pieces = model.pieces.map((p) => p.piece);
    this.scores = model.pieces.map((p) => p.score);
    this.types = model.pieces.map((p) => p.type);

    const unkIndex = model.pieces.findIndex((p) => p.type === PieceType.Unknown);
    if (unkIndex >= 0) this.unkId = unkIndex;

    let minScore = Infinity;
    let maxPieceLength = 1;
    for (let i = 0; i < this.pieces.length; i++) {
      const type = this.types[i];
      if (type === PieceType.Normal) {
        minScore = Math.min(minScore, this.scores[i]);
      }
      if (type === PieceType.Normal || type === PieceType.UserDefined) {
        const piece = this.pieces[i];
        if (piece.length === 0) continue;
        this.pieceIds.set(piece, i);
        const length = Array.from(piece).length;
        if (length > maxPieceLength) maxPieceLength = length;
      }
    }
    this.minScore = minScore === Infinity ? 0 : minScore;
    this.unkScore = this.minScore - 10;
    this.maxPieceLength = Math.min(maxPieceLength, 64);
  }

  /** Equivalent to SentencePieceProcessor.normalize(). */
  normalize(text: string): string {
    const raw = encodeUtf8(text);
    const mapped = this.charsMap.normalizeBytes(raw);
    let normalized = decodeUtf8(mapped);

    if (this.normalizer.removeExtraWhitespaces) {
      normalized = normalized.replace(WHITESPACE_RE, " ").replace(/^ +| +$/g, "");
    }

    if (this.normalizer.addDummyPrefix && normalized.length > 0) {
      normalized = " " + normalized;
    }
    if (this.normalizer.escapeWhitespaces) {
      normalized = normalized.replace(/ /g, "\u2581");
    }
    return normalized;
  }

  /** Unigram Viterbi segmentation. Returns indices into `pieces`. */
  encodeIds(text: string): number[] {
    const normalized = this.normalize(text);
    const chars = Array.from(normalized);
    const n = chars.length;
    if (n === 0) return [];

    const best = new Array<number>(n + 1).fill(-Infinity);
    const backStart = new Array<number>(n + 1).fill(-1);
    const backId = new Array<number>(n + 1).fill(-1);
    best[0] = 0;

    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(best[i])) continue;

      // Unknown character (ElanMT does not use byte fallback).
      const unknownScore = best[i] + this.unkScore;
      if (unknownScore > best[i + 1]) {
        best[i + 1] = unknownScore;
        backStart[i + 1] = i;
        backId[i + 1] = this.unkId;
      }

      let candidate = "";
      const limit = Math.min(n, i + this.maxPieceLength);
      for (let j = i; j < limit; j++) {
        candidate += chars[j];
        const pieceId = this.pieceIds.get(candidate);
        if (pieceId === undefined) continue;
        const score = best[i] + this.scores[pieceId];
        if (score > best[j + 1]) {
          best[j + 1] = score;
          backStart[j + 1] = i;
          backId[j + 1] = pieceId;
        }
      }
    }

    const ids: number[] = [];
    for (let pos = n; pos > 0;) {
      const start = backStart[pos];
      if (start < 0 || backId[pos] < 0) {
        ids.push(this.unkId);
        pos -= 1;
      } else {
        ids.push(backId[pos]);
        pos = start;
      }
    }
    ids.reverse();
    return ids;
  }

  /** Encode and append EOS, truncating the body (not EOS) if requested. */
  encode(text: string, maxTokens = 512): { ids: number[]; truncated: boolean } {
    const body = this.encodeIds(text);
    const limit = Math.max(0, maxTokens - 1);
    const truncated = body.length > limit;
    const ids = truncated ? body.slice(0, limit) : body;
    ids.push(this.eosId);
    return { ids, truncated };
  }

  /** Decode ids the way MarianTokenizer.decode(..., skip_special_tokens=True) does. */
  decode(ids: readonly number[]): string {
    const tokens: string[] = [];
    for (const id of ids) {
      if (id < 0 || id >= this.pieces.length) continue;
      const type = this.types[id];
      if (type === PieceType.Control || type === PieceType.Unknown) continue;
      tokens.push(this.pieces[id]);
    }
    let text = tokens
      .join("")
      .replace(/\u2581/g, " ")
      .trim();
    // Equivalent to the default clean_up_tokenization_spaces=true.
    text = text
      .replace(/ \./g, ".")
      .replace(/ \?/g, "?")
      .replace(/ !/g, "!")
      .replace(/ ,/g, ",")
      .replace(/ ' /g, "'")
      .replace(/ n't/g, "n't")
      .replace(/ 'm/g, "'m")
      .replace(/ 's/g, "'s")
      .replace(/ 've/g, "'ve")
      .replace(/ 're/g, "'re");
    return text;
  }
}
