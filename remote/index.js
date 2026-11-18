// src/safetensors.ts
const F16_TABLE = (() => {
	const table = /* @__PURE__ */ new Float32Array(65536);
	for (let bits = 0; bits < 65536; bits++) {
		const sign = bits & 32768 ? -1 : 1;
		const exponent = bits >> 10 & 31;
		const fraction = bits & 1023;
		let value;
		if (exponent === 0) value = fraction === 0 ? 0 : fraction * 2 ** -24;
		else if (exponent === 31) value = fraction === 0 ? Infinity : NaN;
		else value = (fraction + 1024) * 2 ** (exponent - 25);
		table[bits] = sign * value;
	}
	return table;
})();
function bf16ToFloat(bits) {
	const sign = bits & 32768 ? -1 : 1;
	const exponent = bits >> 7 & 255;
	const fraction = bits & 127;
	if (exponent === 0) return sign * (fraction / 128) * 2 ** -126;
	if (exponent === 255) return fraction === 0 ? sign * Infinity : NaN;
	return sign * (1 + fraction / 128) * 2 ** (exponent - 127);
}
function parseHeader(bytes) {
	if (bytes.byteLength < 8) throw new Error("safetensors: file is too small");
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const dataStart = 8 + Number(view.getBigUint64(0, true));
	if (dataStart > bytes.byteLength) throw new Error("safetensors: invalid header length");
	const text = new TextDecoder().decode(bytes.subarray(8, dataStart));
	return {
		header: JSON.parse(text),
		dataStart
	};
}
/**
* Parse a safetensors buffer. Tensor data is copied into Float32Array values so
* the (large) source buffer can be released afterwards.
*/
function parseSafetensors(input) {
	const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
	const { header, dataStart } = parseHeader(bytes);
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const tensors = /* @__PURE__ */ new Map();
	for (const [name, info] of Object.entries(header)) {
		if (name === "__metadata__") continue;
		const [begin, end] = info.data_offsets;
		const offset = dataStart + begin;
		const count = info.shape.reduce((a, b) => a * b, 1);
		const data = new Float32Array(count);
		const byteLength = end - begin;
		if (info.dtype === "F16") {
			if (byteLength !== count * 2) throw new Error(`safetensors: bad F16 byte length for ${name}`);
			for (let i = 0; i < count; i++) data[i] = F16_TABLE[view.getUint16(offset + i * 2, true)];
		} else if (info.dtype === "BF16") {
			if (byteLength !== count * 2) throw new Error(`safetensors: bad BF16 byte length for ${name}`);
			for (let i = 0; i < count; i++) data[i] = bf16ToFloat(view.getUint16(offset + i * 2, true));
		} else if (info.dtype === "F32") {
			if (byteLength !== count * 4) throw new Error(`safetensors: bad F32 byte length for ${name}`);
			for (let i = 0; i < count; i++) data[i] = view.getFloat32(offset + i * 4, true);
		} else throw new Error(`safetensors: unsupported dtype ${info.dtype} for ${name}`);
		tensors.set(name, {
			name,
			shape: info.shape,
			data
		});
	}
	return tensors;
}
// src/utf8.ts
/**
* UTF-8 helpers for runtimes without TextEncoder/TextDecoder (Bloxd's world
* code VM does not provide them). The implementation follows the public
* domain fast-text-encoding routines used by Brandon-Tiny-10M-Instruct.
*/
function encodeUtf8(text) {
	const out = [];
	for (let i = 0; i < text.length; i++) {
		let code = text.charCodeAt(i);
		if (code >= 55296 && code <= 56319) {
			if (i + 1 < text.length) {
				const low = text.charCodeAt(i + 1);
				if ((low & 64512) === 56320) {
					i++;
					code = ((code & 1023) << 10) + (low & 1023) + 65536;
				}
			}
			if (code >= 55296 && code <= 56319) continue;
		}
		if (code < 128) out.push(code);
		else if (code < 2048) out.push(192 | code >> 6, 128 | code & 63);
		else if (code < 65536) out.push(224 | code >> 12, 128 | code >> 6 & 63, 128 | code & 63);
		else if (code < 2097152) out.push(240 | code >> 18, 128 | code >> 12 & 63, 128 | code >> 6 & 63, 128 | code & 63);
	}
	return out;
}
function decodeUtf8(bytes) {
	const codeUnits = [];
	let i = 0;
	while (i < bytes.length) {
		const b1 = bytes[i++] & 255;
		if (b1 < 128) codeUnits.push(b1);
		else if ((b1 & 224) === 192) {
			const b2 = (bytes[i++] ?? 0) & 63;
			codeUnits.push((b1 & 31) << 6 | b2);
		} else if ((b1 & 240) === 224) {
			const b2 = (bytes[i++] ?? 0) & 63;
			const b3 = (bytes[i++] ?? 0) & 63;
			codeUnits.push((b1 & 15) << 12 | b2 << 6 | b3);
		} else if ((b1 & 248) === 240) {
			const b2 = (bytes[i++] ?? 0) & 63;
			const b3 = (bytes[i++] ?? 0) & 63;
			const b4 = (bytes[i++] ?? 0) & 63;
			let code = (b1 & 7) << 18 | b2 << 12 | b3 << 6 | b4;
			if (code > 65535) {
				code -= 65536;
				codeUnits.push(55296 | code >> 10);
				code = 56320 | code & 1023;
			}
			codeUnits.push(code & 65535);
		}
	}
	let out = "";
	const CHUNK = 4096;
	for (let start = 0; start < codeUnits.length; start += CHUNK) {
		const end = Math.min(start + CHUNK, codeUnits.length);
		const chunk = [];
		for (let j = start; j < end; j++) chunk.push(codeUnits[j]);
		out += String.fromCharCode.apply(null, chunk);
	}
	return out;
}
// src/sentencepiece.ts
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
const WHITESPACE_RE = /[\t\n\v\f\r \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/g;
var ProtobufReader = class {
	bytes;
	view;
	offset = 0;
	constructor(bytes) {
		this.bytes = bytes;
		this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	}
	get done() {
		return this.offset >= this.bytes.byteLength;
	}
	varint() {
		let result = 0;
		let shift = 0;
		while (this.offset < this.bytes.byteLength) {
			const byte = this.bytes[this.offset++];
			result += (byte & 127) * 2 ** shift;
			if ((byte & 128) === 0) return result;
			shift += 7;
			if (shift > 63) throw new Error("sentencepiece: varint is too long");
		}
		throw new Error("sentencepiece: truncated varint");
	}
	tag() {
		const value = this.varint();
		return {
			field: Math.floor(value / 8),
			wire: value & 7
		};
	}
	float32() {
		const value = this.view.getFloat32(this.offset, true);
		this.offset += 4;
		return value;
	}
	bytesField(length) {
		const value = this.bytes.subarray(this.offset, this.offset + length);
		this.offset += length;
		return value;
	}
	skip(wire) {
		if (wire === 0) this.varint();
		else if (wire === 1) this.offset += 8;
		else if (wire === 2) {
			const length = this.varint();
			this.offset += length;
		} else if (wire === 5) this.offset += 4;
		else throw new Error(`sentencepiece: unsupported protobuf wire type ${wire}`);
	}
};
function parseNormalizerSpec(bytes) {
	const reader = new ProtobufReader(bytes);
	const spec = {
		name: "nmt_nfkc",
		charsmap: /* @__PURE__ */ new Uint8Array(0),
		addDummyPrefix: true,
		removeExtraWhitespaces: true,
		escapeWhitespaces: true
	};
	while (!reader.done) {
		const { field, wire } = reader.tag();
		if (field === 1 && wire === 2) spec.name = decodeUtf8(reader.bytesField(reader.varint()));
		else if (field === 2 && wire === 2) spec.charsmap = reader.bytesField(reader.varint());
		else if (field === 3 && wire === 0) spec.addDummyPrefix = reader.varint() !== 0;
		else if (field === 4 && wire === 0) spec.removeExtraWhitespaces = reader.varint() !== 0;
		else if (field === 5 && wire === 0) spec.escapeWhitespaces = reader.varint() !== 0;
		else reader.skip(wire);
	}
	return spec;
}
function parsePiece(bytes) {
	const reader = new ProtobufReader(bytes);
	let piece = "";
	let score = 0;
	let type = 1;
	while (!reader.done) {
		const { field, wire } = reader.tag();
		if (field === 1 && wire === 2) piece = decodeUtf8(reader.bytesField(reader.varint()));
		else if (field === 2 && wire === 5) score = reader.float32();
		else if (field === 3 && wire === 0) type = reader.varint();
		else reader.skip(wire);
	}
	return {
		piece,
		score,
		type
	};
}
function parseModel(bytes) {
	const reader = new ProtobufReader(bytes);
	const pieces = [];
	let normalizer = null;
	let modelType = 1;
	while (!reader.done) {
		const { field, wire } = reader.tag();
		if (field === 1 && wire === 2) pieces.push(parsePiece(reader.bytesField(reader.varint())));
		else if (field === 3 && wire === 2) normalizer = parseNormalizerSpec(reader.bytesField(reader.varint()));
		else if (field === 2 && wire === 2) {
			const trainer = new ProtobufReader(reader.bytesField(reader.varint()));
			while (!trainer.done) {
				const t = trainer.tag();
				if (t.field === 3 && t.wire === 0) modelType = trainer.varint();
				else trainer.skip(t.wire);
			}
		} else reader.skip(wire);
	}
	return {
		pieces,
		normalizer: normalizer ?? {
			name: "nmt_nfkc",
			charsmap: /* @__PURE__ */ new Uint8Array(0),
			addDummyPrefix: true,
			removeExtraWhitespaces: true,
			escapeWhitespaces: true
		},
		modelType
	};
}
/**
* Darts-clone double-array lookup used by SentencePiece's precompiled
* normalization charsmap.
*/
var CharsMap = class CharsMap {
	units;
	blob;
	cache = /* @__PURE__ */ new Map();
	constructor(charsmap) {
		if (charsmap.byteLength < 4) {
			this.units = /* @__PURE__ */ new Uint32Array(0);
			this.blob = /* @__PURE__ */ new Uint8Array(0);
			return;
		}
		const view = new DataView(charsmap.buffer, charsmap.byteOffset, charsmap.byteLength);
		const trieBytes = view.getUint32(0, true);
		if (trieBytes % 4 !== 0 || 4 + trieBytes > charsmap.byteLength) throw new Error("sentencepiece: invalid normalization charsmap");
		const unitCount = trieBytes / 4;
		this.units = new Uint32Array(unitCount);
		for (let i = 0; i < unitCount; i++) this.units[i] = view.getUint32(4 + i * 4, true);
		this.blob = charsmap.subarray(4 + trieBytes);
	}
	static offset(unit) {
		return unit >>> 10 << ((unit & 512) >>> 6) >>> 0;
	}
	valueString(valueOffset) {
		const cached = this.cache.get(valueOffset);
		if (cached !== void 0) return cached;
		let end = valueOffset;
		while (end < this.blob.byteLength && this.blob[end] !== 0) end++;
		const text = decodeUtf8(this.blob.subarray(valueOffset, end));
		this.cache.set(valueOffset, text);
		return text;
	}
	/**
	* Normalize every UTF-8 character (or longest matching byte sequence) using
	* the compiled charsmap. Unmapped characters are copied verbatim.
	*/
	normalizeBytes(input) {
		if (this.units.length === 0) return input;
		const output = [];
		const { units } = this;
		let i = 0;
		while (i < input.byteLength) {
			let pos = 0;
			let matchLength = 0;
			let value = -1;
			for (let j = i; j < input.byteLength; j++) {
				const byte = input[j];
				const unit = units[pos];
				if (unit === void 0) break;
				pos = (pos ^ CharsMap.offset(unit) ^ byte) >>> 0;
				if (pos >= units.length) break;
				const next = units[pos];
				if ((next & 2147483903) !== byte) break;
				if ((next & 256) !== 0) {
					const leaf = units[(pos ^ CharsMap.offset(next)) >>> 0];
					if (leaf !== void 0) {
						matchLength = j - i + 1;
						value = leaf & 2147483647;
					}
				}
			}
			if (matchLength > 0) {
				const mapped = encodeUtf8(this.valueString(value));
				for (let k = 0; k < mapped.length; k++) output.push(mapped[k]);
				i += matchLength;
			} else {
				const lead = input[i];
				const length = lead < 128 ? 1 : lead < 224 ? 2 : lead < 240 ? 3 : 4;
				for (let k = 0; k < length && i + k < input.byteLength; k++) output.push(input[i + k]);
				i += length;
			}
		}
		return new Uint8Array(output);
	}
};
var SentencePieceTokenizer = class {
	pieces;
	scores;
	types;
	eosId;
	unkId;
	minScore;
	unkScore;
	pieceIds = /* @__PURE__ */ new Map();
	maxPieceLength;
	charsMap;
	normalizer;
	constructor(spmBytes, eosId = 0, unkId = 1) {
		const model = parseModel(spmBytes instanceof Uint8Array ? spmBytes : new Uint8Array(spmBytes));
		if (model.modelType !== 1) throw new Error(`sentencepiece: model type ${model.modelType} is not supported (only unigram)`);
		this.eosId = eosId;
		this.unkId = unkId;
		this.normalizer = model.normalizer;
		this.charsMap = new CharsMap(model.normalizer.charsmap);
		this.pieces = model.pieces.map((p) => p.piece);
		this.scores = new Float32Array(model.pieces.map((p) => p.score));
		this.types = new Uint8Array(model.pieces.map((p) => p.type));
		const unkIndex = model.pieces.findIndex((p) => p.type === 2);
		if (unkIndex >= 0) this.unkId = unkIndex;
		let minScore = Infinity;
		let maxPieceLength = 1;
		for (let i = 0; i < this.pieces.length; i++) {
			const type = this.types[i];
			if (type === 1) minScore = Math.min(minScore, this.scores[i]);
			if (type === 1 || type === 4) {
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
	normalize(text) {
		const raw = new Uint8Array(encodeUtf8(text));
		let normalized = decodeUtf8(this.charsMap.normalizeBytes(raw));
		if (this.normalizer.removeExtraWhitespaces) normalized = normalized.replace(WHITESPACE_RE, " ").replace(/^ +| +$/g, "");
		if (this.normalizer.addDummyPrefix && normalized.length > 0) normalized = " " + normalized;
		if (this.normalizer.escapeWhitespaces) normalized = normalized.replace(/ /g, "▁");
		return normalized;
	}
	/** Unigram Viterbi segmentation. Returns indices into `pieces`. */
	encodeIds(text) {
		const normalized = this.normalize(text);
		const chars = Array.from(normalized);
		const n = chars.length;
		if (n === 0) return [];
		const best = new Float64Array(n + 1).fill(-Infinity);
		const backStart = new Int32Array(n + 1).fill(-1);
		const backId = new Int32Array(n + 1).fill(-1);
		best[0] = 0;
		for (let i = 0; i < n; i++) {
			if (!Number.isFinite(best[i])) continue;
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
				if (pieceId === void 0) continue;
				const score = best[i] + this.scores[pieceId];
				if (score > best[j + 1]) {
					best[j + 1] = score;
					backStart[j + 1] = i;
					backId[j + 1] = pieceId;
				}
			}
		}
		const ids = [];
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
	encode(text, maxTokens = 512) {
		const body = this.encodeIds(text);
		const limit = Math.max(0, maxTokens - 1);
		const truncated = body.length > limit;
		const ids = truncated ? body.slice(0, limit) : body;
		ids.push(this.eosId);
		return {
			ids,
			truncated
		};
	}
	/** Decode ids the way MarianTokenizer.decode(..., skip_special_tokens=True) does. */
	decode(ids) {
		const tokens = [];
		for (const id of ids) {
			if (id < 0 || id >= this.pieces.length) continue;
			const type = this.types[id];
			if (type === 3 || type === 2) continue;
			tokens.push(this.pieces[id]);
		}
		let text = tokens.join("").replace(/\u2581/g, " ").trim();
		text = text.replace(/ \./g, ".").replace(/ \?/g, "?").replace(/ !/g, "!").replace(/ ,/g, ",").replace(/ ' /g, "'").replace(/ n't/g, "n't").replace(/ 'm/g, "'m").replace(/ 's/g, "'s").replace(/ 've/g, "'ve").replace(/ 're/g, "'re");
		return text;
	}
};
// src/marian.ts
function sinusoidalPositions(maxPositions, dModel) {
	const table = new Float32Array(maxPositions * dModel);
	const half = Math.floor(dModel / 2);
	const rate = Math.log(1e4) / half;
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
function activation(name, x) {
	if (name === "relu") return x > 0 ? x : 0;
	if (name === "gelu" || name === "gelu_new") return .5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + .044715 * x ** 3)));
	if (name === "tanh") return Math.tanh(x);
	return x / (1 + Math.exp(-x));
}
/** out = x @ W^T + b, with W stored as [outDim, inDim]. */
function linear(out, x, xOffset, weights) {
	const { w, b, inDim, outDim } = weights;
	for (let o = 0; o < outDim; o++) {
		const row = o * inDim;
		let sum = b[o];
		for (let i = 0; i < inDim; i++) sum += x[xOffset + i] * w[row + i];
		out[o] = sum;
	}
}
function layerNorm(x, offset, dim, weight, bias) {
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
	for (let i = 0; i < dim; i++) x[offset + i] = (x[offset + i] - mean) * inv * weight[i] + bias[i];
}
var MarianModel = class {
	config;
	dModel;
	vocabSize;
	maxPositions;
	numBeams;
	maxLength;
	eosId;
	padId;
	decoderStartId;
	badWordIds;
	encoderEmbed;
	decoderEmbed;
	lmHead;
	finalLogitsBias;
	encoderPositions;
	decoderPositions;
	encoderLayers = [];
	decoderLayers = [];
	embedScale;
	activationName;
	encoderHeads;
	decoderHeads;
	constructor(config, tensors) {
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
		if (config.normalize_embedding) throw new Error("marian: normalize_embedding is not supported");
		const get = (name) => {
			const tensor = tensors.get(name);
			if (!tensor) throw new Error(`marian: missing tensor ${name}`);
			return tensor;
		};
		const optional = (name) => tensors.get(name);
		const shared = get("model.shared.weight");
		this.encoderEmbed = (optional("model.encoder.embed_tokens.weight") ?? shared).data;
		this.decoderEmbed = (optional("model.decoder.embed_tokens.weight") ?? shared).data;
		this.lmHead = (optional("lm_head.weight") ?? shared).data;
		const bias = optional("final_logits_bias");
		this.finalLogitsBias = bias ? bias.data : new Float32Array(this.vocabSize);
		const encPos = optional("model.encoder.embed_positions.weight");
		const decPos = optional("model.decoder.embed_positions.weight");
		this.encoderPositions = encPos ? encPos.data : sinusoidalPositions(this.maxPositions, this.dModel);
		this.decoderPositions = decPos ? decPos.data : sinusoidalPositions(this.maxPositions, this.dModel);
		const readLinear = (prefix) => {
			const w = get(`${prefix}.weight`);
			const b = get(`${prefix}.bias`);
			const outDim = w.shape[0];
			const inDim = w.shape[1] ?? 1;
			if (b.data.length !== outDim) throw new Error(`marian: bad bias shape for ${prefix}`);
			return {
				w: w.data,
				b: b.data,
				inDim,
				outDim
			};
		};
		const readAttention = (prefix) => ({
			q: readLinear(`${prefix}.q_proj`),
			k: readLinear(`${prefix}.k_proj`),
			v: readLinear(`${prefix}.v_proj`),
			out: readLinear(`${prefix}.out_proj`)
		});
		for (let i = 0; i < config.encoder_layers; i++) {
			const prefix = `model.encoder.layers.${i}`;
			this.encoderLayers.push({
				selfAttn: readAttention(`${prefix}.self_attn`),
				selfAttnNorm: readLinear(`${prefix}.self_attn_layer_norm`),
				finalNorm: readLinear(`${prefix}.final_layer_norm`),
				fc1: readLinear(`${prefix}.fc1`),
				fc2: readLinear(`${prefix}.fc2`)
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
				fc2: readLinear(`${prefix}.fc2`)
			});
		}
	}
	embed(out, ids, embed, positions, length) {
		const d = this.dModel;
		const scale = this.embedScale;
		for (let t = 0; t < length; t++) {
			const id = ids[t];
			if (id < 0 || id * d >= embed.length) throw new Error(`marian: token id ${id} is out of range`);
			if (t >= this.maxPositions) throw new Error("marian: input is longer than the position table");
			const src = id * d;
			const dst = t * d;
			const pos = t * d;
			for (let i = 0; i < d; i++) out[dst + i] = embed[src + i] * scale + positions[pos + i];
		}
	}
	projectAll(input, length, weights) {
		const d = this.dModel;
		const out = new Float32Array(length * d);
		const row = new Float32Array(d);
		for (let t = 0; t < length; t++) {
			linear(row, input, t * d, weights);
			out.set(row, t * d);
		}
		return out;
	}
	attention(input, length, weights, heads) {
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
					for (let e = 0; e < headDim; e++) dot += q[i * d + base + e] * k[j * d + base + e];
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
					for (let e = 0; e < headDim; e++) context[cBase + e] += p * v[vBase + e];
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
	feedForward(input, length, fc1, fc2) {
		const d = this.dModel;
		const hidden = new Float32Array(fc1.outDim);
		const out = new Float32Array(length * d);
		const row = new Float32Array(d);
		for (let t = 0; t < length; t++) {
			linear(hidden, input, t * d, fc1);
			for (let i = 0; i < hidden.length; i++) hidden[i] = activation(this.activationName, hidden[i]);
			linear(row, hidden, 0, fc2);
			out.set(row, t * d);
		}
		return out;
	}
	normalizeRows(x, length, norm) {
		for (let t = 0; t < length; t++) layerNorm(x, t * this.dModel, this.dModel, norm.w, norm.b);
	}
	/** Run the encoder and return memory [sourceLength, d_model]. */
	encode(ids) {
		const length = ids.length;
		const d = this.dModel;
		if (length === 0) return /* @__PURE__ */ new Float32Array(0);
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
	createCrossCache(memory, sourceLength) {
		const k = [];
		const v = [];
		for (const layer of this.decoderLayers) {
			k.push(this.projectAll(memory, sourceLength, layer.encoderAttn.k));
			v.push(this.projectAll(memory, sourceLength, layer.encoderAttn.v));
		}
		return {
			k,
			v,
			length: sourceLength
		};
	}
	createDecoderState(maxLength, maxNewTokens) {
		return new DecoderState(this.dModel, this.decoderLayers.length, Math.min(this.maxPositions, maxLength + maxNewTokens + 1));
	}
	crossAttention(queryInput, queryLength, weights, heads, keys, values, keyLength) {
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
					for (let e = 0; e < headDim; e++) dot += q[i * d + base + e] * keys[j * d + base + e];
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
					for (let e = 0; e < headDim; e++) context[cBase + e] += p * values[vBase + e];
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
	step(state, token, cross) {
		const d = this.dModel;
		const pos = state.length;
		if (pos >= this.maxPositions) throw new Error("marian: decoder exceeded the position table");
		state.ensureCapacity(pos + 1);
		const x = new Float32Array(d);
		const src = token * d;
		for (let i = 0; i < d; i++) x[i] = this.decoderEmbed[src + i] * this.embedScale + this.decoderPositions[pos * d + i];
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
					for (let e = 0; e < headDim; e++) context[base + e] += p * state.v[layerIndex][vBase + e];
				}
			}
			linear(row, context, 0, layer.selfAttn.out);
			for (let i = 0; i < d; i++) x[i] += row[i];
			layerNorm(x, 0, d, layer.selfAttnNorm.w, layer.selfAttnNorm.b);
			const crossOut = this.crossAttention(x, 1, layer.encoderAttn, this.decoderHeads, cross.k[layerIndex], cross.v[layerIndex], cross.length);
			for (let i = 0; i < d; i++) x[i] += crossOut[i];
			layerNorm(x, 0, d, layer.encoderAttnNorm.w, layer.encoderAttnNorm.b);
			const hidden = new Float32Array(layer.fc1.outDim);
			linear(hidden, x, 0, layer.fc1);
			for (let i = 0; i < hidden.length; i++) hidden[i] = activation(this.activationName, hidden[i]);
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
};
var DecoderState = class DecoderState {
	dModel;
	k;
	v;
	length = 0;
	capacity;
	constructor(dModel, layers, capacity) {
		this.dModel = dModel;
		this.capacity = Math.max(1, capacity);
		this.k = Array.from({ length: layers }, () => new Float32Array(this.capacity * dModel));
		this.v = Array.from({ length: layers }, () => new Float32Array(this.capacity * dModel));
	}
	ensureCapacity(length) {
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
	clone() {
		const copy = new DecoderState(this.dModel, this.k.length, Math.max(this.capacity, this.length + 1));
		copy.length = this.length;
		const used = this.length * this.dModel;
		for (let i = 0; i < this.k.length; i++) {
			copy.k[i].set(this.k[i].subarray(0, used));
			copy.v[i].set(this.v[i].subarray(0, used));
		}
		return copy;
	}
};
// src/index.ts
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
function decodeConfig(config) {
	if (typeof config === "string") return JSON.parse(config);
	if (config instanceof Uint8Array || config instanceof ArrayBuffer) return JSON.parse(new TextDecoder().decode(config));
	return config;
}
function joinUrl(base, file) {
	return `${base.replace(/\/+$/, "")}/${file}`;
}
async function fetchBuffer(url, fetchImpl, onProgress) {
	const response = await fetchImpl(url);
	if (!response.ok) throw new Error(`ElanMT: failed to fetch ${url} (${response.status})`);
	const total = Number(response.headers.get("content-length") ?? 0);
	if (!response.body || !onProgress) return response.arrayBuffer();
	const reader = response.body.getReader();
	const chunks = [];
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
function applyBadWords(logits, badWords) {
	for (const ids of badWords) if (ids.length === 1) {
		const id = ids[0];
		if (id >= 0 && id < logits.length) logits[id] = -Infinity;
	}
}
function topKLogits(logits, k) {
	const ids = [];
	const scores = [];
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
	return {
		ids,
		scores
	};
}
var ElanMT = class {
	config;
	model;
	tokenizer;
	numBeams;
	maxNewTokens;
	maxSourceTokens;
	constructor(config, tokenizer, model, options = {}) {
		this.config = config;
		this.tokenizer = tokenizer;
		this.model = model;
		this.numBeams = Math.max(1, options.numBeams ?? config.num_beams ?? 1);
		this.maxNewTokens = Math.max(1, options.maxNewTokens ?? 128);
		this.maxSourceTokens = Math.max(1, options.maxSourceTokens ?? 512);
	}
	translate(text, options = {}) {
		const maxSourceTokens = Math.max(1, options.maxSourceTokens ?? this.maxSourceTokens);
		const maxNewTokens = Math.max(1, options.maxNewTokens ?? this.maxNewTokens);
		const numBeams = Math.max(1, options.numBeams ?? this.numBeams);
		const { ids, truncated } = this.tokenizer.encode(text, maxSourceTokens);
		if (ids.length === 1 && ids[0] === this.model.eosId) return {
			source: text,
			translation: "",
			sourceTokenIds: ids,
			sourceTruncated: truncated,
			outputTokenIds: [],
			outputTokens: []
		};
		const memory = this.model.encode(ids);
		const cross = this.model.createCrossCache(memory, ids.length);
		const output = numBeams <= 1 ? this.greedy(cross, maxNewTokens) : this.beamSearch(cross, maxNewTokens, numBeams);
		return {
			source: text,
			translation: this.tokenizer.decode(output),
			sourceTokenIds: ids,
			sourceTruncated: truncated,
			outputTokenIds: output,
			outputTokens: output.map((id) => id >= 0 && id < this.tokenizer.pieces.length ? this.tokenizer.pieces[id] : "")
		};
	}
	prepareLogits(logits, step, maxNewTokens) {
		applyBadWords(logits, this.model.badWordIds);
		if (step === maxNewTokens - 1) {
			for (let i = 0; i < logits.length; i++) logits[i] = -Infinity;
			logits[this.model.eosId] = 0;
		}
	}
	greedy(cross, maxNewTokens) {
		const state = this.model.createDecoderState(this.model.maxLength, maxNewTokens);
		const output = [];
		let input = this.model.decoderStartId;
		for (let step = 0; step < maxNewTokens; step++) {
			const logits = this.model.step(state, input, cross);
			this.prepareLogits(logits, step, maxNewTokens);
			let bestId = 0;
			let bestScore = -Infinity;
			for (let id = 0; id < logits.length; id++) if (logits[id] > bestScore) {
				bestScore = logits[id];
				bestId = id;
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
	beamSearch(cross, maxNewTokens, numBeams) {
		const lengthPenalty = 1;
		let active = [{
			tokens: [],
			score: 0,
			state: this.model.createDecoderState(this.model.maxLength, maxNewTokens)
		}];
		const finished = [];
		for (let step = 0; step < maxNewTokens; step++) {
			if (active.length === 0) break;
			const candidates = [];
			for (let beamIndex = 0; beamIndex < active.length; beamIndex++) {
				const beam = active[beamIndex];
				const input = beam.tokens.length === 0 ? this.model.decoderStartId : beam.tokens[beam.tokens.length - 1];
				const logits = this.model.step(beam.state, input, cross);
				this.prepareLogits(logits, step, maxNewTokens);
				let max = -Infinity;
				for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
				let sum = 0;
				for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
				const logSum = Math.log(sum) + max;
				const top = topKLogits(logits, 2 * numBeams);
				for (let i = 0; i < top.ids.length; i++) candidates.push({
					beamIndex,
					token: top.ids[i],
					score: beam.score + (top.scores[i] - logSum)
				});
			}
			candidates.sort((a, b) => b.score - a.score);
			if (candidates.length > 2 * numBeams) candidates.length = 2 * numBeams;
			const nextActive = [];
			for (const candidate of candidates) if (candidate.token === this.model.eosId) finished.push({
				tokens: active[candidate.beamIndex].tokens.slice(),
				score: candidate.score,
				endedWithEos: true
			});
			else if (nextActive.length < numBeams) {
				const parent = active[candidate.beamIndex];
				nextActive.push({
					tokens: [...parent.tokens, candidate.token],
					score: candidate.score,
					state: parent.state.clone()
				});
			}
			active = nextActive;
			if (finished.length >= numBeams && active.length > 0) {
				if (finished.map((h) => h.score / (h.tokens.length + 1) ** lengthPenalty).sort((a, b) => a - b)[0] >= Math.max(...active.map((b) => b.score / (b.tokens.length + 1) ** lengthPenalty))) break;
			}
		}
		for (const beam of active) finished.push({
			tokens: beam.tokens.slice(),
			score: beam.score,
			endedWithEos: false
		});
		if (finished.length === 0) return [];
		finished.sort((a, b) => b.score / (b.tokens.length + 1) ** lengthPenalty - a.score / (a.tokens.length + 1) ** lengthPenalty);
		const best = finished[0];
		const result = best.tokens.slice();
		if (best.endedWithEos) result.push(this.model.eosId);
		return result;
	}
};
async function loadElanMT(source, options = {}) {
	let config;
	let spm;
	let weights;
	if (typeof source === "string") {
		const fetchImpl = options.fetch ?? fetch;
		const configResponse = await fetchImpl(joinUrl(source, "config.json"));
		if (!configResponse.ok) throw new Error(`ElanMT: failed to fetch config.json (${configResponse.status})`);
		config = await configResponse.json();
		const spmResponse = await fetchImpl(joinUrl(source, "source.spm"));
		if (!spmResponse.ok) throw new Error(`ElanMT: failed to fetch source.spm (${spmResponse.status})`);
		spm = await spmResponse.arrayBuffer();
		weights = await fetchBuffer(joinUrl(source, "model.safetensors"), fetchImpl, options.onProgress);
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
export { ElanMT, MarianModel, SentencePieceTokenizer, loadElanMT as default, loadElanMT, parseSafetensors };
