// src/bloxd/loadChunk.ts
let queue$1 = [];
const ready = [];
const waitLoadChunk = (pos, callback) => {
	if (api.isBlockInLoadedChunk(pos[0], pos[1], pos[2])) {
		callback();
		return;
	}
	queue$1.push({
		pos,
		callback
	});
};
const update$1 = () => {
	ready.length = 0;
	let write = 0;
	for (let read = 0; read < queue$1.length; read++) {
		const entry = queue$1[read];
		const [x, y, z] = entry.pos;
		if (api.getBlock(x, y, z) !== "Unloaded") ready.push(entry.callback);
		else queue$1[write++] = entry;
	}
	queue$1.length = write;
	for (let i = 0; i < ready.length; i++) ready[i]();
};
// src/bloxd/blockdata.ts
function readText(pos) {
	try {
		return api.getBlockData(pos[0], pos[1], pos[2])?.persisted?.shared?.text;
	} catch {
		return;
	}
}
function readData(pos, callback) {
	const text = readText(pos);
	if (text !== void 0) {
		callback(text);
		return;
	}
	if (api.isBlockInLoadedChunk(pos[0], pos[1], pos[2])) {
		callback(void 0);
		return;
	}
	waitLoadChunk(pos, () => callback(readText(pos)));
}
// src/bloxd/eventLoop.ts
/**
* Per-tick work budget.
*
* Bloxd's api.isNearInterrupt() is not reliable in every runtime, and Date.now()
* can be frozen inside one tick, so the scheduler charges bytes decoded from
* block data instead of measuring time. Every ByteCursor read reports its size
* through spend(); update() stops starting new tasks once the tick's budget is
* exhausted. Bulk loading uses a separate spendLoad() budget so reading the
* full weight stream is split over many ticks instead of consuming the whole
* inference budget at once. A task already running always finishes, so the
* worst case is budget + the largest single task.
*/
const TASK_OVERHEAD = 1024;
const DEFAULT_TICK_BYTE_BUDGET = 8388608;
const DEFAULT_LOAD_BYTE_BUDGET = 262144;
let tickByteBudget = DEFAULT_TICK_BYTE_BUDGET;
let loadByteBudget = DEFAULT_LOAD_BYTE_BUDGET;
let spent = 0;
let loadSpent = 0;
const queue = [];
const setTickBudget = (bytes) => {
	if (!Number.isFinite(bytes) || bytes < 1) throw new RangeError(`Tick budget must be a positive number: ${bytes}`);
	tickByteBudget = Math.floor(bytes);
};
const setLoadBudget = (bytes) => {
	if (!Number.isFinite(bytes) || bytes < 1) throw new RangeError(`Load budget must be a positive number: ${bytes}`);
	loadByteBudget = Math.floor(bytes);
};
const spend = (bytes) => {
	spent += bytes;
};
const spendLoad = (bytes) => {
	loadSpent += bytes;
};
const queueTask = (task) => {
	queue.push(task);
};
const update = () => {
	let head = 0;
	try {
		while (head < queue.length && spent < tickByteBudget && loadSpent < loadByteBudget) {
			const task = queue[head];
			try {
				spent += TASK_OVERHEAD;
				task();
			} finally {
				head++;
			}
		}
	} finally {
		spent = 0;
		loadSpent = 0;
		if (head > 0) {
			let write = 0;
			for (let read = head; read < queue.length; read++) queue[write++] = queue[read];
			queue.length = write;
		}
	}
};
// src/bloxd/byteCursor.ts
function readFloat32LE$1(bytes, offset) {
	const bits = (bytes[offset] | bytes[offset + 1] << 8 | bytes[offset + 2] << 16 | bytes[offset + 3] << 24) >>> 0;
	const sign = bits & 2147483648 ? -1 : 1;
	const exponent = bits >>> 23 & 255;
	const fraction = bits & 8388607;
	if (exponent === 0) {
		if (fraction === 0) return sign === 1 ? 0 : -0;
		return sign * fraction * 2 ** -149;
	}
	if (exponent === 255) return fraction === 0 ? sign * Infinity : NaN;
	return sign * (1 + fraction * 2 ** -23) * 2 ** (exponent - 127);
}
/**
* Random-access reader over safe8 text stored in block data.
*
* The first read of a chunk may have to wait for Bloxd to load it; in that case
* readData defers the callback and the event loop resumes on a later tick.
* Every completed read reports its decoded byte count to the tick budget.
*/
var ByteCursor = class {
	chunkSize;
	coordFn;
	chunkIndex = -1;
	chunkText = "";
	constructor(chunkSize, coordFn) {
		this.chunkSize = chunkSize;
		this.coordFn = coordFn;
	}
	_ensureChunkLoaded(charOffset, callback) {
		const neededChunk = Math.floor(charOffset / this.chunkSize);
		if (neededChunk === this.chunkIndex) {
			callback(charOffset - neededChunk * this.chunkSize);
			return;
		}
		readData(this.coordFn(neededChunk), (text = "") => {
			this.chunkText = text;
			this.chunkIndex = neededChunk;
			callback(charOffset - neededChunk * this.chunkSize);
		});
	}
	readBytesInto(out, tensorOffset, length, callback, outBase = 0, unsigned = false) {
		if (length < 0 || outBase < 0 || length > out.length - outBase) throw new RangeError(`readBytesInto range [${outBase}, ${outBase + length}) exceeds buffer ${out.length}`);
		let encodedCharOffset = Math.floor(tensorOffset * 8 / 3);
		const skipBits = tensorOffset * 8 % 3;
		let firstChar = true;
		let bitBuffer = 0;
		let bitCount = 0;
		let outIdx = 0;
		let lastChunkIndex = -1;
		const readNext = () => {
			if (outIdx >= length) {
				spend(length);
				callback();
				return;
			}
			this._ensureChunkLoaded(encodedCharOffset, (localPos) => {
				if (this.chunkIndex === lastChunkIndex && localPos >= this.chunkText.length) throw new Error(`stalled safe8 read at chunk ${this.chunkIndex}`);
				lastChunkIndex = this.chunkIndex;
				const text = this.chunkText;
				let i = localPos;
				while (i < text.length && outIdx < length) {
					const digit = text.charCodeAt(i) - 48;
					if (digit < 0 || digit > 7) throw new Error(`Invalid safe8 character code: ${text.charCodeAt(i)}`);
					i++;
					if (firstChar) {
						firstChar = false;
						if (skipBits > 0) {
							bitBuffer = digit & (1 << 3 - skipBits) - 1;
							bitCount = 3 - skipBits;
						} else {
							bitBuffer = digit;
							bitCount = 3;
						}
					} else {
						bitBuffer = bitBuffer << 3 | digit;
						bitCount += 3;
					}
					while (bitCount >= 8 && outIdx < length) {
						bitCount -= 8;
						const byte = bitBuffer >> bitCount & 255;
						out[outBase + outIdx++] = unsigned ? byte : byte << 24 >> 24;
					}
					bitBuffer &= (1 << bitCount) - 1;
				}
				encodedCharOffset += i - localPos;
				readNext();
			});
		};
		readNext();
	}
	readBytes(tensorOffset, length, callback) {
		const out = new Array(length).fill(0);
		this.readBytesInto(out, tensorOffset, length, () => callback(out));
	}
	readRawBytes(tensorOffset, length, callback) {
		const out = new Array(length).fill(0);
		this.readBytesInto(out, tensorOffset, length, () => callback(out), 0, true);
	}
	readFloats(tensorOffset, count, callback) {
		this.readRawBytes(tensorOffset, count * 4, (bytes) => {
			const values = new Array(count).fill(0);
			for (let i = 0; i < count; i++) values[i] = readFloat32LE$1(bytes, i * 4);
			callback(values);
		});
	}
};
// src/bloxd/constants.ts
/** Block-data layout for the ElanMT world. */
const CHUNK_SIZE = 16e3;
/** Weight chunks live on the z=32 layer. */
const WEIGHT_ORIGIN = [
	0,
	1,
	32
];
// src/bloxd/ops.ts
/** Element-wise Marian building blocks. */
function layerNormInto(x, weight, bias, out) {
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
	for (let i = 0; i < n; i++) out[i] = Math.fround((x[i] - mean) * inv * weight[i] - -bias[i]);
}
/** Encoder self-attention for a whole source sentence. */
function encoderAttention(q, k, v, heads, out) {
	const length = q.length;
	const headDim = q[0].length / heads;
	const scale = 1 / Math.sqrt(headDim);
	const scores = new Array(length).fill(0);
	for (let h = 0; h < heads; h++) {
		const base = h * headDim;
		for (let i = 0; i < length; i++) {
			let max = -Infinity;
			for (let j = 0; j < length; j++) {
				let dot = 0;
				for (let e = 0; e < headDim; e++) dot -= -q[i][base - -e] * k[j][base - -e];
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
				for (let e = 0; e < headDim; e++) target[base - -e] = Math.fround(target[base - -e] - -p * v[j][base - -e]);
			}
		}
	}
}
/** Decoder masked self-attention for the current token. */
function selfAttentionStep(q, keys, values, pos, heads, out) {
	const headDim = q.length / heads;
	const scale = 1 / Math.sqrt(headDim);
	const scores = new Array(pos + 1).fill(0);
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
			for (let e = 0; e < headDim; e++) out[base - -e] = Math.fround(out[base - -e] - -p * values[j][base - -e]);
		}
	}
}
/** Cross-attention for the current decoder token over encoder memory. */
function crossAttentionStep(q, keys, values, heads, out) {
	const headDim = q.length / heads;
	const scale = 1 / Math.sqrt(headDim);
	const length = keys.length;
	const scores = new Array(length).fill(0);
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
			for (let e = 0; e < headDim; e++) out[base - -e] = Math.fround(out[base - -e] - -p * values[j][base - -e]);
		}
	}
}
function silu(x) {
	return x / (1 + Math.exp(-x));
}
let positionCache = [];
function positionVector(pos, dModel) {
	const cached = positionCache[pos];
	if (cached !== void 0 && cached.length === dModel) return cached;
	const half = Math.floor(dModel / 2);
	const rate = Math.log(1e4) / half;
	const values = new Array(dModel).fill(0);
	for (let i = 0; i < half; i++) {
		const angle = pos * Math.exp(-i * rate);
		values[i] = Math.sin(angle);
		values[half - -i] = Math.cos(angle);
	}
	positionCache[pos] = values;
	return values;
}
function argmax(logits) {
	let best = 0;
	let score = logits[0];
	for (let i = 1; i < logits.length; i++) if (logits[i] > score) {
		score = logits[i];
		best = i;
	}
	return best;
}
// src/bloxd/generate.ts
function applyBadWords(logits, badWordIds) {
	for (let i = 0; i < badWordIds.length; i++) {
		const ids = badWordIds[i];
		if (ids.length === 1) logits[ids[0]] = -Infinity;
	}
}
function forceEos(logits, eosId) {
	for (let i = 0; i < logits.length; i++) logits[i] = -Infinity;
	logits[eosId] = 0;
}
/** Greedy Marian translation; weights stream from block data every step. */
function translateBlock(model, tokenizer, text, callbacks = {}, options = {}) {
	const maxNewTokens = Math.max(1, options.maxNewTokens ?? 32);
	const maxSourceTokens = Math.max(1, options.maxSourceTokens ?? 512);
	const { ids: sourceTokens } = tokenizer.encode(text, maxSourceTokens);
	if (sourceTokens.length === 1 && sourceTokens[0] === model.config.eosId) {
		callbacks.onComplete?.([], sourceTokens);
		return;
	}
	model.prepare(() => {
		model.encode(sourceTokens, (memory) => {
			model.createCrossCache(memory, (cross) => {
				const state = model.createState();
				const output = [];
				let input = model.config.decoderStartId;
				const step = () => {
					if (output.length >= maxNewTokens) {
						callbacks.onComplete?.(output, sourceTokens);
						return;
					}
					model.decode(input, state, cross, (logits) => {
						applyBadWords(logits, model.config.badWordIds);
						if (output.length === maxNewTokens - 1) forceEos(logits, model.config.eosId);
						const token = argmax(logits);
						output.push(token);
						callbacks.onToken?.(output, tokenizer.decode(output));
						if (token === model.config.eosId) {
							callbacks.onComplete?.(output, sourceTokens);
							return;
						}
						input = token;
						queueTask(step);
					});
				};
				queueTask(step);
			});
		});
	});
}
let scratchRaw;
function scratch(byteLength) {
	if (scratchRaw === void 0 || scratchRaw.length < byteLength) scratchRaw = new Array(byteLength).fill(0);
	return scratchRaw;
}
function gemvRowsInto(raw, rawBase, scales, scaleBase, vec, inDim, out, outBase, rowCount) {
	let o = 0;
	const limit4 = inDim - inDim % 4;
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
		for (; i < inDim; i++) sum -= -raw[rowBase - -i] * vec[i];
		out[outBase - -o] = Math.fround(sum * scales[scaleBase - -o]);
	}
}
/** QLoRA NF4 codebook, normalized to the per-group absmax. */
const NF4 = [
	-1,
	-.6961928009986877,
	-.5250730514526367,
	-.39491748809814453,
	-.28444138169288635,
	-.18477343022823334,
	-.09105003625154495,
	0,
	.07958029955625534,
	.16093020141124725,
	.24611230194568634,
	.33791524171829224,
	.44070982933044434,
	.5626170039176941,
	.7229568362236023,
	1
];
function gemvInt4RowsInto(raw, rawBase, rowBytes, scales, scaleBase, groupsPerRow, groupSize, cols, vec, out, outBase, rowCount) {
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
function linearInto(cursor, scales, weight, inputs, outs, callback) {
	const outDim = weight.rows;
	const bits = weight.bits ?? 8;
	const inDim = weight.cols ?? Math.floor(weight.length / weight.rows);
	const rowBytes = weight.rowBytes ?? (bits === 4 ? Math.ceil(inDim / 2) : inDim);
	const groupSize = weight.groupSize ?? 128;
	const groupsPerRow = bits === 4 ? Math.ceil(inDim / groupSize) : 1;
	if (outDim * rowBytes !== weight.length) throw new Error(`Tensor ${weight.name} is not a dense row matrix`);
	if (inputs.length !== outs.length) throw new RangeError("linearInto expects one output per input");
	const raw = scratch(Math.min(256, outDim) * rowBytes);
	let row = 0;
	const processBatch = () => {
		if (row >= outDim) {
			callback();
			return;
		}
		const firstRow = row;
		const rowCount = Math.min(256, outDim - row);
		const rawLength = rowCount * rowBytes;
		row += rowCount;
		cursor.readBytesInto(raw, weight.offset + firstRow * rowBytes, rawLength, () => {
			for (let t = 0; t < inputs.length; t++) if (bits === 4) gemvInt4RowsInto(raw, 0, rowBytes, scales, weight.scaleBase + firstRow * groupsPerRow, groupsPerRow, groupSize, inDim, inputs[t], outs[t], firstRow, rowCount);
			else gemvRowsInto(raw, 0, scales, weight.scaleBase + firstRow, inputs[t], inDim, outs[t], firstRow, rowCount);
			if (inputs.length > 1) spend(Math.ceil(rawLength * (inputs.length - 1) * .15));
			queueTask(processBatch);
		}, 0, bits === 4);
	};
	processBatch();
}
let logitsBuffer;
/** Logits against the shared embedding matrix. */
function computeLogits(cursor, scales, embed, hidden, callback) {
	const vocab = embed.rows;
	const bits = embed.bits ?? 8;
	const dim = embed.cols ?? Math.floor(embed.length / embed.rows);
	const rowBytes = embed.rowBytes ?? (bits === 4 ? Math.ceil(dim / 2) : dim);
	const groupSize = embed.groupSize ?? 128;
	const groupsPerRow = bits === 4 ? Math.ceil(dim / groupSize) : 1;
	if (vocab * rowBytes !== embed.length) throw new Error("Embedding tensor is not a dense row matrix");
	if (logitsBuffer === void 0 || logitsBuffer.length !== vocab) logitsBuffer = new Array(vocab).fill(0);
	const logits = logitsBuffer;
	const raw = scratch(Math.min(512, vocab) * rowBytes);
	let row = 0;
	const processBatch = () => {
		if (row >= vocab) {
			callback(logits);
			return;
		}
		const firstRow = row;
		const rowCount = Math.min(512, vocab - row);
		row += rowCount;
		cursor.readBytesInto(raw, embed.offset + firstRow * rowBytes, rowCount * rowBytes, () => {
			if (bits === 4) gemvInt4RowsInto(raw, 0, rowBytes, scales, embed.scaleBase + firstRow * groupsPerRow, groupsPerRow, groupSize, dim, hidden, logits, firstRow, rowCount);
			else gemvRowsInto(raw, 0, scales, embed.scaleBase + firstRow, hidden, dim, logits, firstRow, rowCount);
			queueTask(processBatch);
		}, 0, bits === 4);
	};
	processBatch();
}
function readRowDequantized(cursor, scales, tensor, row, callback) {
	const bits = tensor.bits ?? 8;
	const dim = tensor.cols ?? Math.floor(tensor.length / tensor.rows);
	const rowBytes = tensor.rowBytes ?? (bits === 4 ? Math.ceil(dim / 2) : dim);
	if (row < 0 || row >= tensor.rows) throw new RangeError(`Tensor ${tensor.name} row ${row} is out of range`);
	const out = new Array(dim).fill(0);
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
function readFloatVector(cursor, tensor, callback) {
	cursor.readFloats(tensor.offset, tensor.count, callback);
}
function readVectorDequantized(cursor, scales, tensor, callback) {
	readRowDequantized(cursor, scales, tensor, 0, callback);
}
// src/bloxd/marian.ts
function series(tasks, done) {
	let index = 0;
	const next = () => {
		if (index >= tasks.length) {
			done();
			return;
		}
		const task = tasks[index++];
		task(next);
	};
	next();
}
var BlockMarian = class {
	cursor;
	config;
	weights;
	rowScales = [];
	biasCache = /* @__PURE__ */ new Map();
	embedScale = 1;
	finalLogitsBias = [];
	prepared = false;
	prepareCallbacks = [];
	encoderNorms = [];
	decoderNorms = [];
	encoderNormCache = [];
	decoderNormCache = [];
	constructor(cursor, config, weights) {
		this.cursor = cursor;
		this.config = config;
		this.weights = weights;
	}
	prepare(callback) {
		this.embedScale = this.config.scaleEmbedding ? Math.sqrt(this.config.dModel) : 1;
		if (this.prepared) {
			callback();
			return;
		}
		this.prepareCallbacks.push(callback);
		if (this.prepareCallbacks.length > 1) return;
		const tasks = [];
		tasks.push((done) => {
			this.cursor.readFloats(this.weights.rowScales.offset, this.weights.rowScales.count, (values) => {
				this.rowScales = values;
				done();
			});
		});
		tasks.push((done) => {
			this.cursor.readFloats(this.weights.finalLogitsBias.offset, this.weights.finalLogitsBias.count, (values) => {
				this.finalLogitsBias = values;
				done();
			});
		});
		for (let layer = 0; layer < this.config.encoderLayers; layer++) {
			const weights = this.weights.encoderLayers[layer];
			for (const linear of [
				weights.selfAttn.q,
				weights.selfAttn.k,
				weights.selfAttn.v,
				weights.selfAttn.out,
				weights.fc1,
				weights.fc2
			]) tasks.push((done) => this.loadBias(linear, done));
			tasks.push((done) => this.loadNormPair(weights.selfAttnNorm, (norm) => {
				this.encoderNormCache[layer] = { selfAttn: norm };
				done();
			}));
			tasks.push((done) => this.loadNormPair(weights.finalNorm, (norm) => {
				const entry = this.encoderNormCache[layer];
				entry.final = norm;
				done();
			}));
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
				weights.fc2
			]) tasks.push((done) => this.loadBias(linear, done));
			tasks.push((done) => this.loadNormPair(weights.selfAttnNorm, (norm) => {
				this.decoderNormCache[layer] = { selfAttn: norm };
				done();
			}));
			tasks.push((done) => this.loadNormPair(weights.encoderAttnNorm, (norm) => {
				const entry = this.decoderNormCache[layer];
				entry.encoderAttn = norm;
				done();
			}));
			tasks.push((done) => this.loadNormPair(weights.finalNorm, (norm) => {
				const entry = this.decoderNormCache[layer];
				entry.final = norm;
				done();
			}));
		}
		series(tasks, () => {
			this.encoderNorms = new Array(this.config.encoderLayers);
			this.decoderNorms = new Array(this.config.decoderLayers);
			for (let i = 0; i < this.config.encoderLayers; i++) {
				const cached = this.encoderNormCache[i];
				this.encoderNorms[i] = {
					selfAttn: cached.selfAttn,
					final: cached.final
				};
			}
			for (let i = 0; i < this.config.decoderLayers; i++) {
				const cached = this.decoderNormCache[i];
				this.decoderNorms[i] = {
					selfAttn: cached.selfAttn,
					encoderAttn: cached.encoderAttn,
					final: cached.final
				};
			}
			this.prepared = true;
			const callbacks = this.prepareCallbacks;
			this.prepareCallbacks = [];
			for (let i = 0; i < callbacks.length; i++) callbacks[i]();
		});
	}
	loadNormPair(weights, callback) {
		let weight;
		readFloatVector(this.cursor, weights.weight, (w) => {
			weight = w;
			readFloatVector(this.cursor, weights.bias, (b) => {
				callback({
					weight,
					bias: b
				});
			});
		});
	}
	loadBias(weights, callback) {
		if (this.biasCache.has(weights.bias.offset)) {
			callback();
			return;
		}
		readVectorDequantized(this.cursor, this.rowScales, weights.bias, (bias) => {
			this.biasCache.set(weights.bias.offset, bias);
			callback();
		});
	}
	linearWithBias(weights, inputs, outs, callback) {
		linearInto(this.cursor, this.rowScales, weights.weight, inputs, outs, () => {
			const bias = this.biasCache.get(weights.bias.offset);
			if (bias === void 0) throw new Error(`Bias ${weights.bias.name} was not loaded`);
			for (let t = 0; t < outs.length; t++) {
				const out = outs[t];
				for (let i = 0; i < out.length; i++) out[i] = Math.fround(out[i] - -bias[i]);
			}
			callback();
		});
	}
	assertPrepared() {
		if (!this.prepared) throw new Error("BlockMarian.prepare() has not finished");
	}
	embedding(token, pos, callback) {
		readRowDequantized(this.cursor, this.rowScales, this.weights.shared, token, (vec) => {
			const position = positionVector(pos, this.config.dModel);
			for (let i = 0; i < vec.length; i++) vec[i] = Math.fround(vec[i] * this.embedScale - -position[i]);
			callback(vec);
		});
	}
	encoderLayer(layerIndex, x, callback) {
		const layer = this.weights.encoderLayers[layerIndex];
		const norms = this.encoderNorms[layerIndex];
		const length = x.length;
		const d = this.config.dModel;
		const heads = this.config.encoderHeads;
		const q = Array.from({ length }, () => new Array(d).fill(0));
		const k = Array.from({ length }, () => new Array(d).fill(0));
		const v = Array.from({ length }, () => new Array(d).fill(0));
		const attn = Array.from({ length }, () => new Array(d).fill(0));
		const normed = Array.from({ length }, () => new Array(d).fill(0));
		const hidden = Array.from({ length }, () => new Array(this.config.encoderFfn).fill(0));
		const ffn = Array.from({ length }, () => new Array(d).fill(0));
		this.linearWithBias(layer.selfAttn.q, x, q, () => {
			this.linearWithBias(layer.selfAttn.k, x, k, () => {
				this.linearWithBias(layer.selfAttn.v, x, v, () => {
					encoderAttention(q, k, v, heads, attn);
					const projected = Array.from({ length }, () => new Array(d).fill(0));
					this.linearWithBias(layer.selfAttn.out, attn, projected, () => {
						for (let t = 0; t < length; t++) layerNormInto(addVectors(x[t], projected[t]), norms.selfAttn.weight, norms.selfAttn.bias, normed[t]);
						this.linearWithBias(layer.fc1, normed, hidden, () => {
							for (let t = 0; t < length; t++) for (let i = 0; i < hidden[t].length; i++) hidden[t][i] = silu(hidden[t][i]);
							this.linearWithBias(layer.fc2, hidden, ffn, () => {
								const out = new Array(length);
								for (let t = 0; t < length; t++) {
									const next = new Array(d).fill(0);
									layerNormInto(addVectors(normed[t], ffn[t]), norms.final.weight, norms.final.bias, next);
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
	encode(ids, callback) {
		this.assertPrepared();
		const tasks = [];
		const embeddings = new Array(ids.length);
		for (let i = 0; i < ids.length; i++) tasks.push((done) => {
			this.embedding(ids[i], i, (vec) => {
				embeddings[i] = vec;
				done();
			});
		});
		series(tasks, () => {
			let layer = 0;
			const run = (x) => {
				if (layer >= this.config.encoderLayers) {
					callback(x);
					return;
				}
				this.encoderLayer(layer++, x, run);
			};
			run(embeddings);
		});
	}
	createCrossCache(memory, callback) {
		this.assertPrepared();
		const keys = new Array(this.config.decoderLayers);
		const values = new Array(this.config.decoderLayers);
		const tasks = [];
		for (let layer = 0; layer < this.config.decoderLayers; layer++) {
			const weights = this.weights.decoderLayers[layer].encoderAttn;
			keys[layer] = Array.from({ length: memory.length }, () => new Array(this.config.dModel).fill(0));
			values[layer] = Array.from({ length: memory.length }, () => new Array(this.config.dModel).fill(0));
			tasks.push((done) => {
				this.linearWithBias(weights.k, memory, keys[layer], () => {
					this.linearWithBias(weights.v, memory, values[layer], done);
				});
			});
		}
		series(tasks, () => callback({
			keys,
			values
		}));
	}
	createState() {
		return {
			selfKeys: Array.from({ length: this.config.decoderLayers }, () => []),
			selfValues: Array.from({ length: this.config.decoderLayers }, () => []),
			length: 0
		};
	}
	decode(token, state, cross, callback) {
		this.assertPrepared();
		const pos = state.length;
		if (pos >= this.config.maxPositions) throw new RangeError("Decoder exceeded the position table");
		this.embedding(token, pos, (embedding) => {
			const x = embedding;
			const d = this.config.dModel;
			const heads = this.config.decoderHeads;
			const runLayer = (layerIndex) => {
				if (layerIndex >= this.config.decoderLayers) {
					state.length = pos + 1;
					computeLogits(this.cursor, this.rowScales, this.weights.shared, x, (logits) => {
						for (let i = 0; i < logits.length; i++) logits[i] = Math.fround(logits[i] - -this.finalLogitsBias[i]);
						callback(logits);
					});
					return;
				}
				const layer = this.weights.decoderLayers[layerIndex];
				const norms = this.decoderNorms[layerIndex];
				const q = new Array(d).fill(0);
				const k = new Array(d).fill(0);
				const v = new Array(d).fill(0);
				const attn = new Array(d).fill(0);
				this.linearWithBias(layer.selfAttn.q, [x], [q], () => {
					this.linearWithBias(layer.selfAttn.k, [x], [k], () => {
						this.linearWithBias(layer.selfAttn.v, [x], [v], () => {
							state.selfKeys[layerIndex][pos] = k;
							state.selfValues[layerIndex][pos] = v;
							selfAttentionStep(q, state.selfKeys[layerIndex], state.selfValues[layerIndex], pos, heads, attn);
							const selfProjected = new Array(d).fill(0);
							this.linearWithBias(layer.selfAttn.out, [attn], [selfProjected], () => {
								const afterSelf = new Array(d).fill(0);
								layerNormInto(addVectors(x, selfProjected), norms.selfAttn.weight, norms.selfAttn.bias, afterSelf);
								const crossQ = new Array(d).fill(0);
								this.linearWithBias(layer.encoderAttn.q, [afterSelf], [crossQ], () => {
									const crossOut = new Array(d).fill(0);
									crossAttentionStep(crossQ, cross.keys[layerIndex], cross.values[layerIndex], heads, crossOut);
									const crossProjected = new Array(d).fill(0);
									this.linearWithBias(layer.encoderAttn.out, [crossOut], [crossProjected], () => {
										const afterCross = new Array(d).fill(0);
										layerNormInto(addVectors(afterSelf, crossProjected), norms.encoderAttn.weight, norms.encoderAttn.bias, afterCross);
										const hidden = new Array(this.config.decoderFfn).fill(0);
										this.linearWithBias(layer.fc1, [afterCross], [hidden], () => {
											for (let i = 0; i < hidden.length; i++) hidden[i] = silu(hidden[i]);
											const ffn = new Array(d).fill(0);
											this.linearWithBias(layer.fc2, [hidden], [ffn], () => {
												const next = new Array(d).fill(0);
												layerNormInto(addVectors(afterCross, ffn), norms.final.weight, norms.final.bias, next);
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
};
function addVectors(a, b) {
	const out = new Array(a.length).fill(0);
	for (let i = 0; i < a.length; i++) out[i] = Math.fround(a[i] - -b[i]);
	return out;
}
const STRING_CHUNK_SIZE = 1 << 18;
const STRING_CHUNK_MASK = 262143;
const BUILD_PART = 8192;
/** Pack 0..255 bytes two per char. An odd trailing byte is padded with lo = 0. */
function bytesToPacked(bytes, length = bytes.length, start = 0) {
	const parts = [];
	const step = BUILD_PART * 2;
	for (let i = 0; i < length; i += step) {
		const end = Math.min(i + step, length);
		const sub = [];
		for (let j = i; j < end; j += 2) {
			const hi = (bytes[start + j] ?? 0) & 255;
			const lo = j + 1 < end ? (bytes[start + j + 1] ?? 0) & 255 : 0;
			sub.push(hi << 8 | lo);
		}
		parts.push(String.fromCharCode.apply(null, sub));
	}
	return parts.join("");
}
function float32FromBits(bits) {
	const sign = bits & 2147483648 ? -1 : 1;
	const exponent = bits >>> 23 & 255;
	const fraction = bits & 8388607;
	if (exponent === 0) {
		if (fraction === 0) return sign === 1 ? 0 : -0;
		return sign * fraction * 2 ** -149;
	}
	if (exponent === 255) return fraction === 0 ? sign * Infinity : NaN;
	return sign * (1 + fraction * 2 ** -23) * 2 ** (exponent - 127);
}
/** Weight source backed by strings packing two bytes per char. */
var StringCursor = class {
	strings;
	byteLength;
	constructor(strings, byteLength) {
		this.strings = strings;
		this.byteLength = byteLength;
	}
	byteAt(offset) {
		const chunk = this.strings[offset >>> 18];
		if (chunk === void 0) return 0;
		const c = chunk.charCodeAt((offset & STRING_CHUNK_MASK) >>> 1);
		return offset & 1 ? c & 255 : c >>> 8;
	}
	readBytesInto(out, tensorOffset, length, callback, outBase = 0, unsigned = false) {
		if (length < 0 || outBase < 0 || length > out.length - outBase) throw new RangeError(`readBytesInto range [${outBase}, ${outBase + length}) exceeds buffer ${out.length}`);
		if (tensorOffset < 0 || tensorOffset + length > this.byteLength) throw new RangeError(`readBytesInto range [${tensorOffset}, ${tensorOffset + length}) exceeds weights`);
		let i = 0;
		while (i < length) {
			const absolute = tensorOffset + i;
			const chunkIndex = absolute >>> 18;
			const chunk = this.strings[chunkIndex];
			if (chunk === void 0) {
				for (; i < length; i++) out[outBase + i] = 0;
				break;
			}
			const local = absolute & STRING_CHUNK_MASK;
			const run = Math.min(length - i, STRING_CHUNK_SIZE - local);
			for (let j = 0; j < run; j++) {
				const p = local + j;
				const c = chunk.charCodeAt(p >>> 1);
				const byte = p & 1 ? c & 255 : c >>> 8;
				out[outBase + i + j] = unsigned ? byte : byte << 24 >> 24;
			}
			i += run;
		}
		spend(length);
		callback();
	}
	readBytes(tensorOffset, length, callback) {
		const out = new Array(length).fill(0);
		this.readBytesInto(out, tensorOffset, length, () => callback(out));
	}
	readRawBytes(tensorOffset, length, callback) {
		const out = new Array(length).fill(0);
		this.readBytesInto(out, tensorOffset, length, () => callback(out), 0, true);
	}
	readFloats(tensorOffset, count, callback) {
		const values = new Array(count).fill(0);
		for (let i = 0; i < count; i++) {
			const o = tensorOffset + i * 4;
			const bits = (this.byteAt(o) | this.byteAt(o + 1) << 8 | this.byteAt(o + 2) << 16 | this.byteAt(o + 3) << 24) >>> 0;
			values[i] = float32FromBits(bits);
		}
		spend(count * 4);
		callback(values);
	}
};
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
function readUint32LE(bytes, offset) {
	return (bytes[offset] | bytes[offset + 1] << 8 | bytes[offset + 2] << 16 | bytes[offset + 3] << 24) >>> 0;
}
function readFloat32LE(bytes, offset) {
	const bits = readUint32LE(bytes, offset);
	const sign = bits & 2147483648 ? -1 : 1;
	const exponent = bits >>> 23 & 255;
	const fraction = bits & 8388607;
	if (exponent === 0) {
		if (fraction === 0) return sign === 1 ? 0 : -0;
		return sign * fraction * 2 ** -149;
	}
	if (exponent === 255) return fraction === 0 ? sign * Infinity : NaN;
	return sign * (1 + fraction * 2 ** -23) * 2 ** (exponent - 127);
}
var ProtobufReader = class {
	bytes;
	offset = 0;
	constructor(bytes) {
		this.bytes = bytes;
	}
	get done() {
		return this.offset >= this.bytes.length;
	}
	varint() {
		let result = 0;
		let shift = 0;
		while (this.offset < this.bytes.length) {
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
		const value = readFloat32LE(this.bytes, this.offset);
		this.offset += 4;
		return value;
	}
	bytesField(length) {
		const value = this.bytes.slice(this.offset, this.offset + length);
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
		charsmap: [],
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
			charsmap: [],
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
		if (charsmap.length < 4) {
			this.units = [];
			this.blob = [];
			return;
		}
		const trieBytes = readUint32LE(charsmap, 0);
		if (trieBytes % 4 !== 0 || 4 + trieBytes > charsmap.length) throw new Error("sentencepiece: invalid normalization charsmap");
		const unitCount = trieBytes / 4;
		this.units = new Array(unitCount);
		for (let i = 0; i < unitCount; i++) this.units[i] = readUint32LE(charsmap, 4 + i * 4);
		this.blob = charsmap.slice(4 + trieBytes);
	}
	static offset(unit) {
		return unit >>> 10 << ((unit & 512) >>> 6) >>> 0;
	}
	valueString(valueOffset) {
		const cached = this.cache.get(valueOffset);
		if (cached !== void 0) return cached;
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
	normalizeBytes(input) {
		if (this.units.length === 0) return input;
		const output = [];
		const { units } = this;
		let i = 0;
		while (i < input.length) {
			let pos = 0;
			let matchLength = 0;
			let value = -1;
			for (let j = i; j < input.length; j++) {
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
				for (let k = 0; k < length && i + k < input.length; k++) output.push(input[i + k]);
				i += length;
			}
		}
		return output;
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
	constructor(bytes, eosId = 0, unkId = 1) {
		const model = parseModel(bytes);
		if (model.modelType !== 1) throw new Error(`sentencepiece: model type ${model.modelType} is not supported (only unigram)`);
		this.eosId = eosId;
		this.unkId = unkId;
		this.normalizer = model.normalizer;
		this.charsMap = new CharsMap(model.normalizer.charsmap);
		this.pieces = model.pieces.map((p) => p.piece);
		this.scores = model.pieces.map((p) => p.score);
		this.types = model.pieces.map((p) => p.type);
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
		const raw = encodeUtf8(text);
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
		const best = new Array(n + 1).fill(-Infinity);
		const backStart = new Array(n + 1).fill(-1);
		const backId = new Array(n + 1).fill(-1);
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
// src/bloxd/tokenizerBlock.ts
let tokenizer;
/** Load and parse the raw source.spm blob stored in block data. */
function loadTokenizer(cursor, weights, eosId, callback) {
	if (tokenizer !== void 0) {
		callback(tokenizer);
		return;
	}
	cursor.readRawBytes(weights.tokenizer.offset, weights.tokenizer.length, (bytes) => {
		tokenizer = new SentencePieceTokenizer(bytes, eosId, 1);
		callback(tokenizer);
	});
}
// src/bloxd/memory.ts
/**
* Read the whole stream into packed string chunks (two bytes per char), one
* buffer per task. The task size must equal STRING_CHUNK_SIZE so that every
* pushed string lines up with StringCursor's chunk indexing.
*/
function loadWeightsAsStrings(cursor, totalLengthOrWeights, callback) {
	const totalLength = typeof totalLengthOrWeights === "number" ? totalLengthOrWeights : totalLengthOrWeights.totalLength ?? NaN;
	if (!Number.isFinite(totalLength) || totalLength <= 0) throw new RangeError(`invalid weight length: ${String(totalLength)}`);
	const strings = [];
	const chunkSize = STRING_CHUNK_SIZE;
	const temp = new Array(chunkSize).fill(0);
	let offset = 0;
	const next = () => {
		if (offset >= totalLength) {
			callback(new StringCursor(strings, totalLength));
			return;
		}
		console.log(offset);
		const length = Math.min(chunkSize, totalLength - offset);
		const start = offset;
		offset += length;
		cursor.readBytesInto(temp, start, length, () => {
			strings.push(bytesToPacked(temp, length));
			spendLoad(length);
			queueTask(next);
		}, 0, true);
	};
	next();
}
// src/bloxd/manifestInt4.ts
const MODEL_CONFIG = {
	dModel: 512,
	encoderLayers: 6,
	decoderLayers: 6,
	encoderHeads: 8,
	decoderHeads: 8,
	encoderFfn: 2048,
	decoderFfn: 2048,
	vocabSize: 32001,
	maxPositions: 512,
	maxLength: 512,
	eosId: 0,
	padId: 32e3,
	decoderStartId: 32e3,
	scaleEmbedding: true,
	badWordIds: [[32e3]]
};
const MODEL_WEIGHTS = {
	shared: {
		name: "model.shared.weight",
		shape: [32001, 512],
		offset: 0,
		length: 16384512,
		rows: 32001,
		cols: 512,
		rowBytes: 512,
		bits: 8,
		groupSize: 0,
		scaleBase: 0
	},
	encoderLayers: [
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.0.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 16384512,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 32001
					},
					bias: {
						name: "model.encoder.layers.0.self_attn.q_proj.bias",
						shape: [512],
						offset: 16515584,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 34049
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.0.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 16516096,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 34050
					},
					bias: {
						name: "model.encoder.layers.0.self_attn.k_proj.bias",
						shape: [512],
						offset: 16647168,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 36098
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.0.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 16647680,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 36099
					},
					bias: {
						name: "model.encoder.layers.0.self_attn.v_proj.bias",
						shape: [512],
						offset: 16778752,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 38147
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.0.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 16779264,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 38148
					},
					bias: {
						name: "model.encoder.layers.0.self_attn.out_proj.bias",
						shape: [512],
						offset: 16910336,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 40196
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.0.self_attn_layer_norm.weight",
					shape: [512],
					offset: 16910848,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.0.self_attn_layer_norm.bias",
					shape: [512],
					offset: 16912896,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.0.final_layer_norm.weight",
					shape: [512],
					offset: 16914944,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.0.final_layer_norm.bias",
					shape: [512],
					offset: 16916992,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.0.fc1.weight",
					shape: [2048, 512],
					offset: 16919040,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 40197
				},
				bias: {
					name: "model.encoder.layers.0.fc1.bias",
					shape: [2048],
					offset: 17443328,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 48389
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.0.fc2.weight",
					shape: [512, 2048],
					offset: 17445376,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 48390
				},
				bias: {
					name: "model.encoder.layers.0.fc2.bias",
					shape: [512],
					offset: 17969664,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 56582
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.1.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 17970176,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 56583
					},
					bias: {
						name: "model.encoder.layers.1.self_attn.q_proj.bias",
						shape: [512],
						offset: 18101248,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 58631
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.1.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 18101760,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 58632
					},
					bias: {
						name: "model.encoder.layers.1.self_attn.k_proj.bias",
						shape: [512],
						offset: 18232832,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 60680
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.1.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 18233344,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 60681
					},
					bias: {
						name: "model.encoder.layers.1.self_attn.v_proj.bias",
						shape: [512],
						offset: 18364416,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 62729
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.1.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 18364928,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 62730
					},
					bias: {
						name: "model.encoder.layers.1.self_attn.out_proj.bias",
						shape: [512],
						offset: 18496e3,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 64778
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.1.self_attn_layer_norm.weight",
					shape: [512],
					offset: 18496512,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.1.self_attn_layer_norm.bias",
					shape: [512],
					offset: 18498560,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.1.final_layer_norm.weight",
					shape: [512],
					offset: 18500608,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.1.final_layer_norm.bias",
					shape: [512],
					offset: 18502656,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.1.fc1.weight",
					shape: [2048, 512],
					offset: 18504704,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 64779
				},
				bias: {
					name: "model.encoder.layers.1.fc1.bias",
					shape: [2048],
					offset: 19028992,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 72971
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.1.fc2.weight",
					shape: [512, 2048],
					offset: 19031040,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 72972
				},
				bias: {
					name: "model.encoder.layers.1.fc2.bias",
					shape: [512],
					offset: 19555328,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 81164
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.2.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 19555840,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 81165
					},
					bias: {
						name: "model.encoder.layers.2.self_attn.q_proj.bias",
						shape: [512],
						offset: 19686912,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 83213
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.2.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 19687424,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 83214
					},
					bias: {
						name: "model.encoder.layers.2.self_attn.k_proj.bias",
						shape: [512],
						offset: 19818496,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 85262
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.2.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 19819008,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 85263
					},
					bias: {
						name: "model.encoder.layers.2.self_attn.v_proj.bias",
						shape: [512],
						offset: 19950080,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 87311
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.2.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 19950592,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 87312
					},
					bias: {
						name: "model.encoder.layers.2.self_attn.out_proj.bias",
						shape: [512],
						offset: 20081664,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 89360
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.2.self_attn_layer_norm.weight",
					shape: [512],
					offset: 20082176,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.2.self_attn_layer_norm.bias",
					shape: [512],
					offset: 20084224,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.2.final_layer_norm.weight",
					shape: [512],
					offset: 20086272,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.2.final_layer_norm.bias",
					shape: [512],
					offset: 20088320,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.2.fc1.weight",
					shape: [2048, 512],
					offset: 20090368,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 89361
				},
				bias: {
					name: "model.encoder.layers.2.fc1.bias",
					shape: [2048],
					offset: 20614656,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 97553
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.2.fc2.weight",
					shape: [512, 2048],
					offset: 20616704,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 97554
				},
				bias: {
					name: "model.encoder.layers.2.fc2.bias",
					shape: [512],
					offset: 21140992,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 105746
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.3.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 21141504,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 105747
					},
					bias: {
						name: "model.encoder.layers.3.self_attn.q_proj.bias",
						shape: [512],
						offset: 21272576,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 107795
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.3.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 21273088,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 107796
					},
					bias: {
						name: "model.encoder.layers.3.self_attn.k_proj.bias",
						shape: [512],
						offset: 21404160,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 109844
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.3.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 21404672,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 109845
					},
					bias: {
						name: "model.encoder.layers.3.self_attn.v_proj.bias",
						shape: [512],
						offset: 21535744,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 111893
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.3.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 21536256,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 111894
					},
					bias: {
						name: "model.encoder.layers.3.self_attn.out_proj.bias",
						shape: [512],
						offset: 21667328,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 113942
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.3.self_attn_layer_norm.weight",
					shape: [512],
					offset: 21667840,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.3.self_attn_layer_norm.bias",
					shape: [512],
					offset: 21669888,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.3.final_layer_norm.weight",
					shape: [512],
					offset: 21671936,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.3.final_layer_norm.bias",
					shape: [512],
					offset: 21673984,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.3.fc1.weight",
					shape: [2048, 512],
					offset: 21676032,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 113943
				},
				bias: {
					name: "model.encoder.layers.3.fc1.bias",
					shape: [2048],
					offset: 22200320,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 122135
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.3.fc2.weight",
					shape: [512, 2048],
					offset: 22202368,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 122136
				},
				bias: {
					name: "model.encoder.layers.3.fc2.bias",
					shape: [512],
					offset: 22726656,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 130328
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.4.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 22727168,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 130329
					},
					bias: {
						name: "model.encoder.layers.4.self_attn.q_proj.bias",
						shape: [512],
						offset: 22858240,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 132377
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.4.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 22858752,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 132378
					},
					bias: {
						name: "model.encoder.layers.4.self_attn.k_proj.bias",
						shape: [512],
						offset: 22989824,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 134426
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.4.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 22990336,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 134427
					},
					bias: {
						name: "model.encoder.layers.4.self_attn.v_proj.bias",
						shape: [512],
						offset: 23121408,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 136475
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.4.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 23121920,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 136476
					},
					bias: {
						name: "model.encoder.layers.4.self_attn.out_proj.bias",
						shape: [512],
						offset: 23252992,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 138524
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.4.self_attn_layer_norm.weight",
					shape: [512],
					offset: 23253504,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.4.self_attn_layer_norm.bias",
					shape: [512],
					offset: 23255552,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.4.final_layer_norm.weight",
					shape: [512],
					offset: 23257600,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.4.final_layer_norm.bias",
					shape: [512],
					offset: 23259648,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.4.fc1.weight",
					shape: [2048, 512],
					offset: 23261696,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 138525
				},
				bias: {
					name: "model.encoder.layers.4.fc1.bias",
					shape: [2048],
					offset: 23785984,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 146717
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.4.fc2.weight",
					shape: [512, 2048],
					offset: 23788032,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 146718
				},
				bias: {
					name: "model.encoder.layers.4.fc2.bias",
					shape: [512],
					offset: 24312320,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 154910
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.5.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 24312832,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 154911
					},
					bias: {
						name: "model.encoder.layers.5.self_attn.q_proj.bias",
						shape: [512],
						offset: 24443904,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 156959
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.5.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 24444416,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 156960
					},
					bias: {
						name: "model.encoder.layers.5.self_attn.k_proj.bias",
						shape: [512],
						offset: 24575488,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 159008
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.5.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 24576e3,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 159009
					},
					bias: {
						name: "model.encoder.layers.5.self_attn.v_proj.bias",
						shape: [512],
						offset: 24707072,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 161057
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.5.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 24707584,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 161058
					},
					bias: {
						name: "model.encoder.layers.5.self_attn.out_proj.bias",
						shape: [512],
						offset: 24838656,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 163106
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.5.self_attn_layer_norm.weight",
					shape: [512],
					offset: 24839168,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.5.self_attn_layer_norm.bias",
					shape: [512],
					offset: 24841216,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.5.final_layer_norm.weight",
					shape: [512],
					offset: 24843264,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.5.final_layer_norm.bias",
					shape: [512],
					offset: 24845312,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.5.fc1.weight",
					shape: [2048, 512],
					offset: 24847360,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 163107
				},
				bias: {
					name: "model.encoder.layers.5.fc1.bias",
					shape: [2048],
					offset: 25371648,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 171299
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.5.fc2.weight",
					shape: [512, 2048],
					offset: 25373696,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 171300
				},
				bias: {
					name: "model.encoder.layers.5.fc2.bias",
					shape: [512],
					offset: 25897984,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 179492
				}
			}
		}
	],
	decoderLayers: [
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.0.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 25898496,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 179493
					},
					bias: {
						name: "model.decoder.layers.0.self_attn.q_proj.bias",
						shape: [512],
						offset: 26029568,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 181541
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.0.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 26030080,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 181542
					},
					bias: {
						name: "model.decoder.layers.0.self_attn.k_proj.bias",
						shape: [512],
						offset: 26161152,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 183590
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.0.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 26161664,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 183591
					},
					bias: {
						name: "model.decoder.layers.0.self_attn.v_proj.bias",
						shape: [512],
						offset: 26292736,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 185639
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.0.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 26293248,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 185640
					},
					bias: {
						name: "model.decoder.layers.0.self_attn.out_proj.bias",
						shape: [512],
						offset: 26424320,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 187688
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.0.self_attn_layer_norm.weight",
					shape: [512],
					offset: 26424832,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.0.self_attn_layer_norm.bias",
					shape: [512],
					offset: 26426880,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.0.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 26428928,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 187689
					},
					bias: {
						name: "model.decoder.layers.0.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 2656e4,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 189737
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.0.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 26560512,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 189738
					},
					bias: {
						name: "model.decoder.layers.0.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 26691584,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 191786
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.0.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 26692096,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 191787
					},
					bias: {
						name: "model.decoder.layers.0.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 26823168,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 193835
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.0.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 26823680,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 193836
					},
					bias: {
						name: "model.decoder.layers.0.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 26954752,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 195884
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.0.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 26955264,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.0.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 26957312,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.0.final_layer_norm.weight",
					shape: [512],
					offset: 26959360,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.0.final_layer_norm.bias",
					shape: [512],
					offset: 26961408,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.0.fc1.weight",
					shape: [2048, 512],
					offset: 26963456,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 195885
				},
				bias: {
					name: "model.decoder.layers.0.fc1.bias",
					shape: [2048],
					offset: 27487744,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 204077
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.0.fc2.weight",
					shape: [512, 2048],
					offset: 27489792,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 204078
				},
				bias: {
					name: "model.decoder.layers.0.fc2.bias",
					shape: [512],
					offset: 28014080,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 212270
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.1.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 28014592,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 212271
					},
					bias: {
						name: "model.decoder.layers.1.self_attn.q_proj.bias",
						shape: [512],
						offset: 28145664,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 214319
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.1.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 28146176,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 214320
					},
					bias: {
						name: "model.decoder.layers.1.self_attn.k_proj.bias",
						shape: [512],
						offset: 28277248,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 216368
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.1.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 28277760,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 216369
					},
					bias: {
						name: "model.decoder.layers.1.self_attn.v_proj.bias",
						shape: [512],
						offset: 28408832,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 218417
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.1.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 28409344,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 218418
					},
					bias: {
						name: "model.decoder.layers.1.self_attn.out_proj.bias",
						shape: [512],
						offset: 28540416,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 220466
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.1.self_attn_layer_norm.weight",
					shape: [512],
					offset: 28540928,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.1.self_attn_layer_norm.bias",
					shape: [512],
					offset: 28542976,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.1.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 28545024,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 220467
					},
					bias: {
						name: "model.decoder.layers.1.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 28676096,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 222515
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.1.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 28676608,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 222516
					},
					bias: {
						name: "model.decoder.layers.1.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 28807680,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 224564
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.1.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 28808192,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 224565
					},
					bias: {
						name: "model.decoder.layers.1.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 28939264,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 226613
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.1.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 28939776,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 226614
					},
					bias: {
						name: "model.decoder.layers.1.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 29070848,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 228662
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.1.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 29071360,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.1.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 29073408,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.1.final_layer_norm.weight",
					shape: [512],
					offset: 29075456,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.1.final_layer_norm.bias",
					shape: [512],
					offset: 29077504,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.1.fc1.weight",
					shape: [2048, 512],
					offset: 29079552,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 228663
				},
				bias: {
					name: "model.decoder.layers.1.fc1.bias",
					shape: [2048],
					offset: 29603840,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 236855
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.1.fc2.weight",
					shape: [512, 2048],
					offset: 29605888,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 236856
				},
				bias: {
					name: "model.decoder.layers.1.fc2.bias",
					shape: [512],
					offset: 30130176,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 245048
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.2.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 30130688,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 245049
					},
					bias: {
						name: "model.decoder.layers.2.self_attn.q_proj.bias",
						shape: [512],
						offset: 30261760,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 247097
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.2.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 30262272,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 247098
					},
					bias: {
						name: "model.decoder.layers.2.self_attn.k_proj.bias",
						shape: [512],
						offset: 30393344,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 249146
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.2.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 30393856,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 249147
					},
					bias: {
						name: "model.decoder.layers.2.self_attn.v_proj.bias",
						shape: [512],
						offset: 30524928,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 251195
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.2.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 30525440,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 251196
					},
					bias: {
						name: "model.decoder.layers.2.self_attn.out_proj.bias",
						shape: [512],
						offset: 30656512,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 253244
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.2.self_attn_layer_norm.weight",
					shape: [512],
					offset: 30657024,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.2.self_attn_layer_norm.bias",
					shape: [512],
					offset: 30659072,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.2.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 30661120,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 253245
					},
					bias: {
						name: "model.decoder.layers.2.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 30792192,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 255293
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.2.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 30792704,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 255294
					},
					bias: {
						name: "model.decoder.layers.2.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 30923776,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 257342
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.2.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 30924288,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 257343
					},
					bias: {
						name: "model.decoder.layers.2.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 31055360,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 259391
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.2.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 31055872,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 259392
					},
					bias: {
						name: "model.decoder.layers.2.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 31186944,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 261440
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.2.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 31187456,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.2.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 31189504,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.2.final_layer_norm.weight",
					shape: [512],
					offset: 31191552,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.2.final_layer_norm.bias",
					shape: [512],
					offset: 31193600,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.2.fc1.weight",
					shape: [2048, 512],
					offset: 31195648,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 261441
				},
				bias: {
					name: "model.decoder.layers.2.fc1.bias",
					shape: [2048],
					offset: 31719936,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 269633
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.2.fc2.weight",
					shape: [512, 2048],
					offset: 31721984,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 269634
				},
				bias: {
					name: "model.decoder.layers.2.fc2.bias",
					shape: [512],
					offset: 32246272,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 277826
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.3.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 32246784,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 277827
					},
					bias: {
						name: "model.decoder.layers.3.self_attn.q_proj.bias",
						shape: [512],
						offset: 32377856,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 279875
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.3.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 32378368,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 279876
					},
					bias: {
						name: "model.decoder.layers.3.self_attn.k_proj.bias",
						shape: [512],
						offset: 32509440,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 281924
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.3.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 32509952,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 281925
					},
					bias: {
						name: "model.decoder.layers.3.self_attn.v_proj.bias",
						shape: [512],
						offset: 32641024,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 283973
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.3.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 32641536,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 283974
					},
					bias: {
						name: "model.decoder.layers.3.self_attn.out_proj.bias",
						shape: [512],
						offset: 32772608,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 286022
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.3.self_attn_layer_norm.weight",
					shape: [512],
					offset: 32773120,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.3.self_attn_layer_norm.bias",
					shape: [512],
					offset: 32775168,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.3.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 32777216,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 286023
					},
					bias: {
						name: "model.decoder.layers.3.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 32908288,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 288071
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.3.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 32908800,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 288072
					},
					bias: {
						name: "model.decoder.layers.3.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 33039872,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 290120
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.3.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 33040384,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 290121
					},
					bias: {
						name: "model.decoder.layers.3.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 33171456,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 292169
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.3.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 33171968,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 292170
					},
					bias: {
						name: "model.decoder.layers.3.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 33303040,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 294218
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.3.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 33303552,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.3.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 33305600,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.3.final_layer_norm.weight",
					shape: [512],
					offset: 33307648,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.3.final_layer_norm.bias",
					shape: [512],
					offset: 33309696,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.3.fc1.weight",
					shape: [2048, 512],
					offset: 33311744,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 294219
				},
				bias: {
					name: "model.decoder.layers.3.fc1.bias",
					shape: [2048],
					offset: 33836032,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 302411
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.3.fc2.weight",
					shape: [512, 2048],
					offset: 33838080,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 302412
				},
				bias: {
					name: "model.decoder.layers.3.fc2.bias",
					shape: [512],
					offset: 34362368,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 310604
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.4.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 34362880,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 310605
					},
					bias: {
						name: "model.decoder.layers.4.self_attn.q_proj.bias",
						shape: [512],
						offset: 34493952,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 312653
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.4.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 34494464,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 312654
					},
					bias: {
						name: "model.decoder.layers.4.self_attn.k_proj.bias",
						shape: [512],
						offset: 34625536,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 314702
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.4.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 34626048,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 314703
					},
					bias: {
						name: "model.decoder.layers.4.self_attn.v_proj.bias",
						shape: [512],
						offset: 34757120,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 316751
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.4.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 34757632,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 316752
					},
					bias: {
						name: "model.decoder.layers.4.self_attn.out_proj.bias",
						shape: [512],
						offset: 34888704,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 318800
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.4.self_attn_layer_norm.weight",
					shape: [512],
					offset: 34889216,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.4.self_attn_layer_norm.bias",
					shape: [512],
					offset: 34891264,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.4.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 34893312,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 318801
					},
					bias: {
						name: "model.decoder.layers.4.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 35024384,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 320849
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.4.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 35024896,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 320850
					},
					bias: {
						name: "model.decoder.layers.4.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 35155968,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 322898
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.4.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 35156480,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 322899
					},
					bias: {
						name: "model.decoder.layers.4.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 35287552,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 324947
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.4.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 35288064,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 324948
					},
					bias: {
						name: "model.decoder.layers.4.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 35419136,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 326996
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.4.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 35419648,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.4.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 35421696,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.4.final_layer_norm.weight",
					shape: [512],
					offset: 35423744,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.4.final_layer_norm.bias",
					shape: [512],
					offset: 35425792,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.4.fc1.weight",
					shape: [2048, 512],
					offset: 35427840,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 326997
				},
				bias: {
					name: "model.decoder.layers.4.fc1.bias",
					shape: [2048],
					offset: 35952128,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 335189
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.4.fc2.weight",
					shape: [512, 2048],
					offset: 35954176,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 335190
				},
				bias: {
					name: "model.decoder.layers.4.fc2.bias",
					shape: [512],
					offset: 36478464,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 343382
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.5.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 36478976,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 343383
					},
					bias: {
						name: "model.decoder.layers.5.self_attn.q_proj.bias",
						shape: [512],
						offset: 36610048,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 345431
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.5.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 36610560,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 345432
					},
					bias: {
						name: "model.decoder.layers.5.self_attn.k_proj.bias",
						shape: [512],
						offset: 36741632,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 347480
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.5.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 36742144,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 347481
					},
					bias: {
						name: "model.decoder.layers.5.self_attn.v_proj.bias",
						shape: [512],
						offset: 36873216,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 349529
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.5.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 36873728,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 349530
					},
					bias: {
						name: "model.decoder.layers.5.self_attn.out_proj.bias",
						shape: [512],
						offset: 37004800,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 351578
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.5.self_attn_layer_norm.weight",
					shape: [512],
					offset: 37005312,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.5.self_attn_layer_norm.bias",
					shape: [512],
					offset: 37007360,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.5.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 37009408,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 351579
					},
					bias: {
						name: "model.decoder.layers.5.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 37140480,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 353627
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.5.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 37140992,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 353628
					},
					bias: {
						name: "model.decoder.layers.5.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 37272064,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 355676
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.5.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 37272576,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 355677
					},
					bias: {
						name: "model.decoder.layers.5.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 37403648,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 357725
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.5.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 37404160,
						length: 131072,
						rows: 512,
						cols: 512,
						rowBytes: 256,
						bits: 4,
						groupSize: 128,
						scaleBase: 357726
					},
					bias: {
						name: "model.decoder.layers.5.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 37535232,
						length: 512,
						rows: 1,
						cols: 512,
						rowBytes: 512,
						bits: 8,
						groupSize: 0,
						scaleBase: 359774
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.5.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 37535744,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.5.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 37537792,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.5.final_layer_norm.weight",
					shape: [512],
					offset: 37539840,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.5.final_layer_norm.bias",
					shape: [512],
					offset: 37541888,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.5.fc1.weight",
					shape: [2048, 512],
					offset: 37543936,
					length: 524288,
					rows: 2048,
					cols: 512,
					rowBytes: 256,
					bits: 4,
					groupSize: 128,
					scaleBase: 359775
				},
				bias: {
					name: "model.decoder.layers.5.fc1.bias",
					shape: [2048],
					offset: 38068224,
					length: 2048,
					rows: 1,
					cols: 2048,
					rowBytes: 2048,
					bits: 8,
					groupSize: 0,
					scaleBase: 367967
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.5.fc2.weight",
					shape: [512, 2048],
					offset: 38070272,
					length: 524288,
					rows: 512,
					cols: 2048,
					rowBytes: 1024,
					bits: 4,
					groupSize: 128,
					scaleBase: 367968
				},
				bias: {
					name: "model.decoder.layers.5.fc2.bias",
					shape: [512],
					offset: 38594560,
					length: 512,
					rows: 1,
					cols: 512,
					rowBytes: 512,
					bits: 8,
					groupSize: 0,
					scaleBase: 376160
				}
			}
		}
	],
	finalLogitsBias: {
		name: "final_logits_bias",
		shape: [32001],
		offset: 40099716,
		length: 128004,
		count: 32001
	},
	rowScales: {
		name: "row_scales",
		shape: [376161],
		offset: 38595072,
		length: 1504644,
		count: 376161
	},
	tokenizer: {
		offset: 40227720,
		length: 792751
	},
	totalLength: 41020471
};
// src/bloxd/weightCoordFn.ts
const BLOCK_SIZE = 32;
const SLOT_SPACING = 4;
const BLOCKS_PER_SPATIAL_CHUNK = 8;
/** Map a safe8 chunk index to the block that stores it. */
function createWeightCoordFn(origin = WEIGHT_ORIGIN) {
	return (index) => {
		if (!Number.isInteger(index) || index < 0) throw new RangeError(`Invalid weight chunk index: ${index}`);
		const spatialIndex = Math.floor(index / BLOCKS_PER_SPATIAL_CHUNK);
		const slot = index % BLOCKS_PER_SPATIAL_CHUNK;
		return [
			origin[0] + spatialIndex * BLOCK_SIZE + slot * SLOT_SPACING,
			origin[1],
			origin[2]
		];
	};
}
// src/bloxd/int4.ts
/**
* Bloxd world-code entry point for the in-memory int4 model.
*
* The 41 MB NF4 stream lives in block data. On the first think() call it is
* read once into a number[], then every layer runs from memory; tick() only
* has to advance the generator queue, not fetch weights again.
*/
const sourceCursor = new ByteCursor(CHUNK_SIZE, createWeightCoordFn([
	0,
	1,
	64
]));
setTickBudget(67108864);
setLoadBudget(1048576);
let memoryCursor;
let model;
let loading = false;
const waiting = [];
globalThis.tick = () => {
	update$1();
	update();
};
function ensureModel(callback) {
	if (model !== void 0 && memoryCursor !== void 0) {
		callback(model, memoryCursor);
		return;
	}
	waiting.push(callback);
	if (loading) return;
	loading = true;
	const totalLength = MODEL_WEIGHTS.totalLength;
	if (totalLength === void 0 || totalLength <= 0) throw new Error("int4 manifest is missing totalLength");
	loadWeightsAsStrings(sourceCursor, totalLength, (cursor) => {
		memoryCursor = cursor;
		const loaded = new BlockMarian(cursor, MODEL_CONFIG, MODEL_WEIGHTS);
		loaded.prepare(() => {
			model = loaded;
			const callbacks = waiting.splice(0);
			for (let i = 0; i < callbacks.length; i++) callbacks[i](loaded, cursor);
		});
	});
}
function think(text, callbacks = {}, maxNewTokens = 32) {
	queueTask(() => {
		ensureModel((loaded, cursor) => {
			console.log("model loaded");
			loadTokenizer(cursor, MODEL_WEIGHTS, MODEL_CONFIG.eosId, (tokenizer) => {
				console.log("tokenizer loaded");
				translateBlock(loaded, tokenizer, text, {
					onToken: (tokens, decoded) => callbacks.onToken?.(tokens, decoded),
					onComplete: (tokens, sourceTokens) => {
						const decoded = tokenizer.decode(tokens);
						callbacks.onComplete?.(tokens, sourceTokens.length, decoded);
						if (!callbacks.onComplete) console.log(decoded);
					}
				}, { maxNewTokens });
			});
		});
	});
}
const enqueueTranslation = think;
globalThis.think = ((text, callbacks, _stream = true, maxNewTokens = 32) => {
	if (typeof callbacks === "object" && callbacks !== null) {
		enqueueTranslation(text, callbacks, maxNewTokens);
		return;
	}
	enqueueTranslation(text, { onComplete: (tokens, promptLength, decoded) => {
		if (callbacks) callbacks(tokens, promptLength);
		else console.log(decoded);
	} }, maxNewTokens);
});
globalThis.setTickBudget = setTickBudget;
globalThis.setLoadBudget = setLoadBudget;
