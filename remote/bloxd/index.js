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
const DEFAULT_LOAD_BYTE_BUDGET = 1048576;
let tickByteBudget = DEFAULT_TICK_BYTE_BUDGET;
let loadByteBudget = DEFAULT_LOAD_BYTE_BUDGET;
let spent = 0;
let loadSpent = 0;
const queue = [];
const setTickBudget = (bytes) => {
	if (!Number.isFinite(bytes) || bytes < 1) throw new RangeError(`Tick budget must be a positive number: ${bytes}`);
	tickByteBudget = Math.floor(bytes);
};
const getTickBudget = () => tickByteBudget;
const spend = (bytes) => {
	spent += bytes;
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
// src/bloxd/manifest.ts
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
						length: 262144,
						rows: 512,
						scaleBase: 32001
					},
					bias: {
						name: "model.encoder.layers.0.self_attn.q_proj.bias",
						shape: [512],
						offset: 16646656,
						length: 512,
						rows: 1,
						scaleBase: 32513
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.0.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 16647168,
						length: 262144,
						rows: 512,
						scaleBase: 32514
					},
					bias: {
						name: "model.encoder.layers.0.self_attn.k_proj.bias",
						shape: [512],
						offset: 16909312,
						length: 512,
						rows: 1,
						scaleBase: 33026
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.0.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 16909824,
						length: 262144,
						rows: 512,
						scaleBase: 33027
					},
					bias: {
						name: "model.encoder.layers.0.self_attn.v_proj.bias",
						shape: [512],
						offset: 17171968,
						length: 512,
						rows: 1,
						scaleBase: 33539
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.0.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 17172480,
						length: 262144,
						rows: 512,
						scaleBase: 33540
					},
					bias: {
						name: "model.encoder.layers.0.self_attn.out_proj.bias",
						shape: [512],
						offset: 17434624,
						length: 512,
						rows: 1,
						scaleBase: 34052
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.0.self_attn_layer_norm.weight",
					shape: [512],
					offset: 17435136,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.0.self_attn_layer_norm.bias",
					shape: [512],
					offset: 17437184,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.0.final_layer_norm.weight",
					shape: [512],
					offset: 17439232,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.0.final_layer_norm.bias",
					shape: [512],
					offset: 17441280,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.0.fc1.weight",
					shape: [2048, 512],
					offset: 17443328,
					length: 1048576,
					rows: 2048,
					scaleBase: 34053
				},
				bias: {
					name: "model.encoder.layers.0.fc1.bias",
					shape: [2048],
					offset: 18491904,
					length: 2048,
					rows: 1,
					scaleBase: 36101
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.0.fc2.weight",
					shape: [512, 2048],
					offset: 18493952,
					length: 1048576,
					rows: 512,
					scaleBase: 36102
				},
				bias: {
					name: "model.encoder.layers.0.fc2.bias",
					shape: [512],
					offset: 19542528,
					length: 512,
					rows: 1,
					scaleBase: 36614
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.1.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 19543040,
						length: 262144,
						rows: 512,
						scaleBase: 36615
					},
					bias: {
						name: "model.encoder.layers.1.self_attn.q_proj.bias",
						shape: [512],
						offset: 19805184,
						length: 512,
						rows: 1,
						scaleBase: 37127
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.1.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 19805696,
						length: 262144,
						rows: 512,
						scaleBase: 37128
					},
					bias: {
						name: "model.encoder.layers.1.self_attn.k_proj.bias",
						shape: [512],
						offset: 20067840,
						length: 512,
						rows: 1,
						scaleBase: 37640
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.1.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 20068352,
						length: 262144,
						rows: 512,
						scaleBase: 37641
					},
					bias: {
						name: "model.encoder.layers.1.self_attn.v_proj.bias",
						shape: [512],
						offset: 20330496,
						length: 512,
						rows: 1,
						scaleBase: 38153
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.1.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 20331008,
						length: 262144,
						rows: 512,
						scaleBase: 38154
					},
					bias: {
						name: "model.encoder.layers.1.self_attn.out_proj.bias",
						shape: [512],
						offset: 20593152,
						length: 512,
						rows: 1,
						scaleBase: 38666
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.1.self_attn_layer_norm.weight",
					shape: [512],
					offset: 20593664,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.1.self_attn_layer_norm.bias",
					shape: [512],
					offset: 20595712,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.1.final_layer_norm.weight",
					shape: [512],
					offset: 20597760,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.1.final_layer_norm.bias",
					shape: [512],
					offset: 20599808,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.1.fc1.weight",
					shape: [2048, 512],
					offset: 20601856,
					length: 1048576,
					rows: 2048,
					scaleBase: 38667
				},
				bias: {
					name: "model.encoder.layers.1.fc1.bias",
					shape: [2048],
					offset: 21650432,
					length: 2048,
					rows: 1,
					scaleBase: 40715
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.1.fc2.weight",
					shape: [512, 2048],
					offset: 21652480,
					length: 1048576,
					rows: 512,
					scaleBase: 40716
				},
				bias: {
					name: "model.encoder.layers.1.fc2.bias",
					shape: [512],
					offset: 22701056,
					length: 512,
					rows: 1,
					scaleBase: 41228
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.2.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 22701568,
						length: 262144,
						rows: 512,
						scaleBase: 41229
					},
					bias: {
						name: "model.encoder.layers.2.self_attn.q_proj.bias",
						shape: [512],
						offset: 22963712,
						length: 512,
						rows: 1,
						scaleBase: 41741
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.2.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 22964224,
						length: 262144,
						rows: 512,
						scaleBase: 41742
					},
					bias: {
						name: "model.encoder.layers.2.self_attn.k_proj.bias",
						shape: [512],
						offset: 23226368,
						length: 512,
						rows: 1,
						scaleBase: 42254
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.2.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 23226880,
						length: 262144,
						rows: 512,
						scaleBase: 42255
					},
					bias: {
						name: "model.encoder.layers.2.self_attn.v_proj.bias",
						shape: [512],
						offset: 23489024,
						length: 512,
						rows: 1,
						scaleBase: 42767
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.2.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 23489536,
						length: 262144,
						rows: 512,
						scaleBase: 42768
					},
					bias: {
						name: "model.encoder.layers.2.self_attn.out_proj.bias",
						shape: [512],
						offset: 23751680,
						length: 512,
						rows: 1,
						scaleBase: 43280
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.2.self_attn_layer_norm.weight",
					shape: [512],
					offset: 23752192,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.2.self_attn_layer_norm.bias",
					shape: [512],
					offset: 23754240,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.2.final_layer_norm.weight",
					shape: [512],
					offset: 23756288,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.2.final_layer_norm.bias",
					shape: [512],
					offset: 23758336,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.2.fc1.weight",
					shape: [2048, 512],
					offset: 23760384,
					length: 1048576,
					rows: 2048,
					scaleBase: 43281
				},
				bias: {
					name: "model.encoder.layers.2.fc1.bias",
					shape: [2048],
					offset: 24808960,
					length: 2048,
					rows: 1,
					scaleBase: 45329
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.2.fc2.weight",
					shape: [512, 2048],
					offset: 24811008,
					length: 1048576,
					rows: 512,
					scaleBase: 45330
				},
				bias: {
					name: "model.encoder.layers.2.fc2.bias",
					shape: [512],
					offset: 25859584,
					length: 512,
					rows: 1,
					scaleBase: 45842
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.3.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 25860096,
						length: 262144,
						rows: 512,
						scaleBase: 45843
					},
					bias: {
						name: "model.encoder.layers.3.self_attn.q_proj.bias",
						shape: [512],
						offset: 26122240,
						length: 512,
						rows: 1,
						scaleBase: 46355
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.3.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 26122752,
						length: 262144,
						rows: 512,
						scaleBase: 46356
					},
					bias: {
						name: "model.encoder.layers.3.self_attn.k_proj.bias",
						shape: [512],
						offset: 26384896,
						length: 512,
						rows: 1,
						scaleBase: 46868
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.3.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 26385408,
						length: 262144,
						rows: 512,
						scaleBase: 46869
					},
					bias: {
						name: "model.encoder.layers.3.self_attn.v_proj.bias",
						shape: [512],
						offset: 26647552,
						length: 512,
						rows: 1,
						scaleBase: 47381
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.3.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 26648064,
						length: 262144,
						rows: 512,
						scaleBase: 47382
					},
					bias: {
						name: "model.encoder.layers.3.self_attn.out_proj.bias",
						shape: [512],
						offset: 26910208,
						length: 512,
						rows: 1,
						scaleBase: 47894
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.3.self_attn_layer_norm.weight",
					shape: [512],
					offset: 26910720,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.3.self_attn_layer_norm.bias",
					shape: [512],
					offset: 26912768,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.3.final_layer_norm.weight",
					shape: [512],
					offset: 26914816,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.3.final_layer_norm.bias",
					shape: [512],
					offset: 26916864,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.3.fc1.weight",
					shape: [2048, 512],
					offset: 26918912,
					length: 1048576,
					rows: 2048,
					scaleBase: 47895
				},
				bias: {
					name: "model.encoder.layers.3.fc1.bias",
					shape: [2048],
					offset: 27967488,
					length: 2048,
					rows: 1,
					scaleBase: 49943
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.3.fc2.weight",
					shape: [512, 2048],
					offset: 27969536,
					length: 1048576,
					rows: 512,
					scaleBase: 49944
				},
				bias: {
					name: "model.encoder.layers.3.fc2.bias",
					shape: [512],
					offset: 29018112,
					length: 512,
					rows: 1,
					scaleBase: 50456
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.4.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 29018624,
						length: 262144,
						rows: 512,
						scaleBase: 50457
					},
					bias: {
						name: "model.encoder.layers.4.self_attn.q_proj.bias",
						shape: [512],
						offset: 29280768,
						length: 512,
						rows: 1,
						scaleBase: 50969
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.4.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 29281280,
						length: 262144,
						rows: 512,
						scaleBase: 50970
					},
					bias: {
						name: "model.encoder.layers.4.self_attn.k_proj.bias",
						shape: [512],
						offset: 29543424,
						length: 512,
						rows: 1,
						scaleBase: 51482
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.4.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 29543936,
						length: 262144,
						rows: 512,
						scaleBase: 51483
					},
					bias: {
						name: "model.encoder.layers.4.self_attn.v_proj.bias",
						shape: [512],
						offset: 29806080,
						length: 512,
						rows: 1,
						scaleBase: 51995
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.4.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 29806592,
						length: 262144,
						rows: 512,
						scaleBase: 51996
					},
					bias: {
						name: "model.encoder.layers.4.self_attn.out_proj.bias",
						shape: [512],
						offset: 30068736,
						length: 512,
						rows: 1,
						scaleBase: 52508
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.4.self_attn_layer_norm.weight",
					shape: [512],
					offset: 30069248,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.4.self_attn_layer_norm.bias",
					shape: [512],
					offset: 30071296,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.4.final_layer_norm.weight",
					shape: [512],
					offset: 30073344,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.4.final_layer_norm.bias",
					shape: [512],
					offset: 30075392,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.4.fc1.weight",
					shape: [2048, 512],
					offset: 30077440,
					length: 1048576,
					rows: 2048,
					scaleBase: 52509
				},
				bias: {
					name: "model.encoder.layers.4.fc1.bias",
					shape: [2048],
					offset: 31126016,
					length: 2048,
					rows: 1,
					scaleBase: 54557
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.4.fc2.weight",
					shape: [512, 2048],
					offset: 31128064,
					length: 1048576,
					rows: 512,
					scaleBase: 54558
				},
				bias: {
					name: "model.encoder.layers.4.fc2.bias",
					shape: [512],
					offset: 32176640,
					length: 512,
					rows: 1,
					scaleBase: 55070
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.encoder.layers.5.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 32177152,
						length: 262144,
						rows: 512,
						scaleBase: 55071
					},
					bias: {
						name: "model.encoder.layers.5.self_attn.q_proj.bias",
						shape: [512],
						offset: 32439296,
						length: 512,
						rows: 1,
						scaleBase: 55583
					}
				},
				k: {
					weight: {
						name: "model.encoder.layers.5.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 32439808,
						length: 262144,
						rows: 512,
						scaleBase: 55584
					},
					bias: {
						name: "model.encoder.layers.5.self_attn.k_proj.bias",
						shape: [512],
						offset: 32701952,
						length: 512,
						rows: 1,
						scaleBase: 56096
					}
				},
				v: {
					weight: {
						name: "model.encoder.layers.5.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 32702464,
						length: 262144,
						rows: 512,
						scaleBase: 56097
					},
					bias: {
						name: "model.encoder.layers.5.self_attn.v_proj.bias",
						shape: [512],
						offset: 32964608,
						length: 512,
						rows: 1,
						scaleBase: 56609
					}
				},
				out: {
					weight: {
						name: "model.encoder.layers.5.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 32965120,
						length: 262144,
						rows: 512,
						scaleBase: 56610
					},
					bias: {
						name: "model.encoder.layers.5.self_attn.out_proj.bias",
						shape: [512],
						offset: 33227264,
						length: 512,
						rows: 1,
						scaleBase: 57122
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.encoder.layers.5.self_attn_layer_norm.weight",
					shape: [512],
					offset: 33227776,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.5.self_attn_layer_norm.bias",
					shape: [512],
					offset: 33229824,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.encoder.layers.5.final_layer_norm.weight",
					shape: [512],
					offset: 33231872,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.encoder.layers.5.final_layer_norm.bias",
					shape: [512],
					offset: 33233920,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.encoder.layers.5.fc1.weight",
					shape: [2048, 512],
					offset: 33235968,
					length: 1048576,
					rows: 2048,
					scaleBase: 57123
				},
				bias: {
					name: "model.encoder.layers.5.fc1.bias",
					shape: [2048],
					offset: 34284544,
					length: 2048,
					rows: 1,
					scaleBase: 59171
				}
			},
			fc2: {
				weight: {
					name: "model.encoder.layers.5.fc2.weight",
					shape: [512, 2048],
					offset: 34286592,
					length: 1048576,
					rows: 512,
					scaleBase: 59172
				},
				bias: {
					name: "model.encoder.layers.5.fc2.bias",
					shape: [512],
					offset: 35335168,
					length: 512,
					rows: 1,
					scaleBase: 59684
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
						offset: 35335680,
						length: 262144,
						rows: 512,
						scaleBase: 59685
					},
					bias: {
						name: "model.decoder.layers.0.self_attn.q_proj.bias",
						shape: [512],
						offset: 35597824,
						length: 512,
						rows: 1,
						scaleBase: 60197
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.0.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 35598336,
						length: 262144,
						rows: 512,
						scaleBase: 60198
					},
					bias: {
						name: "model.decoder.layers.0.self_attn.k_proj.bias",
						shape: [512],
						offset: 35860480,
						length: 512,
						rows: 1,
						scaleBase: 60710
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.0.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 35860992,
						length: 262144,
						rows: 512,
						scaleBase: 60711
					},
					bias: {
						name: "model.decoder.layers.0.self_attn.v_proj.bias",
						shape: [512],
						offset: 36123136,
						length: 512,
						rows: 1,
						scaleBase: 61223
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.0.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 36123648,
						length: 262144,
						rows: 512,
						scaleBase: 61224
					},
					bias: {
						name: "model.decoder.layers.0.self_attn.out_proj.bias",
						shape: [512],
						offset: 36385792,
						length: 512,
						rows: 1,
						scaleBase: 61736
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.0.self_attn_layer_norm.weight",
					shape: [512],
					offset: 36386304,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.0.self_attn_layer_norm.bias",
					shape: [512],
					offset: 36388352,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.0.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 36390400,
						length: 262144,
						rows: 512,
						scaleBase: 61737
					},
					bias: {
						name: "model.decoder.layers.0.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 36652544,
						length: 512,
						rows: 1,
						scaleBase: 62249
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.0.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 36653056,
						length: 262144,
						rows: 512,
						scaleBase: 62250
					},
					bias: {
						name: "model.decoder.layers.0.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 36915200,
						length: 512,
						rows: 1,
						scaleBase: 62762
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.0.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 36915712,
						length: 262144,
						rows: 512,
						scaleBase: 62763
					},
					bias: {
						name: "model.decoder.layers.0.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 37177856,
						length: 512,
						rows: 1,
						scaleBase: 63275
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.0.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 37178368,
						length: 262144,
						rows: 512,
						scaleBase: 63276
					},
					bias: {
						name: "model.decoder.layers.0.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 37440512,
						length: 512,
						rows: 1,
						scaleBase: 63788
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.0.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 37441024,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.0.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 37443072,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.0.final_layer_norm.weight",
					shape: [512],
					offset: 37445120,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.0.final_layer_norm.bias",
					shape: [512],
					offset: 37447168,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.0.fc1.weight",
					shape: [2048, 512],
					offset: 37449216,
					length: 1048576,
					rows: 2048,
					scaleBase: 63789
				},
				bias: {
					name: "model.decoder.layers.0.fc1.bias",
					shape: [2048],
					offset: 38497792,
					length: 2048,
					rows: 1,
					scaleBase: 65837
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.0.fc2.weight",
					shape: [512, 2048],
					offset: 38499840,
					length: 1048576,
					rows: 512,
					scaleBase: 65838
				},
				bias: {
					name: "model.decoder.layers.0.fc2.bias",
					shape: [512],
					offset: 39548416,
					length: 512,
					rows: 1,
					scaleBase: 66350
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.1.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 39548928,
						length: 262144,
						rows: 512,
						scaleBase: 66351
					},
					bias: {
						name: "model.decoder.layers.1.self_attn.q_proj.bias",
						shape: [512],
						offset: 39811072,
						length: 512,
						rows: 1,
						scaleBase: 66863
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.1.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 39811584,
						length: 262144,
						rows: 512,
						scaleBase: 66864
					},
					bias: {
						name: "model.decoder.layers.1.self_attn.k_proj.bias",
						shape: [512],
						offset: 40073728,
						length: 512,
						rows: 1,
						scaleBase: 67376
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.1.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 40074240,
						length: 262144,
						rows: 512,
						scaleBase: 67377
					},
					bias: {
						name: "model.decoder.layers.1.self_attn.v_proj.bias",
						shape: [512],
						offset: 40336384,
						length: 512,
						rows: 1,
						scaleBase: 67889
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.1.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 40336896,
						length: 262144,
						rows: 512,
						scaleBase: 67890
					},
					bias: {
						name: "model.decoder.layers.1.self_attn.out_proj.bias",
						shape: [512],
						offset: 40599040,
						length: 512,
						rows: 1,
						scaleBase: 68402
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.1.self_attn_layer_norm.weight",
					shape: [512],
					offset: 40599552,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.1.self_attn_layer_norm.bias",
					shape: [512],
					offset: 40601600,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.1.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 40603648,
						length: 262144,
						rows: 512,
						scaleBase: 68403
					},
					bias: {
						name: "model.decoder.layers.1.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 40865792,
						length: 512,
						rows: 1,
						scaleBase: 68915
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.1.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 40866304,
						length: 262144,
						rows: 512,
						scaleBase: 68916
					},
					bias: {
						name: "model.decoder.layers.1.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 41128448,
						length: 512,
						rows: 1,
						scaleBase: 69428
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.1.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 41128960,
						length: 262144,
						rows: 512,
						scaleBase: 69429
					},
					bias: {
						name: "model.decoder.layers.1.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 41391104,
						length: 512,
						rows: 1,
						scaleBase: 69941
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.1.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 41391616,
						length: 262144,
						rows: 512,
						scaleBase: 69942
					},
					bias: {
						name: "model.decoder.layers.1.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 41653760,
						length: 512,
						rows: 1,
						scaleBase: 70454
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.1.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 41654272,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.1.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 41656320,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.1.final_layer_norm.weight",
					shape: [512],
					offset: 41658368,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.1.final_layer_norm.bias",
					shape: [512],
					offset: 41660416,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.1.fc1.weight",
					shape: [2048, 512],
					offset: 41662464,
					length: 1048576,
					rows: 2048,
					scaleBase: 70455
				},
				bias: {
					name: "model.decoder.layers.1.fc1.bias",
					shape: [2048],
					offset: 42711040,
					length: 2048,
					rows: 1,
					scaleBase: 72503
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.1.fc2.weight",
					shape: [512, 2048],
					offset: 42713088,
					length: 1048576,
					rows: 512,
					scaleBase: 72504
				},
				bias: {
					name: "model.decoder.layers.1.fc2.bias",
					shape: [512],
					offset: 43761664,
					length: 512,
					rows: 1,
					scaleBase: 73016
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.2.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 43762176,
						length: 262144,
						rows: 512,
						scaleBase: 73017
					},
					bias: {
						name: "model.decoder.layers.2.self_attn.q_proj.bias",
						shape: [512],
						offset: 44024320,
						length: 512,
						rows: 1,
						scaleBase: 73529
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.2.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 44024832,
						length: 262144,
						rows: 512,
						scaleBase: 73530
					},
					bias: {
						name: "model.decoder.layers.2.self_attn.k_proj.bias",
						shape: [512],
						offset: 44286976,
						length: 512,
						rows: 1,
						scaleBase: 74042
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.2.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 44287488,
						length: 262144,
						rows: 512,
						scaleBase: 74043
					},
					bias: {
						name: "model.decoder.layers.2.self_attn.v_proj.bias",
						shape: [512],
						offset: 44549632,
						length: 512,
						rows: 1,
						scaleBase: 74555
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.2.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 44550144,
						length: 262144,
						rows: 512,
						scaleBase: 74556
					},
					bias: {
						name: "model.decoder.layers.2.self_attn.out_proj.bias",
						shape: [512],
						offset: 44812288,
						length: 512,
						rows: 1,
						scaleBase: 75068
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.2.self_attn_layer_norm.weight",
					shape: [512],
					offset: 44812800,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.2.self_attn_layer_norm.bias",
					shape: [512],
					offset: 44814848,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.2.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 44816896,
						length: 262144,
						rows: 512,
						scaleBase: 75069
					},
					bias: {
						name: "model.decoder.layers.2.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 45079040,
						length: 512,
						rows: 1,
						scaleBase: 75581
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.2.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 45079552,
						length: 262144,
						rows: 512,
						scaleBase: 75582
					},
					bias: {
						name: "model.decoder.layers.2.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 45341696,
						length: 512,
						rows: 1,
						scaleBase: 76094
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.2.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 45342208,
						length: 262144,
						rows: 512,
						scaleBase: 76095
					},
					bias: {
						name: "model.decoder.layers.2.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 45604352,
						length: 512,
						rows: 1,
						scaleBase: 76607
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.2.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 45604864,
						length: 262144,
						rows: 512,
						scaleBase: 76608
					},
					bias: {
						name: "model.decoder.layers.2.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 45867008,
						length: 512,
						rows: 1,
						scaleBase: 77120
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.2.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 45867520,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.2.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 45869568,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.2.final_layer_norm.weight",
					shape: [512],
					offset: 45871616,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.2.final_layer_norm.bias",
					shape: [512],
					offset: 45873664,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.2.fc1.weight",
					shape: [2048, 512],
					offset: 45875712,
					length: 1048576,
					rows: 2048,
					scaleBase: 77121
				},
				bias: {
					name: "model.decoder.layers.2.fc1.bias",
					shape: [2048],
					offset: 46924288,
					length: 2048,
					rows: 1,
					scaleBase: 79169
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.2.fc2.weight",
					shape: [512, 2048],
					offset: 46926336,
					length: 1048576,
					rows: 512,
					scaleBase: 79170
				},
				bias: {
					name: "model.decoder.layers.2.fc2.bias",
					shape: [512],
					offset: 47974912,
					length: 512,
					rows: 1,
					scaleBase: 79682
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.3.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 47975424,
						length: 262144,
						rows: 512,
						scaleBase: 79683
					},
					bias: {
						name: "model.decoder.layers.3.self_attn.q_proj.bias",
						shape: [512],
						offset: 48237568,
						length: 512,
						rows: 1,
						scaleBase: 80195
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.3.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 48238080,
						length: 262144,
						rows: 512,
						scaleBase: 80196
					},
					bias: {
						name: "model.decoder.layers.3.self_attn.k_proj.bias",
						shape: [512],
						offset: 48500224,
						length: 512,
						rows: 1,
						scaleBase: 80708
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.3.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 48500736,
						length: 262144,
						rows: 512,
						scaleBase: 80709
					},
					bias: {
						name: "model.decoder.layers.3.self_attn.v_proj.bias",
						shape: [512],
						offset: 48762880,
						length: 512,
						rows: 1,
						scaleBase: 81221
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.3.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 48763392,
						length: 262144,
						rows: 512,
						scaleBase: 81222
					},
					bias: {
						name: "model.decoder.layers.3.self_attn.out_proj.bias",
						shape: [512],
						offset: 49025536,
						length: 512,
						rows: 1,
						scaleBase: 81734
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.3.self_attn_layer_norm.weight",
					shape: [512],
					offset: 49026048,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.3.self_attn_layer_norm.bias",
					shape: [512],
					offset: 49028096,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.3.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 49030144,
						length: 262144,
						rows: 512,
						scaleBase: 81735
					},
					bias: {
						name: "model.decoder.layers.3.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 49292288,
						length: 512,
						rows: 1,
						scaleBase: 82247
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.3.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 49292800,
						length: 262144,
						rows: 512,
						scaleBase: 82248
					},
					bias: {
						name: "model.decoder.layers.3.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 49554944,
						length: 512,
						rows: 1,
						scaleBase: 82760
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.3.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 49555456,
						length: 262144,
						rows: 512,
						scaleBase: 82761
					},
					bias: {
						name: "model.decoder.layers.3.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 49817600,
						length: 512,
						rows: 1,
						scaleBase: 83273
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.3.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 49818112,
						length: 262144,
						rows: 512,
						scaleBase: 83274
					},
					bias: {
						name: "model.decoder.layers.3.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 50080256,
						length: 512,
						rows: 1,
						scaleBase: 83786
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.3.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 50080768,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.3.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 50082816,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.3.final_layer_norm.weight",
					shape: [512],
					offset: 50084864,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.3.final_layer_norm.bias",
					shape: [512],
					offset: 50086912,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.3.fc1.weight",
					shape: [2048, 512],
					offset: 50088960,
					length: 1048576,
					rows: 2048,
					scaleBase: 83787
				},
				bias: {
					name: "model.decoder.layers.3.fc1.bias",
					shape: [2048],
					offset: 51137536,
					length: 2048,
					rows: 1,
					scaleBase: 85835
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.3.fc2.weight",
					shape: [512, 2048],
					offset: 51139584,
					length: 1048576,
					rows: 512,
					scaleBase: 85836
				},
				bias: {
					name: "model.decoder.layers.3.fc2.bias",
					shape: [512],
					offset: 52188160,
					length: 512,
					rows: 1,
					scaleBase: 86348
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.4.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 52188672,
						length: 262144,
						rows: 512,
						scaleBase: 86349
					},
					bias: {
						name: "model.decoder.layers.4.self_attn.q_proj.bias",
						shape: [512],
						offset: 52450816,
						length: 512,
						rows: 1,
						scaleBase: 86861
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.4.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 52451328,
						length: 262144,
						rows: 512,
						scaleBase: 86862
					},
					bias: {
						name: "model.decoder.layers.4.self_attn.k_proj.bias",
						shape: [512],
						offset: 52713472,
						length: 512,
						rows: 1,
						scaleBase: 87374
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.4.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 52713984,
						length: 262144,
						rows: 512,
						scaleBase: 87375
					},
					bias: {
						name: "model.decoder.layers.4.self_attn.v_proj.bias",
						shape: [512],
						offset: 52976128,
						length: 512,
						rows: 1,
						scaleBase: 87887
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.4.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 52976640,
						length: 262144,
						rows: 512,
						scaleBase: 87888
					},
					bias: {
						name: "model.decoder.layers.4.self_attn.out_proj.bias",
						shape: [512],
						offset: 53238784,
						length: 512,
						rows: 1,
						scaleBase: 88400
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.4.self_attn_layer_norm.weight",
					shape: [512],
					offset: 53239296,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.4.self_attn_layer_norm.bias",
					shape: [512],
					offset: 53241344,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.4.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 53243392,
						length: 262144,
						rows: 512,
						scaleBase: 88401
					},
					bias: {
						name: "model.decoder.layers.4.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 53505536,
						length: 512,
						rows: 1,
						scaleBase: 88913
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.4.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 53506048,
						length: 262144,
						rows: 512,
						scaleBase: 88914
					},
					bias: {
						name: "model.decoder.layers.4.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 53768192,
						length: 512,
						rows: 1,
						scaleBase: 89426
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.4.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 53768704,
						length: 262144,
						rows: 512,
						scaleBase: 89427
					},
					bias: {
						name: "model.decoder.layers.4.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 54030848,
						length: 512,
						rows: 1,
						scaleBase: 89939
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.4.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 54031360,
						length: 262144,
						rows: 512,
						scaleBase: 89940
					},
					bias: {
						name: "model.decoder.layers.4.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 54293504,
						length: 512,
						rows: 1,
						scaleBase: 90452
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.4.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 54294016,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.4.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 54296064,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.4.final_layer_norm.weight",
					shape: [512],
					offset: 54298112,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.4.final_layer_norm.bias",
					shape: [512],
					offset: 54300160,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.4.fc1.weight",
					shape: [2048, 512],
					offset: 54302208,
					length: 1048576,
					rows: 2048,
					scaleBase: 90453
				},
				bias: {
					name: "model.decoder.layers.4.fc1.bias",
					shape: [2048],
					offset: 55350784,
					length: 2048,
					rows: 1,
					scaleBase: 92501
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.4.fc2.weight",
					shape: [512, 2048],
					offset: 55352832,
					length: 1048576,
					rows: 512,
					scaleBase: 92502
				},
				bias: {
					name: "model.decoder.layers.4.fc2.bias",
					shape: [512],
					offset: 56401408,
					length: 512,
					rows: 1,
					scaleBase: 93014
				}
			}
		},
		{
			selfAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.5.self_attn.q_proj.weight",
						shape: [512, 512],
						offset: 56401920,
						length: 262144,
						rows: 512,
						scaleBase: 93015
					},
					bias: {
						name: "model.decoder.layers.5.self_attn.q_proj.bias",
						shape: [512],
						offset: 56664064,
						length: 512,
						rows: 1,
						scaleBase: 93527
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.5.self_attn.k_proj.weight",
						shape: [512, 512],
						offset: 56664576,
						length: 262144,
						rows: 512,
						scaleBase: 93528
					},
					bias: {
						name: "model.decoder.layers.5.self_attn.k_proj.bias",
						shape: [512],
						offset: 56926720,
						length: 512,
						rows: 1,
						scaleBase: 94040
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.5.self_attn.v_proj.weight",
						shape: [512, 512],
						offset: 56927232,
						length: 262144,
						rows: 512,
						scaleBase: 94041
					},
					bias: {
						name: "model.decoder.layers.5.self_attn.v_proj.bias",
						shape: [512],
						offset: 57189376,
						length: 512,
						rows: 1,
						scaleBase: 94553
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.5.self_attn.out_proj.weight",
						shape: [512, 512],
						offset: 57189888,
						length: 262144,
						rows: 512,
						scaleBase: 94554
					},
					bias: {
						name: "model.decoder.layers.5.self_attn.out_proj.bias",
						shape: [512],
						offset: 57452032,
						length: 512,
						rows: 1,
						scaleBase: 95066
					}
				}
			},
			selfAttnNorm: {
				weight: {
					name: "model.decoder.layers.5.self_attn_layer_norm.weight",
					shape: [512],
					offset: 57452544,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.5.self_attn_layer_norm.bias",
					shape: [512],
					offset: 57454592,
					length: 2048,
					count: 512
				}
			},
			encoderAttn: {
				q: {
					weight: {
						name: "model.decoder.layers.5.encoder_attn.q_proj.weight",
						shape: [512, 512],
						offset: 57456640,
						length: 262144,
						rows: 512,
						scaleBase: 95067
					},
					bias: {
						name: "model.decoder.layers.5.encoder_attn.q_proj.bias",
						shape: [512],
						offset: 57718784,
						length: 512,
						rows: 1,
						scaleBase: 95579
					}
				},
				k: {
					weight: {
						name: "model.decoder.layers.5.encoder_attn.k_proj.weight",
						shape: [512, 512],
						offset: 57719296,
						length: 262144,
						rows: 512,
						scaleBase: 95580
					},
					bias: {
						name: "model.decoder.layers.5.encoder_attn.k_proj.bias",
						shape: [512],
						offset: 57981440,
						length: 512,
						rows: 1,
						scaleBase: 96092
					}
				},
				v: {
					weight: {
						name: "model.decoder.layers.5.encoder_attn.v_proj.weight",
						shape: [512, 512],
						offset: 57981952,
						length: 262144,
						rows: 512,
						scaleBase: 96093
					},
					bias: {
						name: "model.decoder.layers.5.encoder_attn.v_proj.bias",
						shape: [512],
						offset: 58244096,
						length: 512,
						rows: 1,
						scaleBase: 96605
					}
				},
				out: {
					weight: {
						name: "model.decoder.layers.5.encoder_attn.out_proj.weight",
						shape: [512, 512],
						offset: 58244608,
						length: 262144,
						rows: 512,
						scaleBase: 96606
					},
					bias: {
						name: "model.decoder.layers.5.encoder_attn.out_proj.bias",
						shape: [512],
						offset: 58506752,
						length: 512,
						rows: 1,
						scaleBase: 97118
					}
				}
			},
			encoderAttnNorm: {
				weight: {
					name: "model.decoder.layers.5.encoder_attn_layer_norm.weight",
					shape: [512],
					offset: 58507264,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.5.encoder_attn_layer_norm.bias",
					shape: [512],
					offset: 58509312,
					length: 2048,
					count: 512
				}
			},
			finalNorm: {
				weight: {
					name: "model.decoder.layers.5.final_layer_norm.weight",
					shape: [512],
					offset: 58511360,
					length: 2048,
					count: 512
				},
				bias: {
					name: "model.decoder.layers.5.final_layer_norm.bias",
					shape: [512],
					offset: 58513408,
					length: 2048,
					count: 512
				}
			},
			fc1: {
				weight: {
					name: "model.decoder.layers.5.fc1.weight",
					shape: [2048, 512],
					offset: 58515456,
					length: 1048576,
					rows: 2048,
					scaleBase: 97119
				},
				bias: {
					name: "model.decoder.layers.5.fc1.bias",
					shape: [2048],
					offset: 59564032,
					length: 2048,
					rows: 1,
					scaleBase: 99167
				}
			},
			fc2: {
				weight: {
					name: "model.decoder.layers.5.fc2.weight",
					shape: [512, 2048],
					offset: 59566080,
					length: 1048576,
					rows: 512,
					scaleBase: 99168
				},
				bias: {
					name: "model.decoder.layers.5.fc2.bias",
					shape: [512],
					offset: 60614656,
					length: 512,
					rows: 1,
					scaleBase: 99680
				}
			}
		}
	],
	finalLogitsBias: {
		name: "final_logits_bias",
		shape: [32001],
		offset: 61013892,
		length: 128004,
		count: 32001
	},
	rowScales: {
		name: "row_scales",
		shape: [99681],
		offset: 60615168,
		length: 398724,
		count: 99681
	},
	tokenizer: {
		offset: 61141896,
		length: 792751
	}
};
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
// src/bloxd/safe8.ts
/**
* safe8: three bits per character, using only '0'..'7'. Eight characters carry
* exactly three bytes, so encoded text contains no characters that can break
* Bloxd's block-data serialization.
*/
const CODES = [
	48,
	49,
	50,
	51,
	52,
	53,
	54,
	55
];
function encode(bytes) {
	const outLen = Math.ceil(bytes.length * 8 / 3);
	const out = new Array(outLen).fill(0);
	let idx = 0;
	let buffer = 0;
	let bits = 0;
	for (let i = 0; i < bytes.length; i++) {
		buffer = buffer << 8 | bytes[i] & 255;
		bits += 8;
		while (bits >= 3) {
			bits -= 3;
			out[idx++] = CODES[buffer >> bits & 7];
		}
		buffer &= (1 << bits) - 1;
	}
	if (bits > 0) out[idx++] = CODES[buffer << 3 - bits & 7];
	return String.fromCharCode.apply(null, out);
}
function decode(text) {
	const bytes = [];
	let buffer = 0;
	let bits = 0;
	for (let i = 0; i < text.length; i++) {
		const value = text.charCodeAt(i) - 48;
		if (value < 0 || value > 7) throw new Error(`Invalid safe8 character code: ${text.charCodeAt(i)}`);
		buffer = buffer << 3 | value;
		bits += 3;
		while (bits >= 8) {
			bits -= 8;
			bytes.push(buffer >> bits & 255);
		}
		buffer &= (1 << bits) - 1;
	}
	return bytes;
}
// src/bloxd/index.ts
/**
* Bloxd world-code entry point for ElanMT.
*
* The weights and the raw source.spm live in block data. Every forward step
* streams int8 rows through ByteCursor and charges the decoded byte count to
* the per-tick budget. Call think() to enqueue a translation; the normal tick
* hook drains the work queue.
*/
const cursor = new ByteCursor(CHUNK_SIZE, createWeightCoordFn());
const model = new BlockMarian(cursor, MODEL_CONFIG, MODEL_WEIGHTS);
globalThis.tick = () => {
	update$1();
	update();
};
function think(text, callbacks = {}, maxNewTokens = 32) {
	queueTask(() => {
		loadTokenizer(cursor, MODEL_WEIGHTS, MODEL_CONFIG.eosId, (tokenizer) => {
			translateBlock(model, tokenizer, text, {
				onToken: (tokens, decoded) => {
					callbacks.onToken?.(tokens, decoded);
					if (streamToPlayers) {
						const client = api.getPlayerIds?.();
						if (client !== void 0 && api.setClientOption !== void 0) for (let i = 0; i < client.length; i++) api.setClientOption(client[i], "middleTextLower", decoded);
					}
				},
				onComplete: (tokens, sourceTokens) => {
					const decoded = tokenizer.decode(tokens);
					callbacks.onComplete?.(tokens, sourceTokens.length, decoded);
					if (!callbacks.onComplete) console.log(decoded);
				}
			}, { maxNewTokens });
		});
	});
}
let streamToPlayers = true;
function setStreamToPlayers(enabled) {
	streamToPlayers = enabled;
}
function reset() {
	streamToPlayers = true;
}
const enqueueTranslation = think;
globalThis.think = ((text, callbacks, stream = true, maxNewTokens = 32) => {
	setStreamToPlayers(stream);
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
export { BlockMarian, ByteCursor, MODEL_CONFIG, MODEL_WEIGHTS, computeLogits, decode as decodeSafe8, encode as encodeSafe8, getTickBudget, linearInto, loadTokenizer, reset, setStreamToPlayers, setTickBudget, think, translateBlock, update };
