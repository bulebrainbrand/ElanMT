import { closeSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import vm from "node:vm";
import { parentPort, workerData } from "node:worker_threads";

const { bundlePath, textPath, prompt, maxNewTokens, chunkSize, weightOrigin } = workerData;

// The ESM bundle is self-contained (no imports). Drop its final export
// statement so the same code can run as a world-code script in node:vm.
const code = readFileSync(bundlePath, "utf8").replace(/^export \{[^}]*\};\s*$/m, "");
const textSize = statSync(textPath).size;
const textFd = openSync(textPath, "r");

/** Read one 16,000-character block. safe8 text is ASCII, so bytes == chars. */
function readChunk(index) {
  const start = index * chunkSize;
  if (start >= textSize) return "";
  const length = Math.min(chunkSize, textSize - start);
  const buffer = Buffer.allocUnsafe(length);
  readSync(textFd, buffer, 0, length, start);
  return buffer.toString("utf8");
}

const [, originY, originZ] = weightOrigin;
const context = vm.createContext({
  console,
  api: {
    isBlockInLoadedChunk: () => true,
    getBlock: () => "Code Block",
    setBlock: () => {},
    setBlockData: () => {},
    getPlayerIds: () => [],
    setClientOption: () => {},
    getBlockData: (x, y, z) => {
      if (y !== originY || z !== originZ) {
        return { persisted: { shared: { text: "" } } };
      }
      const spatial = Math.floor(x / 32);
      const slot = Math.floor((x % 32) / 4);
      return {
        persisted: {
          shared: { text: readChunk(spatial * 8 + slot) },
        },
      };
    },
  },
});

context.__prompt = prompt;
context.__maxNewTokens = maxNewTokens;
context.__decoded = "";
context.__callbacks = {
  onToken: (_tokens, decoded) => {
    context.__decoded = decoded;
  },
  onComplete: (tokens, promptLength) => {
    context.__result = {
      tokens: Array.from(tokens),
      promptLength,
      decoded: context.__decoded,
    };
  },
};

const started = Date.now();
vm.runInContext(code, context, { filename: bundlePath });
vm.runInContext("think(__prompt, __callbacks, false, __maxNewTokens)", context, {
  filename: "emulate-call.js",
});

let ticks = 0;
while (context.__result === undefined && ticks < 10000) {
  vm.runInContext("tick()", context, { filename: "emulate-tick.js" });
  ticks++;
}
const elapsedMs = Date.now() - started;

const memory = process.memoryUsage();
parentPort.postMessage({
  ticks,
  elapsedMs,
  result: context.__result ?? null,
  memory: {
    rssMb: Math.round(memory.rss / 1024 / 1024),
    heapUsedMb: Math.round(memory.heapUsed / 1024 / 1024),
    heapTotalMb: Math.round(memory.heapTotal / 1024 / 1024),
    externalMb: Math.round(memory.external / 1024 / 1024),
  },
});
closeSync(textFd);
