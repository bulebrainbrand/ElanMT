import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";

const here = new URL(".", import.meta.url);
const bundlePath = fileURLToPath(new URL("../remote/bloxd/index.js", here));
const textPath = fileURLToPath(new URL("../out/weights_safe8.txt", here));

const prompt = process.argv[2] ?? "こんにちは。私はAIです。";
const maxNewTokens = Number(process.argv[3] ?? 32);
const memoryMb = 200;

if (!existsSync(textPath)) {
  console.error("out/weights_safe8.txt is missing. Run `npm run export:blockdata` first.");
  process.exit(1);
}

const worker = new Worker(new URL("./vm_worker.mjs", import.meta.url), {
  workerData: {
    bundlePath,
    textPath,
    prompt,
    maxNewTokens,
    chunkSize: 16000,
    weightOrigin: [0, 1, 32],
  },
  resourceLimits: {
    maxOldGenerationSizeMb: memoryMb,
    maxYoungGenerationSizeMb: 32,
    codeRangeSizeMb: 16,
    stackSizeMb: 8,
  },
});

const timer = setTimeout(
  () => {
    console.error("emulation timed out");
    worker.terminate();
    process.exit(1);
  },
  10 * 60 * 1000,
);

worker.on("message", (message) => {
  clearTimeout(timer);
  if (message.result === null) {
    console.error("translation did not finish", message);
    process.exitCode = 1;
    return;
  }
  console.log(
    JSON.stringify(
      {
        prompt,
        decoded: message.result.decoded,
        tokens: message.result.tokens,
        promptLength: message.result.promptLength,
        ticks: message.ticks,
        elapsedMs: message.elapsedMs,
        memoryLimitMb: memoryMb,
        memory: message.memory,
      },
      null,
      2,
    ),
  );
});

worker.on("error", (error) => {
  clearTimeout(timer);
  console.error("worker failed (possibly the 200 MiB heap limit):", error);
  process.exitCode = 1;
});

worker.on("exit", (code) => {
  clearTimeout(timer);
  if (code !== 0) process.exitCode = code;
});
