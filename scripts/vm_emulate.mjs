import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";

const here = new URL(".", import.meta.url);
const int4 = process.argv.includes("--int4");
const positional = process.argv.slice(2).filter((value) => !value.startsWith("--"));
const bundlePath = fileURLToPath(
  new URL(int4 ? "../remote/bloxd-int4/int4.js" : "../remote/bloxd/index.js", here),
);
const textPath = fileURLToPath(
  new URL(int4 ? "../out/weights_int4_safe8.txt" : "../out/weights_safe8.txt", here),
);

const defaultPrompt = "こんにちは。私はAIです。";
const prompt = positional[0] ?? defaultPrompt;
const maxNewTokens = Number(positional[1] ?? 32);
const memoryMb = Number(process.env.MEMORY_MB ?? 200);
const assert = process.argv.includes("--assert");
const expectedText = "Hello. I'm an AI.";
const expectedTokens = [14696, 110, 7, 77, 27, 132, 56, 3, 4208, 7, 0];

if (!existsSync(textPath)) {
  console.error(
    `${textPath} is missing. Run \`npm run export:blockdata\` or \`npm run export:int4\` first.`,
  );
  process.exit(1);
}

// The world bundle must not depend on TypedArray/ArrayBuffer/DataView.
const bundleSource = readFileSync(bundlePath, "utf8");
const forbidden = bundleSource.match(
  /(Uint8Array|Int8Array|Float32Array|Float64Array|Int32Array|DataView|ArrayBuffer)/,
);
if (forbidden) {
  console.error(`forbidden constructor in world bundle: ${forbidden[1]}`);
  process.exit(1);
}

const worker = new Worker(new URL("./vm_worker.mjs", import.meta.url), {
  workerData: {
    bundlePath,
    textPath,
    prompt,
    maxNewTokens,
    chunkSize: 16000,
    weightOrigin: int4 ? [0, 1, 64] : [0, 1, 32],
  },
  resourceLimits: {
    maxOldGenerationSizeMb: memoryMb,
    maxYoungGenerationSizeMb: 16,
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
  const output = {
    variant: int4 ? "int4-memory" : "int8-stream",
    prompt,
    decoded: message.result.decoded,
    tokens: message.result.tokens,
    promptLength: message.result.promptLength,
    ticks: message.ticks,
    elapsedMs: message.elapsedMs,
    second: {
      ticks: message.second.ticks,
      elapsedMs: message.second.elapsedMs,
      decoded: message.second.result?.decoded,
      tokens: message.second.result?.tokens,
    },
    memoryLimitMb: memoryMb,
    memory: message.memory,
  };
  console.log(JSON.stringify(output, null, 2));

  if (assert) {
    if (prompt !== defaultPrompt) {
      console.error("--assert only knows the default test prompt");
      process.exitCode = 1;
    } else if (output.decoded !== expectedText) {
      console.error(`text mismatch: got ${JSON.stringify(output.decoded)}`);
      process.exitCode = 1;
    } else if (JSON.stringify(output.tokens) !== JSON.stringify(expectedTokens)) {
      console.error(`token mismatch: ${JSON.stringify(output.tokens)}`);
      process.exitCode = 1;
    } else if (JSON.stringify(output.second.tokens) !== JSON.stringify(expectedTokens)) {
      console.error("second run (cached weights) returned different tokens");
      process.exitCode = 1;
    }
  }
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
