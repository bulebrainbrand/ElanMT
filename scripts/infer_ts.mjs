#!/usr/bin/env node
/**
 * Node runner for the TypeScript ElanMT implementation.
 *
 *   node scripts/infer_ts.mjs
 *   node scripts/infer_ts.mjs --num-beams 1 --out scripts/out/inference_ts.json
 *
 * The browser bundle (remote/index.js) is imported directly; model files are
 * read from ./model so no HTTP server is needed for local checks.
 */

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { loadElanMT } from "../remote/index.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const { values } = parseArgs({
  options: {
    text: { type: "string" },
    "model-dir": { type: "string", default: resolve(ROOT, "model") },
    sentences: {
      type: "string",
      default: resolve(ROOT, "scripts/test_sentences.txt"),
    },
    out: { type: "string", default: resolve(ROOT, "scripts/out/inference_ts.json") },
    "num-beams": { type: "string" },
    "max-new-tokens": { type: "string", default: "128" },
    "max-source-tokens": { type: "string", default: "512" },
  },
});

function readSentences(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.replace(/^\uFEFF/, "").trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"));
}

const modelDir = resolve(values["model-dir"]);
const [configText, spm, weights] = await Promise.all([
  readFile(resolve(modelDir, "config.json"), "utf8"),
  readFile(resolve(modelDir, "source.spm")),
  readFile(resolve(modelDir, "model.safetensors")),
]);
const config = JSON.parse(configText);

const numBeams = values["num-beams"] === undefined ? undefined : Number(values["num-beams"]);
const maxNewTokens = Number(values["max-new-tokens"]);
const maxSourceTokens = Number(values["max-source-tokens"]);

const translator = await loadElanMT(
  { config, spm, weights },
  { numBeams, maxNewTokens, maxSourceTokens },
);

const sentences =
  values.text === undefined
    ? readSentences(await readFile(values.sentences, "utf8"))
    : [values.text];
const results = [];
const started = performance.now();

for (const [index, source] of sentences.entries()) {
  const result = translator.translate(source);
  results.push({
    id: index,
    source: result.source,
    source_truncated: result.sourceTruncated,
    output_token_ids: result.outputTokenIds,
    output_tokens: result.outputTokens,
    translation: result.translation,
  });
  console.log(`[${index + 1}/${sentences.length}] ${source} -> ${result.translation}`);
}

const elapsed = ((performance.now() - started) / 1000).toFixed(2);
const payload = {
  created_at: new Date().toISOString(),
  model_dir: "model",
  settings: {
    runtime: `node ${process.version}`,
    num_beams: numBeams ?? config.num_beams ?? 4,
    max_new_tokens: maxNewTokens,
    max_source_tokens: maxSourceTokens,
    elapsed_seconds: Number(elapsed),
  },
  sentences: results,
};

await mkdir(dirname(values.out), { recursive: true });
await writeFile(values.out, JSON.stringify(payload, null, 2) + "\n", "utf8");
console.log(`saved: ${values.out} (${elapsed}s)`);
