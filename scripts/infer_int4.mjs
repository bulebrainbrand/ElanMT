#!/usr/bin/env node
import { existsSync, readFileSync } from "node:fs";
import { parseArgs } from "node:util";

import { translateInt4Bytes } from "../remote/bloxd-int4/int4.js";

const { values } = parseArgs({
  options: {
    text: { type: "string", default: "こんにちは。私はAIです。" },
    "max-new-tokens": { type: "string", default: "32" },
    weights: { type: "string", default: "./out/weights_int4.bin" },
  },
});

if (!existsSync(values.weights)) {
  console.error(`${values.weights} is missing. Run \`npm run export:int4\` first.`);
  process.exit(1);
}

const bytes = Array.from(readFileSync(values.weights));
const started = performance.now();
let result;
translateInt4Bytes(
  bytes,
  values.text,
  {
    onToken: (tokens, decoded) => {
      console.error(`\r${decoded}`);
    },
    onComplete: (tokens, promptLength, decoded) => {
      result = { tokens, promptLength, decoded };
    },
  },
  Number(values["max-new-tokens"]),
);
const elapsed = ((performance.now() - started) / 1000).toFixed(2);
console.log(JSON.stringify({ ...result, elapsedSeconds: Number(elapsed) }, null, 2));
