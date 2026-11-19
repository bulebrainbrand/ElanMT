/// <reference types="@types/node" />
import { closeSync, mkdirSync, openSync, readFileSync, writeSync } from "node:fs";
import { dirname } from "node:path";

const INPUT_PATH = process.argv[2] ?? "./out/weights_int8.bin";
const OUTPUT_PATH = process.argv[3] ?? "./out/weights_safe8.txt";

const rawBytes = readFileSync(INPUT_PATH);
mkdirSync(dirname(OUTPUT_PATH), { recursive: true });
const fd = openSync(OUTPUT_PATH, "w");

const CHUNK = 16384;
let out: number[] = [];
let buffer = 0;
let bits = 0;
let written = 0;

const flush = (): void => {
  if (out.length === 0) return;
  const text = String.fromCharCode.apply(null, out);
  writeSync(fd, text, null, "ascii");
  written += out.length;
  out = [];
};

for (let i = 0; i < rawBytes.length; i++) {
  buffer = (buffer << 8) | rawBytes[i];
  bits += 8;
  while (bits >= 3) {
    bits -= 3;
    out.push(48 + ((buffer >> bits) & 0x07));
    if (out.length >= CHUNK) flush();
  }
  buffer &= (1 << bits) - 1;
}

if (bits > 0) {
  out.push(48 + ((buffer << (3 - bits)) & 0x07));
  if (out.length >= CHUNK) flush();
}
flush();
closeSync(fd);

console.log(
  `safe8 text written: ${OUTPUT_PATH} ` +
    `(${rawBytes.byteLength.toLocaleString()} raw bytes -> ` +
    `${written.toLocaleString()} characters)`,
);
