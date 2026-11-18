import assert from "node:assert/strict";
import { test } from "node:test";

// The world bundle installs a tick hook when it is evaluated.
globalThis.tick = () => {};
const { encodeSafe8, decodeSafe8 } = await import("../remote/bloxd/index.js");

test("safe8 round-trips every byte value", () => {
  for (let length = 0; length <= 256; length++) {
    const source = Array.from({ length }, (_, index) => (index * 73 + length) & 0xff);
    const encoded = encodeSafe8(source);
    assert.match(encoded, /^[0-7]*$/);
    assert.deepEqual(decodeSafe8(encoded), source);
  }
});

test("safe8 rejects non-alphabet characters", () => {
  assert.throws(() => decodeSafe8("8"), /Invalid safe8/);
});
