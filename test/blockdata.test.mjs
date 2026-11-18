import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const TEXT_PATH = new URL("../out/weights_safe8.txt", import.meta.url);
const hasWeights = existsSync(TEXT_PATH);
const CHUNK_SIZE = 16000;

if (hasWeights) {
  const text = readFileSync(TEXT_PATH, "utf8");
  const weightOrigin = [0, 1, 32];

  globalThis.tick = () => {};
  globalThis.api = {
    isBlockInLoadedChunk: () => true,
    getBlock: () => "Code Block",
    setBlock: () => {},
    setBlockData: () => {},
    getBlockData: (x, y, z) => {
      if (y !== weightOrigin[1] || z !== weightOrigin[2]) {
        return { persisted: { shared: { text: "" } } };
      }
      const spatial = Math.floor(x / 32);
      const slot = Math.round((x % 32) / 4);
      const index = spatial * 8 + slot;
      return {
        persisted: {
          shared: {
            text: text.slice(index * CHUNK_SIZE, (index + 1) * CHUNK_SIZE),
          },
        },
      };
    },
  };

  const { think } = await import("../remote/bloxd/index.js");

  test("block-data Marian translation matches the reference tokens", { timeout: 300_000 }, () => {
    let result;
    think(
      "こんにちは。私はAIです。",
      {
        onComplete: (tokens, promptLength) => {
          result = { tokens, promptLength };
        },
      },
      16,
    );

    for (let tick = 0; tick < 5000 && result === undefined; tick++) {
      globalThis.tick();
    }
    assert.ok(result, "translation did not finish");
    assert.deepEqual(result.tokens, [14696, 110, 7, 77, 27, 132, 56, 3, 4208, 7, 0]);
  });
} else {
  test.skip("block-data translation (run npm run export:blockdata first)");
}
