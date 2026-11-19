import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const WEIGHTS_PATH = new URL("../out/weights_int4.bin", import.meta.url);
const hasWeights = existsSync(WEIGHTS_PATH);

if (hasWeights) {
  const bytes = Array.from(readFileSync(WEIGHTS_PATH));
  const { translateInt4Bytes } = await import("../remote/bloxd-int4/int4.js");

  test(
    "in-memory int4 translation matches the int8 reference for a short sentence",
    { timeout: 120_000 },
    () => {
      let result;
      translateInt4Bytes(
        bytes,
        "こんにちは。私はAIです。",
        {
          onComplete: (tokens, promptLength, decoded) => {
            result = { tokens, promptLength, decoded };
          },
        },
        16,
      );
      assert.ok(result, "translation did not finish");
      assert.equal(result.decoded, "Hello. I'm an AI.");
      assert.deepEqual(result.tokens, [14696, 110, 7, 77, 27, 132, 56, 3, 4208, 7, 0]);

      const expected = [
        "How are you doing?",
        "It's a nice day, isn't it?",
        "Tokyo is the capital of Japan.",
        "This model translates Japanese into English.",
      ];
      const sources = [
        "お元気ですか？",
        "今日はいい天気ですね。",
        "東京は日本の首都です。",
        "このモデルは日本語を英語に翻訳します。",
      ];
      for (let i = 0; i < sources.length; i++) {
        let decoded;
        translateInt4Bytes(bytes, sources[i], {
          onComplete: (_tokens, _promptLength, text) => {
            decoded = text;
          },
        });
        assert.equal(decoded, expected[i], sources[i]);
      }
    },
  );
} else {
  test.skip("in-memory int4 translation (run npm run export:int4 first)");
}
