import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const TEXT = new URL("../out/weights_int4_safe8.txt", import.meta.url);
const SCRIPT = new URL("../scripts/vm_emulate.mjs", import.meta.url);

if (existsSync(TEXT)) {
  test(
    "node:vm e2e: int4 world code translates through blockdata under 200MiB",
    { timeout: 180_000 },
    () => {
      const stdout = execFileSync(process.execPath, [fileURLToPath(SCRIPT), "--int4", "--assert"], {
        cwd: fileURLToPath(new URL("..", import.meta.url)),
        encoding: "utf8",
        timeout: 150_000,
        env: { ...process.env, MEMORY_MB: "200" },
      });
      const result = JSON.parse(stdout);
      assert.equal(result.variant, "int4-memory");
      assert.equal(result.decoded, "Hello. I'm an AI.");
      assert.deepEqual(result.tokens, [14696, 110, 7, 77, 27, 132, 56, 3, 4208, 7, 0]);
      assert.deepEqual(result.second.tokens, result.tokens);
      assert.ok(result.memoryLimitMb <= 200);
      // The V8 old-gen cap is 200 MiB; RSS also includes the worker shell.
      assert.ok(result.memory.rssMb <= 210, `RSS ${result.memory.rssMb} MiB`);
    },
  );
} else {
  test.skip("node:vm e2e (run npm run export:int4 first)");
}
