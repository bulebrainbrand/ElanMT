import { defineConfig } from "vite-plus";

export default defineConfig({
  pack: {
    outDir: "remote",
    format: "esm",
    fixedExtension: false,
    plugins: [
      {
        name: "strip-empty-export",
        generateBundle(_, bundle) {
          for (const file of Object.values(bundle)) {
            if (file.type !== "chunk") continue;
            file.code = file.code.replace(/^export \{\};?\s*$/m, "");
          }
        },
      },
    ],
  },
});
