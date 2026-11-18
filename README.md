# ElanMT ja→en (MarianMT) TypeScript / Bloxd port

`model/` の MarianMT（d_model 512、encoder/decoder 各6層、8 heads、語彙 32001、
int8 重み）を TypeScript に移植したものです。2つの実行形態があります。

| 形態  | 入口                                           | 重みの読み方                           | 用途                                    |
| ----- | ---------------------------------------------- | -------------------------------------- | --------------------------------------- |
| Dense | `src/index.ts` → `remote/index.js`             | safetensors をメモリに展開             | Node / ブラウザでの検証、正解データ作成 |
| Bloxd | `src/bloxd/index.ts` → `remote/bloxd/index.js` | blockdata を `ByteCursor` でストリーム | bloxd.io の world code                  |

Dense 版は HF `MarianMTModel` とトークン単位で一致します。Bloxd 版は同じ計算を
int8 + 行ごとの scale で行い、tick ごとのバイト予算内で少しずつ進めます。

## セットアップ

```bash
npm install
```

Dense 版のビルドと実行:

```bash
npm run build:dense
node scripts/infer_ts.mjs --text "こんにちは。私はAIです。"
```

`scripts/infer_ts.mjs` は `model/{config.json,source.spm,model.safetensors}` を読み、
`remote/index.js` の `loadElanMT()` を使います。空文字・空白のみの入力は `""` を返します。

## Bloxd 用 blockdata の生成

```bash
npm run export:blockdata
```

次の3つを順に実行します。

1. `scripts/export_blockdata.py`
   - `model/model.safetensors` を読み、2次元の重みを行ごとの symmetric int8 に量子化。
   - LayerNorm の weight/bias は精度優先で float32 のまま。
   - `out/weights_int8.bin`（約 62MB）と `out/manifest.json`、
     `src/bloxd/manifest.ts` を出力。
2. `scripts/createSafe8Text.ts`
   - バイト列を 3 bit / 1 文字（`0`–`7`）の safe8 に変換。
   - `out/weights_safe8.txt`（約 158MB）を出力。
3. `scripts/createWeightSchematics.ts`
   - 16,000 文字ごとに Code Block へ格納し、x 軸方向に分割した
     `schematics/weights_*.bloxdschem`（259ファイル、約 317MB）を出力。

`out/` と `schematics/` はサイズが大きいので `.gitignore` 済みです。必要なときに再生成してください。
`model/source.spm` と `model/target.spm` は同一バイト列なので、blockdata には1つだけ格納しています。

## Bloxd world code

1. `schematics/weights_*.bloxdschem` をワールドに配置する。
   - 重みの原点は `[0, 1, 32]`（`src/bloxd/constants.ts` の `WEIGHT_ORIGIN`）。
   - 各 schematic の `globalPosition` を足した位置に置く。
2. `remote/bloxd/index.js` を world code として貼り付ける。
3. ワールドコードから次のように呼ぶ。

```js
think("こんにちは。私はAIです。", (tokens, promptLength) => {
  console.log("done", tokens, promptLength);
});

// 1 tick あたりのデコード量（既定 8 MiB）
setTickBudget(8 * 1024 * 1024);
```

`tick()` は `globalThis.tick` に登録され、`ByteCursor` の読み込みと作業キューを進めます。
`think()` の第2引数はコールバック関数のほか、次のオブジェクトも受け付けます。

```js
think(
  "今日はいい天気ですね。",
  {
    onToken: (tokens, text) => {},
    onComplete: (tokens, promptLength, text) => {},
  },
  false, // プレイヤーへのストリーミングを無効化
  32, // maxNewTokens
);
```

計算は greedy（beam search なし）です。Dense 版は `numBeams: 1` で同じ結果になります。

## 検証

```bash
npm test          # safe8 の往復 + blockdata で1文を翻訳
npm run emulate   # node:vm + Worker 200MiB で world code を実行
```

`npm run emulate` は `remote/bloxd/index.js` の export 行だけを外して `node:vm` の
context で評価し、`api.getBlockData()` を 16,000 文字ずつファイルから読んで返します。
165MB のテキストをメモリに置かず、`worker_threads` の
`resourceLimits.maxOldGenerationSizeMb = 200` でヒープを制限しています
（`node:vm` 単体にはヒープ上限の指定がないため）。

実行例:

```json
{
  "decoded": "Hello. I'm an AI.",
  "tokens": [14696, 110, 7, 77, 27, 132, 56, 3, 4208, 7, 0],
  "ticks": 57,
  "memoryLimitMb": 200,
  "memory": { "rssMb": 137, "heapUsedMb": 49, "heapTotalMb": 69 }
}
```

## 精度

Dense 版（beam=1 / beam=4）と Bloxd 版（greedy）で、次の出力を確認済みです。

- `こんにちは。私はAIです。` → `Hello. I'm an AI.`
- `お元気ですか？` → `How are you doing?`
- `今日はいい天気ですね。` → `It's a nice day, isn't it?`
- `東京は日本の首都です。` → `Tokyo is the capital of Japan.`
- `このモデルは日本語を英語に翻訳します。` → `This model translates Japanese into English.`

追加10文（長文、固有名詞、𠮷・髙、BLEU など）も Python の `MarianMTModel`
（greedy）とトークン列が一致しています。量子化で変わりやすいのは LayerNorm なので、
そこだけ float32 に残しています。

## 構成

```
src/
  index.ts            Dense 版 API（greedy / beam search）
  marian.ts           Marian encoder-decoder（dense、KV cache）
  sentencepiece.ts    SentencePiece protobuf / Darts-clone normalizer / unigram
  safetensors.ts      safetensors reader（F16/BF16 → F32）
  bloxd/
    index.ts          world code 入口（tick / think / setTickBudget）
    byteCursor.ts     safe8 を blockdata から読むカーソル
    blockdata.ts      api.getBlockData / setBlockData ラッパー
    loadChunk.ts      未ロードチャンク待ち
    eventLoop.ts      tick バイト予算とタスクキュー
    linear.ts         int8 行列積 + 量子化解除
    marian.ts         BlockMarian（encoder / decoder / cross-attention）
    generate.ts       greedy 生成
    tokenizerBlock.ts blockdata から source.spm を読む
    manifest.ts       自動生成されたテンソル位置
scripts/
  export_blockdata.py
  createSafe8Text.ts
  createWeightSchematics.ts
  vm_emulate.mjs      node:vm + 200MiB Worker
  vm_worker.mjs
test/
  safe8.test.mjs
  blockdata.test.mjs
```
