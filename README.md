# ElanMT ja→en (MarianMT) TypeScript / Bloxd port

`model/` の MarianMT（d_model 512、encoder/decoder 各6層、8 heads、語彙 32001）を
TypeScript に移植したものです。次の3つの実行形態があります。

| 形態       | 入口                                              | 重み                                       | 用途                                    |
| ---------- | ------------------------------------------------- | ------------------------------------------ | --------------------------------------- |
| Dense      | `src/index.ts` → `remote/index.js`                | safetensors を float で展開                | Node / ブラウザでの検証、正解データ作成 |
| Bloxd int8 | `src/bloxd/index.ts` → `remote/bloxd/index.js`    | blockdata を毎回 `ByteCursor` でストリーム | 省メモリ・低速                          |
| Bloxd int4 | `src/bloxd/int4.ts` → `remote/bloxd-int4/int4.js` | 41MB の NF4 を起動時に全読み込み           | 省メモリ・高速（推奨）                  |

Dense 版は HF `MarianMTModel` とトークン単位で一致します。int8 / int4 版は同じ計算を
量子化重みで行い、tick ごとのバイト予算内で少しずつ進めます。

**Bloxd runtime は TypedArray / ArrayBuffer / DataView / Buffer を使いません。**
重みは `number[]` に 32bit word（4バイト）単位でパックし、41MB の重みを約10.25M要素で
保持します。トークナイザも `number[]` と手書きの float32 デコードで動きます。
（Dense 版は Node/ブラウザ専用なので `Float32Array` を使っています。Bloxd bundle には
含まれません。）

## セットアップ

```bash
npm install
```

Dense 版:

```bash
npm run build:dense
node scripts/infer_ts.mjs --text "こんにちは。私はAIです。"
```

## int4（NF4）版

`model.shared.weight`（埋め込み・logits、16.4MB）だけ int8 に残し、それ以外の
2次元重みを **NF4（4bit, group size 128）** に量子化します。LayerNorm と
final logits bias は float32、線形層の bias は int8 です。

- `out/weights_int4.bin` … 41,020,471 bytes（約41MB）
- `out/weights_int4_safe8.txt` … 109,387,923 文字（約105MB）
- `schematics_int4/int4_*.bloxdschem` … 171ファイル（約210MB）

生成:

```bash
npm run export:int4
```

内訳:

1. `scripts/export_blockdata.py --bits 4`
   - NF4 の group scale と 4bit index を生成し、2 index を1バイトにパック。
   - `out/weights_int4.bin`、`out/manifest_int4.json`、`src/bloxd/manifestInt4.ts` を出力。
2. `scripts/createSafe8Text.ts out/weights_int4.bin out/weights_int4_safe8.txt`
3. `scripts/createWeightSchematics.ts out/weights_int4_safe8.txt schematics_int4 int4 elanmt-int4-weights`

Node で直接実行:

```bash
npm run build:world:int4
node scripts/infer_int4.mjs --text "今日はいい天気ですね。"
```

`scripts/infer_int4.mjs` は `out/weights_int4.bin` を読み、
`remote/bloxd-int4/int4.js` の `translateInt4Bytes()` を呼びます。最初の1回だけ
41MB を読み込み、同じプロセス内ではモデルを再利用できます。

## Bloxd world code

### int4（推奨）

1. `schematics_int4/int4_*.bloxdschem` をワールドに配置する。
   - 原点は `[0, 1, 64]`（`src/bloxd/int4.ts` の `INT4_ORIGIN`）。
   - 各 schematic の `globalPosition` を足した位置に置く。
2. `remote/bloxd-int4/int4.js` を world code として貼り付ける。
3. ワールドコードから呼ぶ。

```js
think("こんにちは。私はAIです。", (tokens, promptLength) => {
  console.log("done", tokens, promptLength);
});
```

最初の `think()` で 41MB を blockdata から読み、`number[]` の 32bit word に詰めて
メモリに保持します。以降はメモリ上の重みだけで推論します。int4 版の既定 tick 予算は
64 MiB です。

```js
setTickBudget(16 * 1024 * 1024); // 必要なら変更
```

### int8（ストリーム）

`schematics/weights_*.bloxdschem` を原点 `[0, 1, 32]` に置き、
`remote/bloxd/index.js` を貼り付けます。重みは毎回 blockdata から読むため、
int4 版より遅いですが、起動時の 41MB 読み込みがありません。

`think()` の第2引数はコールバック関数のほか、オブジェクトも受け付けます。

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
npm test             # safe8 / int8 blockdata / int4 in-memory / node:vm e2e
npm run test:vm      # int4 の node:vm e2e だけを実行
npm run emulate      # int8 を node:vm + Worker 200MiB で実行
npm run emulate:int4 # int4 を node:vm + Worker 200MiB で実行
```

`npm test` には `test/vm.test.mjs` が含まれます。これは `scripts/vm_emulate.mjs --int4 --assert`
を起動し、blockdata からの読み込み、翻訳結果のトークン列、2回目（重みキャッシュ）の一致、
world bundle に TypedArray / ArrayBuffer / DataView が含まれないことを確認します。

`scripts/vm_emulate.mjs` は `remote/bloxd*/index.js` の export 行だけを外して
`node:vm` の context で評価し、`api.getBlockData()` を 16,000 文字ずつファイルから
返します。テキスト全体をメモリに置かず、`worker_threads` の
`resourceLimits.maxOldGenerationSizeMb = 200`（young 16MiB）で V8 ヒープを制限して
います（`node:vm` 単体にはヒープ上限の指定がないため）。

int4 の実行例（短文、200MiB 制限）:

```json
{
  "variant": "int4-memory",
  "decoded": "Hello. I'm an AI.",
  "tokens": [14696, 110, 7, 77, 27, 132, 56, 3, 4208, 7, 0],
  "ticks": 6,
  "elapsedMs": 2481,
  "second": { "ticks": 5, "elapsedMs": 1628 },
  "memoryLimitMb": 200,
  "memory": { "rssMb": 190, "heapUsedMb": 123, "heapTotalMb": 138 }
}
```

`second` は同じモデルを再利用した2回目です。重みの再読み込みがないことを確認できます。
int8 版は同じ文で 57 ticks / 約8.2秒、int4 版は 6 ticks / 約2.5秒です。
512トークン級の長文は計算量が大きいため、実運用では短い文に分けるのが現実的です。

## 精度

Dense 版（beam=1 / beam=4）と int8 版（greedy）で、次の5文は完全一致します。

- `こんにちは。私はAIです。` → `Hello. I'm an AI.`
- `お元気ですか？` → `How are you doing?`
- `今日はいい天気ですね。` → `It's a nice day, isn't it?`
- `東京は日本の首都です。` → `Tokyo is the capital of Japan.`
- `このモデルは日本語を英語に翻訳します。` → `This model translates Japanese into English.`

int4 版もこの5文は完全一致します。追加10文では11/15文がベースと完全一致し、
残り4文は意味が同じ言い換えです（例: `very popular` → `very delicious`、
`Olympics` → `Olympic Games`）。LayerNorm を float32 に残すことで、
int8 版は追加10文も HF とトークン単位で一致しています。

## 構成

```
src/
  index.ts            Dense 版 API（greedy / beam search）
  marian.ts           Marian encoder-decoder（dense、KV cache）
  sentencepiece.ts    SentencePiece protobuf / Darts-clone normalizer / unigram（number[]）
  safetensors.ts      safetensors reader（dense）
  bloxd/
    index.ts          int8 world code 入口（ストリーム）
    int4.ts           int4 world code 入口（起動時メモリロード）
    memory.ts         重みの一括ロードと in-memory 推論
    memoryCursor.ts   32bit word の number[] による重みカーソル
    byteCursor.ts     safe8 を blockdata から読むカーソル（number[]）
    blockdata.ts      api.getBlockData / setBlockData ラッパー
    loadChunk.ts      未ロードチャンク待ち
    eventLoop.ts      tick バイト予算とタスクキュー
    linear.ts         int8 / NF4 行列積 + 量子化解除
    marian.ts         BlockMarian（encoder / decoder / cross-attention）
    generate.ts       greedy 生成
    tokenizerBlock.ts blockdata から source.spm を読む
    manifest.ts       int8 用テンソル位置（自動生成）
    manifestInt4.ts   int4 用テンソル位置（自動生成）
scripts/
  export_blockdata.py
  createSafe8Text.ts
  createWeightSchematics.ts
  infer_int4.mjs
  vm_emulate.mjs      node:vm + 200MiB Worker
  vm_worker.mjs
test/
  safe8.test.mjs
  blockdata.test.mjs
  int4.test.mjs
```
