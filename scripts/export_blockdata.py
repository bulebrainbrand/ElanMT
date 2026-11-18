#!/usr/bin/env python3
"""Export ElanMT (Marian ja-en) to the Bloxd int8 block-data format.

The runtime reads one flat byte stream through ByteCursor. Each weight row has
its own symmetric int8 scale, which keeps the translation quality much closer
to the float checkpoint than a single scale per tensor.

Outputs:
  out/weights_int8.bin     int8 tensor data + float32 row scales + logits bias
                           + raw source.spm
  out/manifest.json        metadata for tooling and tests
  src/bloxd/manifest.ts    the same metadata as TypeScript constants

Usage:
  python scripts/export_blockdata.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
OUT_DIR = ROOT / "out"
MANIFEST_TS = ROOT / "src" / "bloxd" / "manifest.ts"


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
        payload = handle.read()

    tensors: dict[str, np.ndarray] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = spec["dtype"]
        shape = tuple(int(v) for v in spec["shape"])
        start, end = (int(v) for v in spec["data_offsets"])
        raw = payload[start:end]
        count = int(np.prod(shape, dtype=np.int64))
        if dtype == "F16":
            array = np.frombuffer(raw, dtype="<f2", count=count)
        elif dtype == "BF16":
            bits = np.frombuffer(raw, dtype="<u2", count=count).astype(np.uint32) << 16
            array = bits.view("<f4")
        elif dtype == "F32":
            array = np.frombuffer(raw, dtype="<f4", count=count)
        else:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        tensors[name] = np.asarray(array, dtype=np.float32).reshape(shape)
    return tensors


class Writer:
    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.offset = 0
        self.scales: list[np.ndarray] = []
        self.scale_count = 0
        self.tensors: list[dict[str, Any]] = []

    def write_float(self, name: str, array: np.ndarray) -> dict[str, Any]:
        values = np.asarray(array, dtype='<f4').reshape(-1)
        raw = values.tobytes(order="C")
        meta = {
            "name": name,
            "shape": [int(v) for v in np.shape(array)],
            "offset": self.offset,
            "length": len(raw),
            "count": int(values.size),
        }
        self.parts.append(raw)
        self.offset += len(raw)
        self.tensors.append(meta)
        return meta

    def quantize(self, name: str, array: np.ndarray) -> dict[str, Any]:
        values = np.asarray(array, dtype=np.float32)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim == 1:
            rows = 1
            flat = values.reshape(1, -1)
            shape = [int(v) for v in values.shape]
        else:
            rows = int(values.shape[0])
            flat = values.reshape(rows, -1)
            shape = [int(v) for v in values.shape]

        max_abs = np.max(np.abs(flat), axis=1)
        scales = np.where(max_abs == 0, np.float32(1.0), max_abs / np.float32(127.0))
        quantized = np.clip(np.rint(flat / scales[:, None]), -127, 127).astype(np.int8)
        raw = quantized.tobytes(order="C")

        meta = {
            "name": name,
            "shape": shape,
            "offset": self.offset,
            "length": len(raw),
            "rows": rows,
            "scaleBase": self.scale_count,
        }
        self.parts.append(raw)
        self.offset += len(raw)
        self.scales.append(scales.astype("<f4"))
        self.scale_count += rows
        self.tensors.append(meta)
        return meta


def linear(w: np.ndarray, b: np.ndarray, writer: Writer, prefix: str) -> dict[str, Any]:
    return {
        "weight": writer.quantize(f"{prefix}.weight", w),
        "bias": writer.quantize(f"{prefix}.bias", b),
    }


def attention(tensors: dict[str, np.ndarray], writer: Writer, prefix: str) -> dict[str, Any]:
    return {
        key: linear(
            tensors[f"{prefix}.{key}_proj.weight"],
            tensors[f"{prefix}.{key}_proj.bias"],
            writer,
            f"{prefix}.{key}_proj",
        )
        for key in ("q", "k", "v", "out")
    }


def layer_norm(tensors: dict[str, np.ndarray], writer: Writer, prefix: str) -> dict[str, Any]:
    return {
        "weight": writer.write_float(f"{prefix}.weight", tensors[f"{prefix}.weight"]),
        "bias": writer.write_float(f"{prefix}.bias", tensors[f"{prefix}.bias"]),
    }


def main() -> None:
    tensors = load_safetensors(MODEL_DIR / "model.safetensors")
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    spm = (MODEL_DIR / "source.spm").read_bytes()

    writer = Writer()
    shared = writer.quantize("model.shared.weight", tensors["model.shared.weight"])

    encoder_layers = []
    for i in range(config["encoder_layers"]):
        prefix = f"model.encoder.layers.{i}"
        encoder_layers.append(
            {
                "selfAttn": attention(tensors, writer, f"{prefix}.self_attn"),
                "selfAttnNorm": layer_norm(tensors, writer, f"{prefix}.self_attn_layer_norm"),
                "finalNorm": layer_norm(tensors, writer, f"{prefix}.final_layer_norm"),
                "fc1": linear(
                    tensors[f"{prefix}.fc1.weight"],
                    tensors[f"{prefix}.fc1.bias"],
                    writer,
                    f"{prefix}.fc1",
                ),
                "fc2": linear(
                    tensors[f"{prefix}.fc2.weight"],
                    tensors[f"{prefix}.fc2.bias"],
                    writer,
                    f"{prefix}.fc2",
                ),
            }
        )

    decoder_layers = []
    for i in range(config["decoder_layers"]):
        prefix = f"model.decoder.layers.{i}"
        decoder_layers.append(
            {
                "selfAttn": attention(tensors, writer, f"{prefix}.self_attn"),
                "selfAttnNorm": layer_norm(tensors, writer, f"{prefix}.self_attn_layer_norm"),
                "encoderAttn": attention(tensors, writer, f"{prefix}.encoder_attn"),
                "encoderAttnNorm": layer_norm(
                    tensors, writer, f"{prefix}.encoder_attn_layer_norm"
                ),
                "finalNorm": layer_norm(tensors, writer, f"{prefix}.final_layer_norm"),
                "fc1": linear(
                    tensors[f"{prefix}.fc1.weight"],
                    tensors[f"{prefix}.fc1.bias"],
                    writer,
                    f"{prefix}.fc1",
                ),
                "fc2": linear(
                    tensors[f"{prefix}.fc2.weight"],
                    tensors[f"{prefix}.fc2.bias"],
                    writer,
                    f"{prefix}.fc2",
                ),
            }
        )

    # Float32 row scales, then the logits bias, then the raw SentencePiece model.
    scale_offset = writer.offset
    scale_bytes = b"".join(writer.scales)
    writer.parts.append(scale_bytes)
    writer.offset += len(scale_bytes)

    bias = np.asarray(tensors["final_logits_bias"], dtype="<f4").reshape(-1)
    bias_offset = writer.offset
    bias_bytes = bias.tobytes(order="C")
    writer.parts.append(bias_bytes)
    writer.offset += len(bias_bytes)

    tokenizer_offset = writer.offset
    writer.parts.append(spm)
    writer.offset += len(spm)

    model_weights = {
        "shared": shared,
        "encoderLayers": encoder_layers,
        "decoderLayers": decoder_layers,
        "finalLogitsBias": {
            "name": "final_logits_bias",
            "shape": [int(bias.size)],
            "offset": bias_offset,
            "length": len(bias_bytes),
            "count": int(bias.size),
        },
        "rowScales": {
            "name": "row_scales",
            "shape": [writer.scale_count],
            "offset": scale_offset,
            "length": len(scale_bytes),
            "count": writer.scale_count,
        },
        "tokenizer": {"offset": tokenizer_offset, "length": len(spm)},
    }

    model_config = {
        "dModel": int(config["d_model"]),
        "encoderLayers": int(config["encoder_layers"]),
        "decoderLayers": int(config["decoder_layers"]),
        "encoderHeads": int(config["encoder_attention_heads"]),
        "decoderHeads": int(config["decoder_attention_heads"]),
        "encoderFfn": int(config["encoder_ffn_dim"]),
        "decoderFfn": int(config["decoder_ffn_dim"]),
        "vocabSize": int(config["decoder_vocab_size"]),
        "maxPositions": int(config["max_position_embeddings"]),
        "maxLength": int(config["max_length"]),
        "eosId": int(config["eos_token_id"]),
        "padId": int(config["pad_token_id"]),
        "decoderStartId": int(config["decoder_start_token_id"]),
        "scaleEmbedding": bool(config.get("scale_embedding", True)),
        "badWordIds": [[int(v) for v in row] for row in config.get("bad_words_ids", [])],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "weights_int8.bin").write_bytes(b"".join(writer.parts))
    manifest = {
        "format": "elanmt-marian-int8-row-v1",
        "source": "model/model.safetensors",
        "quantization": "symmetric-per-row-int8",
        "total_raw_bytes": writer.offset,
        "modelConfig": model_config,
        "modelWeights": model_weights,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    MANIFEST_TS.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_TS.write_text(
        "// Generated by scripts/export_blockdata.py. Do not edit by hand.\n"
        'import type { ModelConfig, ModelWeights } from "./types";\n\n'
        f"export const MODEL_CONFIG: ModelConfig = {json.dumps(model_config, indent=2)};\n\n"
        f"export const MODEL_WEIGHTS: ModelWeights = {json.dumps(model_weights, indent=2)};\n",
        encoding="utf-8",
    )

    print(f"wrote {OUT_DIR / 'weights_int8.bin'} ({writer.offset:,} bytes)")
    print(f"wrote {OUT_DIR / 'manifest.json'}")
    print(f"wrote {MANIFEST_TS}")


if __name__ == "__main__":
    main()
