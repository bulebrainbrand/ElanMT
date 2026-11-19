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

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
OUT_DIR = ROOT / "out"
MANIFEST_TS = ROOT / "src" / "bloxd" / "manifest.ts"

# QLoRA NF4 codebook. Values are normalized to the per-group absmax.
NF4 = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)


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
    def __init__(self, bits: int = 8, group_size: int = 128) -> None:
        self.bits = bits
        self.group_size = group_size
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

    def quantize(self, name: str, array: np.ndarray, force_bits: int | None = None) -> dict[str, Any]:
        values = np.asarray(array, dtype=np.float32)
        bits = self.bits if force_bits is None else force_bits
        if bits == 4 and values.ndim >= 2:
            return self.quantize_nf4(name, values)
        return self.quantize_int8(name, values)

    def quantize_int8(self, name: str, array: np.ndarray) -> dict[str, Any]:
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
        cols = int(flat.shape[1])

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
            "cols": cols,
            "rowBytes": cols,
            "bits": 8,
            "groupSize": 0,
            "scaleBase": self.scale_count,
        }
        self.parts.append(raw)
        self.offset += len(raw)
        self.scales.append(scales.astype("<f4"))
        self.scale_count += rows
        self.tensors.append(meta)
        return meta

    def quantize_nf4(self, name: str, array: np.ndarray) -> dict[str, Any]:
        values = np.asarray(array, dtype=np.float32)
        rows = int(values.shape[0])
        cols = int(values.shape[1])
        groups = math.ceil(cols / self.group_size)
        padded_cols = groups * self.group_size
        shape = [int(v) for v in values.shape]

        padded = np.zeros((rows, padded_cols), dtype=np.float32)
        padded[:, :cols] = values.reshape(rows, cols)
        grouped = padded.reshape(rows, groups, self.group_size)

        max_abs = np.max(np.abs(grouped), axis=2)
        scales = np.where(max_abs == 0, np.float32(1.0), max_abs).astype(np.float32)
        normalized = grouped / scales[:, :, None]

        # Nearest NF4 level. A loop keeps the temporary memory small.
        indices = np.zeros(normalized.shape, dtype=np.uint8)
        best = np.full(normalized.shape, np.inf, dtype=np.float32)
        for level, value in enumerate(NF4):
            distance = np.abs(normalized - value)
            mask = distance < best
            indices[mask] = level
            best[mask] = distance[mask]

        flat_indices = indices.reshape(rows, padded_cols)
        packed = (flat_indices[:, 0::2] | (flat_indices[:, 1::2] << 4)).astype(np.uint8)
        row_bytes = padded_cols // 2
        raw = packed.tobytes(order="C")

        meta = {
            "name": name,
            "shape": shape,
            "offset": self.offset,
            "length": len(raw),
            "rows": rows,
            "cols": cols,
            "rowBytes": row_bytes,
            "bits": 4,
            "groupSize": self.group_size,
            "scaleBase": self.scale_count,
        }
        self.parts.append(raw)
        self.offset += len(raw)
        self.scales.append(scales.reshape(-1).astype("<f4"))
        self.scale_count += rows * groups
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--manifest-ts", type=Path)
    args = parser.parse_args()
    out_path = args.out or (OUT_DIR / ("weights_int4.bin" if args.bits == 4 else "weights_int8.bin"))
    manifest_json = args.manifest_json or (
        OUT_DIR / ("manifest_int4.json" if args.bits == 4 else "manifest.json")
    )
    manifest_ts = args.manifest_ts or (
        ROOT / "src" / "bloxd" / ("manifestInt4.ts" if args.bits == 4 else "manifest.ts")
    )

    tensors = load_safetensors(MODEL_DIR / "model.safetensors")
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    spm = (MODEL_DIR / "source.spm").read_bytes()

    writer = Writer(args.bits, args.group_size)
    # The tied embedding / logits matrix stays int8: it is the most
    # quality-sensitive tensor and is still only 16 MB.
    shared = writer.quantize(
        "model.shared.weight", tensors["model.shared.weight"], force_bits=8
    )

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
        "totalLength": writer.offset,
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_ts.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"".join(writer.parts))
    manifest = {
        "format": f"elanmt-marian-int{args.bits}-v1",
        "source": "model/model.safetensors",
        "quantization": (
            "nf4-group128-with-int8-embedding"
            if args.bits == 4
            else "symmetric-per-row-int8"
        ),
        "bits": args.bits,
        "groupSize": args.group_size if args.bits == 4 else 0,
        "total_raw_bytes": writer.offset,
        "modelConfig": model_config,
        "modelWeights": model_weights,
    }
    manifest_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_ts.write_text(
        "// Generated by scripts/export_blockdata.py. Do not edit by hand.\n"
        'import type { ModelConfig, ModelWeights } from "./types";\n\n'
        f"export const MODEL_CONFIG: ModelConfig = {json.dumps(model_config, indent=2)};\n\n"
        f"export const MODEL_WEIGHTS: ModelWeights = {json.dumps(model_weights, indent=2)};\n",
        encoding="utf-8",
    )

    print(f"wrote {out_path} ({writer.offset:,} bytes)")
    print(f"wrote {manifest_json}")
    print(f"wrote {manifest_ts}")


if __name__ == "__main__":
    main()
