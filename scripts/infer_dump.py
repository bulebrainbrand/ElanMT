#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ElanMT (MarianMT) inference with transformers.

For every test sentence this script saves only the output tokens and the final
translation:

  output_token_ids : generated ids without the decoder start token (may end with EOS)
  output_tokens    : the same ids converted to token strings
  translation      : decoded English text

Examples:
    python scripts/infer_dump.py
    python scripts/infer_dump.py --sentences my_tests.txt --num-beams 1
    python scripts/infer_dump.py --out scripts/out/run2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - environment dependent
    sys.exit(
        "transformers could not be imported. Install the inference dependencies "
        "first:\n"
        "  pip install 'transformers>=4.40,<5' 'huggingface-hub<1.0' "
        "sentencepiece sacremoses torch\n"
        f"Original error: {exc}"
    )


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "model"
DEFAULT_SENTENCES = ROOT / "scripts" / "test_sentences.txt"
DEFAULT_OUT = ROOT / "scripts" / "out" / "inference.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run transformers inference on Japanese test sentences and save the "
            "output token ids and translations as JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Local Hugging Face model directory")
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES,
                        help="Test sentences: one sentence per line (# starts a comment)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output JSON with per-sentence tokens and translations")
    parser.add_argument("--num-beams", type=int, default=None,
                        help="Beam size (default: model generation_config, usually 4)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum number of generated tokens")
    parser.add_argument("--max-source-tokens", type=int, default=512,
                        help="Truncate source sentences to this many tokens")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cpu', 'cuda' or 'cuda:0'")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"],
                        help="Model compute dtype (auto: float32 on CPU, float16 on CUDA)")
    return parser.parse_args(argv)


def load_sentences(path: Path) -> List[str]:
    if not path.is_file():
        sys.exit(f"Sentences file not found: {path}")
    sentences = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip().lstrip("\ufeff")
        if not text or text.startswith("#"):
            continue
        sentences.append(text)
    if not sentences:
        sys.exit(f"No test sentences found in {path}")
    return sentences


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return getattr(torch, name)


def infer_sentence(
    text: str,
    sentence_id: int,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    num_beams: int,
    max_new_tokens: int,
    max_source_tokens: int,
) -> Dict[str, Any]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_source_tokens,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    full_length = len(tokenizer(text, add_special_tokens=True)["input_ids"])

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )

    sequence = generated[0]
    # Drop decoder_start_token_id; the remaining ids may end with EOS.
    output_token_ids = sequence[1:].tolist()

    return {
        "id": sentence_id,
        "source": text,
        "source_truncated": full_length > max_source_tokens,
        "output_token_ids": output_token_ids,
        "output_tokens": tokenizer.convert_ids_to_tokens(output_token_ids),
        "translation": tokenizer.decode(sequence.tolist(), skip_special_tokens=True),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    model_dir = args.model_dir.resolve()
    sentences = load_sentences(args.sentences)
    if not model_dir.is_dir():
        sys.exit(f"Model directory not found: {model_dir}")

    device = resolve_device(args.device)
    compute_dtype = resolve_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), torch_dtype=compute_dtype)
    model.to(device)
    model.eval()

    configured_beams = getattr(model.generation_config, "num_beams", None) or 1
    num_beams = args.num_beams if args.num_beams is not None else configured_beams

    settings: Dict[str, Any] = {
        "device": str(device),
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "num_beams": num_beams,
        "max_new_tokens": args.max_new_tokens,
        "max_source_tokens": args.max_source_tokens,
    }

    print(
        f"model={display_path(model_dir)} device={device} dtype={settings['compute_dtype']} "
        f"beams={num_beams} sentences={len(sentences)}"
    )

    results: List[Dict[str, Any]] = []
    for index, text in enumerate(sentences, start=1):
        result = infer_sentence(
            text,
            index - 1,
            model,
            tokenizer,
            device,
            num_beams,
            args.max_new_tokens,
            args.max_source_tokens,
        )
        results.append(result)
        print(f"[{index}/{len(sentences)}] {text} -> {result['translation']}")
        if result["source_truncated"]:
            print(f"  warning: source truncated to {args.max_source_tokens} tokens")

    payload: Dict[str, Any] = {
        "created_at": utc_now(),
        "model_dir": display_path(model_dir),
        "settings": settings,
        "sentences": results,
    }
    write_json(args.out, payload)
    print(f"saved: {display_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
