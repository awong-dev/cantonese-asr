#!/usr/bin/env python3
"""
Convert a fine-tuned Whisper model to CTranslate2 format for fast inference.

Usage:
    python convert_ct2.py --model ./whisper-large-v3-yue/final
    python convert_ct2.py --model ./whisper-large-v3-yue/final --quantization int8
    python convert_ct2.py --model ./whisper-large-v3-yue/final --output_dir ./my-ct2-model

Requirements:
    pip install ctranslate2
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Whisper model to CTranslate2 format"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to model directory (e.g. ./whisper-finetuned/final)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for CT2 model (default: {model}-ct2)",
    )
    parser.add_argument(
        "--quantization", type=str, default="float16",
        choices=["float16", "float32", "int8", "int8_float16", "int8_float32"],
        help="Quantization type (default: float16)",
    )
    parser.add_argument(
        "--copy_files", type=str, nargs="+",
        default=["tokenizer.json", "preprocessor_config.json"],
        help="Extra files to copy to output (default: tokenizer.json preprocessor_config.json)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_dir():
        print(f"Error: model directory not found: {model_path}")
        sys.exit(1)

    output_dir = args.output_dir or f"{model_path}-ct2"

    # Check that files to copy exist
    for f in args.copy_files:
        if not (model_path / f).exists():
            print(f"Warning: {f} not found in {model_path}, skipping copy")

    # Build command
    cmd = [
        "ct2-transformers-converter",
        "--model", str(model_path),
        "--output_dir", output_dir,
        "--quantization", args.quantization,
    ]

    # Only copy files that exist
    existing_copy_files = [f for f in args.copy_files if (model_path / f).exists()]
    if existing_copy_files:
        cmd.extend(["--copy_files"] + existing_copy_files)

    print(f"Converting {model_path} -> {output_dir}")
    print(f"  Quantization: {args.quantization}")
    print(f"  Copy files: {existing_copy_files}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nConversion failed (exit {result.returncode})")
        sys.exit(1)

    print(f"\nDone! CT2 model saved to {output_dir}")


if __name__ == "__main__":
    main()
