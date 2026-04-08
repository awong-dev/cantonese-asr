#!/usr/bin/env python3
"""
Evaluate a fine-tuned wav2vec2 model on test and/or holdback splits.

Usage:
    # Explicit TSV
    python eval_wav2vec2.py --model ./wav2vec2-yue/final \
        --dataset_path cv-corpus/yue --test_tsv test_split.tsv

    # Regenerate splits from seed
    python eval_wav2vec2.py --model ./wav2vec2-yue/final \
        --dataset_path cv-corpus/yue --all_tsv validated.tsv \
        --holdback_tsv test.tsv --pct_test 0.05 --seed 42 \
        --eval_test --eval_holdback

Requirements:
    pip install transformers datasets evaluate jiwer torchcodec
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchcodec.decoders import AudioDecoder
from datasets import load_dataset, concatenate_datasets
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from cer_utils import build_cer_transform, compute_cer, print_examples

warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

# Text cleaning (must match training preprocessing)
import jiwer
import string

_text_normalize = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def clean_text(text: str) -> str:
    """Normalize transcription text (must match train_wav2vec2.py)."""
    text = _text_normalize(text)
    if "d" in text:
        ascii_letters = [c for c in text if c in string.ascii_lowercase]
        if len(ascii_letters) == 1 and ascii_letters[0] == "d":
            text = text.replace("d", "啲")
    return text + " "


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------
def run_evaluation(
    model_path,
    dataset_path,
    test_tsv=None,
    holdback_tsv=None,
    all_tsv=None,
    pct_validation=0.1,
    pct_test=0.1,
    seed=None,
    eval_test=False,
    eval_holdback=False,
    eval_batch_size=64,
    dataloader_num_workers=4,
):
    """Evaluate a wav2vec2 model on test and/or holdback splits."""

    # ---- Resolve datasets ----
    test_datasets = []
    holdback_datasets = []

    if seed is not None and all_tsv is not None:
        from create_splits import create_splits
        splits = create_splits(
            dataset_paths=dataset_path,
            all_tsvs=all_tsv,
            holdback_tsvs=holdback_tsv or "",
            pct_validation=pct_validation,
            pct_test=pct_test,
            seed=seed,
        )
        if eval_test:
            for ts, name in zip(splits["test_splits"], splits["fileset_names"]):
                test_datasets.append((f"test_{name}", ts))
        if eval_holdback:
            for hb, name in zip(splits["holdback_splits"], splits["fileset_names"]):
                if hb is not None:
                    holdback_datasets.append((f"holdback_{name}", hb))
    else:
        ds_paths = [p.strip() for p in dataset_path.split(",")] \
            if isinstance(dataset_path, str) else dataset_path
        primary_path = Path(ds_paths[0])

        if test_tsv:
            tsvs = [t.strip() for t in test_tsv.split(",")]
            for j, tsv in enumerate(tsvs):
                p = Path(tsv)
                if not p.is_absolute():
                    dp = Path(ds_paths[j]) if j < len(ds_paths) else primary_path
                    p = dp / p
                assert p.exists(), f"Test TSV not found: {p}"
                ds = load_dataset("csv", data_files=str(p), delimiter="\t", split="train")
                label = Path(ds_paths[j]).name if j < len(ds_paths) else f"set{j}"
                test_datasets.append((f"test_{label}", ds))

        if holdback_tsv and eval_holdback:
            tsvs = [t.strip() for t in holdback_tsv.split(",")]
            for j, tsv in enumerate(tsvs):
                p = Path(tsv)
                if not p.is_absolute():
                    dp = Path(ds_paths[j]) if j < len(ds_paths) else primary_path
                    p = dp / p
                assert p.exists(), f"Holdback TSV not found: {p}"
                ds = load_dataset("csv", data_files=str(p), delimiter="\t", split="train")
                label = Path(ds_paths[j]).name if j < len(ds_paths) else f"set{j}"
                holdback_datasets.append((f"holdback_{label}", ds))

    all_eval = test_datasets + holdback_datasets
    assert len(all_eval) > 0, "No datasets to evaluate."

    # ---- Load model + processor ----
    print(f"Loading model from: {model_path}")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_bf16 = torch.cuda.is_available()
    print(f"Device: {device}" + (", bf16 autocast" if use_bf16 else ""))

    # ---- Audio loading ----
    ds_paths = [p.strip() for p in dataset_path.split(",")] \
        if isinstance(dataset_path, str) else dataset_path
    default_clips_dir = str(Path(ds_paths[0]) / "clips")

    def _load_audio(clips_dir, path):
        filepath = os.path.join(clips_dir, path)
        decoder = AudioDecoder(filepath, sample_rate=16000, num_channels=1)
        samples = decoder.get_all_samples()
        return samples.data.squeeze().numpy()

    def prepare_fn_batched(batch):
        clips_dirs = batch.get("clips_dir", [default_clips_dir] * len(batch["path"]))
        audio_arrays = [_load_audio(cd, p) for cd, p in zip(clips_dirs, batch["path"])]
        batch["input_values"] = processor(
            audio_arrays, sampling_rate=16000, padding=False,
        ).input_values
        batch["labels"] = [
            processor.tokenizer(s).input_ids for s in batch["sentence"]
        ]
        return batch

    def preprocess(ds, name):
        """Clean text, normalize columns, extract features."""
        text_col = "sentence" if "sentence" in ds.column_names else "text"
        keep = {"path", text_col}
        if "clips_dir" in ds.column_names:
            keep.add("clips_dir")
        else:
            ds = ds.map(lambda x: {"clips_dir": default_clips_dir}, batched=False)
            keep.add("clips_dir")
        cols_to_remove = [c for c in ds.column_names if c not in keep]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)
        if text_col != "sentence":
            ds = ds.rename_column(text_col, "sentence")

        # Apply text cleaning (must match training)
        ds = ds.map(lambda batch: {"sentence": clean_text(batch["sentence"])})

        remove_cols = ds.column_names
        print(f"Preprocessing {name} ({len(ds)} samples)...")
        ds = ds.map(prepare_fn_batched, remove_columns=remove_cols,
                     batched=True, batch_size=16)
        print(f"  Ready: {len(ds)} samples")
        return ds

    # ---- Preprocess ----
    processed = [(name, preprocess(ds, name)) for name, ds in all_eval]

    # ---- Evaluation loop ----
    _cer_transform = build_cer_transform()
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    def evaluate_split(dataset, split_name):
        """Run forward pass per batch, decode immediately, discard logits."""
        dataloader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            collate_fn=data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
            shuffle=False,
        )

        all_pred_str = []
        all_label_str = []
        num_batches = (len(dataset) + eval_batch_size - 1) // eval_batch_size

        for batch_idx, batch in enumerate(dataloader):
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"]  # keep on CPU

            with torch.no_grad():
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(
                            input_values,
                            attention_mask=attention_mask,
                        ).logits
                else:
                    logits = model(
                        input_values,
                        attention_mask=attention_mask,
                    ).logits

            # Argmax and decode immediately — discard logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            del logits

            label_ids = labels.numpy().copy()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

            all_pred_str.extend(processor.batch_decode(pred_ids))
            all_label_str.extend(
                processor.batch_decode(label_ids, group_tokens=False)
            )

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"  [{split_name}] batch {batch_idx + 1}/{num_batches}")

        # Compute CER on collected strings
        result, filtered_preds, filtered_refs = compute_cer(
            all_pred_str, all_label_str, cer_transform=_cer_transform
        )
        print_examples(filtered_preds, filtered_refs)
        return result

    # ---- Run evaluations ----
    results = {}
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for split_name, ds in processed:
        print(f"\nEvaluating on {split_name} ({len(ds)} samples)...")
        try:
            metrics = evaluate_split(ds, split_name)
            cer_raw = metrics.get("cer_raw", None)
            cer_nopunct = metrics.get("cer_nopunct", None)
            if cer_raw is not None:
                print(f"  {split_name} CER (raw):     {cer_raw:.4f}")
            if cer_nopunct is not None:
                print(f"  {split_name} CER (nopunct): {cer_nopunct:.4f}")
            # Store with prefixed keys for consistent summary format
            results[split_name] = {
                f"{split_name}_cer_raw": cer_raw,
            }
            if cer_nopunct is not None:
                results[split_name][f"{split_name}_cer_nopunct"] = cer_nopunct
        except Exception as e:
            print(f"  {split_name} evaluation failed: {e}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, m in results.items():
        cer_raw = m.get(f"{name}_cer_raw", None)
        cer_nopunct = m.get(f"{name}_cer_nopunct", None)
        if cer_raw is not None:
            line = f"  {name:24s} CER: {cer_raw:.4f}"
            if cer_nopunct is not None:
                line += f"  (nopunct: {cer_nopunct:.4f})"
            print(line)
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned wav2vec2 model on test/holdback splits"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Comma-separated paths to dataset directories")
    parser.add_argument("--test_tsv", type=str, default=None)
    parser.add_argument("--all_tsv", type=str, default=None)
    parser.add_argument("--pct_validation", type=float, default=0.1)
    parser.add_argument("--pct_test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--holdback_tsv", type=str, default=None)
    parser.add_argument("--eval_holdback", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    args = parser.parse_args()

    has_explicit = args.test_tsv is not None
    has_regenerate = args.seed is not None and args.all_tsv is not None
    has_holdback = args.holdback_tsv is not None and args.eval_holdback

    assert has_explicit or (has_regenerate and args.eval_test) or has_holdback, \
        "Nothing to evaluate."

    run_evaluation(
        model_path=args.model,
        dataset_path=args.dataset_path,
        test_tsv=args.test_tsv,
        holdback_tsv=args.holdback_tsv,
        all_tsv=args.all_tsv,
        pct_validation=args.pct_validation,
        pct_test=args.pct_test,
        seed=args.seed,
        eval_test=args.eval_test or has_explicit,
        eval_holdback=args.eval_holdback,
        eval_batch_size=args.eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )


if __name__ == "__main__":
    main()
