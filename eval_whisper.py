#!/usr/bin/env python3
"""
Evaluate a fine-tuned Whisper model on test and/or holdback splits.

Supports multiple filesets: evaluates each test split separately, then
computes a combined CER across all test samples.

Usage:
    # Single fileset, explicit TSV
    python eval_whisper.py --model ./whisper-yue/final \
        --dataset_path cv-corpus/yue --test_tsv test_split.tsv

    # Single fileset, regenerate splits from seed
    python eval_whisper.py --model ./whisper-yue/final \
        --dataset_path cv-corpus/yue --all_tsv validated.tsv \
        --holdback_tsv test.tsv --pct_test 0.05 --seed 42 \
        --eval_test --eval_holdback

    # Multiple filesets, regenerate splits
    python eval_whisper.py --model ./whisper-yue/final \
        --dataset_path cv-corpus/yue,cv-corpus/zh-CN \
        --all_tsv validated.tsv,validated.tsv \
        --holdback_tsv test.tsv,test.tsv \
        --pct_test 0.05 --seed 42 --eval_test --eval_holdback
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchcodec.decoders import AudioDecoder
from datasets import load_dataset, concatenate_datasets
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
warnings.filterwarnings("ignore", message=".*using a WhisperTokenizerFast.*")
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [f["labels"] for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", return_attention_mask=True
        )
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features}, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def run_evaluation(
    model_path,
    dataset_path,
    # --- Mode 1: explicit TSV paths ---
    test_tsv=None,
    holdback_tsv=None,
    # --- Mode 2: regenerate splits from seed ---
    all_tsv=None,
    pct_validation=0.1,
    pct_test=0.1,
    seed=None,
    eval_test=False,
    eval_holdback=False,
    # --- Common options ---
    language_full="cantonese",
    eval_batch_size=64,
    dataloader_num_workers=4,
    nopunct_in_eval=False,
):
    """
    Evaluate a Whisper model on test and/or holdback splits.

    Supports multiple filesets via comma-separated dataset_path values.
    When multiple test splits exist, evaluates each separately and
    computes a combined CER across all samples.

    Returns:
        dict with per-split metrics and optionally "combined" metrics
    """
    # ---- Resolve datasets to evaluate ----
    test_datasets = []     # list of (name, Dataset)
    holdback_datasets = [] # list of (name, Dataset)

    if seed is not None and all_tsv is not None:
        # Mode 2: regenerate splits from seed
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
        # Mode 1: explicit TSV paths
        # dataset_path may be comma-separated for audio resolution
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
    assert len(all_eval) > 0, \
        "No datasets to evaluate. Provide --test_tsv/--holdback_tsv, " \
        "or use --seed with --all_tsv and --eval_test/--eval_holdback."

    # ---- Load model ----
    print(f"Loading model from: {model_path}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_path, language=language_full, task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        model_path, language=language_full, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_bf16 = torch.cuda.is_available()
    print(f"Device: {device}" + (", bf16 autocast" if use_bf16 else ""))

    # ---- Audio loading (uses clips_dir column if present, else infers) ----
    ds_paths = [p.strip() for p in dataset_path.split(",")] \
        if isinstance(dataset_path, str) else dataset_path
    default_clips_dir = str(Path(ds_paths[0]) / "clips")

    def _load_audio(clips_dir, path):
        filepath = os.path.join(clips_dir, path)
        decoder = AudioDecoder(filepath, sample_rate=16000, num_channels=1)
        samples = decoder.get_all_samples()
        return samples.data.squeeze(0).numpy()

    def prepare_dataset_batched(batch):
        clips_dirs = batch.get("clips_dir", [default_clips_dir] * len(batch["path"]))
        audio_arrays = [_load_audio(cd, p) for cd, p in zip(clips_dirs, batch["path"])]
        batch["input_features"] = feature_extractor(
            audio_arrays, sampling_rate=16000,
        ).input_features
        batch["labels"] = [tokenizer(s).input_ids for s in batch["sentence"]]
        return batch

    def preprocess(ds, name):
        """Normalize columns and extract features."""
        text_col = "sentence" if "sentence" in ds.column_names else "text"
        # Determine which columns to keep
        keep = {"path", text_col}
        if "clips_dir" in ds.column_names:
            keep.add("clips_dir")
        else:
            # Infer clips_dir from default
            ds = ds.map(lambda x: {"clips_dir": default_clips_dir}, batched=False)
            keep.add("clips_dir")
        cols_to_remove = [c for c in ds.column_names if c not in keep]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)
        if text_col != "sentence":
            ds = ds.rename_column(text_col, "sentence")

        remove_cols = ds.column_names
        print(f"Preprocessing {name} ({len(ds)} samples)...")
        ds = ds.map(prepare_dataset_batched, remove_columns=remove_cols,
                     batched=True, batch_size=16)
        print(f"  Ready: {len(ds)} samples")
        return ds

    # ---- Preprocess all datasets ----
    processed = []
    for name, ds in all_eval:
        processed.append((name, preprocess(ds, name)))

    # ---- Evaluation loop ----
    from cer_utils import build_cer_transform, compute_cer, print_examples

    _cer_transform = build_cer_transform() if nopunct_in_eval else None
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Store all predictions/references for combined CER
    all_preds = []
    all_refs = []

    def evaluate_split(dataset, split_name):
        """Run generate() per batch, decode immediately, discard logits."""
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
            input_features = batch["input_features"].to(device)
            labels = batch["labels"]  # keep on CPU

            with torch.no_grad():
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predicted_ids = model.generate(input_features)
                else:
                    predicted_ids = model.generate(input_features)

            # Decode immediately — discard generated token IDs
            pred_ids = predicted_ids.cpu().numpy()
            del predicted_ids

            label_ids = labels.numpy().copy()
            label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)

            all_pred_str.extend(
                tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            )
            all_label_str.extend(
                tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            )

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"  [{split_name}] batch {batch_idx + 1}/{num_batches}")

        # Compute CER on collected strings
        result, filtered_preds, filtered_refs = compute_cer(
            all_pred_str, all_label_str, cer_transform=_cer_transform
        )

        # Accumulate for combined CER across splits
        all_preds.extend(filtered_preds)
        all_refs.extend(filtered_refs)

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
            results[split_name] = {
                f"{split_name}_cer_raw": cer_raw,
            }
            if cer_nopunct is not None:
                results[split_name][f"{split_name}_cer_nopunct"] = cer_nopunct
        except Exception as e:
            print(f"  {split_name} evaluation failed: {e}")

    # ---- Combined CER (if multiple test splits) ----
    test_results = {k: v for k, v in results.items() if k.startswith("test_")}
    if len(test_results) > 1 and len(all_preds) > 0:
        combined, _, _ = compute_cer(
            all_preds, all_refs, cer_transform=_cer_transform
        )
        results["combined"] = {
            "combined_cer_raw": combined["cer_raw"],
        }
        if "cer_nopunct" in combined:
            results["combined"]["combined_cer_nopunct"] = combined["cer_nopunct"]

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, m in results.items():
        prefix = name
        cer = m.get(f"{prefix}_cer_raw", None)
        cer_norm = m.get(f"{prefix}_cer_nopunct", None)
        if cer is not None:
            line = f"  {name:24s} CER: {cer:.4f}"
            if cer_norm is not None:
                line += f"  (nopunct: {cer_norm:.4f})"
            print(line)
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Whisper model on test/holdback splits"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Comma-separated paths to dataset directories (with clips/)",
    )

    # Mode 1: explicit TSV
    parser.add_argument(
        "--test_tsv", type=str, default=None,
        help="Comma-separated test TSV paths",
    )

    # Mode 2: regenerate splits
    parser.add_argument("--all_tsv", type=str, default=None,
                        help="Comma-separated TSV filenames for regenerating splits")
    parser.add_argument("--pct_validation", type=float, default=0.1)
    parser.add_argument("--pct_test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_test", action="store_true")

    # Shared
    parser.add_argument("--holdback_tsv", type=str, default=None,
                        help="Comma-separated holdback TSV paths")
    parser.add_argument("--eval_holdback", action="store_true")
    parser.add_argument("--language_full", type=str, default="cantonese")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--nopunct_in_eval", action="store_true")
    args = parser.parse_args()

    has_explicit = args.test_tsv is not None
    has_regenerate = args.seed is not None and args.all_tsv is not None
    has_holdback = args.holdback_tsv is not None and args.eval_holdback

    assert has_explicit or (has_regenerate and args.eval_test) or has_holdback, \
        "Nothing to evaluate. Provide --test_tsv, or --seed + --all_tsv + --eval_test, " \
        "or --holdback_tsv + --eval_holdback."

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
        language_full=args.language_full,
        eval_batch_size=args.eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        nopunct_in_eval=args.nopunct_in_eval,
    )


if __name__ == "__main__":
    main()
