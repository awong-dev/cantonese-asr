#!/usr/bin/env python3
"""
Evaluate a fine-tuned Whisper model on test and/or holdback splits.

Two modes:
  1. Explicit TSV: pass --test_tsv and/or --holdback_tsv directly
  2. Regenerate splits: pass --all_tsv, --pct_test, --seed (and optionally
     --holdback_tsv, --pct_validation) to regenerate the same splits used
     during training without needing the training script.

Usage:
    # Mode 1: explicit TSV files
    python eval_whisper.py --model ./whisper-large-v3-yue/final \
        --dataset_path cv-corpus-25.0-2026-03-09/yue \
        --test_tsv test_split.tsv --nopunct_in_eval

    # Mode 2: regenerate splits from seed (same params as training)
    python eval_whisper.py --model ./whisper-large-v3-yue/final \
        --dataset_path cv-corpus-25.0-2026-03-09/yue \
        --all_tsv validated.tsv --holdback_tsv test.tsv \
        --pct_validation 0.005 --pct_test 0.05 --seed 42 \
        --eval_test --eval_holdback --nopunct_in_eval

Requirements:
    pip install transformers datasets evaluate jiwer torchcodec
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import torch
import numpy as np
from torchcodec.decoders import AudioDecoder
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# Suppress noisy warnings
warnings.filterwarnings(
    "ignore", message="None of the inputs have requires_grad=True"
)
warnings.filterwarnings(
    "ignore", message=".*using a WhisperTokenizerFast.*"
)
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Data collator (same as in train_whisper.py)
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
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


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------
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
    eval_batch_size=32,
    eval_accumulation_steps=32,
    nopunct_in_eval=False,
    num_examples=5,
):
    """
    Evaluate a Whisper model on test and/or holdback splits.

    Supports two modes:
      - Explicit TSV: pass test_tsv and/or holdback_tsv file paths
      - Regenerate: pass all_tsv + seed + pct params + eval_test/eval_holdback
        flags, and splits are regenerated deterministically via create_splits

    Returns:
        dict with keys "test" and/or "holdback", each containing metrics dict
    """
    local_path = Path(dataset_path)
    clips_dir = str(local_path / "clips")

    # ---- Determine which datasets to evaluate ----
    test_ds_raw = None
    holdback_ds_raw = None

    if seed is not None and all_tsv is not None:
        # Mode 2: regenerate splits from seed
        from create_splits import create_splits
        splits = create_splits(
            dataset_path=dataset_path,
            all_tsv=all_tsv,
            holdback_tsv=holdback_tsv or "",
            pct_validation=pct_validation,
            pct_test=pct_test,
            seed=seed,
        )
        if eval_test:
            test_ds_raw = splits["test"]
        if eval_holdback and splits["holdback"] is not None:
            holdback_ds_raw = splits["holdback"]
    else:
        # Mode 1: explicit TSV paths
        if test_tsv:
            p = Path(test_tsv)
            if not p.is_absolute():
                p = local_path / p
            assert p.exists(), f"Test TSV not found: {p}"
            test_ds_raw = load_dataset(
                "csv", data_files=str(p), delimiter="\t", split="train"
            )
        if holdback_tsv:
            p = Path(holdback_tsv)
            if not p.is_absolute():
                p = local_path / p
            assert p.exists(), f"Holdback TSV not found: {p}"
            holdback_ds_raw = load_dataset(
                "csv", data_files=str(p), delimiter="\t", split="train"
            )

    assert test_ds_raw is not None or holdback_ds_raw is not None, \
        "No datasets to evaluate. Provide --test_tsv/--holdback_tsv, " \
        "or use --seed with --all_tsv and --eval_test/--eval_holdback."

    # ---- Load processor, tokenizer, model ----
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

    # ---- Audio loading ----
    def _load_audio(path):
        filepath = os.path.join(clips_dir, path)
        decoder = AudioDecoder(filepath, sample_rate=16000, num_channels=1)
        samples = decoder.get_all_samples()
        return samples.data.squeeze(0).numpy()

    def prepare_dataset_batched(batch):
        audio_arrays = [_load_audio(p) for p in batch["path"]]
        batch["input_features"] = feature_extractor(
            audio_arrays, sampling_rate=16000,
        ).input_features
        batch["labels"] = [
            tokenizer(s).input_ids for s in batch["sentence"]
        ]
        return batch

    # ---- Preprocess a dataset: normalize columns → extract features ----
    def preprocess(ds, name):
        text_col = "sentence" if "sentence" in ds.column_names else "text"
        keep_cols = {"path", text_col}
        cols_to_remove = [c for c in ds.column_names if c not in keep_cols]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)
        if text_col != "sentence":
            ds = ds.rename_column(text_col, "sentence")

        remove_cols = ds.column_names
        print(f"Preprocessing {name} ({len(ds)} samples)...")
        ds = ds.map(
            prepare_dataset_batched,
            remove_columns=remove_cols,
            batched=True,
            batch_size=16,
        )
        print(f"  Ready: {len(ds)} samples")
        return ds

    # ---- Preprocess datasets ----
    test_dataset = preprocess(test_ds_raw, "test") if test_ds_raw is not None else None
    holdback_dataset = preprocess(holdback_ds_raw, "holdback") if holdback_ds_raw is not None else None

    # ---- Metrics ----
    cer_metric = evaluate.load("cer")

    _cer_transform = None
    if nopunct_in_eval:
        import jiwer
        _cer_transform = jiwer.Compose([
            jiwer.RemovePunctuation(),
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars(),
        ])

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        pred_ids = np.where(pred_ids == -100, tokenizer.pad_token_id, pred_ids)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pairs = [(p, l) for p, l in zip(pred_str, label_str) if len(l.strip()) > 0]
        if not pairs:
            return {"cer_raw": 1.0}
        pred_str, label_str = zip(*pairs)
        pred_list = list(pred_str)
        label_list = list(label_str)

        cer = cer_metric.compute(predictions=pred_list, references=label_list)
        result = {"cer_raw": cer}

        if _cer_transform is not None:
            import jiwer
            output = jiwer.process_characters(
                label_list, pred_list,
                reference_transform=_cer_transform,
                hypothesis_transform=_cer_transform,
            )
            result["cer_nopunct"] = output.cer

        for i in range(min(num_examples, len(pred_str))):
            print(f"  REF: {label_str[i][:80]}")
            print(f"  HYP: {pred_str[i][:80]}")
            print()

        return result

    # ---- Data collator ----
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ---- Trainer (used only for evaluation) ----
    eval_args = Seq2SeqTrainingArguments(
        output_dir="/tmp/whisper-eval",
        per_device_eval_batch_size=eval_batch_size,
        eval_accumulation_steps=eval_accumulation_steps,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=False,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # ---- Run evaluations ----
    def evaluate_split(dataset, split_name):
        print(f"\nEvaluating on {split_name} ({len(dataset)} samples)...")
        metrics = trainer.evaluate(
            eval_dataset=dataset, metric_key_prefix=split_name
        )
        cer = metrics.get(f"{split_name}_cer_raw", None)
        cer_norm = metrics.get(f"{split_name}_cer_nopunct", None)
        loss = metrics.get(f"{split_name}_loss", None)
        if cer is not None:
            print(f"  {split_name} CER (raw):     {cer:.4f}")
        if cer_norm is not None:
            print(f"  {split_name} CER (nopunct): {cer_norm:.4f}")
        if loss is not None:
            print(f"  {split_name} Loss:           {loss:.4f}")
        return metrics

    results = {}

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    if test_dataset is not None:
        results["test"] = evaluate_split(test_dataset, "test")

    if holdback_dataset is not None:
        results["holdback"] = evaluate_split(holdback_dataset, "holdback")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, prefix in [("Test", "test"), ("Holdback", "holdback")]:
        m = results.get(prefix)
        if m is None:
            continue
        cer = m.get(f"{prefix}_cer_raw", "N/A")
        cer_norm = m.get(f"{prefix}_cer_nopunct", None)
        if isinstance(cer, float):
            line = f"  {name:12s} CER: {cer:.4f}"
            if isinstance(cer_norm, float):
                line += f"  (nopunct: {cer_norm:.4f})"
            print(line)
        else:
            print(f"  {name:12s} CER: {cer}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Whisper model on test/holdback splits"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to model directory (must contain model + tokenizer/processor)",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to Common Voice dataset directory (with clips/)",
    )

    # Mode 1: explicit TSV paths
    parser.add_argument(
        "--test_tsv", type=str, default=None,
        help="Path to test split TSV (relative to dataset_path, or absolute)",
    )

    # Mode 2: regenerate splits from seed
    parser.add_argument(
        "--all_tsv", type=str, default=None,
        help="TSV file with all usable samples (for regenerating splits)",
    )
    parser.add_argument("--pct_validation", type=float, default=0.1)
    parser.add_argument("--pct_test", type=float, default=0.1)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for regenerating splits (must match training seed)",
    )
    parser.add_argument(
        "--eval_test", action="store_true",
        help="Evaluate on the test split (when regenerating splits)",
    )

    # Shared
    parser.add_argument(
        "--holdback_tsv", type=str, default=None,
        help="Path to holdback TSV (relative to dataset_path, or absolute). "
             "Used in both modes.",
    )
    parser.add_argument(
        "--eval_holdback", action="store_true",
        help="Evaluate on the holdback split",
    )
    parser.add_argument("--language_full", type=str, default="cantonese")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_accumulation_steps", type=int, default=32)
    parser.add_argument(
        "--nopunct_in_eval", action="store_true",
        help="Also compute CER with punctuation removed",
    )
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    # Validate: must have something to evaluate
    has_explicit = args.test_tsv is not None
    has_regenerate = args.seed is not None and args.all_tsv is not None
    has_holdback = args.holdback_tsv is not None and args.eval_holdback

    assert has_explicit or (has_regenerate and args.eval_test) or has_holdback, \
        "Nothing to evaluate. Provide --test_tsv, or use --seed + --all_tsv + --eval_test, " \
        "or use --holdback_tsv + --eval_holdback."

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
        eval_accumulation_steps=args.eval_accumulation_steps,
        nopunct_in_eval=args.nopunct_in_eval,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()
