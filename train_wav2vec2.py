#!/usr/bin/env python3
"""
Fine-tune wav2vec2-xls-r (300M or 1B) on Common Voice Cantonese (yue) for ASR.

Usage:
    python train_wav2vec2.py
    python train_wav2vec2.py --model facebook/wav2vec2-xls-r-300m --lr 1e-4
    python train_wav2vec2.py --dataset_path /data/cv-corpus-25.0/yue
    python train_wav2vec2.py --resume_from_checkpoint

Requirements:
    pip install transformers datasets evaluate jiwer accelerate torchcodec
"""

import argparse
import json
import os
import string
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

warnings.filterwarnings(
    "ignore", message="None of the inputs have requires_grad=True"
)


# ---------------------------------------------------------------------------
# 1. Args
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune wav2vec2-xls-r on Common Voice Cantonese"
    )
    # Model
    parser.add_argument(
        "--model", type=str, default="facebook/wav2vec2-xls-r-1b",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--unfreeze", action="store_true",
        help="Unfreeze the feature encoder (frozen by default)",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Use torch.compile (experimental, needs PyTorch 2.x)",
    )

    # Regularization
    parser.add_argument(
        "--attention_dropout", type=float, default=0.1,
        help="Attention dropout rate",
    )
    parser.add_argument(
        "--hidden_dropout", type=float, default=0.1,
        help="Hidden layer dropout rate",
    )
    parser.add_argument(
        "--feat_proj_dropout", type=float, default=0.0,
        help="Feature projection dropout rate",
    )
    parser.add_argument(
        "--mask_time_prob", type=float, default=0.05,
        help="Probability of masking time steps in spectrogram (higher = more regularization)",
    )
    parser.add_argument(
        "--layerdrop", type=float, default=0.1,
        help="Layer dropout rate (probability of dropping a transformer layer)",
    )

    # Dataset
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to local Common Voice dataset (e.g. data/cv-corpus-25.0/yue)",
    )
    parser.add_argument(
        "--all_tsv", type=str, default="validated.tsv",
        help="TSV file containing all usable samples (default: validated.tsv)",
    )
    parser.add_argument(
        "--holdback_tsv", type=str, default="",
        help="TSV file whose samples are excluded from the available pool. "
             "If empty, all samples in all_tsv are used (default: empty)",
    )
    parser.add_argument(
        "--pct_validation", type=float, default=0.1,
        help="Fraction of available_dataset to use for validation (default: 0.1)",
    )
    parser.add_argument(
        "--pct_test", type=float, default=0.1,
        help="Fraction of available_dataset to use for test (default: 0.1)",
    )
    parser.add_argument(
        "--write_splits", action="store_true",
        help="Write train/validation/test split TSV files to current directory",
    )

    # Training
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument(
        "--train_batch_size", type=int, default=4,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4,
        help="Per-device eval batch size (default: 4)",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4,
        help="Gradient accumulation steps (effective batch = train_batch_size * grad_accum)",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=1000, help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=5,
        help="Early stopping patience (number of evals without improvement)",
    )
    parser.add_argument(
        "--eval_accumulation_steps", type=int, default=16,
        help="Flush eval predictions to CPU every N steps to avoid OOM. "
             "Set to 0 for no limit (default: 16)",
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Subsample training set to this many samples (shuffled). Useful for disk/memory constraints.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Subsample eval set to this many samples (shuffled). Speeds up evaluation significantly.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling and subsampling",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./wav2vec2-xls-r-1b-cantonese-yue",
        help="Output directory for checkpoints and final model",
    )
    parser.add_argument(
        "--resume_from_checkpoint", action="store_true",
        help="Resume from latest checkpoint in output_dir",
    )

    # Caching
    parser.add_argument(
        "--no_streaming", action="store_true",
        help="Disable streaming for training: preprocess all data upfront (uses disk cache). "
             "Faster training but requires disk space for cached features.",
    )
    parser.add_argument(
        "--streaming_eval", action="store_true",
        help="Use streaming for eval instead of preprocessing upfront. "
             "Saves disk space but may be slower.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directory for HF dataset cache (default: ~/.cache/huggingface). "
             "Use when default disk is too small.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------
import jiwer

# jiwer pipeline for text normalization: removes punctuation, lowercases,
# and normalizes whitespace. Used both for cleaning training labels and
# for computing normalized CER during evaluation.
_text_normalize = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def clean_text(text: str) -> str:
    """Normalize transcription text for CTC training."""
    text = _text_normalize(text)

    # Common Voice Cantonese quirk: lone ASCII "d" is colloquial 啲
    # Only replace if "d" is the sole ASCII letter in the sentence
    if "d" in text:
        ascii_letters = [c for c in text if c in string.ascii_lowercase]
        if len(ascii_letters) == 1 and ascii_letters[0] == "d":
            text = text.replace("d", "啲")

    return text + " "  # trailing space for CTC word boundary


# ---------------------------------------------------------------------------
# 3. Data collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    """Dynamically pad input_values and labels for CTC training."""

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": f["input_values"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        # Replace padding with -100 so CTC loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# 4. Custom Trainer for torch.compile compatibility
# ---------------------------------------------------------------------------
class CTCTrainer(Trainer):
    """
    Overrides compute_loss to prevent the Trainer from passing
    num_items_in_batch to forward(), which breaks torch.compile.
    """

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Set HF cache directory if specified
    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"Dataset cache dir: {args.cache_dir}")

    print(f"args: {args}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {gpu_mem:.1f} GB")

    # -----------------------------------------------------------------------
    # 6. Load dataset
    # -----------------------------------------------------------------------
    print(f"Loading dataset from local path: {args.dataset_path}")
    local_path = Path(args.dataset_path)
    _local_clips_dir = str(local_path / "clips")

    # Load all usable samples and optionally exclude holdback
    all_tsv = local_path / args.all_tsv

    if not all_tsv.exists():
        raise FileNotFoundError(f"all_tsv not found: {all_tsv}")

    all_dataset = load_dataset(
        "csv", data_files=str(all_tsv), delimiter="\t", split="train",
    )
    print(f"All samples: {len(all_dataset)}")

    if args.holdback_tsv:
        holdback_tsv = local_path / args.holdback_tsv
        if not holdback_tsv.exists():
            raise FileNotFoundError(f"holdback_tsv not found: {holdback_tsv}")
        holdback_dataset = load_dataset(
            "csv", data_files=str(holdback_tsv), delimiter="\t", split="train",
        )
        holdback_paths = set(holdback_dataset["path"])
        available_dataset = all_dataset.filter(
            lambda x: x["path"] not in holdback_paths
        )
        print(f"Holdback samples: {len(holdback_dataset)}")
    else:
        holdback_dataset = None
        available_dataset = all_dataset

    print(f"Available samples: {len(available_dataset)}")

    # Detect text column
    text_col = "sentence" if "sentence" in available_dataset.column_names else "text"

    # Remove unnecessary columns — keep path + text
    keep_cols = {"path", text_col}
    cols_to_remove = [
        c for c in available_dataset.column_names if c not in keep_cols
    ]
    if cols_to_remove:
        available_dataset = available_dataset.remove_columns(cols_to_remove)

    # Normalize column names
    if text_col != "sentence":
        available_dataset = available_dataset.rename_column(text_col, "sentence")

    # Split available_dataset into train / validation / test
    assert args.pct_validation + args.pct_test < 1.0, (
        "pct_validation + pct_test must be less than 1.0"
    )
    available_dataset = available_dataset.shuffle(seed=args.seed)
    n_available = len(available_dataset)
    n_validation = int(n_available * args.pct_validation)
    n_test = int(n_available * args.pct_test)
    n_train = n_available - n_validation - n_test

    common_voice = DatasetDict({
        "train": available_dataset.select(range(n_train)),
        "validation": available_dataset.select(range(n_train, n_train + n_validation)),
        "test": available_dataset.select(range(n_train + n_validation, n_available)),
    })

    print(f"Train samples: {len(common_voice['train'])}")
    print(f"Validation samples: {len(common_voice['validation'])}")
    print(f"Test samples: {len(common_voice['test'])}")

    # Optionally write split files
    if args.write_splits:
        for split_name in ("train", "validation", "test"):
            out_path = f"{split_name}_split.tsv"
            common_voice[split_name].to_csv(out_path, sep="\t", index=False)
            print(f"Wrote {out_path} ({len(common_voice[split_name])} samples)")

    # Optionally subsample training set
    if args.max_train_samples and args.max_train_samples < len(common_voice["train"]):
        common_voice["train"] = common_voice["train"].shuffle(seed=args.seed).select(
            range(args.max_train_samples)
        )
        print(f"Subsampled to {args.max_train_samples} train samples")

    # -----------------------------------------------------------------------
    # 7. Text preprocessing + vocabulary
    # -----------------------------------------------------------------------
    print("Cleaning transcriptions...")

    def apply_text_cleaning(batch):
        batch["sentence"] = clean_text(batch["sentence"])
        return batch

    common_voice = common_voice.map(apply_text_cleaning)

    if holdback_dataset is not None:
        # Clean holdback text and keep only path + sentence
        text_col_hb = "sentence" if "sentence" in holdback_dataset.column_names else "text"
        keep_cols_hb = {"path", text_col_hb}
        cols_remove_hb = [c for c in holdback_dataset.column_names if c not in keep_cols_hb]
        if cols_remove_hb:
            holdback_dataset = holdback_dataset.remove_columns(cols_remove_hb)
        if text_col_hb != "sentence":
            holdback_dataset = holdback_dataset.rename_column(text_col_hb, "sentence")
        holdback_dataset = holdback_dataset.map(apply_text_cleaning)

    # -----------------------------------------------------------------------
    # 7b. Vocabulary + processor
    # -----------------------------------------------------------------------
    # If the model path already contains a saved processor (fine-tuned
    # checkpoint), reuse it so the vocab/token mapping stays consistent.
    # Otherwise build vocabulary from scratch.
    _model_processor_path = Path(args.model)
    if (_model_processor_path / "preprocessor_config.json").exists() and (
        _model_processor_path / "vocab.json"
    ).exists():
        print(f"Loading existing processor from {args.model}")
        processor = Wav2Vec2Processor.from_pretrained(args.model)
        os.makedirs(args.output_dir, exist_ok=True)
        processor.save_pretrained(args.output_dir)
        print(f"Vocabulary size: {len(processor.tokenizer)} (reused from checkpoint)")
    else:
        # Build character vocabulary from train + validation + test
        print("Building vocabulary...")

        def extract_chars(batch):
            all_text = " ".join(batch["sentence"])
            return {"vocab": [list(set(all_text))]}

        all_chars = set()
        for split_name in ("train", "validation", "test"):
            vocab_split = common_voice[split_name].map(
                extract_chars, batched=True, batch_size=1000,
                remove_columns=common_voice[split_name].column_names,
            )
            for row in vocab_split:
                all_chars.update(row["vocab"])

        # Remove ASCII characters (map to [UNK]), keep space for word delimiter
        vocab_list = sorted([c for c in all_chars if not c.isascii()])
        vocab_list.append(" ")

        vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}
        # Use pipe as word delimiter (CTC convention)
        vocab_dict["|"] = vocab_dict.pop(" ")
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        vocab_path = os.path.join(args.output_dir, "vocab.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False)

        print(f"Vocabulary size: {len(vocab_dict)} chars (saved to {vocab_path})")

        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path, unk_token="[UNK]", pad_token="[PAD]",
            word_delimiter_token="|",
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=tokenizer,
        )
        processor.save_pretrained(args.output_dir)

    # -----------------------------------------------------------------------
    # 9. Audio loading + feature extraction
    # -----------------------------------------------------------------------
    train_len = len(common_voice["train"])

    from torchcodec.decoders import AudioDecoder

    def _load_audio(path):
        filepath = os.path.join(_local_clips_dir, path)
        decoder = AudioDecoder(filepath, sample_rate=16000, num_channels=1)
        samples = decoder.get_all_samples()
        return samples.data.squeeze().numpy()

    def prepare_fn(batch):
        audio_np = _load_audio(batch["path"])
        batch["input_values"] = processor(
            audio_np, sampling_rate=16000,
        ).input_values[0]
        batch["labels"] = processor.tokenizer(
            batch["sentence"],
        ).input_ids
        batch["input_length"] = len(audio_np)
        return batch

    def prepare_fn_batched(batch):
        audio_arrays = [_load_audio(p) for p in batch["path"]]
        batch["input_values"] = processor(
            audio_arrays, sampling_rate=16000,
            padding=False,
        ).input_values
        batch["labels"] = [
            processor.tokenizer(s).input_ids for s in batch["sentence"]
        ]
        batch["input_length"] = [len(a) for a in audio_arrays]
        return batch

    remove_cols_train = common_voice.column_names["train"]
    remove_cols_val = common_voice.column_names["validation"]
    remove_cols_test = common_voice.column_names["test"]

    stream_train = not args.no_streaming
    cache_eval = not args.streaming_eval

    if not stream_train:
        # ---- Disk cache mode: preprocess everything upfront (batched) ----
        print("Preprocessing training dataset (disk cache, batched)...")
        train_dataset = common_voice["train"].map(
            prepare_fn_batched,
            remove_columns=remove_cols_train,
            batched=True,
            batch_size=16,
        )
        train_len = len(train_dataset)
        print(f"Train samples: {train_len}")
    else:
        # ---- Streaming mode: process on-the-fly, no disk cache ----
        print("Setting up streaming training dataset...")
        train_dataset = (
            common_voice["train"]
            .to_iterable_dataset(num_shards=max(1, train_len // 5000))
            .map(prepare_fn, remove_columns=remove_cols_train)
        )
        print(f"Train samples (approx): {train_len}")

    if cache_eval:
        print("Preprocessing validation dataset (disk cache, batched)...")
        eval_dataset = common_voice["validation"].map(
            prepare_fn_batched,
            remove_columns=remove_cols_val,
            batched=True,
            batch_size=16,
        )
        print(f"Validation samples: {len(eval_dataset)}")
    else:
        print("Preprocessing validation dataset...")
        eval_dataset = common_voice["validation"].map(
            prepare_fn,
            remove_columns=remove_cols_val,
            keep_in_memory=True,
        )
        print(f"Validation samples: {len(eval_dataset)}")

    # Preprocess test dataset (always cached, used for final evaluation)
    print("Preprocessing test dataset (disk cache, batched)...")
    test_dataset = common_voice["test"].map(
        prepare_fn_batched,
        remove_columns=remove_cols_test,
        batched=True,
        batch_size=16,
    )
    print(f"Test samples: {len(test_dataset)}")

    # Preprocess holdback dataset if present (for final evaluation only)
    if holdback_dataset is not None:
        print("Preprocessing holdback dataset (disk cache, batched)...")
        remove_cols_holdback = holdback_dataset.column_names
        holdback_prepared = holdback_dataset.map(
            prepare_fn_batched,
            remove_columns=remove_cols_holdback,
            batched=True,
            batch_size=16,
        )
        print(f"Holdback samples: {len(holdback_prepared)}")
    else:
        holdback_prepared = None

    # Subsample number of eval.
    if args.max_eval_samples and args.max_eval_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(
            range(args.max_eval_samples)
        )
        print(f"Subsampled eval to {args.max_eval_samples} samples "
              f"(seed={args.seed})")

    # -----------------------------------------------------------------------
    # 10. Load model
    # -----------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_proj_dropout=args.feat_proj_dropout,
        mask_time_prob=args.mask_time_prob,
        layerdrop=args.layerdrop,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    if not args.unfreeze:
        model.freeze_feature_encoder()

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.1f}M")

    # -----------------------------------------------------------------------
    # 11. Metrics
    # -----------------------------------------------------------------------
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Replace -100 with pad token for decoding
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        # Filter out empty references to avoid division by zero
        pairs = [(p, l) for p, l in zip(pred_str, label_str) if len(l.strip()) > 0]
        if not pairs:
            return {"cer_raw": 1.0, "cer_nopunct": 1.0}
        pred_list, label_list = zip(*pairs)
        pred_list = list(pred_list)
        label_list = list(label_list)

        # Raw CER (on decoded text as-is)
        cer_raw = cer_metric.compute(predictions=pred_list, references=label_list)

        # Normalized CER (punctuation removed, lowercased, whitespace normalized)
        cer_nopunct_output = jiwer.process_characters(
            label_list, pred_list,
            reference_transform=_text_normalize,
            hypothesis_transform=_text_normalize,
        )
        cer_nopunct = cer_nopunct_output.cer

        return {"cer_raw": cer_raw, "cer_nopunct": cer_nopunct}

    # -----------------------------------------------------------------------
    # 12. Training arguments
    # -----------------------------------------------------------------------
    # IterableDataset needs max_steps; regular Dataset can use num_train_epochs
    if not stream_train:
        epoch_or_steps_args = {"num_train_epochs": args.epochs}
    else:
        steps_per_epoch = train_len // (args.train_batch_size * args.grad_accum)
        max_steps = steps_per_epoch * args.epochs
        epoch_or_steps_args = {"max_steps": max_steps}
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Max steps: {max_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # Batch / accumulation
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Schedule
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,
        **epoch_or_steps_args,
        # Precision
        fp16=False,
        bf16=torch.cuda.is_available(),
        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        # Logging
        logging_strategy="steps",
        logging_steps=100,
        report_to=["tensorboard"],
        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer_nopunct",
        greater_is_better=False,
        # Performance
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        # Gradient
        max_grad_norm=1.0,
        # Accumulate eval predictions on CPU to avoid OOM from storing all
        # logits (num_samples × seq_len × vocab_size) on GPU at once.
        eval_accumulation_steps=args.eval_accumulation_steps or None,
    )

    # -----------------------------------------------------------------------
    # 13. Trainer
    # -----------------------------------------------------------------------
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Use CTCTrainer if torch.compile is enabled (fixes num_items_in_batch issue)
    TrainerClass = CTCTrainer if args.compile else Trainer

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # 14. Train
    # -----------------------------------------------------------------------
    print("Starting training...")
    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint or None,
    )

    # -----------------------------------------------------------------------
    # 15. Save final model + processor
    # -----------------------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    print(f"Saving final model to {final_dir}")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    # -----------------------------------------------------------------------
    # 16. Final evaluation on validation and test splits
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Final evaluation (best model)")
    print("=" * 60)

    def evaluate_split(dataset, split_name):
        try:
            metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=split_name)
            cer_raw = metrics.get(f"{split_name}_cer_raw", None)
            cer_nopunct = metrics.get(f"{split_name}_cer_nopunct", None)
            if cer_raw is not None:
                print(f"  {split_name} CER (raw):     {cer_raw:.4f}")
            if cer_nopunct is not None:
                print(f"  {split_name} CER (nopunct): {cer_nopunct:.4f}")
            return metrics
        except Exception as e:
            print(f"  {split_name} evaluation failed: {e}")
            return None

    val_metrics = evaluate_split(eval_dataset, "validation")
    test_metrics = evaluate_split(test_dataset, "test")

    holdback_metrics = None
    if holdback_prepared is not None:
        holdback_metrics = evaluate_split(holdback_prepared, "holdback")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, m in [("Validation", val_metrics),
                    ("Test", test_metrics),
                    ("Holdback", holdback_metrics)]:
        if m is None:
            continue
        prefix = name.lower()
        cer_raw = m.get(f"{prefix}_cer_raw", None)
        cer_nopunct = m.get(f"{prefix}_cer_nopunct", None)
        if cer_raw is not None:
            line = f"  {name:12s} CER (raw): {cer_raw:.4f}"
            if cer_nopunct is not None:
                line += f"  (nopunct: {cer_nopunct:.4f})"
            print(line)
    print("=" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
