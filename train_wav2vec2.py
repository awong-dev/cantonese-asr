#!/usr/bin/env python3
"""
Fine-tune wav2vec2-xls-r (300M or 1B) on Common Voice Cantonese (yue) for ASR.

Usage:
    python train_wav2vec2.py
    python train_wav2vec2.py --model facebook/wav2vec2-xls-r-300m --lr 1e-4
    python train_wav2vec2.py --dataset_path /data/cv-corpus-25.0/yue
    python train_wav2vec2.py --resume_from_checkpoint

Requirements:
    pip install transformers datasets evaluate jiwer accelerate soundfile librosa
"""

import argparse
import json
import os
import re
import string
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
from datasets import Audio, DatasetDict, load_dataset, load_from_disk
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
        "--dataset", type=str, default="mozilla-foundation/common_voice_17_0",
        help="HuggingFace dataset ID (used when --dataset_path is not set)",
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help="Path to local Common Voice dataset (e.g. data/cv-corpus-25.0/yue)",
    )
    parser.add_argument(
        "--train_tsv", type=str, default="train.tsv",
        help="TSV file for training data: 'train.tsv' (default, smaller) or 'validated.tsv' (all validated samples)",
    )
    parser.add_argument(
        "--language", type=str, default="yue",
        help="Language code for HF Hub loading",
    )

    # Training
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument(
        "--max_audio_length", type=float, default=15.0,
        help="Max audio length in seconds. Longer samples are skipped. (default: 15s)",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=None,
        help="Per-device eval batch size (default: same as train_batch_size)",
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
        help="Disable streaming and preprocess all data upfront (uses disk cache). "
             "Faster training but requires disk space for cached features.",
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
# Punctuation / symbols to strip from transcriptions
CHARS_TO_REMOVE = re.compile(
    r'[\丶\,\?\.\!\-\;\:"\"\%\'\"\�\．\⋯\！\－\：\–\。'
    r'\》\,\）\,\？\；\～\~\…\︰\，\（\」\‧\《\﹔\、\—'
    r'\／\,\「\﹖\·\']'
)


def clean_text(text: str) -> str:
    """Normalize transcription text for CTC training."""
    text = CHARS_TO_REMOVE.sub("", text).lower().strip()

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
    _local_clips_dir = None  # set when loading local TSV layout

    if args.dataset_path:
        print(f"Loading dataset from local path: {args.dataset_path}")
        local_path = Path(args.dataset_path)

        if (local_path / "dataset_dict.json").exists():
            # Previously saved via dataset.save_to_disk()
            common_voice = load_from_disk(str(local_path))
            if "train" in common_voice and "validation" in common_voice:
                from datasets import concatenate_datasets
                common_voice = DatasetDict({
                    "train": concatenate_datasets([
                        common_voice["train"], common_voice["validation"]
                    ]),
                    "test": common_voice["test"],
                })
        elif (local_path / args.train_tsv).exists():
            # Standard Common Voice extracted archive:
            #   yue/clips/*.mp3, yue/train.tsv, yue/test.tsv, etc.
            _local_clips_dir = str(local_path / "clips")
            common_voice = DatasetDict()
            common_voice["train"] = load_dataset(
                "csv",
                data_files=str(local_path / args.train_tsv),
                delimiter="\t",
                split="train",
            )
            common_voice["test"] = load_dataset(
                "csv",
                data_files=str(local_path / "test.tsv"),
                delimiter="\t",
                split="train",  # CSV loader only has "train" split
            )
        else:
            common_voice = DatasetDict()
            common_voice["train"] = load_dataset(
                str(local_path), split="train+validation", trust_remote_code=True
            )
            common_voice["test"] = load_dataset(
                str(local_path), split="test", trust_remote_code=True
            )
    else:
        print(f"Loading dataset: {args.dataset} ({args.language})")
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(
            args.dataset, args.language, split="train+validation",
            trust_remote_code=True,
        )
        common_voice["test"] = load_dataset(
            args.dataset, args.language, split="test", trust_remote_code=True,
        )

    # Detect text column
    text_col = "sentence" if "sentence" in common_voice["train"].column_names else "text"

    # Remove unnecessary columns — keep audio/path + text
    audio_col = "audio" if "audio" in common_voice["train"].column_names else "path"
    keep_cols = {audio_col, text_col}
    cols_to_remove = [
        c for c in common_voice["train"].column_names if c not in keep_cols
    ]
    if cols_to_remove:
        common_voice = common_voice.remove_columns(cols_to_remove)

    # Normalize column names
    if text_col != "sentence":
        common_voice = common_voice.rename_column(text_col, "sentence")

    # ---- Remove any test samples that leaked into the training set ----
    # Use the "path" column (or audio["path"]) as unique identifier.
    if "path" in common_voice["test"].column_names:
        test_paths = set(common_voice["test"]["path"])
        train_before = len(common_voice["train"])
        common_voice["train"] = common_voice["train"].filter(
            lambda x: x["path"] not in test_paths
        )
        n_removed = train_before - len(common_voice["train"])
        if n_removed:
            print(f"Removed {n_removed} test-overlapping samples from training set")
    elif "audio" in common_voice["test"].column_names:
        # HF datasets store path inside the audio dict
        test_paths = set(
            ex["path"] for ex in common_voice["test"]["audio"] if ex.get("path")
        )
        if test_paths:
            train_before = len(common_voice["train"])
            common_voice["train"] = common_voice["train"].filter(
                lambda x: x["audio"]["path"] not in test_paths
            )
            n_removed = train_before - len(common_voice["train"])
            if n_removed:
                print(f"Removed {n_removed} test-overlapping samples from training set")
    else:
        print("Warning: no 'path' or 'audio' column found — skipping deduplication")

    print(f"Train samples: {len(common_voice['train'])}")
    print(f"Test samples: {len(common_voice['test'])}")

    # Shuffle and optionally subsample training set
    common_voice["train"] = common_voice["train"].shuffle(seed=args.seed)
    if args.max_train_samples and args.max_train_samples < len(common_voice["train"]):
        common_voice["train"] = common_voice["train"].select(
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

    # Build character vocabulary from train + test
    print("Building vocabulary...")

    def extract_chars(batch):
        all_text = " ".join(batch["sentence"])
        return {"vocab": [list(set(all_text))]}

    vocab_train = common_voice["train"].map(
        extract_chars, batched=True, batch_size=1000,
        remove_columns=common_voice["train"].column_names,
    )
    vocab_test = common_voice["test"].map(
        extract_chars, batched=True, batch_size=1000,
        remove_columns=common_voice["test"].column_names,
    )

    all_chars = set()
    for row in vocab_train:
        all_chars.update(row["vocab"])
    for row in vocab_test:
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

    # -----------------------------------------------------------------------
    # 8. Create processor
    # -----------------------------------------------------------------------
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
    has_audio_column = "audio" in common_voice["train"].column_names
    train_len = len(common_voice["train"])
    max_input_samples = int(args.max_audio_length * 16000)

    if has_audio_column:
        common_voice = common_voice.cast_column(
            "audio", Audio(sampling_rate=16000)
        )

        def prepare_features(batch):
            audio = batch["audio"]
            batch["input_values"] = processor(
                audio["array"], sampling_rate=16000,
            ).input_values[0]
            batch["labels"] = processor.tokenizer(
                batch["sentence"],
            ).input_ids
            batch["input_length"] = len(audio["array"])
            return batch

        def prepare_features_batched(batch):
            batch["input_values"] = processor(
                [a["array"] for a in batch["audio"]],
                sampling_rate=16000,
                padding=False,
            ).input_values
            batch["labels"] = [
                processor.tokenizer(s).input_ids for s in batch["sentence"]
            ]
            batch["input_length"] = [len(a["array"]) for a in batch["audio"]]
            return batch

        prepare_fn = prepare_features
        prepare_fn_batched = prepare_features_batched
    else:
        import torchaudio

        if _local_clips_dir is None:
            raise ValueError(
                "Dataset has 'path' column but no clips directory found. "
                "Use --dataset_path pointing to the Common Voice language dir."
            )

        resamplers = {}

        def _load_audio(path):
            filepath = os.path.join(_local_clips_dir, path)
            speech_array, sr = torchaudio.load(filepath)
            if sr != 16000:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
                speech_array = resamplers[sr](speech_array)
            return speech_array.squeeze().numpy()

        def prepare_features_from_path(batch):
            audio_np = _load_audio(batch["path"])
            batch["input_values"] = processor(
                audio_np, sampling_rate=16000,
            ).input_values[0]
            batch["labels"] = processor.tokenizer(
                batch["sentence"],
            ).input_ids
            batch["input_length"] = len(audio_np)
            return batch

        def prepare_features_from_path_batched(batch):
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

        prepare_fn = prepare_features_from_path
        prepare_fn_batched = prepare_features_from_path_batched

    remove_cols_train = common_voice.column_names["train"]
    remove_cols_test = common_voice.column_names["test"]

    if args.no_streaming:
        # ---- Disk cache mode: preprocess everything upfront (batched) ----
        print("Preprocessing training dataset (disk cache, batched)...")
        train_dataset = common_voice["train"].map(
            prepare_fn_batched,
            remove_columns=remove_cols_train,
            batched=True,
            batch_size=16,
        )
        train_dataset = train_dataset.filter(
            lambda x: x["input_length"] < max_input_samples
        )
        train_dataset = train_dataset.remove_columns(["input_length"])

        print("Preprocessing eval dataset (disk cache, batched)...")
        eval_dataset = common_voice["test"].map(
            prepare_fn_batched,
            remove_columns=remove_cols_test,
            batched=True,
            batch_size=16,
        )
        eval_dataset = eval_dataset.filter(
            lambda x: x["input_length"] < max_input_samples
        )
        eval_dataset = eval_dataset.remove_columns(["input_length"])

        train_len = len(train_dataset)
        print(f"Train samples after filtering: {train_len}")
        print(f"Eval samples after filtering: {len(eval_dataset)}")
    else:
        # ---- Streaming mode: process on-the-fly, no disk cache ----
        print("Setting up streaming training dataset...")
        train_dataset = (
            common_voice["train"]
            .to_iterable_dataset(num_shards=max(1, train_len // 5000))
            .map(prepare_fn, remove_columns=remove_cols_train)
            .filter(lambda x: x["input_length"] < max_input_samples)
        )

        print("Preprocessing eval dataset...")
        eval_dataset = common_voice["test"].map(
            prepare_fn,
            remove_columns=remove_cols_test,
            keep_in_memory=True,
        )

        print(f"Train samples (approx): {train_len}")
        print(f"Eval samples: {len(eval_dataset)}")

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

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    # -----------------------------------------------------------------------
    # 12. Training arguments
    # -----------------------------------------------------------------------
    # IterableDataset needs max_steps; regular Dataset can use num_train_epochs
    if args.no_streaming:
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
        per_device_eval_batch_size=args.eval_batch_size or args.train_batch_size,
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
        metric_for_best_model="cer",
        greater_is_better=False,
        # Performance
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        # Gradient
        max_grad_norm=1.0,
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

    print("Done!")


if __name__ == "__main__":
    main()
