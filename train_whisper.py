#!/usr/bin/env python3
"""
Fine-tune openai/whisper-large-v3-turbo on Common Voice Cantonese (yue) for ASR.

Usage:
    python train_whisper.py
    python train_whisper.py --lr 1e-5 --warmup 500 --epochs 10
    python train_whisper.py --resume_from_checkpoint

Requirements:
    pip install transformers datasets evaluate jiwer accelerate soundfile librosa
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import evaluate
import torch
import numpy as np
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# Suppress gradient checkpoint warning for frozen layers
warnings.filterwarnings(
    "ignore", message="None of the inputs have requires_grad=True"
)


# ---------------------------------------------------------------------------
# 1. Args
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper large-v3-turbo on Common Voice Cantonese"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mozilla-foundation/common_voice_17_0",
        help="HuggingFace dataset ID (used when --dataset_path is not set)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to local Common Voice dataset directory (e.g. /data/common_voice/yue)",
    )
    parser.add_argument("--language", type=str, default="yue", help="Language code")
    parser.add_argument(
        "--language_full", type=str, default="cantonese", help="Full language name"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Per-device train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Per-device eval batch size"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./whisper-large-v3-turbo-yue",
        help="Output directory",
    )
    parser.add_argument(
        "--max_input_length",
        type=float,
        default=30.0,
        help="Max audio length in seconds (Whisper limit = 30s)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=1000, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Early stopping patience (number of evals)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Data collator for Whisper seq2seq
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads inputs and labels for Whisper seq2seq training.
    Labels are padded with -100 so they are ignored in the loss.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they need different padding
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features

        ]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so it's ignored in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove the decoder_start_token if it was prepended during tokenization
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print(f"args: {args}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # -----------------------------------------------------------------------
    # 4. Load processor (feature extractor + tokenizer)
    # -----------------------------------------------------------------------
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model, language=args.language_full, task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        args.model, language=args.language_full, task="transcribe"
    )

    # -----------------------------------------------------------------------
    # 5. Load dataset
    # -----------------------------------------------------------------------
    if args.dataset_path:
        # Load from local directory.
        # Supports two common layouts:
        #   1. Arrow/parquet dataset saved via dataset.save_to_disk() or
        #      downloaded with HF cache structure — pass the top-level dir
        #   2. Directory containing train/validation/test subdirs with
        #      audio files + metadata (CSV/TSV)
        print(f"Loading dataset from local path: {args.dataset_path}")
        from pathlib import Path

        local_path = Path(args.dataset_path)

        # Check if it looks like a HF-style dataset directory with split folders
        has_split_dirs = (local_path / "train").exists() or (
            local_path / "train.tsv"
        ).exists()

        if (local_path / "dataset_dict.json").exists():
            # Saved via save_to_disk()
            from datasets import load_from_disk
            common_voice = load_from_disk(str(local_path))
        elif has_split_dirs:
            # Standard Common Voice extracted archive layout:
            #   /path/to/yue/train/ , /path/to/yue/test/ , etc.
            common_voice = DatasetDict()
            common_voice["train"] = load_dataset(
                "audiofolder",
                data_dir=str(local_path),
                split="train+validation",
            )
            common_voice["test"] = load_dataset(
                "audiofolder",
                data_dir=str(local_path),
                split="test",
            )
        else:
            # Try loading as a generic HF dataset directory
            common_voice = DatasetDict()
            common_voice["train"] = load_dataset(
                str(local_path), split="train+validation", trust_remote_code=True
            )
            common_voice["test"] = load_dataset(
                str(local_path), split="test", trust_remote_code=True
            )
    else:
        # Load from HuggingFace Hub
        print(f"Loading dataset: {args.dataset} ({args.language})")
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(
            args.dataset, args.language, split="train+validation", trust_remote_code=True
        )
        common_voice["test"] = load_dataset(
            args.dataset, args.language, split="test", trust_remote_code=True
        )

    # Remove unnecessary columns (keep only audio + sentence/text)
    text_col = "sentence" if "sentence" in common_voice["train"].column_names else "text"
    cols_to_remove = [
        c
        for c in common_voice["train"].column_names
        if c not in ("audio", text_col)
    ]
    if cols_to_remove:
        common_voice = common_voice.remove_columns(cols_to_remove)

    # Normalize text column name to "sentence"
    if text_col != "sentence":
        common_voice = common_voice.rename_column(text_col, "sentence")

    # Resample to 16kHz
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Train samples: {len(common_voice['train'])}")
    print(f"Test samples: {len(common_voice['test'])}")

    # -----------------------------------------------------------------------
    # 6. Preprocessing
    # -----------------------------------------------------------------------
    max_input_length_samples = args.max_input_length * 16000  # 30s * 16kHz

    def prepare_dataset(batch):
        audio = batch["audio"]

        # Compute log-Mel spectrogram
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    def is_audio_in_length_range(length):
        return length < max_input_length_samples

    print("Preprocessing dataset...")
    common_voice = common_voice.map(
        prepare_dataset,
        remove_columns=common_voice.column_names["train"],
        num_proc=1,  # audio processing doesn't benefit much from multiproc
    )

    # Filter out samples longer than 30s
    common_voice["train"] = common_voice["train"].filter(
        is_audio_in_length_range,
        input_columns=["input_features"],
        fn_kwargs=None,
    )

    print(f"Train samples after filtering: {len(common_voice['train'])}")

    # -----------------------------------------------------------------------
    # 7. Load model
    # -----------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        use_cache=False,  # incompatible with gradient checkpointing
    )

    # Set language and task for generation
    model.generation_config.language = args.language_full
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Enable gradient checkpointing to save VRAM
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Freeze the encoder — Whisper's encoder is already very good;
    # fine-tuning only the decoder is faster and often sufficient.
    # Comment this out if you want to fine-tune the full model.
    model.freeze_encoder()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.1f}M")

    # -----------------------------------------------------------------------
    # 8. Data collator
    # -----------------------------------------------------------------------
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # -----------------------------------------------------------------------
    # 9. Metrics — CER (Character Error Rate) for Cantonese
    # -----------------------------------------------------------------------
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id for decoding
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    # -----------------------------------------------------------------------
    # 10. Training arguments
    # -----------------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        # Batch / accumulation
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Schedule
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        # Precision
        fp16=False,
        bf16=torch.cuda.is_available(),
        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        predict_with_generate=True,  # required for seq2seq metrics
        generation_max_length=225,
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
    # 11. Trainer
    # -----------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    # -----------------------------------------------------------------------
    # 12. Train
    # -----------------------------------------------------------------------
    print("Starting training...")
    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint else None,
    )

    # -----------------------------------------------------------------------
    # 13. Save final model + processor
    # -----------------------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    print(f"Saving final model to {final_dir}")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Done!")


if __name__ == "__main__":
    main()
