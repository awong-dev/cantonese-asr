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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import torch
import numpy as np
from datasets import Audio, DatasetDict, load_dataset, load_from_disk
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

    # Regularization
    parser.add_argument(
        "--attention_dropout", type=float, default=0.0,
        help="Attention dropout rate",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout for all FC layers in embeddings, encoder, and decoder",
    )
    parser.add_argument(
        "--activation_dropout", type=float, default=0.0,
        help="Dropout for activations inside FC layers",
    )
    parser.add_argument(
        "--encoder_layerdrop", type=float, default=0.0,
        help="Encoder layer dropout probability",
    )
    parser.add_argument(
        "--decoder_layerdrop", type=float, default=0.0,
        help="Decoder layer dropout probability",
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
    parser.add_argument(
        "--train_tsv",
        type=str,
        default="train.tsv",
        help="TSV file for training data: 'train.tsv' (default, smaller) or 'validated.tsv' (all validated samples)",
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
        "--eval_batch_size", type=int, default=None, help="Per-device eval batch size (default: same as train_batch_size)"
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
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Subsample training set to this many samples (shuffled). Useful for disk/memory constraints.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Subsample eval set to this many samples (shuffled). Speeds up evaluation significantly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and subsampling",
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
        label_features = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", return_attention_mask=True
        )

        # Pad labels — pass as {"input_ids": [...]} which works with both
        # fast and slow tokenizers without triggering the warning
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features}, return_tensors="pt"
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

    # Set HF cache directory if specified
    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"Dataset cache dir: {args.cache_dir}")

    print(f"args: {args}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
    # Track whether we need manual audio loading (local TSV with path column)
    _local_tsv_mode = False
    _clips_dir = None

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
            _clips_dir = str(local_path / "clips")
            _local_tsv_mode = True
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

    # Normalize text column name to "sentence"
    if text_col != "sentence":
        common_voice = common_voice.rename_column(text_col, "sentence")

    print(f"Train samples: {len(common_voice['train'])}")
    print(f"Test samples: {len(common_voice['test'])}")

    # Note: Unlike wav2vec2 (CTC with custom char vocab), Whisper has its own
    # tokenizer that handles punctuation, casing, and mixed scripts natively.
    # Text cleaning/normalization is NOT applied here — it would force the
    # model to unlearn its pretrained text distribution, causing divergence.
    # CER is computed on the raw tokenizer output, which is standard for Whisper.

    # Shuffle and optionally subsample training set
    common_voice["train"] = common_voice["train"].shuffle(seed=args.seed)
    if args.max_train_samples and args.max_train_samples < len(common_voice["train"]):
        common_voice["train"] = common_voice["train"].select(
            range(args.max_train_samples)
        )
        print(f"Subsampled to {args.max_train_samples} train samples")

    # -----------------------------------------------------------------------
    # 6. Preprocessing
    # -----------------------------------------------------------------------
    max_input_length_samples = int(args.max_input_length * 16000)  # 30s * 16kHz

    train_len = len(common_voice["train"])

    if _local_tsv_mode:
        import torchaudio
        resamplers = {}

        def _load_audio(path):
            filepath = os.path.join(_clips_dir, path)
            speech_array, sr = torchaudio.load(filepath)
            if sr != 16000:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
                speech_array = resamplers[sr](speech_array)
            return speech_array.squeeze().numpy()

        def prepare_dataset(batch):
            audio_np = _load_audio(batch["path"])
            batch["input_features"] = feature_extractor(
                audio_np, sampling_rate=16000,
            ).input_features[0]
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
            batch["input_length"] = len(audio_np)
            return batch

        def prepare_dataset_batched(batch):
            audio_arrays = [_load_audio(p) for p in batch["path"]]
            batch["input_features"] = feature_extractor(
                audio_arrays, sampling_rate=16000,
            ).input_features
            batch["labels"] = [
                tokenizer(s).input_ids for s in batch["sentence"]
            ]
            batch["input_length"] = [len(a) for a in audio_arrays]
            return batch
    else:
        common_voice = common_voice.cast_column(
            "audio", Audio(sampling_rate=16000)
        )

        def prepare_dataset(batch):
            audio = batch["audio"]
            batch["input_features"] = feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"],
            ).input_features[0]
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
            batch["input_length"] = len(audio["array"])
            return batch

        def prepare_dataset_batched(batch):
            batch["input_features"] = feature_extractor(
                [a["array"] for a in batch["audio"]],
                sampling_rate=16000,
            ).input_features
            batch["labels"] = [
                tokenizer(s).input_ids for s in batch["sentence"]
            ]
            batch["input_length"] = [len(a["array"]) for a in batch["audio"]]
            return batch

    remove_cols_train = common_voice.column_names["train"]
    remove_cols_test = common_voice.column_names["test"]

    if args.no_streaming:
        # ---- Disk cache mode: preprocess everything upfront (batched) ----
        print("Preprocessing training dataset (disk cache, batched)...")
        train_dataset = common_voice["train"].map(
            prepare_dataset_batched,
            remove_columns=remove_cols_train,
            batched=True,
            batch_size=16,
        )
        train_dataset = train_dataset.filter(
            lambda x: x["input_length"] < max_input_length_samples
        )
        train_dataset = train_dataset.remove_columns(["input_length"])

        print("Preprocessing eval dataset (disk cache, batched)...")
        eval_dataset = common_voice["test"].map(
            prepare_dataset_batched,
            remove_columns=remove_cols_test,
            batched=True,
            batch_size=16,
        )
        eval_dataset = eval_dataset.filter(
            lambda x: x["input_length"] < max_input_length_samples
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
            .map(prepare_dataset, remove_columns=remove_cols_train)
            .filter(lambda x: x["input_length"] < max_input_length_samples)
        )

        print("Preprocessing eval dataset...")
        eval_dataset = common_voice["test"].map(
            prepare_dataset,
            remove_columns=remove_cols_test,
            keep_in_memory=True,
        )
        eval_dataset = eval_dataset.filter(
            lambda x: x["input_length"] < max_input_length_samples,
            keep_in_memory=True,
        )
        eval_dataset = eval_dataset.remove_columns(["input_length"])

        print(f"Train samples (approx): {train_len}")
        print(f"Eval samples: {len(eval_dataset)}")

    # Subsample number of eval.
    if args.max_eval_samples and args.max_eval_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(
            range(args.max_eval_samples)
        )
        print(f"Subsampled eval to {args.max_eval_samples} samples (seed={args.seed})")

    # -----------------------------------------------------------------------
    # 7. Load model
    # -----------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        use_cache=False,  # incompatible with gradient checkpointing
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        encoder_layerdrop=args.encoder_layerdrop,
        decoder_layerdrop=args.decoder_layerdrop,
    )

    # Set language and task for generation
    model.generation_config.language = args.language_full
    model.generation_config.task = "transcribe"
    # Note: forced_decoder_ids is set to None so the model uses the language/task
    # tokens from generation_config. Setting forced_decoder_ids explicitly can
    # conflict with how labels are tokenized during training.
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

        # predict_with_generate can return a tuple (ids, scores)
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Replace -100 with pad token id for decoding
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        # Also handle -100 in predictions (shouldn't happen, but just in case)
        pred_ids = np.where(pred_ids == -100, tokenizer.pad_token_id, pred_ids)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Filter out empty pairs to avoid division by zero in CER
        pairs = [(p, l) for p, l in zip(pred_str, label_str) if len(l.strip()) > 0]
        if not pairs:
            return {"cer": 1.0}
        pred_str, label_str = zip(*pairs)

        cer = cer_metric.compute(predictions=list(pred_str), references=list(label_str))

        # Log a few examples for debugging
        for i in range(min(3, len(pred_str))):
            print(f"  REF: {label_str[i][:80]}")
            print(f"  HYP: {pred_str[i][:80]}")
            print()

        return {"cer": cer}

    # -----------------------------------------------------------------------
    # 10. Training arguments
    # -----------------------------------------------------------------------
    if args.no_streaming:
        epoch_or_steps_args = {"num_train_epochs": args.epochs}
    else:
        steps_per_epoch = train_len // (args.train_batch_size * args.grad_accum)
        max_steps = steps_per_epoch * args.epochs
        epoch_or_steps_args = {"max_steps": max_steps}
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Max steps: {max_steps}")

    training_args = Seq2SeqTrainingArguments(
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
        eval_accumulation_steps=1,  # offload predictions to CPU to prevent OOM during eval
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
