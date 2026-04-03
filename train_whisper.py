#!/usr/bin/env python3
"""
Fine-tune Whisper on a local Common Voice dataset with custom train/val/test splits.

Usage:
    python train_whisper.py --dataset_path cv-corpus-25.0-2026-03-09/yue
    python train_whisper.py --dataset_path /data/yue --holdback_tsv test.tsv --lr 1e-5
    python train_whisper.py --dataset_path /data/yue --write_splits --pct_test 0.05

Requirements:
    pip install transformers datasets evaluate jiwer accelerate torchaudio
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import torch
import torchaudio
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.optimization import get_scheduler

# Suppress noisy warnings
warnings.filterwarnings(
    "ignore", message="None of the inputs have requires_grad=True"
)
warnings.filterwarnings(
    "ignore", message=".*using a WhisperTokenizerFast.*"
)
warnings.filterwarnings(
    "ignore", message=".*torchaudio.*"
)
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# 1. Args
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on a local Common Voice dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="HuggingFace model ID",
    )

    # Dataset
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to local Common Voice dataset directory containing clips/ and TSV files",
    )
    parser.add_argument(
        "--all_tsv",
        type=str,
        default="validated.tsv",
        help="TSV file containing all usable samples (default: validated.tsv)",
    )
    parser.add_argument(
        "--holdback_tsv",
        type=str,
        default="",
        help="TSV file of samples to exclude from the available pool (default: none). "
             "These samples are reserved for final holdback evaluation.",
    )
    parser.add_argument(
        "--pct_validation",
        type=float,
        default=0.1,
        help="Fraction of available samples for validation split (default: 0.1)",
    )
    parser.add_argument(
        "--pct_test",
        type=float,
        default=0.1,
        help="Fraction of available samples for test split (default: 0.1)",
    )
    parser.add_argument(
        "--write_splits", action="store_true",
        help="Write train/validation/test split TSVs to the current directory",
    )

    # Language
    parser.add_argument(
        "--language_full", type=str, default="cantonese", help="Full language name"
    )

    # Freeze / unfreeze
    parser.add_argument(
        "--freeze_encoder", action="store_true",
        help="Freeze the encoder (only train decoder). Default: train full model.",
    )
    parser.add_argument(
        "--unfreeze_encoder_layers", type=int, default=None,
        help="Freeze the encoder except for the last N layers. "
             "Overrides --freeze_encoder.",
    )
    parser.add_argument(
        "--freeze_decoder_layers", type=int, default=None,
        help="Freeze the first N decoder layers.",
    )
    parser.add_argument(
        "--encoder_lr", type=float, default=None,
        help="Separate learning rate for encoder (default: same as --lr).",
    )

    # LoRA
    parser.add_argument(
        "--lora", action="store_true",
        help="Apply LoRA. Requires peft: pip install peft",
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)",
    )
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=None,
        help="Modules to apply LoRA to (default: all attention + FFN projections)",
    )
    parser.add_argument(
        "--lora_merge_on_save", action="store_true",
        help="Merge LoRA weights into base model when saving.",
    )

    # Regularization
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation_dropout", type=float, default=0.0)
    parser.add_argument("--encoder_layerdrop", type=float, default=0.0)
    parser.add_argument("--decoder_layerdrop", type=float, default=0.0)

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument(
        "--eval_accumulation_steps", type=int, default=32,
        help="Number of eval batches to accumulate before offloading predictions "
             "to CPU. Higher = faster eval but more GPU memory. (default: 32)",
    )
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument(
        "--output_dir", type=str, default="./whisper-finetuned",
    )
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument(
        "--early_stopping_patience", type=int, default=0,
        help="0 = disabled (default).",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Streaming / caching
    parser.add_argument(
        "--no_streaming", action="store_true",
        help="Disable streaming for both train and eval.",
    )
    parser.add_argument("--no_streaming_train", action="store_true")
    parser.add_argument(
        "--streaming_eval", action="store_true",
        help="Stream eval data instead of preprocessing upfront. "
             "Default is disk-cached eval for faster evaluation.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Data collator for Whisper seq2seq
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads inputs and labels for Whisper seq2seq training.
    Inputs (mel spectrograms) and labels (token IDs) need different padding
    strategies: inputs are padded by the feature extractor, labels by the tokenizer.
    Label padding tokens are replaced with -100 so they are ignored in the loss.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Separate inputs and labels — they need different padding
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [f["labels"] for f in features]

        # Pad mel spectrogram inputs to the longest in the batch
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", return_attention_mask=True
        )

        # Pad label token IDs to the longest in the batch
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features}, return_tensors="pt"
        )

        # Replace tokenizer padding with -100 so cross-entropy loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip the decoder start token if the tokenizer prepended it,
        # since the model will add it automatically during training
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# 2b. Custom Trainer with differential learning rates
# ---------------------------------------------------------------------------
class DifferentialLRTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer subclass with two extensions:
    1. Strips unexpected keyword arguments (e.g. input_ids) that peft may
       inject but WhisperForConditionalGeneration.forward() does not accept.
    2. Supports separate learning rates for encoder and decoder via custom
       AdamW parameter groups. When encoder_lr is None, falls back to the
       default single-LR optimizer.
    """

    # Whitelist of keys that Whisper's forward() actually accepts.
    # Any other keys in the batch dict are stripped before forwarding.
    _WHISPER_FORWARD_KEYS = {
        "input_features", "attention_mask", "decoder_input_ids",
        "decoder_attention_mask", "head_mask", "decoder_head_mask",
        "cross_attn_head_mask", "encoder_outputs", "past_key_values",
        "decoder_inputs_embeds", "labels", "use_cache",
        "output_attentions", "output_hidden_states", "return_dict",
        "cache_position",
    }

    def __init__(self, *args, encoder_lr=None, **kwargs):
        self.encoder_lr = encoder_lr
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, *args, **kwargs):
        # Strip keys Whisper doesn't accept (e.g. input_ids injected by peft)
        inputs = {
            k: v for k, v in inputs.items()
            if k in self._WHISPER_FORWARD_KEYS
        }
        return super().compute_loss(model, inputs, *args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if self.encoder_lr is None:
            # No differential LR requested — use default single-LR optimizer
            return super().create_optimizer()

        # Split trainable params into encoder vs decoder groups
        encoder_params = []
        decoder_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("model.encoder."):
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        # Create AdamW with two param groups at different learning rates
        optimizer_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.encoder_lr},
                {"params": decoder_params, "lr": self.args.learning_rate},
            ],
            **optimizer_kwargs,
        )
        print(f"  Optimizer param groups:")
        print(f"    Encoder: {len(encoder_params)} tensors, lr={self.encoder_lr}")
        print(f"    Decoder: {len(decoder_params)} tensors, lr={self.args.learning_rate}")
        return self.optimizer


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir
        os.makedirs(args.cache_dir, exist_ok=True)

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
    # 5. Load dataset and create splits
    # -----------------------------------------------------------------------
    local_path = Path(args.dataset_path)
    clips_dir = str(local_path / "clips")

    assert (local_path / args.all_tsv).exists(), \
        f"TSV not found: {local_path / args.all_tsv}"
    assert args.pct_validation + args.pct_test < 1.0, \
        f"pct_validation ({args.pct_validation}) + pct_test ({args.pct_test}) must be < 1.0"

    print(f"Loading dataset from: {local_path / args.all_tsv}")
    all_dataset = load_dataset(
        "csv",
        data_files=str(local_path / args.all_tsv),
        delimiter="\t",
        split="train",
    )
    print(f"Total samples in {args.all_tsv}: {len(all_dataset)}")

    # ---- Holdback exclusion ----
    holdback_dataset = None
    if args.holdback_tsv:
        holdback_path = local_path / args.holdback_tsv
        assert holdback_path.exists(), f"Holdback TSV not found: {holdback_path}"
        holdback_dataset = load_dataset(
            "csv",
            data_files=str(holdback_path),
            delimiter="\t",
            split="train",
        )
        holdback_paths = set(holdback_dataset["path"])
        before = len(all_dataset)
        all_dataset = all_dataset.filter(
            lambda x: x["path"] not in holdback_paths
        )
        print(f"Holdback: {len(holdback_paths)} samples, "
              f"removed {before - len(all_dataset)} from available pool")

    available_count = len(all_dataset)
    print(f"Available samples: {available_count}")

    # ---- Shuffle and split ----
    all_dataset = all_dataset.shuffle(seed=args.seed)

    n_val = int(available_count * args.pct_validation)
    n_test = int(available_count * args.pct_test)
    n_train = available_count - n_val - n_test

    assert n_train > 0, \
        f"No training samples left (val={n_val}, test={n_test}, total={available_count})"

    train_split = all_dataset.select(range(n_train))
    val_split = all_dataset.select(range(n_train, n_train + n_val))
    test_split = all_dataset.select(range(n_train + n_val, n_train + n_val + n_test))

    print(f"Split sizes — train: {n_train}, validation: {n_val}, test: {n_test}")

    # ---- Optionally write split TSVs ----
    if args.write_splits:
        import pandas as pd
        for name, ds in [("train_split", train_split),
                         ("validation_split", val_split),
                         ("test_split", test_split)]:
            df = ds.to_pandas()
            outpath = f"{name}.tsv"
            df.to_csv(outpath, sep="\t", index=False)
            print(f"Wrote {outpath} ({len(df)} rows)")

    # ---- Detect and normalize text column ----
    text_col = "sentence" if "sentence" in train_split.column_names else "text"
    keep_cols = {"path", text_col}
    cols_to_remove = [c for c in train_split.column_names if c not in keep_cols]

    if cols_to_remove:
        train_split = train_split.remove_columns(cols_to_remove)
        val_split = val_split.remove_columns(cols_to_remove)
        test_split = test_split.remove_columns(cols_to_remove)

    if text_col != "sentence":
        train_split = train_split.rename_column(text_col, "sentence")
        val_split = val_split.rename_column(text_col, "sentence")
        test_split = test_split.rename_column(text_col, "sentence")

    # Optionally subsample training set
    if args.max_train_samples and args.max_train_samples < len(train_split):
        train_split = train_split.shuffle(seed=args.seed).select(
            range(args.max_train_samples)
        )
        print(f"Subsampled train to {args.max_train_samples} samples")

    # -----------------------------------------------------------------------
    # 6. Preprocessing functions (torchaudio-based)
    # -----------------------------------------------------------------------
    # Cache resamplers by source sample rate to avoid recreating them
    resamplers = {}

    def _load_audio(path):
        """Load an audio file and resample to 16kHz if needed."""
        filepath = os.path.join(clips_dir, path)
        speech_array, sr = torchaudio.load(filepath)
        if sr != 16000:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
            speech_array = resamplers[sr](speech_array)
        return speech_array.squeeze().numpy()

    def prepare_dataset(batch):
        """Preprocess a single sample: load audio → mel spectrogram → tokenize text."""
        audio_np = _load_audio(batch["path"])
        batch["input_features"] = feature_extractor(
            audio_np, sampling_rate=16000,
        ).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    def prepare_dataset_batched(batch):
        """Batched version of prepare_dataset for faster disk-cached preprocessing."""
        audio_arrays = [_load_audio(p) for p in batch["path"]]
        batch["input_features"] = feature_extractor(
            audio_arrays, sampling_rate=16000,
        ).input_features
        batch["labels"] = [
            tokenizer(s).input_ids for s in batch["sentence"]
        ]
        return batch

    train_len = len(train_split)
    remove_cols = train_split.column_names  # ["path", "sentence"]

    # Resolve streaming flags
    # Train: streams by default, --no_streaming or --no_streaming_train disables
    # Eval: disk-cached by default, --streaming_eval enables streaming
    no_streaming_train = args.no_streaming or args.no_streaming_train
    no_streaming_eval = not args.streaming_eval or args.no_streaming

    # ---- Training data ----
    if no_streaming_train:
        print("Preprocessing training dataset (disk cache, batched)...")
        train_dataset = train_split.map(
            prepare_dataset_batched,
            remove_columns=remove_cols,
            batched=True,
            batch_size=16,
        )
        train_len = len(train_dataset)
        print(f"Train samples: {train_len}")
    else:
        print("Setting up streaming training dataset...")
        train_dataset = (
            train_split
            .to_iterable_dataset(num_shards=max(1, train_len // 5000))
            .map(prepare_dataset, remove_columns=remove_cols)
        )
        print(f"Train samples (approx): {train_len}")

    # ---- Validation data (used during training eval) ----
    if no_streaming_eval:
        print("Preprocessing validation dataset (disk cache, batched)...")
        eval_dataset = val_split.map(
            prepare_dataset_batched,
            remove_columns=remove_cols,
            batched=True,
            batch_size=16,
        )
    else:
        print("Preprocessing validation dataset...")
        eval_dataset = val_split.map(
            prepare_dataset,
            remove_columns=remove_cols,
            keep_in_memory=True,
        )

    print(f"Validation samples: {len(eval_dataset)}")

    if args.max_eval_samples and args.max_eval_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(
            range(args.max_eval_samples)
        )
        print(f"Subsampled validation to {args.max_eval_samples} samples")

    # ---- Test data (preprocessed upfront for final evaluation) ----
    print("Preprocessing test dataset (disk cache, batched)...")
    test_dataset = test_split.map(
        prepare_dataset_batched,
        remove_columns=remove_cols,
        batched=True,
        batch_size=16,
    )
    print(f"Test samples: {len(test_dataset)}")

    # ---- Holdback data (preprocessed for final evaluation if provided) ----
    holdback_eval_dataset = None
    if holdback_dataset is not None:
        print("Preprocessing holdback dataset (disk cache, batched)...")
        # Clean holdback columns to match
        hb_text_col = "sentence" if "sentence" in holdback_dataset.column_names else "text"
        hb_keep = {"path", hb_text_col}
        hb_remove = [c for c in holdback_dataset.column_names if c not in hb_keep]
        if hb_remove:
            holdback_dataset = holdback_dataset.remove_columns(hb_remove)
        if hb_text_col != "sentence":
            holdback_dataset = holdback_dataset.rename_column(hb_text_col, "sentence")

        holdback_eval_dataset = holdback_dataset.map(
            prepare_dataset_batched,
            remove_columns=holdback_dataset.column_names,
            batched=True,
            batch_size=16,
        )
        print(f"Holdback samples: {len(holdback_eval_dataset)}")

    # -----------------------------------------------------------------------
    # 7. Load model
    # -----------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        use_cache=False,              # incompatible with gradient checkpointing
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        encoder_layerdrop=args.encoder_layerdrop,
        decoder_layerdrop=args.decoder_layerdrop,
    )

    # Configure generation to produce the target language
    model.generation_config.language = args.language_full
    model.generation_config.task = "transcribe"
    # Set to None so the model uses language/task tokens from generation_config
    # instead of hardcoded forced_decoder_ids (which can conflict with training)
    model.generation_config.forced_decoder_ids = None

    # Enable gradient checkpointing to reduce VRAM usage (~30% savings).
    # use_reentrant=False is required for compatibility with partially frozen
    # models in newer PyTorch (avoids "backward through graph a second time" error)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # --- Encoder freeze/unfreeze ---
    # Three modes: fully frozen, partially unfrozen (last N layers), or fully unfrozen
    if args.unfreeze_encoder_layers is not None:
        # Freeze everything, then selectively unfreeze last N layers + final layer norm
        model.freeze_encoder()
        encoder_layers = model.model.encoder.layers
        total_encoder_layers = len(encoder_layers)
        n_unfreeze = min(args.unfreeze_encoder_layers, total_encoder_layers)
        for layer in encoder_layers[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in model.model.encoder.layer_norm.parameters():
            param.requires_grad = True
        print(f"Encoder: last {n_unfreeze}/{total_encoder_layers} layers UNFROZEN.")
        if args.encoder_lr and args.encoder_lr != args.lr:
            print(f"  Encoder LR: {args.encoder_lr}, Decoder LR: {args.lr}")
    elif args.freeze_encoder:
        model.freeze_encoder()
        print("Encoder is FROZEN — training decoder only.")
    else:
        for param in model.model.encoder.parameters():
            param.requires_grad = True
        print("Encoder is UNFROZEN — training full model.")
        if args.encoder_lr and args.encoder_lr != args.lr:
            print(f"  Encoder LR: {args.encoder_lr}, Decoder LR: {args.lr}")

    # --- Decoder partial freeze ---
    # Freeze lower decoder layers to reduce trainable params and stabilize training
    if args.freeze_decoder_layers is not None:
        decoder_layers = model.model.decoder.layers
        total_decoder_layers = len(decoder_layers)
        n_freeze = min(args.freeze_decoder_layers, total_decoder_layers)
        for layer in decoder_layers[:n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"Decoder: first {n_freeze}/{total_decoder_layers} layers FROZEN.")

    # -----------------------------------------------------------------------
    # 7b. Optionally apply LoRA
    # -----------------------------------------------------------------------
    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType

        target_modules = args.lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2",
        ]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        print(f"LoRA applied (r={args.lora_r}, alpha={args.lora_alpha}, "
              f"dropout={args.lora_dropout})")
        print(f"  Target modules: {target_modules}")
        model.print_trainable_parameters()
    else:
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
    # 9. Metrics — CER (Character Error Rate)
    # -----------------------------------------------------------------------
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # predict_with_generate may return (token_ids, scores) tuple
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Replace -100 (ignored label padding) with pad token so the
        # tokenizer can decode them without errors
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        pred_ids = np.where(pred_ids == -100, tokenizer.pad_token_id, pred_ids)

        # Decode token IDs back to text strings
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Filter out empty references to avoid division by zero in CER
        pairs = [(p, l) for p, l in zip(pred_str, label_str) if len(l.strip()) > 0]
        if not pairs:
            return {"cer": 1.0}
        pred_str, label_str = zip(*pairs)

        # Compute Character Error Rate
        cer = cer_metric.compute(predictions=list(pred_str), references=list(label_str))

        # Log sample predictions for debugging
        for i in range(min(5, len(pred_str))):
            print(f"  REF: {label_str[i][:80]}")
            print(f"  HYP: {pred_str[i][:80]}")
            print()

        return {"cer": cer}

    # -----------------------------------------------------------------------
    # 10. Training arguments
    # -----------------------------------------------------------------------
    # Disk-cached datasets know their length so we can use num_train_epochs.
    # Streaming datasets don't, so we compute max_steps manually.
    if no_streaming_train:
        epoch_or_steps_args = {"num_train_epochs": args.epochs}
    else:
        steps_per_epoch = train_len // (args.train_batch_size * args.grad_accum)
        max_steps = steps_per_epoch * args.epochs
        epoch_or_steps_args = {"max_steps": max_steps}
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Max steps: {max_steps}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,

        # --- Batch size and gradient accumulation ---
        # Effective batch = train_batch_size * grad_accum (per GPU)
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # --- Learning rate schedule ---
        # Cosine decay with linear warmup; LR ramps up over warmup_steps
        # then decays following a cosine curve to near-zero
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,
        **epoch_or_steps_args,

        # --- Precision ---
        # bf16 is preferred on Ampere+ GPUs (A100, 3090, 4090);
        # fp16 can cause loss spikes with Whisper due to dynamic range issues
        fp16=False,
        bf16=torch.cuda.is_available(),

        # --- Evaluation ---
        # Run eval every eval_steps; eval_accumulation_steps controls how many
        # batches of predictions to accumulate on GPU before offloading to CPU.
        # Higher = faster eval, but uses more GPU memory during generation.
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        # predict_with_generate runs autoregressive decoding for CER metrics
        # (slower than forward-only eval but required for text-based metrics)
        predict_with_generate=True,
        generation_max_length=225,

        # --- Logging ---
        logging_strategy="steps",
        logging_steps=100,
        report_to=["tensorboard"],

        # --- Checkpointing ---
        # Keep the 3 most recent checkpoints; load the best one (by CER) at end
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,  # lower CER = better

        # --- Performance ---
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Don't auto-remove columns — our collator handles the batch format
        remove_unused_columns=False,

        # --- Gradient clipping ---
        # Prevents gradient explosions that cause loss spikes / hallucination
        max_grad_norm=1.0,

        # --- Labels ---
        # Explicitly set so Trainer doesn't infer input_ids for peft models
        label_names=["labels"],
    )

    # -----------------------------------------------------------------------
    # 11. Trainer
    # -----------------------------------------------------------------------
    # Enable differential LR only when encoder is unfrozen with a separate LR
    # and LoRA is not active (peft manages its own trainable params)
    encoder_lr = None
    if not args.lora and not args.freeze_encoder and args.encoder_lr:
        encoder_lr = args.encoder_lr

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = DifferentialLRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        encoder_lr=encoder_lr,
        callbacks=callbacks,
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

    if args.lora and args.lora_merge_on_save:
        # Merge LoRA adapter weights back into the base model so the saved
        # model is standalone and doesn't require peft to load
        print("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(final_dir)
    else:
        trainer.save_model(final_dir)

    # Save tokenizer and processor alongside the model for easy loading
    processor.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # -----------------------------------------------------------------------
    # 14. Final evaluation on all splits
    # -----------------------------------------------------------------------
    # After training, evaluate the best model on validation, test, and
    # optionally holdback splits. This gives a complete picture of model
    # performance on seen (validation) vs unseen (test/holdback) data.
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    def evaluate_split(dataset, split_name):
        """Run evaluation on a dataset split and print CER + loss."""
        print(f"\nEvaluating on {split_name} ({len(dataset)} samples)...")
        metrics = trainer.evaluate(
            eval_dataset=dataset, metric_key_prefix=split_name
        )
        cer = metrics.get(f"{split_name}_cer", None)
        loss = metrics.get(f"{split_name}_loss", None)
        if cer is not None:
            print(f"  {split_name} CER:  {cer:.4f}")
        if loss is not None:
            print(f"  {split_name} Loss: {loss:.4f}")
        return metrics

    val_metrics = evaluate_split(eval_dataset, "validation")
    test_metrics = evaluate_split(test_dataset, "test")

    holdback_metrics = None
    if holdback_eval_dataset is not None:
        holdback_metrics = evaluate_split(holdback_eval_dataset, "holdback")

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
        cer = m.get(f"{prefix}_cer", "N/A")
        if isinstance(cer, float):
            print(f"  {name:12s} CER: {cer:.4f}")
        else:
            print(f"  {name:12s} CER: {cer}")
    print("=" * 60)
    print("\nDone!")


if __name__ == "__main__":
    main()
