#!/usr/bin/env python3
"""
Fine-tune Whisper on a local Common Voice dataset with custom train/val/test splits.

Usage:
    python train_whisper.py --dataset_path cv-corpus-25.0-2026-03-09/yue
    python train_whisper.py --dataset_path /data/yue --holdback_tsv test.tsv --lr 1e-5
    python train_whisper.py --dataset_path /data/yue --write_splits --pct_test 0.05

Requirements:
    pip install transformers datasets evaluate jiwer accelerate torchcodec
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import numpy as np
from torchcodec.decoders import AudioDecoder
from transformers import (
    EarlyStoppingCallback,
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

    # Dataset — supports comma-separated values for multiple filesets
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Comma-separated paths to dataset directories (each with clips/ and TSVs)",
    )
    parser.add_argument(
        "--all_tsv",
        type=str,
        default="validated.tsv",
        help="Comma-separated TSV filenames with all usable samples (default: validated.tsv)",
    )
    parser.add_argument(
        "--holdback_tsv",
        type=str,
        default="",
        help="Comma-separated holdback TSV filenames to exclude (default: none)",
    )
    parser.add_argument(
        "--validation_tsv",
        type=str,
        default=None,
        help="Comma-separated explicit validation TSV filenames. "
             "When set, pct_validation is ignored.",
    )
    parser.add_argument(
        "--test_tsv",
        type=str,
        default=None,
        help="Comma-separated explicit test TSV filenames. "
             "When set, pct_test is ignored.",
    )
    parser.add_argument(
        "--pct_validation",
        type=float,
        default=0.1,
        help="Fraction of pooled samples for validation split (default: 0.1)",
    )
    parser.add_argument(
        "--pct_test",
        type=float,
        default=0.1,
        help="Fraction per fileset for test split (default: 0.1)",
    )
    parser.add_argument(
        "--write_splits", action="store_true",
        help="Write train/validation/test split TSVs to the current directory",
    )
    parser.add_argument(
        "--dataset_ratio", type=str, default=None,
        help="Colon-separated ratios for dataset sampling, e.g. '2:1' means "
             "twice as many samples from the first dataset. Default: use all samples.",
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
    parser.add_argument(
        "--layerwise_lr_decay", type=float, default=None,
        help="When set, applies layerwise learning rate decay to decoder layers. "
             "Layer 0 gets lr * decay^(N-1), top layer gets lr. "
             "Typical values: 0.8-0.95. When None, uses flat LR for all decoder layers.",
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
        "--lora_modules", type=str, nargs="+", default=None,
        dest="lora_target_modules",
        help="Modules to apply LoRA to (default: all attention + FFN projections)",
    )
    parser.add_argument(
        "--lora_target", type=str, default="both",
        choices=["encoder", "decoder", "both"],
        help="Apply LoRA to encoder, decoder, or both (default: both)",
    )
    parser.add_argument(
        "--lora_merge_on_save", action="store_true",
        help="Merge LoRA weights into base model when saving.",
    )

    # Performance
    parser.add_argument(
        "--no_gradient_checkpointing", action="store_true",
        help="Disable gradient checkpointing. Faster training but uses more VRAM. "
             "Recommended with LoRA where most parameters are frozen.",
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

    # LR schedule (cosine or tri_stage)
    from lr_schedule import add_lr_schedule_args
    add_lr_schedule_args(parser)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument(
        "--eval_accumulation_steps", type=int, default=32,
        help="Number of eval batches to accumulate before offloading predictions "
             "to CPU. Higher = faster eval but more GPU memory. (default: 32)",
    )
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument(
        "--output_dir", type=str, default="./whisper-finetuned",
    )
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument(
        "--save_pre_decay", action="store_true",
        help="Save a checkpoint before the decay phase in tri-stage LR schedule.",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=0,
        help="0 = disabled (default).",
    )
    parser.add_argument(
        "--eval_base", action="store_true",
        help="Evaluate the base model before training to establish a baseline.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nopunct_in_eval", action="store_true",
        help="Compute an additional normalized CER alongside raw CER. "
             "Normalization uses a jiwer pipeline that removes punctuation, "
             "lowercases, and normalizes whitespace before computing CER.",
    )
    parser.add_argument(
        "--best_model_metric", type=str, default=None,
        choices=["cer_raw", "cer_nopunct"],
        help="Metric for checkpoint selection. Defaults to cer_nopunct when "
             "--nopunct_in_eval is set, cer_raw otherwise.",
    )

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

    def __init__(self, *args, encoder_lr=None, processor=None, tokenizer=None,
                 tri_stage_args=None, **kwargs):
        self.encoder_lr = encoder_lr
        self._whisper_processor = processor
        self._whisper_tokenizer = tokenizer
        self._tri_stage_args = tri_stage_args
        super().__init__(*args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        """Override to also save processor and tokenizer into every checkpoint."""
        super()._save(output_dir=output_dir, state_dict=state_dict)
        # Save processor and tokenizer so checkpoints are self-contained
        if output_dir and self._whisper_processor is not None:
            self._whisper_processor.save_pretrained(output_dir)
        if output_dir and self._whisper_tokenizer is not None:
            self._whisper_tokenizer.save_pretrained(output_dir)

    def prediction_step(self, model, inputs, prediction_loss_only, **kwargs):
        # Peft's generate wrapper passes labels through to Whisper's generate()
        # which doesn't accept them. Temporarily patch generate to strip labels.
        import peft
        if isinstance(model, peft.PeftModel):
            _orig_generate = model.generate
            def _patched_generate(**gen_kwargs):
                gen_kwargs.pop("labels", None)
                return _orig_generate(**gen_kwargs)
            model.generate = _patched_generate
            result = super().prediction_step(model, inputs, prediction_loss_only, **kwargs)
            model.generate = _orig_generate
            return result
        return super().prediction_step(model, inputs, prediction_loss_only, **kwargs)

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # Strip keys Whisper doesn't accept (e.g. input_ids injected by peft)
        inputs = {
            k: v for k, v in inputs.items()
            if k in self._WHISPER_FORWARD_KEYS
        }
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

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

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self._tri_stage_args is not None:
            from lr_schedule import get_tri_stage_schedule
            self.lr_scheduler = get_tri_stage_schedule(
                optimizer if optimizer is not None else self.optimizer,
                **self._tri_stage_args,
            )
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)


class LayerwiseLRTrainer(DifferentialLRTrainer):
    """DifferentialLRTrainer with layerwise LR decay for decoder layers."""

    def __init__(self, *args, lr_decay=0.9, **kwargs):
        self._lr_decay = lr_decay
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        from layerwise_lr import create_layerwise_optimizer
        self.optimizer = create_layerwise_optimizer(
            model=self.model,
            base_lr=self.args.learning_rate,
            lr_decay=self._lr_decay,
            encoder_lr=self.encoder_lr,
            weight_decay=self.args.weight_decay,
            adam_beta1=self.args.adam_beta1,
            adam_beta2=self.args.adam_beta2,
            adam_epsilon=self.args.adam_epsilon,
        )
        return self.optimizer


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Guard against accidentally overwriting a previous run
    if os.path.isdir(args.output_dir) and not args.resume_from_checkpoint:
        contents = os.listdir(args.output_dir)
        if any(c.startswith("checkpoint-") for c in contents):
            raise SystemExit(
                f"Error: output_dir '{args.output_dir}' already contains checkpoints. "
                f"Use --resume_from_checkpoint to continue, or choose a different --output_dir."
            )

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
    from create_splits import create_splits

    splits = create_splits(
        dataset_paths=args.dataset_path,
        all_tsvs=args.all_tsv,
        holdback_tsvs=args.holdback_tsv,
        validation_tsvs=args.validation_tsv,
        test_tsvs=args.test_tsv,
        pct_validation=args.pct_validation,
        pct_test=args.pct_test,
        seed=args.seed,
        write_splits=args.write_splits,
        dataset_ratio=args.dataset_ratio,
    )

    train_split = splits["train"]
    val_split = splits["validation"]

    # Optionally subsample training set
    if args.max_train_samples and args.max_train_samples < len(train_split):
        train_split = train_split.shuffle(seed=args.seed).select(
            range(args.max_train_samples)
        )
        print(f"Subsampled train to {args.max_train_samples} samples")

    # -----------------------------------------------------------------------
    # 6. Preprocessing functions (torchcodec-based)
    # -----------------------------------------------------------------------
    # Each sample has a "clips_dir" column indicating where its audio lives.
    # This supports multiple filesets with different clips/ directories.
    def _load_audio(clips_dir, path):
        """Load an audio file and resample to 16kHz using torchcodec."""
        filepath = os.path.join(clips_dir, path)
        decoder = AudioDecoder(filepath, sample_rate=16000, num_channels=1)
        samples = decoder.get_all_samples()
        return samples.data.squeeze(0).numpy()

    def prepare_dataset(batch):
        """Preprocess a single sample: load audio → mel spectrogram → tokenize text."""
        audio_np = _load_audio(batch["clips_dir"], batch["path"])
        batch["input_features"] = feature_extractor(
            audio_np, sampling_rate=16000,
        ).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    def prepare_dataset_batched(batch):
        """Batched version of prepare_dataset for faster disk-cached preprocessing."""
        audio_arrays = [_load_audio(cd, p)
                        for cd, p in zip(batch["clips_dir"], batch["path"])]
        batch["input_features"] = feature_extractor(
            audio_arrays, sampling_rate=16000,
        ).input_features
        batch["labels"] = [
            tokenizer(s).input_ids for s in batch["sentence"]
        ]
        return batch

    train_len = len(train_split)
    remove_cols = train_split.column_names  # ["clips_dir", "path", "sentence"]

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

    # Gradient checkpointing trades compute for VRAM (~30% savings but slower).
    # Disable with --no_gradient_checkpointing (recommended for LoRA).
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing: enabled")
    else:
        print("Gradient checkpointing: disabled")

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

        if args.lora_target_modules:
            target_modules = args.lora_target_modules
        else:
            base_modules = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
            if args.lora_target == "both":
                target_modules = sorted(base_modules)
            else:
                # Enumerate matching module names from the model directly
                prefix = f"model.{args.lora_target}."
                target_modules = sorted(
                    name for name, _ in model.named_modules()
                    if name.startswith(prefix) and name.split(".")[-1] in base_modules
                )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)

        # Patch: peft's PeftModelForSeq2SeqLM.forward() injects kwargs like
        # input_ids and inputs_embeds that Whisper's forward() doesn't accept.
        # Filter to only the keys Whisper actually supports.
        _orig_forward = model.base_model.model.forward
        _valid_keys = DifferentialLRTrainer._WHISPER_FORWARD_KEYS
        def _patched_forward(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k in _valid_keys}
            return _orig_forward(*args, **kwargs)
        model.base_model.model.forward = _patched_forward

        print(f"LoRA applied (r={args.lora_r}, alpha={args.lora_alpha}, "
              f"dropout={args.lora_dropout}, target={args.lora_target})")
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
    from cer_utils import build_cer_transform, compute_cer, print_examples

    _cer_transform = None
    if args.nopunct_in_eval:
        _cer_transform = build_cer_transform()
        print("Normalized CER enabled (RemovePunctuation + ToLowerCase + whitespace normalization)")

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

        # Compute CER (raw + optionally nopunct)
        result, filtered_preds, filtered_refs = compute_cer(
            pred_str, label_str, cer_transform=_cer_transform
        )
        print_examples(filtered_preds, filtered_refs)
        return result

    # -----------------------------------------------------------------------
    # 10. Training arguments
    # -----------------------------------------------------------------------
    # Disk-cached datasets know their length so we can use num_train_epochs.
    # Streaming datasets don't, so we compute max_steps manually.
    steps_per_epoch = train_len // (args.train_batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    if no_streaming_train:
        epoch_or_steps_args = {"num_train_epochs": args.epochs}
    else:
        epoch_or_steps_args = {"max_steps": total_steps}
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Max steps: {total_steps}")

    # Resolve LR schedule (cosine vs tri_stage)
    from lr_schedule import resolve_lr_schedule_args
    hf_scheduler_type, hf_warmup_steps, tri_stage_args = resolve_lr_schedule_args(args, total_steps)

    best_model_metric = args.best_model_metric
    if best_model_metric is None:
        best_model_metric = "cer_nopunct" if args.nopunct_in_eval else "cer_raw"
    print(f"Checkpoint selection metric: {best_model_metric}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,

        # --- Batch size and gradient accumulation ---
        # Effective batch = train_batch_size * grad_accum (per GPU)
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # --- Learning rate schedule ---
        learning_rate=args.lr,
        lr_scheduler_type=hf_scheduler_type,
        warmup_steps=hf_warmup_steps,
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
        # Keep the 3 most recent checkpoints; load the best one (by CER) at end.
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=best_model_metric,
        greater_is_better=False,  # lower CER = better

        # --- Performance ---
        dataloader_num_workers=args.dataloader_num_workers,
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
    encoder_lr = args.encoder_lr if (args.encoder_lr and not args.freeze_encoder) else None

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )
    tri_stage_cb = None
    if tri_stage_args is not None and args.save_pre_decay:
        from lr_schedule import TriStageCheckpointCallback
        tri_stage_cb = TriStageCheckpointCallback(
            num_training_steps=tri_stage_args["num_training_steps"],
            warmup_pct=tri_stage_args["warmup_pct"],
            hold_pct=tri_stage_args["hold_pct"],
        )
        callbacks.append(tri_stage_cb)

    TrainerClass = DifferentialLRTrainer
    extra_trainer_kwargs = {}
    if args.layerwise_lr_decay is not None:
        TrainerClass = LayerwiseLRTrainer
        extra_trainer_kwargs["lr_decay"] = args.layerwise_lr_decay

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        encoder_lr=encoder_lr,
        processor=processor,
        tokenizer=tokenizer,
        callbacks=callbacks,
        tri_stage_args=tri_stage_args,
        **extra_trainer_kwargs,
    )
    if tri_stage_cb is not None:
        tri_stage_cb.trainer = trainer

    # -----------------------------------------------------------------------
    # 12. Train
    # -----------------------------------------------------------------------
    if args.eval_base:
        print("Evaluating base model before training...")
        from eval_whisper import run_evaluation as run_eval_base
        run_eval_base(
            model_path=args.model,
            dataset_path=args.dataset_path,
            all_tsv=args.all_tsv,
            holdback_tsv=args.holdback_tsv if args.holdback_tsv else None,
            validation_tsv=args.validation_tsv,
            test_tsv=args.test_tsv,
            pct_validation=args.pct_validation,
            pct_test=args.pct_test,
            seed=args.seed,
            eval_test=True,
            eval_holdback=bool(args.holdback_tsv),
            language_full=args.language_full,
            eval_batch_size=args.eval_batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
            nopunct_in_eval=args.nopunct_in_eval,
        )

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
    # Use the standalone eval script to evaluate the saved model.
    # Pass the split parameters so it can regenerate the same splits.
    from eval_whisper import run_evaluation

    run_evaluation(
        model_path=final_dir,
        dataset_path=args.dataset_path,
        all_tsv=args.all_tsv,
        holdback_tsv=args.holdback_tsv if args.holdback_tsv else None,
        validation_tsv=args.validation_tsv,
        test_tsv=args.test_tsv,
        pct_validation=args.pct_validation,
        pct_test=args.pct_test,
        seed=args.seed,
        eval_test=True,
        eval_holdback=bool(args.holdback_tsv),
        language_full=args.language_full,
        eval_batch_size=args.eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        nopunct_in_eval=args.nopunct_in_eval,
        results_json=os.path.join(args.output_dir, "eval_results.json"),
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
