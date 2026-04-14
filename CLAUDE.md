# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuning wav2vec2 and Whisper models for Cantonese (yue) automatic speech recognition using Mozilla Common Voice datasets. Training targets CUDA GPUs with bf16 support.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ required. The venv is at `./venv/`.

## Key Commands

### Training
```bash
# Whisper (uses Seq2SeqTrainer)
python3 train_whisper.py --dataset_path data/cv-corpus-25.0-2026-03-09/yue --model openai/whisper-large-v3

# wav2vec2 (uses Trainer with CTC loss)
python3 train_wav2vec2.py --dataset_path data/cv-corpus-25.0-2026-03-09/yue --model facebook/wav2vec2-xls-r-1b
```

### Evaluation
```bash
python3 eval_whisper.py --model_path ./output/final --dataset_path data/cv-corpus-25.0-2026-03-09/yue
python3 eval_wav2vec2.py --model_path ./output/final --dataset_path data/cv-corpus-25.0-2026-03-09/yue
```

### Data Download
```bash
python3 download_dataset.py --token YOUR_API_TOKEN --languages yue zh-HK
```

### Upload to HuggingFace
```bash
python3 upload_model.py --model_path ./output/final --repo_name my-model --base_model openai/whisper-large-v3
```

## Architecture

### Two parallel training pipelines

Both pipelines share the same dataset handling, LR scheduling, and CER evaluation code, but use different model architectures and HuggingFace Trainer classes:

- **Whisper** (`train_whisper.py`): Encoder-decoder model using `Seq2SeqTrainer`. Supports `--freeze_encoder`, `--encoder_lr` (separate encoder/decoder LR), and LoRA via `--lora` flag.
- **wav2vec2** (`train_wav2vec2.py`): CTC-based model using `Trainer`. Supports `--unfreeze` for the feature encoder and `--compile` for torch.compile.

### Shared modules

- `create_splits.py` — Deterministic train/val/test splitting from Common Voice TSVs. Supports multiple dataset paths (comma-separated `--dataset_path`) with per-fileset holdback. Both training scripts call `create_splits()` internally.
- `cer_utils.py` — CER computation using jiwer. Provides both raw CER and punctuation-normalized CER (`cer_nopunct`). The `--nopunct_in_eval` flag makes nopunct CER the metric for checkpoint selection.
- `lr_schedule.py` — Tri-stage LR schedule (warmup/hold/decay) modeled after fairseq. Used via `--lr_schedule tri_stage`. Also provides `TriStageCheckpointCallback` to save model state before the decay phase.

### Audio loading

Both trainers use `torchcodec.decoders.AudioDecoder` for audio loading (not torchaudio/soundfile), resampling to 16kHz.

### Dataset format

Expects Mozilla Common Voice layout: a directory containing `clips/` (MP3 audio files) and TSV metadata files (`validated.tsv`, `test.tsv`, etc.). Dataset directories go under `data/`.
