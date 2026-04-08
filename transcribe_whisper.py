#!/usr/bin/env python3
"""
Transcribe an audio file using a fine-tuned Whisper model.

Handles long audio by splitting into 30-second chunks (Whisper's native
window size) with a configurable overlap to avoid cutting words at
chunk boundaries.

Usage:
    python transcribe_whisper.py --model ./whisper-finetuned/final --audio recording.mp3
    python transcribe_whisper.py --model ./whisper-finetuned/final --audio recording.wav --language cantonese
"""

import argparse

import numpy as np
import torch
from torchcodec.decoders import AudioDecoder
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SAMPLE_RATE = 16000
# Whisper's native input window is 30 seconds
CHUNK_SECONDS = 30
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE


def transcribe(model_path, audio_path, language="cantonese", task="transcribe",
               overlap_seconds=2):
    processor = WhisperProcessor.from_pretrained(
        model_path, language=language, task=task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available()
    model.to(device)

    # Load and resample audio to 16kHz mono
    decoder = AudioDecoder(audio_path, sample_rate=SAMPLE_RATE, num_channels=1)
    audio_np = decoder.get_all_samples().data.squeeze(0).numpy()

    total_samples = len(audio_np)
    overlap_samples = int(overlap_seconds * SAMPLE_RATE)
    stride = CHUNK_SAMPLES - overlap_samples

    # Split into overlapping chunks
    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + CHUNK_SAMPLES, total_samples)
        chunks.append(audio_np[start:end])
        if end >= total_samples:
            break
        start += stride

    num_chunks = len(chunks)
    if num_chunks > 1:
        duration = total_samples / SAMPLE_RATE
        print(f"Audio duration: {duration:.1f}s, processing in {num_chunks} chunks")

    texts = []
    for i, chunk in enumerate(chunks):
        inputs = processor.feature_extractor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted_ids = model.generate(input_features)
            else:
                predicted_ids = model.generate(input_features)

        text = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True,
        )[0].strip()

        if text:
            texts.append(text)

        if num_chunks > 1:
            print(f"  chunk {i + 1}/{num_chunks} done")

    return "".join(texts)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with a fine-tuned Whisper model"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--language", type=str, default="cantonese")
    parser.add_argument(
        "--overlap", type=float, default=2,
        help="Overlap between chunks in seconds (default: 2)",
    )
    args = parser.parse_args()

    text = transcribe(args.model, args.audio, language=args.language,
                      overlap_seconds=args.overlap)
    print(text)


if __name__ == "__main__":
    main()
