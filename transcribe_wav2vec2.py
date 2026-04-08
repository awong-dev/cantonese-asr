#!/usr/bin/env python3
"""
Transcribe an audio file using a fine-tuned wav2vec2 model.

Handles long audio by splitting into chunks (default 20s) with overlap
to avoid cutting words at boundaries. Each chunk is processed independently
and decoded to text.

Usage:
    python transcribe_wav2vec2.py --model ./wav2vec2-cantonese/final --audio recording.mp3
    python transcribe_wav2vec2.py --model ./wav2vec2-cantonese/final --audio long.wav --chunk_seconds 15
"""

import argparse

import numpy as np
import torch
from torchcodec.decoders import AudioDecoder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SAMPLE_RATE = 16000


def transcribe(model_path, audio_path, chunk_seconds=20, overlap_seconds=2):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available()
    model.to(device)

    # Load and resample audio to 16kHz mono
    decoder = AudioDecoder(audio_path, sample_rate=SAMPLE_RATE, num_channels=1)
    audio_np = decoder.get_all_samples().data.squeeze().numpy()

    total_samples = len(audio_np)
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    overlap_samples = int(overlap_seconds * SAMPLE_RATE)
    stride = chunk_samples - overlap_samples

    # Split into overlapping chunks
    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
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
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(
                        input_values, attention_mask=attention_mask,
                    ).logits
            else:
                logits = model(
                    input_values, attention_mask=attention_mask,
                ).logits

        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0].strip()
        del logits

        if text:
            texts.append(text)

        if num_chunks > 1:
            print(f"  chunk {i + 1}/{num_chunks} done")

    return " ".join(texts)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with a fine-tuned wav2vec2 model"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument(
        "--chunk_seconds", type=float, default=20,
        help="Chunk size in seconds (default: 20)",
    )
    parser.add_argument(
        "--overlap", type=float, default=2,
        help="Overlap between chunks in seconds (default: 2)",
    )
    args = parser.parse_args()

    text = transcribe(args.model, args.audio, chunk_seconds=args.chunk_seconds,
                      overlap_seconds=args.overlap)
    print(text)


if __name__ == "__main__":
    main()
