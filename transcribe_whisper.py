#!/usr/bin/env python3
"""
Transcribe an audio file using a fine-tuned Whisper model.

Usage:
    python transcribe_whisper.py --model ./whisper-finetuned/final --audio recording.mp3
    python transcribe_whisper.py --model ./whisper-finetuned/final --audio recording.wav --language cantonese
"""

import argparse

import torch
from torchcodec.decoders import AudioDecoder
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def transcribe(model_path, audio_path, language="cantonese", task="transcribe"):
    processor = WhisperProcessor.from_pretrained(
        model_path, language=language, task=task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and resample audio to 16kHz mono
    decoder = AudioDecoder(audio_path, sample_rate=16000, num_channels=1)
    audio_np = decoder.get_all_samples().data.squeeze(0).numpy()

    # Extract mel features
    inputs = processor.feature_extractor(
        audio_np, sampling_rate=16000, return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    # Generate tokens
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted_ids = model.generate(input_features)
        else:
            predicted_ids = model.generate(input_features)

    # Decode to text
    text = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True,
    )[0]
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with a fine-tuned Whisper model"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--language", type=str, default="cantonese")
    args = parser.parse_args()

    text = transcribe(args.model, args.audio, language=args.language)
    print(text)


if __name__ == "__main__":
    main()
