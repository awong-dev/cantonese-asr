#!/usr/bin/env python3
"""
Transcribe an audio file using a fine-tuned wav2vec2 model.

Usage:
    python transcribe_wav2vec2.py --model ./wav2vec2-cantonese/final --audio recording.mp3
"""

import argparse

import torch
from torchcodec.decoders import AudioDecoder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def transcribe(model_path, audio_path):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and resample audio to 16kHz mono
    decoder = AudioDecoder(audio_path, sample_rate=16000, num_channels=1)
    audio_np = decoder.get_all_samples().data.squeeze().numpy()

    # Extract features
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Forward pass and decode
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    input_values, attention_mask=attention_mask,
                ).logits
        else:
            logits = model(
                input_values, attention_mask=attention_mask,
            ).logits

    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with a fine-tuned wav2vec2 model"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    args = parser.parse_args()

    text = transcribe(args.model, args.audio)
    print(text)


if __name__ == "__main__":
    main()
