#!/usr/bin/env python3
"""
Upload a trained model to HuggingFace Hub with a model card.

Generates a README.md with:
  - Base model metadata
  - Training stats (CER, loss) from trainer_state.json
  - Dataset and language info
  - Usage examples

Usage:
    python upload_model.py --model_path ./whisper-large-v3-turbo-yue/final \
        --repo_name whisper-large-v3-turbo-cantonese-yue \
        --base_model openai/whisper-large-v3-turbo \
        --cer 0.123

    python upload_model.py --model_path ./wav2vec2-xls-r-1b-cantonese-yue/checkpoint-15000 \
        --repo_name wav2vec2-xls-r-1b-cantonese-yue \
        --base_model facebook/wav2vec2-xls-r-1b \
        --trainer_state ./wav2vec2-xls-r-1b-cantonese-yue/checkpoint-15000/trainer_state.json

Requirements:
    pip install huggingface_hub transformers
    huggingface-cli login  (or set HF_TOKEN env var)
"""

import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def parse_args():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model directory (checkpoint or final)",
    )
    parser.add_argument(
        "--repo_name", type=str, required=True,
        help="Repository name (e.g. whisper-large-v3-turbo-cantonese-yue)",
    )
    parser.add_argument(
        "--username", type=str, default="awong-dev",
        help="HuggingFace username",
    )
    parser.add_argument(
        "--base_model", type=str, default=None,
        help="Base model ID (e.g. openai/whisper-large-v3-turbo, facebook/wav2vec2-xls-r-1b)",
    )
    parser.add_argument(
        "--dataset", type=str, default="mozilla-foundation/common_voice_17_0",
        help="Training dataset ID",
    )
    parser.add_argument(
        "--language", type=str, default="yue",
        help="Language code",
    )
    parser.add_argument(
        "--language_name", type=str, default="Cantonese",
        help="Language display name",
    )
    parser.add_argument(
        "--cer", type=float, default=None,
        help="Best CER to report (overrides auto-detection from trainer_state)",
    )
    parser.add_argument(
        "--trainer_state", type=str, default=None,
        help="Path to trainer_state.json (auto-detected from model_path if not set)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--commit_message", type=str, default="Upload fine-tuned model",
        help="Commit message",
    )
    return parser.parse_args()


def find_trainer_state(model_path: Path) -> Path | None:
    """Search for trainer_state.json in model_path and parent directories."""
    # Check in the model directory itself
    candidate = model_path / "trainer_state.json"
    if candidate.exists():
        return candidate

    # Check parent (e.g. if model_path is .../final, check .../trainer_state.json)
    candidate = model_path.parent / "trainer_state.json"
    if candidate.exists():
        return candidate

    # Check sibling checkpoint dirs (find the latest one)
    parent = model_path.parent
    checkpoints = sorted(parent.glob("checkpoint-*/trainer_state.json"))
    if checkpoints:
        return checkpoints[-1]

    return None


def parse_training_stats(trainer_state_path: Path) -> dict:
    """Extract training stats from trainer_state.json."""
    with open(trainer_state_path) as f:
        state = json.load(f)

    stats = {
        "total_steps": state.get("global_step", None),
        "best_metric": state.get("best_metric", None),
        "best_model_checkpoint": state.get("best_model_checkpoint", None),
        "epoch": None,
        "eval_history": [],
    }

    # Extract eval history from log_history
    for entry in state.get("log_history", []):
        if "eval_cer" in entry:
            stats["eval_history"].append({
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "eval_loss": entry.get("eval_loss"),
                "eval_cer": entry.get("eval_cer"),
            })
        if "epoch" in entry:
            stats["epoch"] = entry["epoch"]

    # Find best CER from eval history
    if stats["eval_history"]:
        best_eval = min(stats["eval_history"], key=lambda x: x["eval_cer"])
        stats["best_cer"] = best_eval["eval_cer"]
        stats["best_cer_step"] = best_eval["step"]
        stats["best_cer_epoch"] = best_eval["epoch"]
        stats["best_eval_loss"] = best_eval["eval_loss"]
    else:
        stats["best_cer"] = stats["best_metric"]

    return stats


def detect_model_type(model_path: Path) -> str:
    """Detect if this is a Whisper or wav2vec2 model."""
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type", "")
        if "whisper" in model_type:
            return "whisper"
        if "wav2vec2" in model_type:
            return "wav2vec2"
    return "unknown"


def generate_model_card(args, stats: dict | None, model_type: str) -> str:
    """Generate a HuggingFace model card (README.md)."""
    repo_id = f"{args.username}/{args.repo_name}"

    # Determine best CER
    best_cer = args.cer
    if best_cer is None and stats:
        best_cer = stats.get("best_cer")

    # YAML frontmatter
    tags = ["automatic-speech-recognition", args.language, args.language_name.lower()]
    if model_type == "whisper":
        tags.append("whisper")
    elif model_type == "wav2vec2":
        tags.extend(["wav2vec2", "CTC"])

    yaml_lines = [
        "---",
        f"language: {args.language}",
        "license: apache-2.0",
        "tags:",
    ]
    for tag in tags:
        yaml_lines.append(f"  - {tag}")

    if args.base_model:
        yaml_lines.append(f"base_model: {args.base_model}")

    yaml_lines.extend([
        "pipeline_tag: automatic-speech-recognition",
        "datasets:",
        f"  - {args.dataset}",
    ])

    if best_cer is not None:
        yaml_lines.extend([
            "model-index:",
            f"  - name: {args.repo_name}",
            "    results:",
            "      - task:",
            "          type: automatic-speech-recognition",
            "          name: Speech Recognition",
            "        dataset:",
            f"          name: Common Voice ({args.language_name})",
            f"          type: {args.dataset}",
            f"          config: {args.language}",
            "          split: test",
            "        metrics:",
            "          - type: cer",
            f"            value: {best_cer:.4f}",
            "            name: CER",
        ])

    yaml_lines.append("---")

    # Markdown body
    md_lines = [
        "",
        f"# {args.repo_name}",
        "",
    ]

    if args.base_model:
        md_lines.append(
            f"Fine-tuned [{args.base_model}](https://huggingface.co/{args.base_model}) "
            f"for {args.language_name} ({args.language}) speech recognition "
            f"on [Common Voice](https://huggingface.co/datasets/{args.dataset})."
        )
    else:
        md_lines.append(
            f"Fine-tuned model for {args.language_name} ({args.language}) speech recognition."
        )

    # Results
    if best_cer is not None:
        md_lines.extend([
            "",
            "## Evaluation Results",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **CER** | **{best_cer:.2%}** |",
        ])
        if stats and stats.get("best_eval_loss"):
            md_lines.append(f"| Eval Loss | {stats['best_eval_loss']:.4f} |")
        if stats and stats.get("best_cer_step"):
            md_lines.append(f"| Best Step | {stats['best_cer_step']} |")
        if stats and stats.get("best_cer_epoch"):
            md_lines.append(f"| Best Epoch | {stats['best_cer_epoch']:.2f} |")

    # Training history table
    if stats and stats.get("eval_history"):
        md_lines.extend([
            "",
            "### Training History",
            "",
            "| Step | Epoch | Eval Loss | CER |",
            "|------|-------|-----------|-----|",
        ])
        for entry in stats["eval_history"]:
            epoch_str = f"{entry['epoch']:.2f}" if entry.get("epoch") else "-"
            loss_str = f"{entry['eval_loss']:.4f}" if entry.get("eval_loss") else "-"
            cer_str = f"{entry['eval_cer']:.2%}" if entry.get("eval_cer") else "-"
            md_lines.append(f"| {entry.get('step', '-')} | {epoch_str} | {loss_str} | {cer_str} |")

    # Training details
    md_lines.extend([
        "",
        "## Training Details",
        "",
    ])
    if args.base_model:
        md_lines.append(f"- **Base model:** [{args.base_model}](https://huggingface.co/{args.base_model})")
    md_lines.extend([
        f"- **Dataset:** [{args.dataset}](https://huggingface.co/datasets/{args.dataset}) ({args.language})",
        f"- **Language:** {args.language_name} ({args.language})",
        f"- **Task:** Automatic Speech Recognition (ASR)",
    ])
    if model_type == "wav2vec2":
        md_lines.append("- **Architecture:** CTC (Connectionist Temporal Classification)")
    elif model_type == "whisper":
        md_lines.append("- **Architecture:** Encoder-Decoder (Seq2Seq)")
    md_lines.append("- **Metric:** Character Error Rate (CER)")
    if stats and stats.get("total_steps"):
        md_lines.append(f"- **Total training steps:** {stats['total_steps']}")

    # Usage
    md_lines.extend(["", "## Usage", ""])

    if model_type == "whisper":
        md_lines.extend([
            "```python",
            "from transformers import WhisperForConditionalGeneration, WhisperProcessor",
            "import torchaudio",
            "",
            f'processor = WhisperProcessor.from_pretrained("{repo_id}")',
            f'model = WhisperForConditionalGeneration.from_pretrained("{repo_id}")',
            "",
            "# Load audio",
            'audio, sr = torchaudio.load("audio.mp3")',
            "if sr != 16000:",
            "    audio = torchaudio.transforms.Resample(sr, 16000)(audio)",
            "",
            "input_features = processor(",
            '    audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"',
            ").input_features",
            "",
            "predicted_ids = model.generate(input_features)",
            "transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]",
            "print(transcription)",
            "```",
        ])
    elif model_type == "wav2vec2":
        md_lines.extend([
            "```python",
            "from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor",
            "import torchaudio",
            "import torch",
            "",
            f'processor = Wav2Vec2Processor.from_pretrained("{repo_id}")',
            f'model = Wav2Vec2ForCTC.from_pretrained("{repo_id}")',
            "",
            "# Load audio",
            'audio, sr = torchaudio.load("audio.mp3")',
            "if sr != 16000:",
            "    audio = torchaudio.transforms.Resample(sr, 16000)(audio)",
            "",
            'inputs = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")',
            "with torch.no_grad():",
            "    logits = model(**inputs).logits",
            "predicted_ids = torch.argmax(logits, dim=-1)",
            "transcription = processor.batch_decode(predicted_ids)[0]",
            "print(transcription)",
            "```",
        ])

    return "\n".join(yaml_lines) + "\n" + "\n".join(md_lines) + "\n"


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    repo_id = f"{args.username}/{args.repo_name}"

    # Detect model type
    model_type = detect_model_type(model_path)
    print(f"Model type: {model_type}")

    # Auto-detect base model from config if not provided
    if args.base_model is None:
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            candidate = config.get("_name_or_path", "")
            if "/" in candidate:
                args.base_model = candidate
                print(f"Auto-detected base model: {args.base_model}")

    # Find and parse training stats
    stats = None
    trainer_state_path = (
        Path(args.trainer_state) if args.trainer_state
        else find_trainer_state(model_path)
    )
    if trainer_state_path and trainer_state_path.exists():
        print(f"Found trainer_state: {trainer_state_path}")
        stats = parse_training_stats(trainer_state_path)
        if stats.get("best_cer"):
            print(f"Best CER from training: {stats['best_cer']:.4f} "
                  f"(step {stats.get('best_cer_step')}, "
                  f"epoch {stats.get('best_cer_epoch', '?')})")
        if stats.get("eval_history"):
            print(f"Eval history: {len(stats['eval_history'])} checkpoints")
    else:
        print("No trainer_state.json found — using manual --cer if provided")

    # Generate model card
    model_card = generate_model_card(args, stats, model_type)
    readme_path = model_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    print(f"Generated model card: {readme_path}")

    # List files to upload
    files = list(model_path.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"\nFiles to upload: {len(files)}")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.relative_to(model_path)} ({size_mb:.1f} MB)")

    total_size = sum(f.stat().st_size for f in files) / 1e9
    print(f"Total size: {total_size:.2f} GB")

    # Preview model card
    print("\n" + "=" * 60)
    print("MODEL CARD PREVIEW:")
    print("=" * 60)
    print(model_card)
    print("=" * 60)

    # Confirm
    response = input(f"\nUpload to {repo_id}? [y/N] ")
    if response.lower() != "y":
        print("Cancelled.")
        return

    # Create repo and upload
    api = HfApi()
    create_repo(repo_id, private=args.private, exist_ok=True)
    print(f"Repository ready: https://huggingface.co/{repo_id}")

    print("Uploading...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        commit_message=args.commit_message,
    )

    print(f"Done! Model available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
