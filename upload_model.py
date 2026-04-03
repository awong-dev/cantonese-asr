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

    # Push to a specific branch (creates branch if needed):
    python upload_model.py --model_path ./whisper-large-v3-turbo-yue/final \
        --repo_name whisper-large-v3-turbo-cantonese-yue \
        --revision v2 --commit_message "v2: retrained with augmented data"

    # Create a tagged revision from the current main:
    python upload_model.py --model_path ./whisper-large-v3-turbo-yue/final \
        --repo_name whisper-large-v3-turbo-cantonese-yue \
        --revision checkpoint-15000 --create_tag

    # List existing revisions for a repo:
    python upload_model.py --repo_name whisper-large-v3-turbo-cantonese-yue \
        --list_revisions

    # Upload with TensorBoard runs (auto-detected from model_path parent):
    python upload_model.py --model_path ./whisper-large-v3-turbo-yue/final \
        --repo_name whisper-large-v3-turbo-cantonese-yue

    # Specify a custom runs directory:
    python upload_model.py --model_path ./whisper-large-v3-turbo-yue/final \
        --repo_name whisper-large-v3-turbo-cantonese-yue \
        --runs_dir ./whisper-large-v3-turbo-yue/runs

    # Skip uploading TensorBoard runs:
    python upload_model.py --model_path ./whisper-large-v3-turbo-yue/final \
        --repo_name whisper-large-v3-turbo-cantonese-yue --no_runs

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
        "--model_path", type=str, default=None,
        help="Path to the model directory (checkpoint or final). "
             "Required unless using --list_revisions.",
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
    parser.add_argument(
        "--revision", type=str, default=None,
        help="Branch/revision name to push to (e.g. v2, checkpoint-15000). "
             "Creates the branch if it doesn't exist. Defaults to 'main'.",
    )
    parser.add_argument(
        "--create_tag", action="store_true",
        help="Create a git tag instead of a branch for --revision",
    )
    parser.add_argument(
        "--list_revisions", action="store_true",
        help="List existing branches and tags for the repo, then exit",
    )
    parser.add_argument(
        "--runs_dir", type=str, default=None,
        help="Path to TensorBoard runs directory to upload. "
             "Auto-detected from model_path parent if not set (looks for 'runs/' dir).",
    )
    parser.add_argument(
        "--no_runs", action="store_true",
        help="Skip uploading TensorBoard runs even if a runs directory is found",
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


def find_runs_dir(model_path: Path) -> Path | None:
    """Search for a TensorBoard runs directory near model_path.

    Looks for directories named 'runs' containing tfevents files, checking:
      1. model_path/runs/
      2. model_path/../runs/
      3. model_path/../../runs/
    """
    search_roots = [model_path, model_path.parent, model_path.parent.parent]
    for root in search_roots:
        candidate = root / "runs"
        if candidate.is_dir() and list(candidate.rglob("events.out.tfevents.*")):
            return candidate
    return None


def summarize_runs_dir(runs_dir: Path) -> None:
    """Print a summary of a TensorBoard runs directory."""
    tfevents = sorted(runs_dir.rglob("events.out.tfevents.*"))
    subdirs = sorted({f.parent.relative_to(runs_dir) for f in tfevents})

    total_size = sum(f.stat().st_size for f in runs_dir.rglob("*") if f.is_file())
    print(f"\nTensorBoard runs: {runs_dir}")
    print(f"  Event files: {len(tfevents)}")
    print(f"  Run subdirs: {len(subdirs)}")
    for sd in subdirs:
        n = sum(1 for f in tfevents if f.parent.relative_to(runs_dir) == sd)
        print(f"    {sd}/ ({n} event file{'s' if n != 1 else ''})")
    print(f"  Total size: {total_size / 1e6:.1f} MB")


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


def generate_model_card(args, stats: dict | None, model_type: str, has_runs: bool = False) -> str:
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

    revision = getattr(args, "revision", None)
    if revision:
        md_lines.append(f"> **Revision:** `{revision}`")
        md_lines.append("")

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

    # TensorBoard
    if has_runs:
        md_lines.extend([
            "",
            "## Training Metrics",
            "",
            "TensorBoard logs are included in the `runs/` directory of this repository.",
            "",
            "```bash",
            f"# Clone and view locally",
            f"git clone https://huggingface.co/{repo_id}",
            f"tensorboard --logdir {args.repo_name}/runs",
            "```",
        ])

    # Usage
    md_lines.extend(["", "## Usage", ""])

    # Build revision kwarg string for usage examples
    revision = getattr(args, "revision", None)
    rev_kwarg = f', revision="{revision}"' if revision else ""

    if model_type == "whisper":
        md_lines.extend([
            "```python",
            "from transformers import WhisperForConditionalGeneration, WhisperProcessor",
            "import torchaudio",
            "",
            f'processor = WhisperProcessor.from_pretrained("{repo_id}"{rev_kwarg})',
            f'model = WhisperForConditionalGeneration.from_pretrained("{repo_id}"{rev_kwarg})',
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
            f'processor = Wav2Vec2Processor.from_pretrained("{repo_id}"{rev_kwarg})',
            f'model = Wav2Vec2ForCTC.from_pretrained("{repo_id}"{rev_kwarg})',
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


def list_repo_revisions(repo_id: str) -> None:
    """List all branches and tags for a HuggingFace repo."""
    api = HfApi()
    try:
        refs = api.list_repo_refs(repo_id)
    except Exception as e:
        print(f"Error listing revisions for {repo_id}: {e}")
        return

    print(f"\nRevisions for {repo_id}:")
    print(f"  {'='*50}")

    print(f"\n  Branches:")
    if refs.branches:
        for branch in refs.branches:
            prefix = "  * " if branch.name == "main" else "    "
            print(f"{prefix}{branch.name} ({branch.target_commit[:8]})")
    else:
        print("    (none)")

    print(f"\n  Tags:")
    if refs.tags:
        for tag in refs.tags:
            print(f"    {tag.name} ({tag.target_commit[:8]})")
    else:
        print("    (none)")

    print()


def main():
    args = parse_args()

    repo_id = f"{args.username}/{args.repo_name}"

    # Handle --list_revisions: no model_path needed
    if args.list_revisions:
        list_repo_revisions(repo_id)
        return

    model_path = Path(args.model_path) if args.model_path else None
    if model_path is None or not model_path.exists():
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

    # Find TensorBoard runs directory
    runs_dir = None
    if not args.no_runs:
        if args.runs_dir:
            runs_dir = Path(args.runs_dir)
            if not runs_dir.is_dir():
                print(f"Warning: --runs_dir not found: {runs_dir}")
                runs_dir = None
        else:
            runs_dir = find_runs_dir(model_path)

        if runs_dir:
            summarize_runs_dir(runs_dir)
        else:
            print("No TensorBoard runs directory found (use --runs_dir to specify)")

    # Generate model card
    model_card = generate_model_card(args, stats, model_type, has_runs=runs_dir is not None)
    readme_path = model_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    print(f"Generated model card: {readme_path}")

    # List files to upload
    files = list(model_path.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"\nModel files to upload: {len(files)}")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.relative_to(model_path)} ({size_mb:.1f} MB)")

    runs_files = []
    if runs_dir:
        runs_files = [f for f in runs_dir.rglob("*") if f.is_file()]
        print(f"\nTensorBoard files to upload: {len(runs_files)}")
        for f in sorted(runs_files):
            size_mb = f.stat().st_size / 1e6
            print(f"  runs/{f.relative_to(runs_dir)} ({size_mb:.1f} MB)")

    all_files = files + runs_files
    total_size = sum(f.stat().st_size for f in all_files) / 1e9
    print(f"Total size: {total_size:.2f} GB")

    # Preview model card
    print("\n" + "=" * 60)
    print("MODEL CARD PREVIEW:")
    print("=" * 60)
    print(model_card)
    print("=" * 60)

    # Confirm
    revision_label = f" (revision: {args.revision})" if args.revision else ""
    tag_label = " [as tag]" if args.create_tag else ""
    response = input(f"\nUpload to {repo_id}{revision_label}{tag_label}? [y/N] ")
    if response.lower() != "y":
        print("Cancelled.")
        return

    # Create repo and upload
    api = HfApi()
    create_repo(repo_id, private=args.private, exist_ok=True)
    print(f"Repository ready: https://huggingface.co/{repo_id}")

    # Determine the target branch for upload
    upload_branch = args.revision if (args.revision and not args.create_tag) else None

    # Create branch if needed (skip for tags — we upload to main first, then tag)
    if upload_branch and upload_branch != "main":
        try:
            refs = api.list_repo_refs(repo_id)
            existing_branches = {b.name for b in refs.branches}
            if upload_branch not in existing_branches:
                print(f"Creating branch '{upload_branch}'...")
                api.create_branch(repo_id, branch=upload_branch)
        except Exception as e:
            print(f"Note: could not verify/create branch: {e}")
            print(f"Attempting upload to '{upload_branch}' anyway...")

    print(f"Uploading model to {'branch ' + upload_branch if upload_branch else 'main'}...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        revision=upload_branch,
        commit_message=args.commit_message,
    )

    # Upload TensorBoard runs directory
    if runs_dir:
        runs_commit_msg = f"{args.commit_message} (TensorBoard runs)"
        print(f"Uploading TensorBoard runs from {runs_dir}...")
        api.upload_folder(
            folder_path=str(runs_dir),
            repo_id=repo_id,
            path_in_repo="runs",
            revision=upload_branch,
            commit_message=runs_commit_msg,
        )

    # Create tag if requested
    if args.create_tag and args.revision:
        print(f"Creating tag '{args.revision}'...")
        try:
            api.create_tag(
                repo_id,
                tag=args.revision,
                tag_message=args.commit_message,
            )
            print(f"Tag '{args.revision}' created.")
        except Exception as e:
            print(f"Warning: failed to create tag '{args.revision}': {e}")

    revision_suffix = f"?revision={args.revision}" if args.revision else ""
    print(f"Done! Model available at: https://huggingface.co/{repo_id}{revision_suffix}")


if __name__ == "__main__":
    main()
