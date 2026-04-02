#!/usr/bin/env python3
"""Run a finetuned Wav2Vec2 model, export SRT, and compute CER (single file or batch).

Adapted from CanCLID/asr-bench sensevoice_srt_cer.py.
Replaces SenseVoice (FunASR) with a Hugging Face Wav2Vec2ForCTC model.
VAD-based segmentation is replaced with a simpler chunked approach
since Wav2Vec2 CTC models process raw waveforms directly.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# ── asr-bench common utilities (unchanged from original) ────────────────────
# These imports assume you're running from the asr-bench repo root, or have
# the `common` package on your PYTHONPATH.  Adjust as needed.
from common.batch_summary import write_batch_analysis_summary
from common.cantonese_postprocess import CantonesePostProcessor
from common.cer_utils import compute_cer, sequence_for_cer
from common.error_analysis import analyze_char_errors, build_file_analysis_markdown
from common.io_utils import (
    find_audio_files,
    get_audio_duration_sec,
    parse_extensions,
    write_srt,
)
from common.text_utils import clean_asr_text, parse_srt_text


# ── Data class (same fields as original) ────────────────────────────────────

@dataclass
class EvalResult:
    audio_path: Path
    reference_path: Path
    output_srt_path: Path
    audio_duration_sec: float
    segment_count: int
    runtime_sec: float
    asr_runtime_sec: float
    rtf: float
    asr_rtf: float
    reference_chars: int
    hypothesis_chars: int
    edit_distance: int
    cer: float
    reference_chars_no_punc: int
    hypothesis_chars_no_punc: int
    edit_distance_no_punc: int
    cer_no_punc: float
    analysis_report_path: Path
    substitution_count: int
    deletion_count: int
    insertion_count: int
    substitution_counter: Counter[tuple[str, str]]
    deletion_counter: Counter[str]
    insertion_counter: Counter[str]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wav2Vec2 transcription -> SRT + CER vs golden SRT"
    )

    # --- model ---
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Path or HuggingFace Hub ID of the finetuned Wav2Vec2 model "
            "(e.g. 'my-org/wav2vec2-large-cantonese' or './checkpoints/wav2vec2-ft')"
        ),
    )
    parser.add_argument(
        "--processor",
        default=None,
        help=(
            "Path or Hub ID of the processor/tokenizer. "
            "Defaults to the same value as --model."
        ),
    )

    # --- single-file mode ---
    parser.add_argument("--audio", help="Input audio path (single-file mode)")
    parser.add_argument(
        "--golden-srt", help="Golden reference SRT path for CER (single-file mode)"
    )
    parser.add_argument(
        "--output-srt", help="Output path for generated SRT (single-file mode)"
    )

    # --- batch mode ---
    parser.add_argument(
        "--input-dir",
        default="input",
        help='Input audio directory (batch mode). Default: "input".',
    )
    parser.add_argument(
        "--reference-dir",
        default="reference",
        help='Reference SRT directory (batch mode). Default: "reference".',
    )
    parser.add_argument(
        "--output-dir",
        default="predicted/wav2vec2",
        help='Output SRT directory (batch mode). Default: "predicted/wav2vec2".',
    )
    parser.add_argument(
        "--audio-extensions",
        default=".opus,.wav,.mp3,.m4a,.flac,.ogg,.aac",
        help="Comma-separated audio extensions for batch mode.",
    )
    parser.add_argument(
        "--strict-missing-reference",
        action="store_true",
        help="Fail if any audio in batch mode has no matching reference .srt.",
    )

    # --- inference knobs ---
    parser.add_argument(
        "--device",
        default="auto",
        help='Inference device: "auto" (default), "cpu", "cuda:0", etc.',
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate expected by the model. Default: 16000.",
    )
    parser.add_argument(
        "--chunk-length-sec",
        type=float,
        default=30.0,
        help=(
            "Split long audio into chunks of this duration (seconds) to avoid OOM. "
            "Set to 0 to disable chunking. Default: 30."
        ),
    )
    parser.add_argument(
        "--overlap-sec",
        type=float,
        default=2.0,
        help="Overlap between consecutive chunks (seconds). Default: 2.",
    )

    # --- postprocessing ---
    parser.add_argument(
        "--language",
        default="yue",
        help='Language hint for postprocessing. Default: "yue" (Cantonese).',
    )

    # --- summary ---
    parser.add_argument(
        "--summary-dir",
        default="summary",
        help='Batch summary output directory. Default: "summary".',
    )
    parser.add_argument(
        "--summary-name",
        default="wav2vec2",
        help='Batch summary filename stem. Default: "wav2vec2".',
    )

    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_single_mode(args: argparse.Namespace) -> bool:
    return bool(args.audio and args.golden_srt and args.output_srt)


def has_any_single_arg(args: argparse.Namespace) -> bool:
    return bool(args.audio or args.golden_srt or args.output_srt)


def validate_args(args: argparse.Namespace) -> None:
    if has_any_single_arg(args) and not is_single_mode(args):
        raise ValueError(
            "Single-file mode requires all args: --audio --golden-srt --output-srt"
        )


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_summary_path(summary_dir: Path, summary_name: str) -> Path:
    name = summary_name.strip() or "wav2vec2"
    if not name.lower().endswith(".md"):
        name = f"{name}.md"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir / name


# ── Audio loading ────────────────────────────────────────────────────────────

def load_audio(audio_path: Path, target_sr: int) -> np.ndarray:
    """Load audio file and resample to target_sr if needed.

    Returns a 1-D float32 numpy array.
    """
    import librosa

    waveform, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return waveform.astype(np.float32)


# ── Wav2Vec2 inference ───────────────────────────────────────────────────────

def transcribe_wav2vec2(
    waveform: np.ndarray,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
    sample_rate: int = 16000,
    chunk_length_sec: float = 30.0,
    overlap_sec: float = 2.0,
) -> list[tuple[int, int, str]]:
    """Transcribe a waveform and return a list of (start_ms, end_ms, text) entries.

    For short audio (or chunk_length_sec <= 0), the whole file is one segment.
    For long audio, it is split into overlapping chunks; overlap is discarded
    from the right side of each chunk except the last.
    """
    total_samples = len(waveform)
    total_duration_ms = int(total_samples / sample_rate * 1000)

    # ── No chunking ──────────────────────────────────────────────────────
    if chunk_length_sec <= 0 or total_samples <= int(chunk_length_sec * sample_rate):
        text = _decode_chunk(waveform, processor, model, device, sample_rate)
        if text.strip():
            return [(0, total_duration_ms, text.strip())]
        return []

    # ── Chunked inference ────────────────────────────────────────────────
    chunk_samples = int(chunk_length_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    stride = chunk_samples - overlap_samples

    entries: list[tuple[int, int, str]] = []
    offset = 0

    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
        chunk = waveform[offset:end]

        text = _decode_chunk(chunk, processor, model, device, sample_rate)

        start_ms = int(offset / sample_rate * 1000)
        end_ms = int(end / sample_rate * 1000)

        if text.strip():
            entries.append((start_ms, end_ms, text.strip()))

        # Advance by stride (or break if we've reached the end)
        if end >= total_samples:
            break
        offset += stride

    return entries


def _decode_chunk(
    waveform: np.ndarray,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
    sample_rate: int,
) -> str:
    """Run CTC decoding on a single waveform chunk. Returns raw text."""
    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription


# ── Per-file evaluation (mirrors original run_one_file) ─────────────────────

def run_one_file(
    audio_path: Path,
    reference_srt_path: Path,
    output_srt_path: Path,
    processor: Wav2Vec2Processor,
    asr_model: Wav2Vec2ForCTC,
    postprocessor: CantonesePostProcessor,
    device: torch.device,
    sample_rate: int,
    chunk_length_sec: float,
    overlap_sec: float,
) -> EvalResult:
    print(f"\n=== Processing: {audio_path.name} ===")
    file_start_ts = time.perf_counter()

    audio_duration_sec = get_audio_duration_sec(audio_path)

    # Load & resample
    waveform = load_audio(audio_path, target_sr=sample_rate)

    # ASR inference
    asr_start_ts = time.perf_counter()
    entries = transcribe_wav2vec2(
        waveform=waveform,
        processor=processor,
        model=asr_model,
        device=device,
        sample_rate=sample_rate,
        chunk_length_sec=chunk_length_sec,
        overlap_sec=overlap_sec,
    )
    asr_runtime_sec = time.perf_counter() - asr_start_ts

    # Post-process text
    cleaned_entries: list[tuple[int, int, str]] = []
    for start_ms, end_ms, raw_text in entries:
        text = clean_asr_text(raw_text)
        text = postprocessor.apply(text)
        if text:
            cleaned_entries.append((start_ms, end_ms, text))

    segment_count = len(cleaned_entries)

    runtime_sec = time.perf_counter() - file_start_ts
    rtf = float("nan") if audio_duration_sec <= 0 else runtime_sec / audio_duration_sec
    asr_rtf = (
        float("nan")
        if audio_duration_sec <= 0
        else asr_runtime_sec / audio_duration_sec
    )

    # Write SRT
    output_srt_path.parent.mkdir(parents=True, exist_ok=True)
    write_srt(output_srt_path, cleaned_entries)

    # CER evaluation
    ref_text = parse_srt_text(reference_srt_path)
    hyp_text = "".join(text for _, _, text in cleaned_entries)

    ref_len, hyp_len, dist, cer = compute_cer(
        ref_text, hyp_text, include_punctuation=True
    )
    ref_len_no_punc, hyp_len_no_punc, dist_no_punc, cer_no_punc = compute_cer(
        ref_text, hyp_text, include_punctuation=False
    )

    # Error analysis
    ref_seq_no_punc = sequence_for_cer(ref_text, include_punctuation=False)
    hyp_seq_no_punc = sequence_for_cer(hyp_text, include_punctuation=False)

    substitutions, deletions, insertions, equal_count, examples = analyze_char_errors(
        ref_seq_no_punc, hyp_seq_no_punc
    )

    analysis_report_path = output_srt_path.with_suffix(".analysis.md")

    result = EvalResult(
        audio_path=audio_path,
        reference_path=reference_srt_path,
        output_srt_path=output_srt_path,
        audio_duration_sec=audio_duration_sec,
        segment_count=segment_count,
        runtime_sec=runtime_sec,
        asr_runtime_sec=asr_runtime_sec,
        rtf=rtf,
        asr_rtf=asr_rtf,
        reference_chars=ref_len,
        hypothesis_chars=hyp_len,
        edit_distance=dist,
        cer=cer,
        reference_chars_no_punc=ref_len_no_punc,
        hypothesis_chars_no_punc=hyp_len_no_punc,
        edit_distance_no_punc=dist_no_punc,
        cer_no_punc=cer_no_punc,
        analysis_report_path=analysis_report_path,
        substitution_count=sum(substitutions.values()),
        deletion_count=sum(deletions.values()),
        insertion_count=sum(insertions.values()),
        substitution_counter=substitutions,
        deletion_counter=deletions,
        insertion_counter=insertions,
    )

    analysis_md = build_file_analysis_markdown(
        audio_path=audio_path,
        reference_path=reference_srt_path,
        output_srt_path=output_srt_path,
        result=result,
        ref_seq_no_punc=ref_seq_no_punc,
        hyp_seq_no_punc=hyp_seq_no_punc,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        equal_count=equal_count,
        examples=examples,
    )
    analysis_report_path.write_text(analysis_md, encoding="utf-8")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    validate_args(args)

    resolved_device = resolve_device(args.device)
    device = torch.device(resolved_device)
    print(f"Using device: {resolved_device}")

    # ── Load finetuned Wav2Vec2 model ────────────────────────────────────
    model_id = args.model
    processor_id = args.processor or model_id
    print(f"Loading processor from: {processor_id}")
    print(f"Loading model from:     {model_id}")

    processor = AutoProcessor.from_pretrained(processor_id)
    asr_model = Wav2Vec2ForCTC.from_pretrained(model_id)
    asr_model.to(device)
    asr_model.eval()

    postprocessor = CantonesePostProcessor()

    # ── Single-file mode ─────────────────────────────────────────────────
    if is_single_mode(args):
        audio_path = Path(args.audio)
        golden_srt_path = Path(args.golden_srt)
        output_srt_path = Path(args.output_srt)

        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not golden_srt_path.is_file():
            raise FileNotFoundError(f"Golden SRT file not found: {golden_srt_path}")

        result = run_one_file(
            audio_path=audio_path,
            reference_srt_path=golden_srt_path,
            output_srt_path=output_srt_path,
            processor=processor,
            asr_model=asr_model,
            postprocessor=postprocessor,
            device=device,
            sample_rate=args.sample_rate,
            chunk_length_sec=args.chunk_length_sec,
            overlap_sec=args.overlap_sec,
        )

        print("\nDone.")
        print(f"Output SRT: {result.output_srt_path}")
        print(f"Reference chars: {result.reference_chars}")
        print(f"Hypothesis chars: {result.hypothesis_chars}")
        print(f"Edit distance: {result.edit_distance}")
        print(
            f"CER (with punctuation): {result.cer:.6f}"
            if not math.isnan(result.cer)
            else "CER (with punctuation): NaN (empty reference)"
        )
        print(f"Reference chars (no punctuation): {result.reference_chars_no_punc}")
        print(f"Hypothesis chars (no punctuation): {result.hypothesis_chars_no_punc}")
        print(f"Edit distance (no punctuation): {result.edit_distance_no_punc}")
        print(
            f"CER (without punctuation): {result.cer_no_punc:.6f}"
            if not math.isnan(result.cer_no_punc)
            else "CER (without punctuation): NaN (empty reference)"
        )
        print(f"Audio duration (s): {result.audio_duration_sec:.3f}")
        print(f"Runtime (s): {result.runtime_sec:.3f}")
        print(f"ASR runtime only (s): {result.asr_runtime_sec:.3f}")
        print(
            f"End-to-end RTF: {result.rtf:.6f}"
            if not math.isnan(result.rtf)
            else "End-to-end RTF: NaN"
        )
        print(
            f"ASR-only RTF: {result.asr_rtf:.6f}"
            if not math.isnan(result.asr_rtf)
            else "ASR-only RTF: NaN"
        )
        print(f"Analysis report: {result.analysis_report_path}")
        return

    # ── Batch mode ───────────────────────────────────────────────────────
    input_dir = Path(args.input_dir)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if not reference_dir.is_dir():
        raise NotADirectoryError(f"Reference directory not found: {reference_dir}")

    extensions = parse_extensions(args.audio_extensions)
    audio_files = find_audio_files(input_dir, extensions)
    if not audio_files:
        raise RuntimeError(
            f"No audio files found in {input_dir} with extensions: {sorted(extensions)}"
        )

    print(f"\nBatch mode: found {len(audio_files)} audio files")

    results: list[EvalResult] = []
    missing_refs: list[Path] = []

    for audio_path in audio_files:
        reference_path = reference_dir / f"{audio_path.stem}.srt"
        if not reference_path.is_file():
            missing_refs.append(reference_path)
            print(f"Missing reference, skipped: {reference_path}")
            continue

        output_srt_path = output_dir / f"{audio_path.stem}.wav2vec2.srt"

        result = run_one_file(
            audio_path=audio_path,
            reference_srt_path=reference_path,
            output_srt_path=output_srt_path,
            processor=processor,
            asr_model=asr_model,
            postprocessor=postprocessor,
            device=device,
            sample_rate=args.sample_rate,
            chunk_length_sec=args.chunk_length_sec,
            overlap_sec=args.overlap_sec,
        )
        results.append(result)
        print(
            f"Result {audio_path.name}: CER={result.cer:.6f}, "
            f"CER_no_punc={result.cer_no_punc:.6f} "
            f"(edit={result.edit_distance}, "
            f"edit_no_punc={result.edit_distance_no_punc}, "
            f"runtime={result.runtime_sec:.3f}s, rtf={result.rtf:.6f}, "
            f"analysis={result.analysis_report_path.name})"
        )

    if missing_refs and args.strict_missing_reference:
        missing_str = "\n".join(str(p) for p in missing_refs)
        raise RuntimeError(f"Missing references:\n{missing_str}")

    if not results:
        raise RuntimeError("No files were evaluated (all references missing?)")

    # ── Aggregate metrics ────────────────────────────────────────────────
    total_ref_chars = sum(r.reference_chars for r in results)
    total_edit = sum(r.edit_distance for r in results)
    micro_cer = float("nan") if total_ref_chars == 0 else total_edit / total_ref_chars

    total_ref_chars_no_punc = sum(r.reference_chars_no_punc for r in results)
    total_edit_no_punc = sum(r.edit_distance_no_punc for r in results)
    micro_cer_no_punc = (
        float("nan")
        if total_ref_chars_no_punc == 0
        else total_edit_no_punc / total_ref_chars_no_punc
    )

    valid_cers = [r.cer for r in results if not math.isnan(r.cer)]
    macro_cer = float("nan") if not valid_cers else sum(valid_cers) / len(valid_cers)

    valid_cers_no_punc = [
        r.cer_no_punc for r in results if not math.isnan(r.cer_no_punc)
    ]
    macro_cer_no_punc = (
        float("nan")
        if not valid_cers_no_punc
        else sum(valid_cers_no_punc) / len(valid_cers_no_punc)
    )

    total_audio_sec = sum(r.audio_duration_sec for r in results)
    total_runtime_sec = sum(r.runtime_sec for r in results)
    total_asr_runtime_sec = sum(r.asr_runtime_sec for r in results)
    overall_rtf = (
        float("nan") if total_audio_sec <= 0 else total_runtime_sec / total_audio_sec
    )
    overall_asr_rtf = (
        float("nan")
        if total_audio_sec <= 0
        else total_asr_runtime_sec / total_audio_sec
    )

    print("\nBatch summary")
    for r in results:
        print(
            f"- {r.audio_path.name}: CER={r.cer:.6f}, "
            f"CER_no_punc={r.cer_no_punc:.6f}, "
            f"runtime={r.runtime_sec:.3f}s, "
            f"asr_runtime={r.asr_runtime_sec:.3f}s, "
            f"rtf={r.rtf:.6f}, asr_rtf={r.asr_rtf:.6f}, "
            f"output={r.output_srt_path}"
        )
    print(f"Files evaluated: {len(results)}")
    print(f"Micro CER (with punctuation): {micro_cer:.6f}")
    print(f"Micro CER (without punctuation): {micro_cer_no_punc:.6f}")
    print(
        f"Macro CER (with punctuation): {macro_cer:.6f}"
        if not math.isnan(macro_cer)
        else "Macro CER (with punctuation): NaN"
    )
    print(
        f"Macro CER (without punctuation): {macro_cer_no_punc:.6f}"
        if not math.isnan(macro_cer_no_punc)
        else "Macro CER (without punctuation): NaN"
    )
    print(f"Total audio duration (s): {total_audio_sec:.3f}")
    print(f"Total runtime (s): {total_runtime_sec:.3f}")
    print(f"Total ASR runtime only (s): {total_asr_runtime_sec:.3f}")
    print(
        f"End-to-end RTF (batch): {overall_rtf:.6f}"
        if not math.isnan(overall_rtf)
        else "End-to-end RTF (batch): NaN"
    )
    print(
        f"ASR-only RTF (batch): {overall_asr_rtf:.6f}"
        if not math.isnan(overall_asr_rtf)
        else "ASR-only RTF (batch): NaN"
    )

    batch_analysis_path = resolve_summary_path(
        summary_dir=Path(args.summary_dir),
        summary_name=args.summary_name,
    )
    write_batch_analysis_summary(
        output_path=batch_analysis_path,
        results=results,
        micro_cer=micro_cer,
        micro_cer_no_punc=micro_cer_no_punc,
        macro_cer=macro_cer,
        macro_cer_no_punc=macro_cer_no_punc,
    )
    print(f"Output dir: {output_dir}")
    print(f"Batch summary report: {batch_analysis_path}")


if __name__ == "__main__":
    main()
