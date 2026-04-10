"""
Shared CER (Character Error Rate) utilities for Whisper and wav2vec2 training/eval.

Provides:
  - build_text_normalize(): creates the jiwer text normalization pipeline
  - build_cer_transform(): creates the jiwer CER transform pipeline
  - compute_cer(): computes raw + nopunct CER from prediction/reference lists
  - evaluate_and_summarize(): runs evaluation on multiple splits and prints summary
"""

import evaluate
import jiwer


def build_text_normalize():
    """
    Build a jiwer Compose transform for text normalization.
    Removes punctuation, lowercases, and normalizes whitespace.
    Used for cleaning training labels.
    """
    return jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])


def build_cer_transform():
    """
    Build a jiwer Compose transform for normalized CER evaluation.
    Removes punctuation, lowercases, normalizes whitespace, and splits
    into character lists (required for jiwer.process_characters).
    """
    return jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])


# Module-level CER metric (loaded once, reused)
_cer_metric = evaluate.load("cer")


def compute_cer(pred_list, label_list, cer_transform=None):
    """
    Compute raw CER and optionally normalized (nopunct) CER.

    Args:
        pred_list: list of predicted strings
        label_list: list of reference strings
        cer_transform: jiwer Compose transform for normalized CER (or None)

    Returns:
        dict with "cer_raw" and optionally "cer_nopunct"
    """
    # Filter out empty references to avoid division by zero
    pairs = [(p, l) for p, l in zip(pred_list, label_list) if len(l.strip()) > 0]
    if not pairs:
        result = {"cer_raw": 1.0}
        if cer_transform is not None:
            result["cer_nopunct"] = 1.0
        return result, [], []

    filtered_preds, filtered_refs = zip(*pairs)
    filtered_preds = list(filtered_preds)
    filtered_refs = list(filtered_refs)

    # Raw CER
    cer_raw = _cer_metric.compute(predictions=filtered_preds, references=filtered_refs)
    result = {"cer_raw": cer_raw}

    # Normalized CER (punctuation removed, lowercased, whitespace normalized)
    # Process in chunks to avoid memory issues with large eval sets,
    # then aggregate edit-distance counters for an exact global CER.
    if cer_transform is not None:
        chunk_size = 500
        total_hits = 0
        total_subs = 0
        total_ins = 0
        total_dels = 0
        for i in range(0, len(filtered_refs), chunk_size):
            chunk_refs = filtered_refs[i:i + chunk_size]
            chunk_preds = filtered_preds[i:i + chunk_size]
            output = jiwer.process_characters(
                chunk_refs, chunk_preds,
                reference_transform=cer_transform,
                hypothesis_transform=cer_transform,
            )
            total_hits += output.hits
            total_subs += output.substitutions
            total_ins += output.insertions
            total_dels += output.deletions
        total_ref_len = total_hits + total_subs + total_dels
        result["cer_nopunct"] = (
            (total_subs + total_ins + total_dels) / max(1, total_ref_len)
        )

    return result, filtered_preds, filtered_refs


def print_examples(pred_list, label_list, num_stable=3, num_random=3):
    """Print sample REF/HYP pairs for debugging.

    Shows num_stable fixed examples (evenly spaced indices) followed by
    num_random randomly chosen examples.
    """
    import random

    n = len(pred_list)
    if n == 0:
        return

    # Stable: evenly spaced indices so the same samples appear every eval
    if n <= num_stable:
        stable_indices = list(range(n))
    else:
        stable_indices = [i * n // num_stable for i in range(num_stable)]
    stable_set = set(stable_indices)

    # Random: pick from the remaining indices
    remaining = [i for i in range(n) if i not in stable_set]
    random_indices = random.sample(remaining, min(num_random, len(remaining)))

    for tag, indices in [("stable", stable_indices), ("random", random_indices)]:
        for i in indices:
            print(f"  [{tag}] REF: {label_list[i][:80]}")
            print(f"  [{tag}] HYP: {pred_list[i][:80]}")
            print()


def evaluate_and_summarize(trainer, eval_splits, results_json=None):
    """
    Run evaluation on multiple (name, dataset) pairs and print a summary.

    Args:
        trainer: HuggingFace Trainer instance
        eval_splits: list of (split_name, dataset) tuples

    Returns:
        dict mapping split names to their metrics dicts
    """
    results = {}

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for split_name, dataset in eval_splits:
        print(f"\nEvaluating on {split_name} ({len(dataset)} samples)...")
        try:
            metrics = trainer.evaluate(
                eval_dataset=dataset, metric_key_prefix=split_name
            )
        except Exception as e:
            print(f"  {split_name} evaluation failed: {e}")
            continue

        cer_raw = metrics.get(f"{split_name}_cer_raw", None)
        cer_nopunct = metrics.get(f"{split_name}_cer_nopunct", None)
        loss = metrics.get(f"{split_name}_loss", None)

        if cer_raw is not None:
            print(f"  {split_name} CER (raw):     {cer_raw:.4f}")
        if cer_nopunct is not None:
            print(f"  {split_name} CER (nopunct): {cer_nopunct:.4f}")
        if loss is not None:
            print(f"  {split_name} Loss:           {loss:.4f}")

        results[split_name] = metrics

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, m in results.items():
        cer_raw = m.get(f"{name}_cer_raw", None)
        cer_nopunct = m.get(f"{name}_cer_nopunct", None)
        if cer_raw is not None:
            line = f"  {name:24s} CER: {cer_raw:.4f}"
            if cer_nopunct is not None:
                line += f"  (nopunct: {cer_nopunct:.4f})"
            print(line)
    print("=" * 60)

    if results_json:
        import json
        import os
        os.makedirs(os.path.dirname(results_json) or ".", exist_ok=True)
        with open(results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {results_json}")

    return results
