#!/usr/bin/env python3
"""
Create deterministic train/validation/test splits from a Common Voice dataset.

Shared by train_whisper.py and eval_whisper.py. Given the same parameters and
seed, this always produces identical splits — so eval_whisper.py can regenerate
the exact test split used during training without needing the training script.

Usage as standalone script:
    python create_splits.py --dataset_path cv-corpus-25.0-2026-03-09/yue \
        --holdback_tsv test.tsv --pct_validation 0.005 --pct_test 0.05 \
        --seed 42 --write_splits

Usage as module:
    from create_splits import create_splits
    splits = create_splits(dataset_path="...", pct_test=0.05, seed=42)
    test_split = splits["test"]
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def create_splits(
    dataset_path,
    all_tsv="validated.tsv",
    holdback_tsv="",
    pct_validation=0.1,
    pct_test=0.1,
    seed=42,
    write_splits=False,
    write_dir=".",
):
    """
    Load a Common Voice TSV, optionally exclude holdback samples, shuffle
    with a fixed seed, and split into train/validation/test.

    The split is fully deterministic: same inputs + same seed = same splits.

    Args:
        dataset_path: Path to Common Voice dataset directory
        all_tsv: TSV file with all usable samples (default: validated.tsv)
        holdback_tsv: TSV file of samples to exclude (default: "" = none)
        pct_validation: Fraction for validation split (default: 0.1)
        pct_test: Fraction for test split (default: 0.1)
        seed: Random seed for shuffling (default: 42)
        write_splits: Whether to write split TSVs to disk
        write_dir: Directory to write split TSVs into (default: cwd)

    Returns:
        dict with keys:
            "train": Dataset with columns path + sentence
            "validation": Dataset with columns path + sentence
            "test": Dataset with columns path + sentence
            "holdback": Dataset or None (raw, all original columns)
    """
    local_path = Path(dataset_path)

    assert (local_path / all_tsv).exists(), \
        f"TSV not found: {local_path / all_tsv}"
    assert pct_validation + pct_test < 1.0, \
        f"pct_validation ({pct_validation}) + pct_test ({pct_test}) must be < 1.0"

    # ---- Load all samples ----
    print(f"Loading dataset from: {local_path / all_tsv}")
    all_dataset = load_dataset(
        "csv",
        data_files=str(local_path / all_tsv),
        delimiter="\t",
        split="train",
    )
    print(f"Total samples in {all_tsv}: {len(all_dataset)}")

    # ---- Holdback exclusion ----
    holdback_dataset = None
    if holdback_tsv:
        holdback_path = local_path / holdback_tsv
        assert holdback_path.exists(), f"Holdback TSV not found: {holdback_path}"
        holdback_dataset = load_dataset(
            "csv",
            data_files=str(holdback_path),
            delimiter="\t",
            split="train",
        )
        holdback_paths = set(holdback_dataset["path"])
        before = len(all_dataset)
        all_dataset = all_dataset.filter(
            lambda x: x["path"] not in holdback_paths
        )
        print(f"Holdback: {len(holdback_paths)} samples, "
              f"removed {before - len(all_dataset)} from available pool")

    available_count = len(all_dataset)
    print(f"Available samples: {available_count}")

    # ---- Shuffle and split ----
    all_dataset = all_dataset.shuffle(seed=seed)

    n_val = int(available_count * pct_validation)
    n_test = int(available_count * pct_test)
    n_train = available_count - n_val - n_test

    assert n_train > 0, \
        f"No training samples left (val={n_val}, test={n_test}, total={available_count})"

    train_split = all_dataset.select(range(n_train))
    val_split = all_dataset.select(range(n_train, n_train + n_val))
    test_split = all_dataset.select(range(n_train + n_val, n_train + n_val + n_test))

    print(f"Split sizes — train: {n_train}, validation: {n_val}, test: {n_test}")

    # ---- Optionally write split TSVs ----
    if write_splits:
        import pandas as pd
        os.makedirs(write_dir, exist_ok=True)
        for name, ds in [("train_split", train_split),
                         ("validation_split", val_split),
                         ("test_split", test_split)]:
            outpath = os.path.join(write_dir, f"{name}.tsv")
            ds.to_pandas().to_csv(outpath, sep="\t", index=False)
            print(f"Wrote {outpath} ({len(ds)} rows)")

    # ---- Normalize columns: keep only path + sentence ----
    text_col = "sentence" if "sentence" in train_split.column_names else "text"
    keep_cols = {"path", text_col}
    cols_to_remove = [c for c in train_split.column_names if c not in keep_cols]

    if cols_to_remove:
        train_split = train_split.remove_columns(cols_to_remove)
        val_split = val_split.remove_columns(cols_to_remove)
        test_split = test_split.remove_columns(cols_to_remove)

    if text_col != "sentence":
        train_split = train_split.rename_column(text_col, "sentence")
        val_split = val_split.rename_column(text_col, "sentence")
        test_split = test_split.rename_column(text_col, "sentence")

    return {
        "train": train_split,
        "validation": val_split,
        "test": test_split,
        "holdback": holdback_dataset,
    }


# ---------------------------------------------------------------------------
# CLI entry point — useful for inspecting or writing splits without training
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation/test splits from a Common Voice dataset"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to Common Voice dataset directory",
    )
    parser.add_argument(
        "--all_tsv", type=str, default="validated.tsv",
        help="TSV file with all usable samples (default: validated.tsv)",
    )
    parser.add_argument(
        "--holdback_tsv", type=str, default="",
        help="TSV file of samples to exclude (default: none)",
    )
    parser.add_argument(
        "--pct_validation", type=float, default=0.1,
    )
    parser.add_argument(
        "--pct_test", type=float, default=0.1,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--write_splits", action="store_true",
        help="Write train/validation/test split TSVs to current directory",
    )
    parser.add_argument(
        "--write_dir", type=str, default=".",
        help="Directory to write split TSVs into (default: current directory)",
    )
    args = parser.parse_args()

    splits = create_splits(
        dataset_path=args.dataset_path,
        all_tsv=args.all_tsv,
        holdback_tsv=args.holdback_tsv,
        pct_validation=args.pct_validation,
        pct_test=args.pct_test,
        seed=args.seed,
        write_splits=args.write_splits,
        write_dir=args.write_dir,
    )

    print(f"\nTrain:      {len(splits['train'])} samples")
    print(f"Validation: {len(splits['validation'])} samples")
    print(f"Test:       {len(splits['test'])} samples")
    if splits["holdback"] is not None:
        print(f"Holdback:   {len(splits['holdback'])} samples")


if __name__ == "__main__":
    main()
