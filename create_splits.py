#!/usr/bin/env python3
"""
Create deterministic train/validation/test splits from one or more Common Voice
dataset filesets.

Single fileset (backward compatible):
    python create_splits.py --dataset_path cv-corpus/yue \
        --holdback_tsv test.tsv --pct_validation 0.005 --pct_test 0.05 --seed 42

Multiple filesets:
    python create_splits.py \
        --dataset_path cv-corpus/yue,cv-corpus/zh-CN \
        --all_tsv validated.tsv,validated.tsv \
        --holdback_tsv test.tsv,test.tsv \
        --pct_validation 0.005 --pct_test 0.05 --seed 42

Multi-fileset logic:
  1. For each fileset: load all_tsv, subtract holdback_tsv, carve out a
     per-fileset test split (sized by pct_test).
  2. Pool all remaining samples across filesets. Each sample gets a
     "clips_dir" column so audio loading knows where to find it.
  3. Shuffle the pool and split into train/validation.
  4. Returns per-fileset test splits for separate evaluation, plus the
     combined train/validation splits.

Deterministic: same inputs + same seed = same splits.
"""

import argparse
import os
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset


def create_splits(
    dataset_paths,
    all_tsvs=None,
    holdback_tsvs=None,
    pct_validation=0.1,
    pct_test=0.1,
    seed=42,
    write_splits=False,
    write_dir=".",
):
    """
    Load one or more Common Voice TSV filesets, exclude holdback samples,
    create per-fileset test splits, pool the remainder, and split into
    train/validation.

    Args:
        dataset_paths: str or list of str — paths to dataset directories
        all_tsvs: str or list of str — TSV filenames (default: "validated.tsv")
        holdback_tsvs: str or list of str — holdback TSV filenames (default: "")
        pct_validation: Fraction for validation split (default: 0.1)
        pct_test: Fraction for test split per fileset (default: 0.1)
        seed: Random seed for shuffling (default: 42)
        write_splits: Whether to write split TSVs to disk
        write_dir: Directory to write split TSVs into

    Returns:
        dict with keys:
            "train": Dataset with columns clips_dir, path, sentence
            "validation": Dataset with columns clips_dir, path, sentence
            "test_splits": list of Datasets (one per fileset), each with
                           clips_dir, path, sentence
            "holdback_splits": list of Datasets or Nones (one per fileset),
                               raw with all original columns plus clips_dir
            "fileset_names": list of str — directory basenames for labeling
    """
    # ---- Normalize inputs to lists ----
    if isinstance(dataset_paths, str):
        dataset_paths = [p.strip() for p in dataset_paths.split(",")]
    n_filesets = len(dataset_paths)

    if all_tsvs is None:
        all_tsvs = ["validated.tsv"] * n_filesets
    elif isinstance(all_tsvs, str):
        all_tsvs = [t.strip() for t in all_tsvs.split(",")]
    if len(all_tsvs) == 1 and n_filesets > 1:
        all_tsvs = all_tsvs * n_filesets

    if holdback_tsvs is None:
        holdback_tsvs = [""] * n_filesets
    elif isinstance(holdback_tsvs, str):
        holdback_tsvs = [t.strip() for t in holdback_tsvs.split(",")]
    if len(holdback_tsvs) == 1 and n_filesets > 1:
        holdback_tsvs = holdback_tsvs * n_filesets

    assert len(all_tsvs) == n_filesets, \
        f"all_tsvs count ({len(all_tsvs)}) must match dataset_paths count ({n_filesets})"
    assert len(holdback_tsvs) == n_filesets, \
        f"holdback_tsvs count ({len(holdback_tsvs)}) must match dataset_paths count ({n_filesets})"
    assert pct_validation + pct_test < 1.0, \
        f"pct_validation ({pct_validation}) + pct_test ({pct_test}) must be < 1.0"

    fileset_names = [Path(p).name for p in dataset_paths]
    test_splits = []
    holdback_splits = []
    train_pools = []  # per-fileset remaining samples after test/holdback removal

    # ---- Process each fileset ----
    for i, (ds_path, all_tsv, holdback_tsv) in enumerate(
        zip(dataset_paths, all_tsvs, holdback_tsvs)
    ):
        local_path = Path(ds_path)
        clips_dir = str(local_path / "clips")
        label = fileset_names[i]

        assert (local_path / all_tsv).exists(), \
            f"[{label}] TSV not found: {local_path / all_tsv}"

        print(f"\n[{label}] Loading: {local_path / all_tsv}")
        ds = load_dataset(
            "csv",
            data_files=str(local_path / all_tsv),
            delimiter="\t",
            split="train",
        )
        print(f"[{label}] Total samples: {len(ds)}")

        # ---- Holdback exclusion ----
        holdback_ds = None
        if holdback_tsv:
            holdback_path = local_path / holdback_tsv
            assert holdback_path.exists(), \
                f"[{label}] Holdback TSV not found: {holdback_path}"
            holdback_ds = load_dataset(
                "csv",
                data_files=str(holdback_path),
                delimiter="\t",
                split="train",
            )
            # Add clips_dir to holdback for audio loading
            holdback_ds = holdback_ds.map(
                lambda x: {"clips_dir": clips_dir}, batched=False
            )
            hb_paths = set(holdback_ds["path"])
            before = len(ds)
            ds = ds.filter(lambda x: x["path"] not in hb_paths)
            print(f"[{label}] Holdback: {len(hb_paths)} samples, "
                  f"removed {before - len(ds)} from available pool")
        holdback_splits.append(holdback_ds)

        # ---- Normalize text column ----
        text_col = "sentence" if "sentence" in ds.column_names else "text"
        keep_cols = {"path", text_col}
        cols_to_remove = [c for c in ds.column_names if c not in keep_cols]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)
        if text_col != "sentence":
            ds = ds.rename_column(text_col, "sentence")

        # ---- Add clips_dir column ----
        ds = ds.map(lambda x: {"clips_dir": clips_dir}, batched=False)

        # ---- Shuffle and carve out per-fileset test split ----
        ds = ds.shuffle(seed=seed)
        n_test = int(len(ds) * pct_test)
        test_split = ds.select(range(len(ds) - n_test, len(ds)))
        remaining = ds.select(range(len(ds) - n_test))

        print(f"[{label}] Available after holdback: {len(ds)}, "
              f"test: {n_test}, remaining for train pool: {len(remaining)}")

        test_splits.append(test_split)
        train_pools.append(remaining)

    # ---- Pool all remaining samples ----
    if len(train_pools) == 1:
        pooled = train_pools[0]
    else:
        pooled = concatenate_datasets(train_pools)
    print(f"\nPooled train+val samples: {len(pooled)}")

    # ---- Shuffle the pool and split into train/validation ----
    pooled = pooled.shuffle(seed=seed)
    n_val = int(len(pooled) * pct_validation)
    n_train = len(pooled) - n_val

    assert n_train > 0, \
        f"No training samples left (val={n_val}, total={len(pooled)})"

    train_split = pooled.select(range(n_train))
    val_split = pooled.select(range(n_train, n_train + n_val))

    total_test = sum(len(t) for t in test_splits)
    print(f"Final split sizes — train: {n_train}, validation: {n_val}, "
          f"test: {total_test} ({'+'.join(str(len(t)) for t in test_splits)})")

    # ---- Optionally write split TSVs ----
    if write_splits:
        os.makedirs(write_dir, exist_ok=True)
        for name, ds in [("train_split", train_split),
                         ("validation_split", val_split)]:
            outpath = os.path.join(write_dir, f"{name}.tsv")
            ds.to_pandas().to_csv(outpath, sep="\t", index=False)
            print(f"Wrote {outpath} ({len(ds)} rows)")
        for i, ts in enumerate(test_splits):
            label = fileset_names[i]
            outpath = os.path.join(write_dir, f"test_split_{label}.tsv")
            ts.to_pandas().to_csv(outpath, sep="\t", index=False)
            print(f"Wrote {outpath} ({len(ts)} rows)")

    return {
        "train": train_split,
        "validation": val_split,
        "test_splits": test_splits,
        "holdback_splits": holdback_splits,
        "fileset_names": fileset_names,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation/test splits "
                    "from one or more Common Voice datasets"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Comma-separated paths to dataset directories",
    )
    parser.add_argument(
        "--all_tsv", type=str, default="validated.tsv",
        help="Comma-separated TSV filenames (default: validated.tsv for all)",
    )
    parser.add_argument(
        "--holdback_tsv", type=str, default="",
        help="Comma-separated holdback TSV filenames (default: none)",
    )
    parser.add_argument("--pct_validation", type=float, default=0.1)
    parser.add_argument("--pct_test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write_splits", action="store_true")
    parser.add_argument("--write_dir", type=str, default=".")
    args = parser.parse_args()

    splits = create_splits(
        dataset_paths=args.dataset_path,
        all_tsvs=args.all_tsv,
        holdback_tsvs=args.holdback_tsv,
        pct_validation=args.pct_validation,
        pct_test=args.pct_test,
        seed=args.seed,
        write_splits=args.write_splits,
        write_dir=args.write_dir,
    )

    print(f"\nTrain:      {len(splits['train'])} samples")
    print(f"Validation: {len(splits['validation'])} samples")
    for i, (ts, name) in enumerate(
        zip(splits["test_splits"], splits["fileset_names"])
    ):
        print(f"Test [{name}]: {len(ts)} samples")
    for i, (hb, name) in enumerate(
        zip(splits["holdback_splits"], splits["fileset_names"])
    ):
        if hb is not None:
            print(f"Holdback [{name}]: {len(hb)} samples")


if __name__ == "__main__":
    main()
