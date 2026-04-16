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

Explicit validation/test TSVs (overrides percentage-based splitting):
    python create_splits.py --dataset_path cv-corpus/yue \
        --validation_tsv dev.tsv --test_tsv test.tsv --seed 42

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


def _normalize_tsv_list(value, n_filesets, name):
    """Normalize a comma-separated string or list to a list of length n_filesets."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = [t.strip() for t in value.split(",")]
    if len(value) == 1 and n_filesets > 1:
        value = value * n_filesets
    assert len(value) == n_filesets, \
        f"{name} count ({len(value)}) must match dataset_paths count ({n_filesets})"
    return value


def _load_and_normalize_tsv(tsv_path, clips_dir, label):
    """Load a TSV, normalize to (path, sentence, clips_dir) columns."""
    ds = load_dataset("csv", data_files=str(tsv_path), delimiter="\t", split="train")
    text_col = "sentence" if "sentence" in ds.column_names else "text"
    keep_cols = {"path", text_col}
    cols_to_remove = [c for c in ds.column_names if c not in keep_cols]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)
    if text_col != "sentence":
        ds = ds.rename_column(text_col, "sentence")
    ds = ds.map(lambda x: {"clips_dir": clips_dir}, batched=False)
    return ds


def create_splits(
    dataset_paths,
    all_tsvs=None,
    holdback_tsvs=None,
    validation_tsvs=None,
    test_tsvs=None,
    pct_validation=0.1,
    pct_test=0.1,
    seed=42,
    write_splits=False,
    write_dir=".",
    dataset_ratio=None,
):
    """
    Load one or more Common Voice TSV filesets, exclude holdback samples,
    create per-fileset test splits, pool the remainder, and split into
    train/validation.

    Args:
        dataset_paths: str or list of str — paths to dataset directories
        all_tsvs: str or list of str — TSV filenames (default: "validated.tsv")
        holdback_tsvs: str or list of str — holdback TSV filenames (default: "")
        validation_tsvs: str or list of str or None — explicit validation TSV
            filenames. When provided, these are used directly instead of
            carving validation from all_tsvs (pct_validation is ignored).
        test_tsvs: str or list of str or None — explicit test TSV filenames.
            When provided, these are used directly instead of carving test
            from all_tsvs (pct_test is ignored).
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

    all_tsvs = _normalize_tsv_list(all_tsvs, n_filesets, "all_tsvs") \
        or ["validated.tsv"] * n_filesets
    holdback_tsvs = _normalize_tsv_list(holdback_tsvs, n_filesets, "holdback_tsvs") \
        or [""] * n_filesets
    validation_tsvs = _normalize_tsv_list(validation_tsvs, n_filesets, "validation_tsvs")
    test_tsvs = _normalize_tsv_list(test_tsvs, n_filesets, "test_tsvs")

    use_explicit_validation = validation_tsvs is not None
    use_explicit_test = test_tsvs is not None

    if not use_explicit_validation and not use_explicit_test:
        assert pct_validation + pct_test < 1.0, \
            f"pct_validation ({pct_validation}) + pct_test ({pct_test}) must be < 1.0"

    fileset_names = [Path(p).name for p in dataset_paths]
    test_splits = []
    holdback_splits = []
    explicit_val_splits = []  # per-fileset, only when use_explicit_validation
    train_pools = []

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

        # ---- Load and exclude explicit test/validation TSVs ----
        explicit_test_ds = None
        if use_explicit_test:
            test_tsv_path = local_path / test_tsvs[i]
            assert test_tsv_path.exists(), \
                f"[{label}] Test TSV not found: {test_tsv_path}"
            explicit_test_ds = _load_and_normalize_tsv(test_tsv_path, clips_dir, label)
            before = len(ds)
            ep = set(explicit_test_ds["path"])
            ds = ds.filter(lambda x, _ep=ep: x["path"] not in _ep)
            print(f"[{label}] Explicit test: {len(ep)} samples, "
                  f"removed {before - len(ds)} from training pool")

        explicit_val_ds = None
        if use_explicit_validation:
            val_tsv_path = local_path / validation_tsvs[i]
            assert val_tsv_path.exists(), \
                f"[{label}] Validation TSV not found: {val_tsv_path}"
            explicit_val_ds = _load_and_normalize_tsv(val_tsv_path, clips_dir, label)
            before = len(ds)
            ep = set(explicit_val_ds["path"])
            ds = ds.filter(lambda x, _ep=ep: x["path"] not in _ep)
            print(f"[{label}] Explicit validation: {len(ep)} samples, "
                  f"removed {before - len(ds)} from training pool")
            explicit_val_splits.append(explicit_val_ds)

        # ---- Normalize columns on the main dataset ----
        text_col = "sentence" if "sentence" in ds.column_names else "text"
        keep_cols = {"path", text_col}
        cols_to_remove = [c for c in ds.column_names if c not in keep_cols]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)
        if text_col != "sentence":
            ds = ds.rename_column(text_col, "sentence")
        ds = ds.map(lambda x: {"clips_dir": clips_dir}, batched=False)

        # ---- Test split: explicit or percentage-based ----
        if explicit_test_ds is not None:
            test_split = explicit_test_ds
            remaining = ds.shuffle(seed=seed)
            print(f"[{label}] Using explicit test TSV: {len(test_split)} samples, "
                  f"remaining for train pool: {len(remaining)}")
        else:
            ds = ds.shuffle(seed=seed)
            n_test = int(len(ds) * pct_test)
            test_split = ds.select(range(len(ds) - n_test, len(ds)))
            remaining = ds.select(range(len(ds) - n_test))
            print(f"[{label}] Available after holdback: {len(ds)}, "
                  f"test: {n_test}, remaining for train pool: {len(remaining)}")

        test_splits.append(test_split)
        train_pools.append(remaining)

    # ---- Apply dataset ratio sampling ----
    if dataset_ratio is not None:
        if isinstance(dataset_ratio, str):
            ratio_parts = [int(r) for r in dataset_ratio.split(":")]
        else:
            ratio_parts = list(dataset_ratio)
        if len(ratio_parts) == 1:
            ratio_parts = ratio_parts * n_filesets
        assert len(ratio_parts) == n_filesets, \
            f"dataset_ratio has {len(ratio_parts)} parts but there are {n_filesets} filesets"

        pool_sizes = [len(p) for p in train_pools]
        base_count = min(pool_sizes[i] / ratio_parts[i] for i in range(n_filesets))

        sampled_pools = []
        for i, pool in enumerate(train_pools):
            n_take = int(base_count * ratio_parts[i])
            sampled = pool.shuffle(seed=seed).select(range(n_take))
            print(f"[{fileset_names[i]}] Ratio {ratio_parts[i]}: "
                  f"taking {n_take}/{pool_sizes[i]} samples")
            sampled_pools.append(sampled)
        train_pools = sampled_pools

    # ---- Pool all remaining samples ----
    if len(train_pools) == 1:
        pooled = train_pools[0]
    else:
        pooled = concatenate_datasets(train_pools)
    print(f"\nPooled train+val samples: {len(pooled)}")

    # ---- Validation split: explicit or percentage-based ----
    if use_explicit_validation:
        # Pool explicit validation datasets from all filesets
        if len(explicit_val_splits) == 1:
            val_split = explicit_val_splits[0]
        else:
            val_split = concatenate_datasets(explicit_val_splits)
        # All pooled samples go to train
        train_split = pooled.shuffle(seed=seed)
        print(f"Using explicit validation TSVs: {len(val_split)} samples")
        print(f"Final split sizes — train: {len(train_split)}, "
              f"validation: {len(val_split)}, "
              f"test: {sum(len(t) for t in test_splits)} "
              f"({'+'.join(str(len(t)) for t in test_splits)})")
    else:
        # Shuffle the pool and split into train/validation
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
    parser.add_argument(
        "--validation_tsv", type=str, default=None,
        help="Comma-separated explicit validation TSV filenames. "
             "When set, pct_validation is ignored.",
    )
    parser.add_argument(
        "--test_tsv", type=str, default=None,
        help="Comma-separated explicit test TSV filenames. "
             "When set, pct_test is ignored.",
    )
    parser.add_argument("--pct_validation", type=float, default=0.1)
    parser.add_argument("--pct_test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset_ratio", type=str, default=None,
        help="Colon-separated ratios for dataset sampling, e.g. '2:1' means "
             "twice as many samples from the first dataset. Default: use all samples.",
    )
    parser.add_argument("--write_splits", action="store_true")
    parser.add_argument("--write_dir", type=str, default=".")
    args = parser.parse_args()

    splits = create_splits(
        dataset_paths=args.dataset_path,
        all_tsvs=args.all_tsv,
        holdback_tsvs=args.holdback_tsv,
        validation_tsvs=args.validation_tsv,
        test_tsvs=args.test_tsv,
        pct_validation=args.pct_validation,
        pct_test=args.pct_test,
        seed=args.seed,
        write_splits=args.write_splits,
        write_dir=args.write_dir,
        dataset_ratio=args.dataset_ratio,
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
