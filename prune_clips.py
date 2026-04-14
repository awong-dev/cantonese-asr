#!/usr/bin/env python3
"""
Delete clip files not referenced in validated.tsv.

Usage:
    python prune_clips.py --dataset_path data/cv-corpus-25.0-2026-03-09/zh-HK
    python prune_clips.py --dataset_path data/cv-corpus-25.0-2026-03-09/zh-HK --dry_run
    python prune_clips.py --dataset_path data/cv-corpus-25.0-2026-03-09/yue,data/cv-corpus-25.0-2026-03-09/zh-HK
"""

import argparse
import csv
import os
from pathlib import Path


def prune_clips(dataset_path, tsv_name="validated.tsv", dry_run=False):
    ds_path = Path(dataset_path)
    clips_dir = ds_path / "clips"
    tsv_path = ds_path / tsv_name

    assert clips_dir.is_dir(), f"clips dir not found: {clips_dir}"
    assert tsv_path.is_file(), f"TSV not found: {tsv_path}"

    # Collect referenced paths from TSV
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        keep = {row["path"] for row in reader}

    on_disk = set(os.listdir(clips_dir))
    to_delete = on_disk - keep

    total_bytes = 0
    for name in to_delete:
        total_bytes += (clips_dir / name).stat().st_size

    label = ds_path.name
    print(f"[{label}] {len(on_disk)} clips on disk, {len(keep)} in {tsv_name}, "
          f"{len(to_delete)} to prune ({total_bytes / 1e9:.2f} GB)")

    if dry_run:
        print(f"[{label}] dry run — no files deleted")
        return

    for i, name in enumerate(to_delete, 1):
        (clips_dir / name).unlink()
        if i % 10000 == 0:
            print(f"[{label}] deleted {i}/{len(to_delete)}...")

    print(f"[{label}] deleted {len(to_delete)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Delete clips not referenced in validated.tsv"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Comma-separated paths to dataset directories",
    )
    parser.add_argument(
        "--tsv", type=str, default="validated.tsv",
        help="TSV file defining which clips to keep (default: validated.tsv)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Report what would be deleted without deleting",
    )
    args = parser.parse_args()

    for ds_path in args.dataset_path.split(","):
        prune_clips(ds_path.strip(), tsv_name=args.tsv, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
