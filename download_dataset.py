#!/usr/bin/env python3
"""
Download Common Voice datasets from Mozilla's Data Collective API.

Usage:
    python download_dataset.py --token YOUR_API_TOKEN
    python download_dataset.py --token YOUR_API_TOKEN --languages yue zh-HK
    python download_dataset.py --token YOUR_API_TOKEN --languages all
"""

import argparse
import json
import subprocess
import sys

DATASETS = {
    "yue":   {"id": "cmn29rqn9016to107eniyak65", "name": "Cantonese (yue)"},
    "zh-HK": {"id": "cmn2g8zqd01m2mm07prcmehku", "name": "Chinese (Hong Kong)"},
    "zh-CN": {"id": "cmn3iaztg00e4mb070uvufz7q", "name": "Chinese (China)"},
    "zh-TW": {"id": "cmn2g7eaj01fio10769r1m96n", "name": "Chinese (Taiwan)"},
}


def download_dataset(token, language, dataset_id, name, output_dir="./data"):
    """Download and extract a single dataset."""
    print(f"\n{'=' * 60}")
    print(f"Downloading {name} ({language})")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Output dir: {output_dir}")
    print(f"{'=' * 60}")

    # Get download URL
    url = f"https://datacollective.mozillafoundation.org/api/datasets/{dataset_id}/download"
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", url,
         "-H", f"Authorization: Bearer {token}",
         "-H", "Content-Type: application/json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Error getting download URL: {result.stderr}")
        return False

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  Error parsing response: {result.stdout[:200]}")
        return False

    download_url = response.get("downloadUrl")
    if not download_url:
        print(f"  No download URL in response: {response}")
        return False

    # Download and extract directly (pipe curl into tar)
    import os
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Downloading and extracting...")
    dl_result = subprocess.run(
        f'curl -sL "{download_url}" | tar xz -C "{output_dir}"',
        shell=True,
    )
    if dl_result.returncode != 0:
        print(f"  Download/extract failed")
        return False

    print(f"  Extracted to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Common Voice datasets"
    )
    parser.add_argument(
        "--token", type=str, required=True,
        help="Mozilla Data Collective API bearer token",
    )
    parser.add_argument(
        "--languages", type=str, nargs="+",
        default=["all"],
        help=f"Languages to download (choices: {', '.join(DATASETS.keys())}, all). "
             f"Default: all",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Directory to extract datasets into (default: ./data)",
    )
    args = parser.parse_args()

    if "all" in args.languages:
        languages = list(DATASETS.keys())
    else:
        for lang in args.languages:
            if lang not in DATASETS:
                print(f"Unknown language: {lang}")
                print(f"Available: {', '.join(DATASETS.keys())}")
                sys.exit(1)
        languages = args.languages

    success = 0
    for lang in languages:
        ds = DATASETS[lang]
        if download_dataset(args.token, lang, ds["id"], ds["name"], args.output_dir):
            success += 1

    print(f"\nDownloaded {success}/{len(languages)} datasets")


if __name__ == "__main__":
    main()
