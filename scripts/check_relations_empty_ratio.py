#!/usr/bin/env python

# python scripts/check_relations_empty_ratio.py \
#  --base_dir data/nyu_relational/statistical_train

#  python scripts/check_relations_empty_ratio.py \
#  --base_dir data/vkitti2_relational

import argparse
import json
import os
from typing import Tuple


def scan_relations(base_dir: str) -> Tuple[int, int]:
    """
    Scan all *_relations.json files under base_dir and count
    how many are empty ([], {}).
    """
    total = 0
    empty = 0
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith("_relations.json"):
                continue
            fpath = os.path.join(root, fname)
            total += 1
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not data:  # [] or {}
                    empty += 1
            except Exception as e:
                print(f"[WARN] Failed to read {fpath}: {e}")
    return total, empty


def main():
    parser = argparse.ArgumentParser(
        description="Compute fraction of empty *_relations.json files under a base directory."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing *_relations.json files "
             "(e.g., data/nyu_relational/statistical_train or data/vkitti2_relational).",
    )
    args = parser.parse_args()

    base_dir = os.path.expanduser(args.base_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"base_dir is not a directory: {base_dir}")

    print(f"Scanning relations under: {base_dir}")
    total, empty = scan_relations(base_dir)

    if total == 0:
        print("No *_relations.json files found.")
        return

    ratio = empty / total * 100.0
    print(f"Total relations files : {total}")
    print(f"Empty relations files : {empty}")
    print(f"Empty ratio           : {ratio:.2f}%")


if __name__ == "__main__":
    main()

