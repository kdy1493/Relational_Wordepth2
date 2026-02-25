#!/usr/bin/env python3
"""
Split train into train + val (90% / 10%). Overwrites train split with remaining lines.
Uses fixed seed for reproducibility.

실행 명령어 (from repo root):
  python ai_code/split_train_val.py
"""
import os
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(REPO_ROOT, "data_splits", "nyudepthv2_train_split.txt")
VAL_FILE = os.path.join(REPO_ROOT, "data_splits", "nyudepthv2_val_split.txt")
VAL_RATIO = 0.10
SEED = 42


def main() -> None:
    with open(TRAIN_FILE) as f:
        lines = [line.strip() for line in f if line.strip()]
    n = len(lines)
    random.seed(SEED)
    shuffled = lines.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(n * VAL_RATIO))
    val_lines = shuffled[:n_val]
    train_lines = shuffled[n_val:]
    with open(VAL_FILE, "w") as f:
        for line in val_lines:
            f.write(line + "\n")
    with open(TRAIN_FILE, "w") as f:
        for line in train_lines:
            f.write(line + "\n")
    print(f"Train: {len(train_lines)}, Val: {len(val_lines)}")
    print(f"Wrote {VAL_FILE}, updated {TRAIN_FILE}")


if __name__ == "__main__":
    main()
