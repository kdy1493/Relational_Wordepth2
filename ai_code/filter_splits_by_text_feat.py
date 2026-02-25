#!/usr/bin/env python3
"""Keep only split lines for which text_feat/nyu/train/<scene>/rgb_XXXXX.pt exists."""
import os
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", type=str, default="./data_splits/nyudepthv2_train_files_with_gt.txt")
    parser.add_argument("--out_file", type=str, default="./data_splits/nyudepthv2_train_files_with_gt.txt")
    parser.add_argument("--text_feat_dir", type=str, default="./text_feat/nyu/train")
    args = parser.parse_args()

    text_feat_dir = os.path.abspath(args.text_feat_dir)
    kept = []
    with open(args.split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rgb_rel = parts[0].lstrip("/")   # e.g. basement_0001a/rgb_00000.jpg
            pt_path = os.path.join(text_feat_dir, rgb_rel.replace(".jpg", ".pt"))
            if os.path.isfile(pt_path):
                kept.append(line)
    with open(args.out_file, "w") as f:
        for line in kept:
            f.write(line + "\n")
    print(f"Kept {len(kept)} samples (with .pt) -> {args.out_file}")


if __name__ == "__main__":
    main()
