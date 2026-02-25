#!/usr/bin/env python3
"""
Generate nyudepthv2_train_files_with_gt.txt and nyudepthv2_test_files_with_gt.txt
from data/nyu_v2_sync layout: scene/rgb_*.jpg and scene/dense/sync_depth_dense_*.png
"""
import os
import argparse

FOCAL = 518.8579
TEST_SIZE = 654  # match original NYU test set size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/nyu_v2_sync")
    parser.add_argument("--out_dir", type=str, default="./data_splits")
    parser.add_argument("--test_size", type=int, default=TEST_SIZE)
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    pairs: list[tuple[str, str]] = []
    for scene in sorted(os.listdir(data_root)):
        scene_dir = os.path.join(data_root, scene)
        if not os.path.isdir(scene_dir):
            continue
        dense_dir = os.path.join(scene_dir, "dense")
        if not os.path.isdir(dense_dir):
            continue
        for f in sorted(os.listdir(scene_dir)):
            if not f.startswith("rgb_") or not f.endswith(".jpg"):
                continue
            # rgb_00045.jpg -> sync_depth_dense_00045.png
            base = f.replace("rgb_", "").replace(".jpg", "")
            depth_name = f"sync_depth_dense_{base}.png"
            depth_path = os.path.join(dense_dir, depth_name)
            if os.path.isfile(depth_path):
                # WorDepth dataloader expects leading slash then [1:] strips it
                rgb_rel = f"/{scene}/{f}"
                depth_rel = f"/{scene}/dense/{depth_name}"
                pairs.append((rgb_rel, depth_rel))

    if not pairs:
        raise RuntimeError(f"No valid pairs found under {data_root}")

    # Split: last test_size for test (to get fixed test set), rest train
    n_test = min(args.test_size, len(pairs) // 10)
    n_train = len(pairs) - n_test
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_file = os.path.join(args.out_dir, "nyudepthv2_train_files_with_gt.txt")
    test_file = os.path.join(args.out_dir, "nyudepthv2_test_files_with_gt.txt")

    with open(train_file, "w") as f:
        for rgb_rel, depth_rel in train_pairs:
            f.write(f"{rgb_rel} {depth_rel} {FOCAL}\n")
    with open(test_file, "w") as f:
        for rgb_rel, depth_rel in test_pairs:
            f.write(f"{rgb_rel} {depth_rel} {FOCAL}\n")

    print(f"Wrote {len(train_pairs)} train -> {train_file}")
    print(f"Wrote {len(test_pairs)} test -> {test_file}")


if __name__ == "__main__":
    main()
