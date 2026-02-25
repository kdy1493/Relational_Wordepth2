#!/usr/bin/env python3
"""
Validate data/nyu_v2_sync and data_splits against text_feat.
- Every train sample: rgb exists, depth exists, text_feat .pt exists.
- Optionally: sample image/depth shape and depth value range.
"""
import os
import sys

# project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

DATA_ROOT = os.path.join(ROOT, "data", "nyu_v2_sync")
TEXT_FEAT_DIR = os.path.join(ROOT, "text_feat", "nyu", "train")
TRAIN_SPLIT = os.path.join(ROOT, "data_splits", "nyudepthv2_train_files_with_gt.txt")


def main() -> None:
    if not os.path.isdir(DATA_ROOT):
        print(f"FAIL: data root not found: {DATA_ROOT}")
        return
    if not os.path.isfile(TRAIN_SPLIT):
        print(f"FAIL: train split not found: {TRAIN_SPLIT}")
        return

    with open(TRAIN_SPLIT) as f:
        lines = [l.strip() for l in f if l.strip()]

    missing_rgb: list[str] = []
    missing_depth: list[str] = []
    missing_pt: list[str] = []
    bad_focal: list[str] = []

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 3:
            bad_focal.append(line[:60])
            continue
        rgb_rel = parts[0].lstrip("/")
        depth_rel = parts[1].lstrip("/")
        try:
            float(parts[2])
        except ValueError:
            bad_focal.append(line[:60])
            continue

        rgb_path = os.path.join(DATA_ROOT, rgb_rel)
        depth_path = os.path.join(DATA_ROOT, depth_rel)
        pt_path = os.path.join(TEXT_FEAT_DIR, rgb_rel.replace(".jpg", ".pt"))

        if not os.path.isfile(rgb_path):
            missing_rgb.append(rgb_rel)
        if not os.path.isfile(depth_path):
            missing_depth.append(depth_rel)
        if not os.path.isfile(pt_path):
            missing_pt.append(rgb_rel.replace(".jpg", ".pt"))

    n = len(lines)
    print(f"Train split: {n} lines")
    print(f"  Missing RGB:   {len(missing_rgb)}")
    print(f"  Missing depth: {len(missing_depth)}")
    print(f"  Missing .pt:   {len(missing_pt)}")
    print(f"  Bad line/focal: {len(bad_focal)}")

    if missing_rgb:
        for p in missing_rgb[:5]:
            print(f"    - {p}")
        if len(missing_rgb) > 5:
            print(f"    ... and {len(missing_rgb) - 5} more")
    if missing_depth:
        for p in missing_depth[:5]:
            print(f"    - {p}")
        if len(missing_depth) > 5:
            print(f"    ... and {len(missing_depth) - 5} more")
    if missing_pt:
        for p in missing_pt[:5]:
            print(f"    - {p}")
        if len(missing_pt) > 5:
            print(f"    ... and {len(missing_pt) - 5} more")

    # Samples where rgb + depth + .pt all exist
    ok_count = 0
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        rgb_rel = parts[0].lstrip("/")
        depth_rel = parts[1].lstrip("/")
        pt_path = os.path.join(TEXT_FEAT_DIR, rgb_rel.replace(".jpg", ".pt"))
        if (
            os.path.isfile(os.path.join(DATA_ROOT, rgb_rel))
            and os.path.isfile(os.path.join(DATA_ROOT, depth_rel))
            and os.path.isfile(pt_path)
        ):
            ok_count += 1
    print(f"  Samples with rgb + depth + .pt: {ok_count} / {n}")

    # Sample a few: open image and depth, check shape and depth range
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("(Skip sample open: PIL/numpy not available)")
        return

    errors_open: list[str] = []
    for i in [0, n // 2, n - 1]:
        if i >= len(lines):
            continue
        line = lines[i]
        parts = line.split()
        if len(parts) < 2:
            continue
        rgb_rel = parts[0].lstrip("/")
        depth_rel = parts[1].lstrip("/")
        rgb_path = os.path.join(DATA_ROOT, rgb_rel)
        depth_path = os.path.join(DATA_ROOT, depth_rel)
        if not os.path.isfile(rgb_path) or not os.path.isfile(depth_path):
            continue
        try:
            img = np.array(Image.open(rgb_path))
            if img.ndim != 3 or img.shape[2] != 3:
                errors_open.append(f"{rgb_rel} shape {img.shape}")
            depth = np.array(Image.open(depth_path))
            if depth.ndim != 2 and (depth.ndim != 3 or depth.shape[-1] != 1):
                errors_open.append(f"{depth_rel} shape {depth.shape}")
            d_min, d_max = depth.min(), depth.max()
            if d_max <= 0:
                errors_open.append(f"{depth_rel} range [{d_min}, {d_max}]")
        except Exception as e:
            errors_open.append(f"{rgb_rel} / {depth_rel}: {e}")

    if errors_open:
        print("Sample open errors:")
        for e in errors_open:
            print(f"  - {e}")
    else:
        print("Sample open (first/mid/last): OK")

    if ok_count == n and not errors_open and not bad_focal:
        print("Result: PASS")
    else:
        print("Result: FAIL")


if __name__ == "__main__":
    main()
