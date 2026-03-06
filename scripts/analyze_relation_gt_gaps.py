#!/usr/bin/env python3
"""
Train set relation pair들의 GT depth gap 분포를 계산합니다.
학습 시 사용하는 rel_margin을 정할 때 참고: margin을 25th percentile 이하로 두면
대부분의 쌍에서 "순서만 맞추면 됨"이고, 무리한 간격 강제를 피할 수 있습니다.

사용 예 (repo root에서):
  python scripts/analyze_relation_gt_gaps.py \
    --relations_dir_train ./data/nyu_relational/statistical_train \
    --filenames_file ./data_splits/nyudepthv2_train_split.txt \
    --gt_path ./data/nyu_v2_sync \
    --depth_scale 6553.5 \
    --rel_statistical_alpha 1.0

  # 가장 빠름: CPU 멀티프로세싱 (I/O·연산 병렬화)
  python scripts/analyze_relation_gt_gaps.py ... --workers 8

  # GPU 사용 (단일 프로세스, 전송 비용으로 보통 CPU보다 느림):
  python scripts/analyze_relation_gt_gaps.py ... --use_gpu
"""
import os
import sys
import argparse
import json
import multiprocessing as mp
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def _scene_and_basename(rgb_file):
    """filenames_file의 rgb 경로에서 scene( train/ 제거 )과 rgb_basename 반환."""
    rgb_file = rgb_file.replace("\\", "/").lstrip("/")
    scene_name = os.path.dirname(rgb_file)
    first = scene_name.split("/", 1)[0]
    if first in ("train", "test"):
        scene_name = scene_name.split("/", 1)[1] if "/" in scene_name else ""
    rgb_basename = os.path.basename(rgb_file).replace(".jpg", "").replace(".png", "")
    return scene_name, rgb_basename


def _depth_path_from_split(gt_path, line, use_dense_depth=True):
    """한 줄에서 depth 파일 경로 구성."""
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    depth_file = parts[1].lstrip("/").replace("\\", os.sep)
    return os.path.normpath(os.path.join(gt_path, depth_file))


def _rgb_file_from_split(line):
    """한 줄에서 rgb 경로 (상대) 추출."""
    parts = line.strip().split()
    if not parts:
        return None
    return parts[0].lstrip("/").replace("\\", "/")


def compute_rep(depth_meters, mask, alpha=1.0, min_valid=1e-3):
    """마스크 내 유효 깊이로 대표값 rep = mu + alpha*sigma 계산. 실패 시 None."""
    depth_flat = depth_meters[mask]
    depth_flat = depth_flat[depth_flat > min_valid]
    if depth_flat.size == 0:
        return None
    mu = float(np.mean(depth_flat))
    sigma = float(np.std(depth_flat))
    return mu + alpha * sigma


def compute_rep_gpu(depth_t, mask_t, alpha=1.0, min_valid=1e-3):
    """GPU 텐서로 rep = mu + alpha*sigma 계산. 실패 시 None."""
    depth_flat = depth_t[mask_t]
    valid = depth_flat > min_valid
    if valid.sum().item() == 0:
        return None
    depth_valid = depth_flat[valid]
    mu = depth_valid.mean()
    sigma = depth_valid.std()
    if torch.isnan(sigma) or sigma.item() < 1e-9:
        sigma = 0.0
    return (mu + alpha * sigma).item()


def _process_chunk(args_tuple):
    """워커 프로세스용: lines 청크 처리 후 (gaps, n_skip_rel, n_skip_depth, n_skip_masks, n_pairs, n_violations) 반환."""
    (lines_chunk, relations_dir_train, gt_path, depth_scale, rel_statistical_alpha,
     rel_min_pixels, use_dense_depth) = args_tuple
    import cv2
    gaps = []
    n_skip_rel = n_skip_depth = n_skip_masks = n_violations = 0
    for line in lines_chunk:
        rgb_file = _rgb_file_from_split(line)
        if not rgb_file:
            continue
        scene_name, rgb_basename = _scene_and_basename(rgb_file)
        depth_path = _depth_path_from_split(gt_path, line, use_dense_depth)
        if not depth_path or not os.path.isfile(depth_path):
            n_skip_depth += 1
            continue
        rel_path = os.path.join(relations_dir_train, scene_name, f"{rgb_basename}_relations.json")
        mask_path = os.path.join(relations_dir_train, scene_name, f"{rgb_basename}_masks.npy")
        if not os.path.isfile(rel_path):
            n_skip_rel += 1
            continue
        if not os.path.isfile(mask_path):
            n_skip_masks += 1
            continue
        try:
            depth_raw = np.load(depth_path) if depth_path.endswith(".npy") else None
            if depth_raw is None:
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                n_skip_depth += 1
                continue
            depth_meters = np.asarray(depth_raw, dtype=np.float64) / depth_scale
            while depth_meters.ndim > 2:
                depth_meters = depth_meters.squeeze(-1) if depth_meters.shape[-1] == 1 else depth_meters[:, :, 0]
            masks = np.load(mask_path)
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]
            if masks.shape[1:] != depth_meters.shape[:2]:
                d_h, d_w = depth_meters.shape[:2]
                masks_resized = np.zeros((masks.shape[0], d_h, d_w), dtype=masks.dtype)
                for k in range(masks.shape[0]):
                    masks_resized[k] = cv2.resize(
                        masks[k].astype(np.uint8), (d_w, d_h), interpolation=cv2.INTER_NEAREST
                    )
                masks = masks_resized
            with open(rel_path, "r") as f:
                relations = json.load(f)
            if not relations:
                continue
            for rel in relations:
                i, j = rel.get("subject_idx"), rel.get("object_idx")
                if i is None or j is None or i >= masks.shape[0] or j >= masks.shape[0]:
                    continue
                rel_type = str(rel.get("relation", "front")).lower()
                if rel_type not in ("front", "behind"):
                    continue
                # front => subject closer (rep_i < rep_j); behind => object closer (rep_j < rep_i)
                closer_idx, farther_idx = (i, j) if rel_type == "front" else (j, i)
                mask_i = masks[closer_idx] > 0.5
                mask_j = masks[farther_idx] > 0.5
                if np.sum(mask_i) < rel_min_pixels or np.sum(mask_j) < rel_min_pixels:
                    continue
                rep_c = compute_rep(depth_meters, mask_i, rel_statistical_alpha)
                rep_f = compute_rep(depth_meters, mask_j, rel_statistical_alpha)
                if rep_c is None or rep_f is None:
                    continue
                gap = abs(rep_c - rep_f)
                gaps.append(gap)
                # violation: GT says "closer" should be nearer; if rep_closer >= rep_farther, label is inconsistent
                if rep_c >= rep_f:
                    n_violations += 1
        except Exception:
            continue
    return (gaps, n_skip_rel, n_skip_depth, n_skip_masks, len(gaps), n_violations)


def main():
    parser = argparse.ArgumentParser(description="Analyze GT depth gaps of train relation pairs")
    parser.add_argument("--relations_dir_train", type=str, required=True,
                        help="Path to relations dir (e.g. ./data/nyu_relational/statistical_train)")
    parser.add_argument("--filenames_file", type=str, required=True,
                        help="Train split file (e.g. nyudepthv2_train_split.txt)")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="GT depth root (e.g. ./data/nyu_v2_sync)")
    parser.add_argument("--depth_scale", type=float, default=6553.5,
                        help="Raw depth / depth_scale = meters (default 6553.5 for NYU)")
    parser.add_argument("--rel_statistical_alpha", type=float, default=1.0,
                        help="Same as rel_statistical_alpha in config (rep = mu + alpha*sigma)")
    parser.add_argument("--rel_min_pixels", type=int, default=20,
                        help="Skip objects with valid pixels < this (match training)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of train samples to process (default: all)")
    parser.add_argument("--use_dense_depth", action="store_true", default=True,
                        help="Use dense depth path from split (default True)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for rep (single process only); usually slower than --workers")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of CPU processes (default 1). 4~8 권장, 가장 빠름.")
    args = parser.parse_args()

    n_workers = max(1, int(args.workers))
    use_gpu = n_workers == 1 and args.use_gpu and torch is not None and torch.cuda.is_available()
    if args.use_gpu and n_workers > 1:
        print("Note: --workers > 1 이면 --use_gpu 무시, CPU 멀티프로세싱 사용.", file=sys.stderr)
    if args.use_gpu and not use_gpu and n_workers == 1:
        if torch is None:
            print("Warning: PyTorch not found, using CPU.", file=sys.stderr)
        elif not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU.", file=sys.stderr)
    device = torch.device("cuda" if use_gpu else "cpu") if torch is not None else None

    if not os.path.isfile(args.filenames_file):
        print(f"Error: filenames_file not found: {args.filenames_file}", file=sys.stderr)
        sys.exit(1)
    with open(args.filenames_file, "r") as f:
        lines = [ln for ln in f if ln.strip()]
    if args.max_samples is not None:
        lines = lines[: args.max_samples]

    gaps = []
    num_skipped_no_rel = 0
    num_skipped_no_depth = 0
    num_skipped_no_masks = 0
    num_pairs = 0
    num_violations = 0

    if n_workers > 1:
        # 멀티프로세싱: 청크별로 병렬 처리 (가장 빠름)
        chunk_size = max(1, (len(lines) + n_workers - 1) // n_workers)
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
        args_tuples = [
            (chunk, args.relations_dir_train, args.gt_path, args.depth_scale,
             args.rel_statistical_alpha, args.rel_min_pixels, args.use_dense_depth)
            for chunk in chunks
        ]
        with mp.Pool(n_workers) as pool:
            results = pool.map(_process_chunk, args_tuples)
        for (gaps_chunk, a, b, c, n, v) in results:
            gaps.extend(gaps_chunk)
            num_skipped_no_rel += a
            num_skipped_no_depth += b
            num_skipped_no_masks += c
            num_pairs += n
            num_violations += v
    else:
        # 단일 프로세스 (기존 로직, optional GPU)
        import cv2
        for idx, line in enumerate(lines):
            rgb_file = _rgb_file_from_split(line)
            if not rgb_file:
                continue
            scene_name, rgb_basename = _scene_and_basename(rgb_file)
            depth_path = _depth_path_from_split(args.gt_path, line, args.use_dense_depth)
            if not depth_path or not os.path.isfile(depth_path):
                num_skipped_no_depth += 1
                continue
            rel_path = os.path.join(
                args.relations_dir_train, scene_name, f"{rgb_basename}_relations.json"
            )
            mask_path = os.path.join(
                args.relations_dir_train, scene_name, f"{rgb_basename}_masks.npy"
            )
            if not os.path.isfile(rel_path):
                num_skipped_no_rel += 1
                continue
            if not os.path.isfile(mask_path):
                num_skipped_no_masks += 1
                continue
            try:
                depth_raw = np.load(depth_path) if depth_path.endswith(".npy") else None
                if depth_raw is None:
                    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_raw is None:
                    num_skipped_no_depth += 1
                    continue
                depth_meters = np.asarray(depth_raw, dtype=np.float64) / args.depth_scale
                while depth_meters.ndim > 2:
                    depth_meters = depth_meters.squeeze(-1) if depth_meters.shape[-1] == 1 else depth_meters[:, :, 0]
                masks = np.load(mask_path)
                if masks.ndim == 2:
                    masks = masks[np.newaxis, ...]
                if masks.shape[1:] != depth_meters.shape[:2]:
                    d_h, d_w = depth_meters.shape[:2]
                    masks_resized = np.zeros((masks.shape[0], d_h, d_w), dtype=masks.dtype)
                    for k in range(masks.shape[0]):
                        masks_resized[k] = cv2.resize(
                            masks[k].astype(np.uint8), (d_w, d_h), interpolation=cv2.INTER_NEAREST
                        )
                    masks = masks_resized
                with open(rel_path, "r") as f:
                    relations = json.load(f)
                if not relations:
                    continue
                if use_gpu and device is not None:
                    depth_t = torch.from_numpy(depth_meters.astype(np.float32)).to(device)
                    masks_t = torch.from_numpy(masks.astype(np.float32)).to(device)
                    for rel in relations:
                        i, j = rel.get("subject_idx"), rel.get("object_idx")
                        if i is None or j is None or i >= masks_t.shape[0] or j >= masks_t.shape[0]:
                            continue
                        rel_type = str(rel.get("relation", "front")).lower()
                        if rel_type not in ("front", "behind"):
                            continue
                        closer_idx, farther_idx = (i, j) if rel_type == "front" else (j, i)
                        mask_c = masks_t[closer_idx] > 0.5
                        mask_f = masks_t[farther_idx] > 0.5
                        if mask_c.sum().item() < args.rel_min_pixels or mask_f.sum().item() < args.rel_min_pixels:
                            continue
                        rep_c = compute_rep_gpu(depth_t, mask_c, args.rel_statistical_alpha)
                        rep_f = compute_rep_gpu(depth_t, mask_f, args.rel_statistical_alpha)
                        if rep_c is None or rep_f is None:
                            continue
                        gaps.append(abs(rep_c - rep_f))
                        if rep_c >= rep_f:
                            num_violations += 1
                else:
                    for rel in relations:
                        i, j = rel.get("subject_idx"), rel.get("object_idx")
                        if i is None or j is None or i >= masks.shape[0] or j >= masks.shape[0]:
                            continue
                        rel_type = str(rel.get("relation", "front")).lower()
                        if rel_type not in ("front", "behind"):
                            continue
                        closer_idx, farther_idx = (i, j) if rel_type == "front" else (j, i)
                        mask_c = masks[closer_idx] > 0.5
                        mask_f = masks[farther_idx] > 0.5
                        if np.sum(mask_c) < args.rel_min_pixels or np.sum(mask_f) < args.rel_min_pixels:
                            continue
                        rep_c = compute_rep(depth_meters, mask_c, args.rel_statistical_alpha)
                        rep_f = compute_rep(depth_meters, mask_f, args.rel_statistical_alpha)
                        if rep_c is None or rep_f is None:
                            continue
                        gaps.append(abs(rep_c - rep_f))
                        if rep_c >= rep_f:
                            num_violations += 1
            except Exception as e:
                if idx < 3:
                    print(f"Warning: {rel_path}: {e}", file=sys.stderr)
                continue

    num_pairs = len(gaps)
    gaps = np.array(gaps, dtype=np.float64)
    print("=" * 60)
    print("GT depth gap distribution (train relation pairs)")
    print("  rep = mu + alpha*sigma per object, gap = |rep_i - rep_j|")
    print("=" * 60)
    print(f"  relations_dir_train: {args.relations_dir_train}")
    print(f"  filenames_file:       {args.filenames_file}")
    print(f"  rel_statistical_alpha: {args.rel_statistical_alpha}")
    print(f"  depth_scale:          {args.depth_scale}")
    print(f"  workers:              {n_workers}")
    print(f"  use_gpu:              {use_gpu}")
    print(f"  Samples processed:   {len(lines)}")
    print(f"  Skipped (no rel):     {num_skipped_no_rel}")
    print(f"  Skipped (no depth):   {num_skipped_no_depth}")
    print(f"  Skipped (no masks):   {num_skipped_no_masks}")
    print(f"  Total pairs:          {num_pairs}")
    if num_pairs > 0:
        vrate = 100.0 * num_violations / num_pairs
        print(f"  GT label violations:  {num_violations} ({vrate:.2f}%)  [rep_closer >= rep_farther]")
        print("  -> High violation rate suggests relation/mask noise or rep definition mismatch.")
    print()
    if num_pairs == 0:
        print("No pairs collected. Check paths and relation files.")
        sys.exit(1)
    print("Gap (meters) distribution:")
    print(f"  mean:    {np.mean(gaps):.4f}")
    print(f"  median:  {np.median(gaps):.4f}")
    print(f"  std:     {np.std(gaps):.4f}")
    print(f"  min:     {np.min(gaps):.4f}")
    print(f"  max:     {np.max(gaps):.4f}")
    print()
    print("Percentiles (for margin choice):")
    for p in [90, 75, 50, 25, 10, 5]:
        print(f"  {p:2d}th:   {np.percentile(gaps, p):.4f}")
    print()
    print("Recommendation:")
    med = np.median(gaps)
    p75 = np.percentile(gaps, 75)
    p25 = np.percentile(gaps, 25)
    p10 = np.percentile(gaps, 10)
    print(f"  - If rel_margin > median ({med:.3f}), many pairs are forced apart more than GT -> lower margin.")
    print(f"  - If rel_margin > 75th ({p75:.3f}) -> strong over-constraint; consider margin <= median or 25th.")
    print(f"  - margin <= 25th ({p25:.3f}) -> for most pairs only order matters (minimal gap required).")
    print(f"  - margin <= 10th ({p10:.3f}) -> very conservative (order-only for even more pairs).")
    print("=" * 60)


if __name__ == "__main__":
    main()
