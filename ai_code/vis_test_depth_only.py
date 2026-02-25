#!/usr/bin/env python3
"""
Test set depth map 시각화 전용: GT depth만 로드해서 컬러맵(0–8 m, jet)으로 저장.

실행 (repo root에서):
  python ai_code/vis_test_depth_only.py \\
    --data_root ./data/nyu_v2_sync \\
    --filenames_file ./data_splits/nyudepthv2_test_split.txt \\
    --output_dir ./runs/nyu_train_paper/vis_depth_test \\
    --depth_scale 6553.5 \\
    --num_samples 8

옵션: --num_samples 생략 시 파일 전체 사용 (한 figure에 다 넣거나 여러 figure로 나눔).
"""

import argparse
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEPTH_VIS_VMIN = 0.0
DEPTH_VIS_VMAX = 8.0
DEPTH_VIS_CMAP = "jet"
DEPTH_VIS_MAX_VALID_M = 10.0
DEFAULT_CELL_HEIGHT = 384
# Matplotlib has ~2^16 pixel limit per axis; chunk figures to avoid overflow
MAX_ROWS_PER_FIGURE = 32


def _load_depth_meters(
    depth_path: str,
    depth_scale: float,
    input_height: int = 480,
    input_width: int = 640,
) -> np.ndarray:
    """Load depth PNG, scale to meters (raw / depth_scale). Return (H,W) float32."""
    depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_cv is None:
        raise FileNotFoundError(f"Cannot read depth: {depth_path}")
    depth = depth_cv.astype(np.float32) / depth_scale
    if depth.shape[0] != input_height or depth.shape[1] != input_width:
        depth = cv2.resize(
            depth, (input_width, input_height), interpolation=cv2.INTER_NEAREST
        )
    return depth


def _depth_to_color(
    depth: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_name: str = DEPTH_VIS_CMAP,
    invalid_color: Tuple[int, int, int] = (128, 128, 128),
    max_valid: Optional[float] = None,
) -> np.ndarray:
    """Depth (H,W) in meters → RGB (H,W,3) 0–255. Invalid → gray."""
    valid = np.logical_and(np.isfinite(depth), depth > 0)
    if max_valid is not None:
        valid = np.logical_and(valid, depth <= max_valid)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    out[:, :, 0] = invalid_color[0]
    out[:, :, 1] = invalid_color[1]
    out[:, :, 2] = invalid_color[2]
    if valid.sum() == 0:
        return out

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized = np.zeros_like(depth, dtype=np.float32)
    if vmax > vmin:
        normalized[valid] = (np.clip(depth[valid], vmin, vmax) - vmin) / (vmax - vmin)
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    out[valid] = colored[valid]
    return out


def _resize_for_cell(img: np.ndarray, cell_height: int) -> np.ndarray:
    """Resize so height = cell_height, keep aspect ratio."""
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = cell_height / h
    new_w = int(round(w * scale))
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    return cv2.resize(img, (new_w, cell_height), interpolation=interp)


def _read_sample_list(
    filenames_file: str,
    data_root: str,
    num_samples: Optional[int],
) -> List[str]:
    """Return list of depth paths (absolute). Each line: rgb_path depth_path focal."""
    data_root = data_root.rstrip("/")
    with open(filenames_file) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    depth_paths: List[str] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 2:
            continue
        # depth path may have leading slash
        depth_rel = parts[1].lstrip("/")
        full = os.path.join(data_root, depth_rel)
        if os.path.isfile(full):
            depth_paths.append(full)
        if num_samples is not None and len(depth_paths) >= num_samples:
            break
    return depth_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize test set depth maps only (no model)."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/nyu_v2_sync",
        help="Root for data (same as gt_path; paths in split are relative to this).",
    )
    parser.add_argument(
        "--filenames_file",
        type=str,
        default="./data_splits/nyudepthv2_test_split.txt",
        help="Test split file: each line rgb_path depth_path focal.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./runs/vis_depth_test",
        help="Directory to save figure(s).",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=6553.5,
        help="raw / depth_scale = meters. 6553.5 for converted sync.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of test samples in one figure (default 8). Ignored if --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all samples in filenames_file (one figure; can be very tall).",
    )
    parser.add_argument(
        "--input_height",
        type=int,
        default=480,
    )
    parser.add_argument(
        "--input_width",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--cell_height",
        type=int,
        default=DEFAULT_CELL_HEIGHT,
        help="Row height in pixels for each depth in the figure.",
    )
    parser.add_argument(
        "--vis_scale",
        type=str,
        default="fixed",
        choices=("fixed", "adaptive"),
        help="fixed: 0–8 m; adaptive: percentile 1/99.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.filenames_file):
        logger.error("Not found: %s", args.filenames_file)
        return
    if not os.path.isdir(args.data_root):
        logger.error("Data root not found: %s", args.data_root)
        return

    n_limit = None if args.all else args.num_samples
    depth_paths = _read_sample_list(
        args.filenames_file, args.data_root, n_limit
    )
    if not depth_paths:
        logger.error("No depth paths found from %s under %s", args.filenames_file, args.data_root)
        return

    logger.info("Loading %s depth maps...", len(depth_paths))
    all_depth: List[np.ndarray] = []
    for p in depth_paths:
        try:
            d = _load_depth_meters(
                p, args.depth_scale, args.input_height, args.input_width
            )
            all_depth.append(d)
        except Exception as e:
            logger.warning("Skip %s: %s", p, e)

    if not all_depth:
        logger.error("No depth loaded.")
        return

    # Depth range for colormap
    if args.vis_scale == "adaptive":
        all_flat = []
        for d in all_depth:
            v = d[np.logical_and(np.isfinite(d), np.logical_and(d > 0, d <= DEPTH_VIS_MAX_VALID_M))]
            if v.size > 0:
                all_flat.append(v.ravel())
        if all_flat:
            concat = np.concatenate(all_flat)
            vmin = float(np.percentile(concat, 1))
            vmax = float(np.percentile(concat, 99))
            vmin = max(DEPTH_VIS_VMIN, vmin)
            vmax = min(DEPTH_VIS_VMAX, max(vmax, vmin + 0.5))
            logger.info("Adaptive vis range: %.2f – %.2f m", vmin, vmax)
        else:
            vmin, vmax = DEPTH_VIS_VMIN, DEPTH_VIS_VMAX
    else:
        vmin, vmax = DEPTH_VIS_VMIN, DEPTH_VIS_VMAX

    # Build colored cells for all depths
    rows = []
    for d in all_depth:
        color = _depth_to_color(d, vmin, vmax, max_valid=None)
        cell = _resize_for_cell(color, args.cell_height)
        rows.append(cell)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    w_max = max(r.shape[1] for r in rows)
    n_total = len(rows)
    chunk_size = MAX_ROWS_PER_FIGURE
    num_figs = (n_total + chunk_size - 1) // chunk_size

    for fig_idx in range(num_figs):
        start = fig_idx * chunk_size
        end = min(start + chunk_size, n_total)
        chunk = rows[start:end]
        padded = []
        for r in chunk:
            if r.shape[1] < w_max:
                pad = np.zeros((r.shape[0], w_max - r.shape[1], 3), dtype=r.dtype)
                pad[:] = 128
                r = np.concatenate([r, pad], axis=1)
            else:
                r = r[:, :w_max]
            padded.append(r)
        fig_img = np.concatenate(padded, axis=0)

        fig, ax = plt.subplots(
            1, 1, figsize=(fig_img.shape[1] / 100 + 1.2, fig_img.shape[0] / 100)
        )
        ax.imshow(fig_img)
        ax.axis("off")
        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap(DEPTH_VIS_CMAP),
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, shrink=0.4)
        cbar.set_label("Depth Value (m)", fontsize=11)
        plt.tight_layout(pad=0)

        if num_figs == 1:
            out_path = os.path.join(args.output_dir, "vis_test_depth_only.png")
        else:
            out_path = os.path.join(
                args.output_dir, f"vis_test_depth_only_{fig_idx + 1:03d}.png"
            )
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.2, dpi=150)
        plt.close()
        logger.info("Saved: %s (samples %s–%s)", out_path, start + 1, end)

    logger.info("Done. %s figures, %s samples total.", num_figs, n_total)


if __name__ == "__main__":
    main()
