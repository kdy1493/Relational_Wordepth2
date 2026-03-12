"""
Compare three models in one figure: Input | GT | Baseline | WorDepth | WorDepth+Rel.

Usage (from repo root):
  python src/visualize_three_models.py @configs/arguments_eval_nyu_all_wordepth.yaml \\
    --ckpt_baseline ./runs/nyu_train_baseline_exp3_effbatch24_adameps0001_nolegacy/model-45000-best_abs_rel_0.05853 \\
    --ckpt_wordepth ./runs/nyu_train_wordepth_paper_exp2_effbatch24/model-45000-best_abs_rel_0.03969 \\
    --ckpt_relational ./runs/nyu_train_wordepth_relational_exp4_effbatch24_stat_alpha10_lambda005_margin00/model-45000-best_abs_rel_0.03893 \\
    --output_dir ./runs/vis_three_models --num_samples 4
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import convert_arg_line_to_args, expand_argv_yaml
from networks.wordepth import WorDepth

# Reuse helpers from pred_visualize
from pred_visualize import (
    _load_and_preprocess_image,
    _load_gt_depth,
    _depth_to_color,
    _resize_for_cell,
    _text_feat_pt_path,
    DEPTH_VIS_CMAP,
    DEPTH_VIS_MAX_VALID_M,
    DEPTH_VIS_VMIN,
    DEPTH_VIS_VMAX,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COLUMN_LABELS = ["Input", "GT", "Baseline", "WorDepth", "WorDepth+Rel"]


def _load_model(
    checkpoint_path: str,
    baseline_arch: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model") or ckpt
    if any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
    if any(k.startswith("_orig_mod.") for k in model_state.keys()):
        model_state = {k.replace("_orig_mod.", "", 1): v for k, v in model_state.items()}

    model = WorDepth(
        pretrained=getattr(args, "pretrain", None),
        max_depth=args.max_depth,
        prior_mean=getattr(args, "prior_mean", 1.54),
        img_size=(args.input_height, args.input_width),
        weight_kld=getattr(args, "weight_kld", 1e-3),
        alter_prob=getattr(args, "alter_prob", 0.5),
        legacy=getattr(args, "legacy", False),
        baseline_arch=baseline_arch,
    )
    model.load_state_dict(model_state, strict=not baseline_arch)
    model = model.to(device)
    model.eval()
    return model


def _pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    if img.shape[1] >= target_w:
        return img[:, :target_w]
    pad_w = target_w - img.shape[1]
    if img.ndim == 3:
        pad = np.zeros((img.shape[0], pad_w, img.shape[2]), dtype=img.dtype)
    else:
        pad = np.zeros((img.shape[0], pad_w), dtype=img.dtype)
    return np.concatenate([img, pad], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Baseline vs WorDepth vs WorDepth+Rel", fromfile_prefix_chars="@")
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument("--ckpt_baseline", type=str, required=True, help="Baseline checkpoint path")
    parser.add_argument("--ckpt_wordepth", type=str, required=True, help="WorDepth checkpoint path")
    parser.add_argument("--ckpt_relational", type=str, required=True, help="WorDepth+Relational checkpoint path")
    parser.add_argument("--output_dir", type=str, default="./runs/vis_three_models")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None, help="random seed for sampling images; omit for different images each run")
    parser.add_argument("--cell_height", type=int, default=320)
    parser.add_argument("--vis_scale", type=str, default="fixed", choices=("fixed", "adaptive"))
    parser.add_argument("--no_align_pred_scale", action="store_true", help="do not median-scale align pred to GT")
    # Eval config (from YAML)
    parser.add_argument("--dataset", type=str, default="nyu")
    parser.add_argument("--data_path_eval", type=str, default=None)
    parser.add_argument("--gt_path_eval", type=str, default=None)
    parser.add_argument("--filenames_file_eval", type=str, default=None)
    parser.add_argument("--depth_scale", type=float, default=6553.5)
    parser.add_argument("--input_height", type=int, default=480)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_depth_eval", type=float, default=1e-3)
    parser.add_argument("--max_depth_eval", type=float, default=80.0)
    parser.add_argument("--prior_mean", type=float, default=1.54)
    parser.add_argument("--weight_kld", type=float, default=1e-3)
    parser.add_argument("--alter_prob", type=float, default=0.5)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--do_kb_crop", action="store_true")
    parser.add_argument("--eigen_crop", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default=None)
    # Ignored (from eval YAML when using @config.yaml)
    parser.add_argument("--checkpoint_path", type=str, default="", help="ignored; we use --ckpt_*")
    parser.add_argument("--post_process", action="store_true", help="ignored")
    parser.add_argument("--baseline_arch", action="store_true", help="ignored")

    argv = expand_argv_yaml(sys.argv[1:], _REPO_ROOT)
    args = parser.parse_args(argv)

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids).strip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = args.data_path_eval or getattr(args, "data_path", None)
    gt_path = args.gt_path_eval or getattr(args, "gt_path", None)
    if not args.filenames_file_eval or not data_path or not gt_path:
        logger.error("Need --filenames_file_eval, --data_path_eval, --gt_path_eval (or @eval YAML).")
        sys.exit(1)

    # Read all samples, then randomly pick num_samples (different images each run unless --seed set)
    all_lines: List[Tuple[str, str, str]] = []
    with open(args.filenames_file_eval) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rgb_rel = parts[0].lstrip("/")
            depth_rel = parts[1].lstrip("/")
            image_path = os.path.join(data_path, rgb_rel)
            depth_path = os.path.join(gt_path, depth_rel)
            sample_key = rgb_rel[:-4] if rgb_rel.endswith(".jpg") else rgb_rel
            all_lines.append((image_path, depth_path, sample_key))
    if args.seed is not None:
        random.seed(args.seed)
    n_take = min(args.num_samples, len(all_lines))
    samples = random.sample(all_lines, n_take)
    if not samples:
        logger.error("No samples read.")
        sys.exit(1)
    logger.info("Randomly selected %d samples (seed=%s).", len(samples), args.seed)

    logger.info("Loading three models...")
    model_baseline = _load_model(args.ckpt_baseline, True, args, device)
    model_wordepth = _load_model(args.ckpt_wordepth, False, args, device)
    model_relational = _load_model(args.ckpt_relational, False, args, device)

    text_feat_base = os.path.join(_REPO_ROOT, "data", "text_feat", args.dataset)
    all_rgb: List[np.ndarray] = []
    all_gt: List[np.ndarray] = []
    all_pred_b: List[np.ndarray] = []
    all_pred_w: List[np.ndarray] = []
    all_pred_r: List[np.ndarray] = []

    for image_path, depth_path, sample_key in samples:
        pt_path = _text_feat_pt_path(os.path.join(text_feat_base, "test"), sample_key)
        if pt_path is None:
            pt_path = _text_feat_pt_path(os.path.join(text_feat_base, "train"), sample_key)
        if pt_path is None:
            logger.warning("No text feature for %s; skip.", sample_key)
            continue

        tensor, rgb_display = _load_and_preprocess_image(
            image_path, args.input_height, args.input_width, args.do_kb_crop
        )
        tensor = tensor.to(device)
        gt = _load_gt_depth(
            depth_path, args.dataset, args.do_kb_crop,
            args.input_height, args.input_width, args.depth_scale
        )

        # Text: baseline = zeros; others = load
        text_b = torch.zeros(1, 1024, device=device, dtype=torch.float32)
        text_w = torch.load(pt_path, map_location=device)
        if isinstance(text_w, torch.Tensor) and text_w.dim() == 1:
            text_w = text_w.unsqueeze(0)
        text_w = text_w.to(device)
        text_r = text_w.clone()

        with torch.no_grad():
            pred_b = model_baseline(tensor, text_b, sample_from_gaussian=False)
            pred_w = model_wordepth(tensor, text_w, sample_from_gaussian=False)
            pred_r = model_relational(tensor, text_r, sample_from_gaussian=False)

        def to_np(x: torch.Tensor) -> np.ndarray:
            out = x.squeeze().cpu().numpy()
            if out.shape != gt.shape:
                out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            out = np.clip(out, args.min_depth_eval, args.max_depth_eval)
            out[np.isinf(out)] = args.max_depth_eval
            out[np.isnan(out)] = args.min_depth_eval
            return out

        all_rgb.append(rgb_display)
        all_gt.append(gt)
        all_pred_b.append(to_np(pred_b))
        all_pred_w.append(to_np(pred_w))
        all_pred_r.append(to_np(pred_r))

    if not all_rgb:
        logger.error("No samples processed.")
        sys.exit(1)

    # Optional median scale alignment per image
    if not args.no_align_pred_scale:
        for i in range(len(all_gt)):
            gt_v = all_gt[i][np.logical_and(np.isfinite(all_gt[i]), all_gt[i] > 0)]
            for pred_list in (all_pred_b, all_pred_w, all_pred_r):
                pr_v = pred_list[i][np.logical_and(np.isfinite(pred_list[i]), pred_list[i] > 0)]
                if gt_v.size > 0 and pr_v.size > 0:
                    mg, mp = np.median(gt_v), np.median(pr_v)
                    if mp > 1e-6:
                        pred_list[i] = pred_list[i] * (mg / mp)

    # Depth color range
    if args.vis_scale == "adaptive":
        all_depths = []
        for d in all_gt + all_pred_b + all_pred_w + all_pred_r:
            v = d[np.logical_and(np.isfinite(d), np.logical_and(d > 0, d <= DEPTH_VIS_MAX_VALID_M))]
            if v.size > 0:
                all_depths.append(v.ravel())
        if all_depths:
            concat = np.concatenate(all_depths)
            vmin = float(np.percentile(concat, 1))
            vmax = float(np.percentile(concat, 99))
            vmin = max(DEPTH_VIS_VMIN, vmin)
            vmax = min(DEPTH_VIS_VMAX, max(vmax, vmin + 0.5))
        else:
            vmin, vmax = DEPTH_VIS_VMIN, DEPTH_VIS_VMAX
    else:
        vmin, vmax = DEPTH_VIS_VMIN, DEPTH_VIS_VMAX

    cell_h = args.cell_height
    ncols = 5
    rows = []
    for i in range(len(all_rgb)):
        rgb_cell = _resize_for_cell(all_rgb[i], cell_h)
        gt_color = _depth_to_color(all_gt[i], vmin, vmax, max_valid=None)
        pred_b_color = _depth_to_color(all_pred_b[i], vmin, vmax, max_valid=DEPTH_VIS_MAX_VALID_M)
        pred_w_color = _depth_to_color(all_pred_w[i], vmin, vmax, max_valid=DEPTH_VIS_MAX_VALID_M)
        pred_r_color = _depth_to_color(all_pred_r[i], vmin, vmax, max_valid=DEPTH_VIS_MAX_VALID_M)

        gt_cell = _resize_for_cell(gt_color, cell_h)
        pred_b_cell = _resize_for_cell(pred_b_color, cell_h)
        pred_w_cell = _resize_for_cell(pred_w_color, cell_h)
        pred_r_cell = _resize_for_cell(pred_r_color, cell_h)

        w_max = max(
            rgb_cell.shape[1], gt_cell.shape[1],
            pred_b_cell.shape[1], pred_w_cell.shape[1], pred_r_cell.shape[1],
        )
        rgb_cell = _pad_to_width(rgb_cell, w_max)
        gt_cell = _pad_to_width(gt_cell, w_max)
        pred_b_cell = _pad_to_width(pred_b_cell, w_max)
        pred_w_cell = _pad_to_width(pred_w_cell, w_max)
        pred_r_cell = _pad_to_width(pred_r_cell, w_max)

        row = np.concatenate([rgb_cell, gt_cell, pred_b_cell, pred_w_cell, pred_r_cell], axis=1)
        rows.append(row)

    fig_img = np.concatenate(rows, axis=0)

    # Column labels
    cell_w = fig_img.shape[1] // ncols
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    for c, label in enumerate(COLUMN_LABELS):
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        x = c * cell_w + (cell_w - tw) // 2
        y = 22
        cv2.putText(fig_img, label, (x, y), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(fig_img, label, (x, y), font, font_scale, (255, 255, 255), thickness)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(fig_img.shape[1] / 100 + 1.5, fig_img.shape[0] / 100))
    ax.imshow(fig_img)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(DEPTH_VIS_CMAP), norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, shrink=0.35)
    cbar.set_label("Depth (m)", fontsize=10)
    plt.tight_layout(pad=0)
    os.makedirs(args.output_dir, exist_ok=True)
    # Avoid overwrite: comparison_three_models.png, _2, _3, ...
    base_name = "comparison_three_models"
    ext = ".png"
    out_path = os.path.join(args.output_dir, base_name + ext)
    if os.path.exists(out_path):
        idx = 2
        while True:
            cand = os.path.join(args.output_dir, f"{base_name}_{idx}{ext}")
            if not os.path.exists(cand):
                out_path = cand
                break
            idx += 1
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.2, dpi=150)
    plt.close()
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
