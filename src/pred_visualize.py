"""
Visualization script: Input | GT | Ours layout, N rows per figure.

실행 명령어 (from repo root):
  # Train/val split 예시
  python src/pred_visualize.py @configs/arguments_run_nyu_paper.txt --checkpoint_path ./runs/nyu_train_paper/model-xxx --output_dir ./runs/nyu_train_paper/vis --num_samples 4
  # Test set (eval_results 와 동일 체크포인트)
  python src/pred_visualize.py @configs/arguments_eval_nyu_paper.txt --output_dir ./runs/nyu_train_paper/vis --num_samples 8
"""

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from utils import convert_arg_line_to_args
from networks.wordepth import WorDepth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _text_feat_pt_path(text_feat_dir: str, rgb_key: str) -> Optional[str]:
    """Resolve .pt path under text_feat_dir (train or test). Same logic as eval.py."""
    rel = rgb_key.lstrip("/")
    p1 = os.path.join(text_feat_dir, rel + ".pt")
    if os.path.isfile(p1):
        return p1
    if rel.startswith("train/"):
        p2 = os.path.join(text_feat_dir, rel[len("train/"):] + ".pt")
        if os.path.isfile(p2):
            return p2
    if rel.startswith("test/"):
        p2 = os.path.join(text_feat_dir, rel[len("test/"):] + ".pt")
        if os.path.isfile(p2):
            return p2
    return None


# ImageNet normalize (same as dataloader)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_VIS_SAMPLES = 4
COLUMN_LABELS = ["(a) Input", "(b) GT", "(c) Ours"]


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WorDepth visualization: Input | GT | Ours.",
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="directory to save figure")
    parser.add_argument("--num_samples", type=int, default=NUM_VIS_SAMPLES, help="number of rows (samples) per figure")
    parser.add_argument("--cell_height", type=int, default=384, help="height in pixels of each row cell (larger = sharper)")
    parser.add_argument("--dataset", type=str, default="nyu")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=6553.5,
        help="raw depth value / depth_scale = meters. Default 6553.5 for converted sync. Use 1000 for original mm.",
    )
    parser.add_argument("--data_path_eval", type=str, default=None)
    parser.add_argument("--gt_path_eval", type=str, default=None)
    parser.add_argument("--filenames_file_eval", type=str, default=None)
    parser.add_argument("--input_height", type=int, default=256)
    parser.add_argument("--input_width", type=int, default=320)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_depth_eval", type=float, default=1e-3)
    parser.add_argument("--max_depth_eval", type=float, default=80.0)
    parser.add_argument("--prior_mean", type=float, default=1.54)
    parser.add_argument("--weight_kld", type=float, default=1e-3)
    parser.add_argument("--alter_prob", type=float, default=0.5)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--baseline_arch", action="store_true", help="baseline model (no text); use with baseline checkpoint")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--do_kb_crop", action="store_true")
    parser.add_argument("--eigen_crop", action="store_true")

    # Ignored (from run's arguments file)
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0)
    parser.add_argument("--end_learning_rate", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--adam_eps", type=float, default=0)
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--do_random_rotate", action="store_true")
    parser.add_argument("--degree", type=float, default=0)
    parser.add_argument("--log_directory", type=str, default="")
    parser.add_argument("--log_freq", type=int, default=0)
    parser.add_argument("--do_online_eval", action="store_true")
    parser.add_argument("--eval_freq", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--cache_images", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--gt_path", type=str, default="")
    parser.add_argument("--filenames_file", type=str, default="")
    parser.add_argument("--post_process", action="store_true", help="ignored (eval config compatibility)")
    parser.add_argument(
        "--vis_scale",
        type=str,
        default="fixed",
        choices=("fixed", "adaptive"),
        help="fixed=0-8m; adaptive=1p/99p of valid depth",
    )
    parser.add_argument(
        "--no_align_pred_scale",
        action="store_true",
        help="do not scale pred to GT median (default: scale so pred colors spread like GT)",
    )

    return parser


def _parse_args() -> argparse.Namespace:
    return _make_parser().parse_args()


def _read_sample_list(
    filenames_file: str,
    data_path: str,
    gt_path: str,
    num_samples: int,
) -> List[Tuple[str, str, str]]:
    """Read first num_samples lines; return list of (image_path, depth_path, sample_key)."""
    with open(filenames_file) as f:
        lines = [line.strip() for line in f if line.strip()][:num_samples]
    out: List[Tuple[str, str, str]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        rgb_rel = parts[0].lstrip("/")
        depth_rel = parts[1].lstrip("/")
        image_path = os.path.join(data_path, rgb_rel)
        depth_path = os.path.join(gt_path, depth_rel)
        sample_key = rgb_rel[:-4] if rgb_rel.endswith(".jpg") else rgb_rel
        out.append((image_path, depth_path, sample_key))
    return out


def _load_and_preprocess_image(
    image_path: str,
    input_height: int,
    input_width: int,
    do_kb_crop: bool,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Load RGB, preprocess for model (resize, normalize), return tensor and RGB for display (0–255)."""
    image_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_cv is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if do_kb_crop:
        h, w = image.shape[:2]
        top = int(h - 352)
        left = int((w - 1216) / 2)
        image = image[top : top + 352, left : left + 1216, :]

    if image.shape[0] != input_height or image.shape[1] != input_width:
        image = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_LINEAR)

    rgb_for_display = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    image_norm = (image - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, rgb_for_display


def _load_gt_depth(
    depth_path: str,
    dataset: str,
    do_kb_crop: bool,
    input_height: int,
    input_width: int,
    depth_scale: float = 6553.5,
) -> np.ndarray:
    """Load GT depth (H,W), scale to meters (raw / depth_scale), optional crop/resize."""
    depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_cv is None:
        raise FileNotFoundError(f"Cannot read depth: {depth_path}")
    depth = depth_cv.astype(np.float32)
    if dataset == "nyu":
        depth = depth / depth_scale
    else:
        depth = depth / 256.0

    if do_kb_crop:
        h, w = depth.shape[:2]
        top = int(h - 352)
        left = int((w - 1216) / 2)
        depth = depth[top : top + 352, left : left + 1216]

    if depth.shape[0] != input_height or depth.shape[1] != input_width:
        depth = cv2.resize(depth, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    return depth


# Depth visualization: 0–8 m, blue (close) → red (far), matching paper-style colorbar
DEPTH_VIS_VMIN = 0.0
DEPTH_VIS_VMAX = 8.0
DEPTH_VIS_CMAP = "jet"  # blue → cyan → green → yellow → red
# Pixels with depth > this (m) are treated as invalid and shown gray (avoids "all red" from bad/max values)
DEPTH_VIS_MAX_VALID_M = 10.0


def _depth_to_color(
    depth: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_name: str = DEPTH_VIS_CMAP,
    invalid_color: Tuple[int, int, int] = (128, 128, 128),
    max_valid: Optional[float] = None,
) -> np.ndarray:
    """Normalize depth to [0,1] with vmin/vmax, apply colormap. Invalid (<=0 or non-finite) → gray; if max_valid is set, also gray out depth>max_valid. Return RGB (H,W,3) 0–255."""
    valid = np.logical_and(np.isfinite(depth), depth > 0)
    if max_valid is not None:
        valid = np.logical_and(valid, depth <= max_valid)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    out[:, :, 0] = invalid_color[0]
    out[:, :, 1] = invalid_color[1]
    out[:, :, 2] = invalid_color[2]
    if valid.sum() == 0:
        return out

    normalized = np.zeros_like(depth, dtype=np.float32)
    if vmax > vmin:
        normalized[valid] = (np.clip(depth[valid], vmin, vmax) - vmin) / (vmax - vmin)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    out[valid] = colored[valid]
    return out


def _resize_for_cell(img: np.ndarray, cell_height: int) -> np.ndarray:
    """Resize image so that height is cell_height, keep aspect ratio; then pad/crop to exact cell_height width."""
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = cell_height / h
    new_w = int(round(w * scale))
    resized = cv2.resize(img, (new_w, cell_height), interpolation=cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST)
    return resized


def main() -> None:
    args = _parse_args()

    if getattr(args, "gpu_ids", None) and str(args.gpu_ids).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids).strip()

    if not args.filenames_file_eval or not os.path.isfile(args.filenames_file_eval):
        logger.error("--filenames_file_eval is required and must exist (e.g. from @arguments_run_nyu.txt).")
        sys.exit(1)
    data_path = args.data_path_eval or getattr(args, "data_path", None)
    gt_path = args.gt_path_eval or getattr(args, "gt_path", None)
    if not data_path or not gt_path:
        logger.error("Data paths required. Use @run_dir/arguments_run_nyu.txt or set --data_path_eval and --gt_path_eval.")
        sys.exit(1)

    samples = _read_sample_list(
        args.filenames_file_eval,
        data_path,
        gt_path,
        args.num_samples,
    )
    if len(samples) == 0:
        logger.error("No samples read from %s", args.filenames_file_eval)
        sys.exit(1)
    logger.info("Visualizing %d samples.", len(samples))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model")
    if model_state is None:
        logger.error("Checkpoint has no 'model' key.")
        sys.exit(1)
    # Checkpoint may be from DataParallel/DDP (keys like "module.backbone....")
    if any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}

    baseline_arch = getattr(args, "baseline_arch", False)
    model = WorDepth(
        pretrained=args.pretrain,
        max_depth=args.max_depth,
        prior_mean=args.prior_mean,
        img_size=(args.input_height, args.input_width),
        weight_kld=args.weight_kld,
        alter_prob=args.alter_prob,
        legacy=args.legacy,
        baseline_arch=baseline_arch,
    )
    load_ok = model.load_state_dict(model_state, strict=not baseline_arch)
    if load_ok.missing_keys or load_ok.unexpected_keys:
        logger.warning(
            "Checkpoint has key mismatches (missing: %d, unexpected: %d). Visualization may be wrong if backbone differs.",
            len(load_ok.missing_keys),
            len(load_ok.unexpected_keys),
        )
    model = model.to(device)
    model.eval()

    text_feat_base = os.path.join(_REPO_ROOT, "data", "text_feat", args.dataset)
    all_rgb: List[np.ndarray] = []
    all_gt: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for image_path, depth_path, sample_key in samples:
        # Baseline: zero text; else load text feature
        if baseline_arch:
            text_feat = torch.zeros(1, 1024, device=device, dtype=torch.float32)
        else:
            pt_path = _text_feat_pt_path(os.path.join(text_feat_base, "test"), sample_key)
            if pt_path is None:
                pt_path = _text_feat_pt_path(os.path.join(text_feat_base, "train"), sample_key)
            if pt_path is None:
                logger.warning("Text feature not found for %s; skipping sample.", sample_key)
                continue
            text_feat = torch.load(pt_path, map_location=device)
            if isinstance(text_feat, torch.Tensor):
                if text_feat.dim() == 1:
                    text_feat = text_feat.unsqueeze(0)  # [1024] -> [1, 1024]
            else:
                text_feat = torch.from_numpy(np.array(text_feat)).float().to(device)
                if text_feat.dim() == 1:
                    text_feat = text_feat.unsqueeze(0)

        tensor, rgb_display = _load_and_preprocess_image(
            image_path,
            args.input_height,
            args.input_width,
            args.do_kb_crop,
        )
        tensor = tensor.to(device)
        gt = _load_gt_depth(
            depth_path,
            args.dataset,
            args.do_kb_crop,
            args.input_height,
            args.input_width,
            depth_scale=getattr(args, "depth_scale", 6553.5),
        )

        # First sample: verify raw GT file (before /1000) so we can tell if gray = dataset or our bug
        if len(all_gt) == 0:
            raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if raw is not None:
                raw_f = raw.astype(np.float32)
                nz = np.sum(raw_f > 0)
                total = raw.size
                logger.info(
                    "GT raw file (before /1000): path=%s shape=%s dtype=%s min=%s max=%s nonzero_ratio=%.2f",
                    depth_path,
                    raw.shape,
                    raw.dtype,
                    int(raw.min()) if raw.size else "n/a",
                    int(raw.max()) if raw.size else "n/a",
                    nz / total if total else 0,
                )
                if nz > 0:
                    scale = getattr(args, "depth_scale", 6553.5)
                    logger.info(
                        "GT raw nonzero: min=%.0f max=%.0f  (with depth_scale=%.0f -> %.2f–%.2f m)",
                        float(raw_f[raw_f > 0].min()),
                        float(raw_f[raw_f > 0].max()),
                        scale,
                        raw_f[raw_f > 0].min() / scale,
                        raw_f[raw_f > 0].max() / scale,
                    )

        with torch.no_grad():
            pred = model(tensor, text_feat, sample_from_gaussian=False)
        pred = pred.squeeze().cpu().numpy()
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred = np.clip(pred, args.min_depth_eval, args.max_depth_eval)
        pred[np.isinf(pred)] = args.max_depth_eval
        pred[np.isnan(pred)] = args.min_depth_eval

        all_rgb.append(rgb_display)
        all_gt.append(gt)
        all_pred.append(pred)

    # Log depth range for first sample (verify scale: expect GT/pred in meters, ~0.1–8 for NYU)
    gt0, pred0 = all_gt[0], all_pred[0]
    gt_valid = gt0[np.logical_and(np.isfinite(gt0), gt0 > 0)]
    pred_valid = pred0[np.logical_and(np.isfinite(pred0), pred0 > 0)]
    if gt_valid.size > 0:
        logger.info(
            "GT depth (m): min=%.3f max=%.3f mean=%.3f valid_ratio=%.2f",
            float(gt_valid.min()), float(gt_valid.max()), float(gt_valid.mean()), float(gt_valid.size) / gt0.size,
        )
    if pred_valid.size > 0:
        logger.info(
            "Pred depth (m): min=%.3f max=%.3f mean=%.3f",
            float(pred_valid.min()), float(pred_valid.max()), float(pred_valid.mean()),
        )

    if len(all_rgb) == 0:
        logger.error("No samples processed (missing text features?).")
        sys.exit(1)

    # Default: scale pred so median(pred) = median(gt) per image (so Ours spreads blue–red like GT, not all red)
    if not getattr(args, "no_align_pred_scale", False):
        logger.info("Applying per-image median scale alignment (pred *= median_gt/median_pred).")
        for i in range(len(all_gt)):
            gt_v = all_gt[i][np.logical_and(np.isfinite(all_gt[i]), all_gt[i] > 0)]
            pr_v = all_pred[i][np.logical_and(np.isfinite(all_pred[i]), all_pred[i] > 0)]
            if gt_v.size > 0 and pr_v.size > 0:
                mg, mp = np.median(gt_v), np.median(pr_v)
                if mp > 1e-6:
                    all_pred[i] = all_pred[i] * (mg / mp)

    # Depth color scale: fixed 0–8 m or adaptive (1/99 percentile) so colors spread
    vis_scale = getattr(args, "vis_scale", "fixed")
    if vis_scale == "adaptive":
        all_depths = []
        for d in all_gt + all_pred:
            v = d[np.logical_and(np.isfinite(d), np.logical_and(d > 0, d <= DEPTH_VIS_MAX_VALID_M))]
            if v.size > 0:
                all_depths.append(v.ravel())
        if all_depths:
            concat = np.concatenate(all_depths)
            vmin = float(np.percentile(concat, 1))
            vmax = float(np.percentile(concat, 99))
            vmin = max(DEPTH_VIS_VMIN, vmin)
            vmax = min(DEPTH_VIS_VMAX, max(vmax, vmin + 0.5))
            logger.info("Adaptive depth vis range: vmin=%.2f m vmax=%.2f m", vmin, vmax)
        else:
            vmin, vmax = DEPTH_VIS_VMIN, DEPTH_VIS_VMAX
    else:
        vmin, vmax = DEPTH_VIS_VMIN, DEPTH_VIS_VMAX

    cell_h = args.cell_height
    nrows = len(all_rgb)
    ncols = 3
    # Build figure: each row is [Input | GT | Ours], same cell height
    rows = []
    for i in range(nrows):
        rgb_cell = _resize_for_cell(all_rgb[i], cell_h)
        # GT: only depth<=0 or non-finite → gray (do not gray out far depth 8–10m)
        gt_color = _depth_to_color(all_gt[i], vmin, vmax, max_valid=None)
        # Pred: gray out >10m to avoid showing crazy large values as red
        pred_color = _depth_to_color(all_pred[i], vmin, vmax, max_valid=DEPTH_VIS_MAX_VALID_M)
        gt_cell = _resize_for_cell(gt_color, cell_h)
        pred_cell = _resize_for_cell(pred_color, cell_h)
        # Same width for alignment: use max width and pad
        w_max = max(rgb_cell.shape[1], gt_cell.shape[1], pred_cell.shape[1])
        def pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
            if img.shape[1] >= target_w:
                return img[:, :target_w]
            pad = np.zeros((img.shape[0], target_w - img.shape[1], img.shape[2]) if img.ndim == 3 else (img.shape[0], target_w - img.shape[1]), dtype=img.dtype)
            return np.concatenate([img, pad], axis=1)
        rgb_cell = pad_to_width(rgb_cell, w_max)
        gt_cell = pad_to_width(gt_cell, w_max)
        pred_cell = pad_to_width(pred_cell, w_max)
        row = np.concatenate([rgb_cell, gt_cell, pred_cell], axis=1)
        rows.append(row)
    fig_img = np.concatenate(rows, axis=0)

    # Draw column labels on top of first row
    cell_w = fig_img.shape[1] // ncols
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    for c, label in enumerate(COLUMN_LABELS):
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        x = c * cell_w + (cell_w - tw) // 2
        y = 24
        cv2.putText(fig_img, label, (x, y), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(fig_img, label, (x, y), font, font_scale, (255, 255, 255), thickness)

    # Add "Depth Value (m)" colorbar (0–8 m, blue→red) on the right
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(fig_img.shape[1] / 100 + 1.2, fig_img.shape[0] / 100))
    ax.imshow(fig_img)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(DEPTH_VIS_CMAP), norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, shrink=0.4)
    cbar.set_label("Depth Value (m)", fontsize=11)
    if vis_scale == "adaptive":
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cbar.set_ticklabels([f"{vmin:.1f}", f"{(vmin + vmax) / 2:.1f}", f"{vmax:.1f}"])
    else:
        cbar.set_ticks(list(range(int(vmin), int(vmax) + 1)))
    plt.tight_layout(pad=0)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "vis_input_gt_ours.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.2, dpi=150)
    plt.close()
    logger.info("Saved figure: %s", out_path)


if __name__ == "__main__":
    main()
