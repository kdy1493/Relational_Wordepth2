"""
Online evaluation for WorDepth: run model on eval/test split and compute metrics. Eval 전용 (단독 실행).

실행 명령어 (from repo root):
  python src/eval.py configs/arguments_eval_nyu_paper.yaml
  python src/eval.py @configs/arguments_eval_nyu_paper.txt

학습 후 eval까지 한 번에: scripts/run_train_then_eval.py 참고.
GPU: --gpu_ids 미지정 시 cuda:0, 지정 시 해당 GPU만 사용.

Exposes: online_eval(), _text_feat_pt_path() for use from train.py and elsewhere.
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import compute_errors, eval_metrics

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _text_feat_pt_path(text_feat_dir: str, rgb_key: str) -> Optional[str]:
    """Resolve .pt path under text_feat_dir. Tries two conventions: train/scene/rgb_X.pt and scene/rgb_X.pt. Returns path if file exists, else None."""
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


def online_eval(
    model: torch.nn.Module,
    dataloader_eval: Any,
    args: argparse.Namespace,
    post_process: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Run online evaluation; returns (9-element tensor of metrics, num_valid_samples)."""
    device = getattr(args, "device", None) or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    eval_measures = torch.zeros(10, device=device)
    eval_model = model.module if hasattr(model, "module") else model

    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data, desc="Eval")):
        with torch.no_grad():
            image = eval_sample_batched["image"].to(device, non_blocking=True)
            gt_depth = eval_sample_batched["depth"]
            has_valid_depth = eval_sample_batched["has_valid_depth"]
            if isinstance(has_valid_depth, torch.Tensor):
                has_valid_depth = has_valid_depth.item()
            elif isinstance(has_valid_depth, (list, tuple)):
                has_valid_depth = has_valid_depth[0]
            if not has_valid_depth:
                continue

            baseline_arch = getattr(args, "baseline_arch", False)
            if baseline_arch:
                batch_size = image.size(0)
                text_feature_list = torch.zeros(batch_size, 1024, device=image.device, dtype=torch.float32)
            else:
                text_feature_list = []
                first_path = eval_sample_batched["sample_path"][0].split(" ")[0]
                text_feat_mode = "test" if "/test/" in first_path or first_path.lstrip("/").startswith("test/") else "train"
                text_feat_dir = os.path.join(_REPO_ROOT, "data", "text_feat", "nyu" if args.dataset == "nyu" else "kitti", text_feat_mode)
                for i in range(len(eval_sample_batched["sample_path"])):
                    sample_key = eval_sample_batched["sample_path"][i].split(" ")[0][:-4]
                    pt_path = _text_feat_pt_path(text_feat_dir, sample_key)
                    if pt_path is None:
                        raise FileNotFoundError(f"Text feature not found for {sample_key} under {text_feat_dir}")
                    text_feature_list.append(torch.load(pt_path, map_location=image.device))
                text_feature_list = torch.cat(text_feature_list, dim=0)

            pred_depth = eval_model(image, text_feature_list, sample_from_gaussian=False)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin : top_margin + 352, left_margin : left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth = np.clip(pred_depth, args.min_depth_eval, args.max_depth_eval)
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)
            if args.garg_crop:
                eval_mask[
                    int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            elif args.eigen_crop:
                if args.dataset == "kitti":
                    eval_mask[
                        int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                    ] = 1
                else:
                    r0, r1 = int(45 / 480 * gt_height), int(471 / 480 * gt_height)
                    c0, c1 = int(41 / 640 * gt_width), int(601 / 640 * gt_width)
                    eval_mask[r0:r1, c0:c1] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)

        if valid_mask.sum() == 0:
            continue

        gt_valid = gt_depth[valid_mask]
        pred_valid = pred_depth[valid_mask].copy()
        if post_process:
            scale = np.median(gt_valid) / (np.median(pred_valid) + 1e-8)
            pred_valid = pred_valid * scale
        measures = compute_errors(gt_valid, pred_valid)
        eval_measures[:9] += torch.tensor(measures, device=device)
        eval_measures[9] += 1

    cnt = eval_measures[9].item()
    eval_measures_cpu = (eval_measures / cnt).cpu()[:9] if cnt > 0 else torch.zeros(9)
    if cnt > 0:
        logger.info("Eval samples: %d, post_process: %s", int(cnt), post_process)
        logger.info(
            "silog=%.4f abs_rel=%.4f log10=%.4f rms=%.4f sq_rel=%.4f log_rms=%.4f d1=%.4f d2=%.4f d3=%.4f",
            *[eval_measures_cpu[i].item() for i in range(9)],
        )
    return eval_measures_cpu, int(cnt)


# -----------------------------------------------------------------------------
# Standalone eval entry point (eval only, no training)
#
# Run from repo root, e.g.:
#   python src/eval.py @configs/arguments_eval_nyu.txt
# Or from src/:  python eval.py @../configs/arguments_eval_nyu.txt
# -----------------------------------------------------------------------------

def _make_eval_parser() -> argparse.ArgumentParser:
    """Minimal parser for standalone eval; supports @config.txt and configs/*.yaml."""
    from utils import convert_arg_line_to_args

    parser = argparse.ArgumentParser(
        description="WorDepth evaluation only (no training).",
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="nyu", help="nyu or kitti")
    parser.add_argument("--data_path_eval", type=str, required=True, help="path to eval images")
    parser.add_argument("--gt_path_eval", type=str, required=True, help="path to eval ground truth")
    parser.add_argument("--filenames_file_eval", type=str, required=True, help="path to eval filenames list")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=6553.5,
        help="NYU: raw/depth_scale = m. Default 6553.5 for converted sync. Use 1000 for original mm.",
    )
    parser.add_argument("--input_height", type=int, default=256)
    parser.add_argument("--input_width", type=int, default=320)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--prior_mean", type=float, default=1.54)
    parser.add_argument("--min_depth_eval", type=float, default=1e-3)
    parser.add_argument("--max_depth_eval", type=float, default=80.0)
    parser.add_argument("--do_kb_crop", action="store_true")
    parser.add_argument("--eigen_crop", action="store_true")
    parser.add_argument("--garg_crop", action="store_true")
    parser.add_argument("--pretrain", type=str, default=None, help="encoder pretrain (for model init)")
    parser.add_argument("--weight_kld", type=float, default=1e-3)
    parser.add_argument("--alter_prob", type=float, default=0.5)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--baseline_arch", action="store_true", help="Evaluate baseline (Swin-L + depth decoder only); no text path, use with baseline checkpoint")
    parser.add_argument("--post_process", action="store_true", help="median scaling before metrics")
    parser.add_argument("--gpu_ids", type=str, default=None, help="comma-separated GPU ids")
    return parser


def run_eval_only() -> None:
    """Parse args, load model and checkpoint, build eval dataloader, run online_eval. Use from __main__."""
    from utils import expand_argv_yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    argv = expand_argv_yaml(sys.argv[1:], repo_root)
    parser = _make_eval_parser()
    args = parser.parse_args(argv)

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if device.type == "cpu":
        logger.warning("CUDA not available; running evaluation on CPU (slower).")

    from dataloaders.dataloader import NewDataLoader
    from networks.wordepth import WorDepth

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
    # PyTorch 2.6+: torch.load defaults to weights_only=True, which can break
    # loading older checkpoints that rely on full pickling. Explicitly disable
    # weights_only where supported, with a backward-compatible fallback.
    try:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    model_state = ckpt["model"]
    # Checkpoint may be from DataParallel/DDP (keys like "module.backbone....");
    # load into unwrapped model first, so strip "module." prefix if present.
    if any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=not baseline_arch)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    dataloader_eval = NewDataLoader(args, "online_eval")
    post_process = getattr(args, "post_process", False)
    metrics_tensor, num_samples = online_eval(model, dataloader_eval, args, post_process=post_process)

    # Save eval results next to the checkpoint (same run dir)
    if num_samples > 0:
        out_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        out_path = os.path.join(out_dir, f"eval_results_{ckpt_stem}.json")
        results = {
            "checkpoint_path": args.checkpoint_path,
            "post_process": post_process,
            "num_samples": num_samples,
            "filenames_file_eval": args.filenames_file_eval,
            "metrics": {name: metrics_tensor[i].item() for i, name in enumerate(eval_metrics)},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Eval results saved to %s", out_path)


if __name__ == "__main__":
    run_eval_only()
