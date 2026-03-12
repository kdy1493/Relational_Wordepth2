"""
VKITTI2 evaluation script for WorDepth (+/- Relational).

기능:
- VKITTI2 RGB+Depth + (선택적으로) relational annotation을 사용해서
  SILog / AbsRel / RMSE 등 기본 depth metric을 계산한다.
- NYU/KITTI용 eval.py는 text_feat(.pt)를 요구하지만,
  VKITTI2는 caption/CLIP embedding(768D)을 vkitti2_relational_dataloader에서 바로 받아서 사용한다.

사용 예시 (repo root 기준):
  python src/eval_vkitti2.py @configs/arguments_eval_vkitti2.yaml
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import compute_errors, eval_metrics, convert_arg_line_to_args, expand_argv_yaml
from networks.wordepth import WorDepth
from dataloaders.vkitti2_relational_dataloader import create_vkitti2_dataloader


logger = logging.getLogger(__name__)


def _make_vkitti2_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WorDepth VKITTI2 evaluation (no training).",
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # 필수 입력
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to WorDepth checkpoint (.pt)")
    parser.add_argument("--vkitti2_data_path", type=str, required=True, help="VKITTI2 root directory")
    parser.add_argument("--vkitti2_caption_cache", type=str, required=True, help="VKITTI2 caption/embedding cache")
    parser.add_argument(
        "--vkitti2_relations_dir",
        type=str,
        default=None,
        help="(optional) VKITTI2 relations directory; if provided, masks/relations도 dataloader에서 함께 로드",
    )

    # Scene / condition 필터 (train config와 동일한 의미)
    parser.add_argument(
        "--vkitti2_scenes",
        type=str,
        nargs="+",
        default=["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"],
        help="VKITTI2 scenes to use",
    )
    parser.add_argument(
        "--vkitti2_conditions",
        type=str,
        nargs="+",
        default=["clone"],
        help="VKITTI2 conditions to use",
    )

    # WorDepth / 입력 설정
    parser.add_argument("--input_height", type=int, default=352)
    parser.add_argument("--input_width", type=int, default=704)
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--prior_mean", type=float, default=20.0)
    parser.add_argument("--min_depth_eval", type=float, default=0.1)
    parser.add_argument("--max_depth_eval", type=float, default=80.0)
    parser.add_argument("--weight_kld", type=float, default=1e-3)
    parser.add_argument("--alter_prob", type=float, default=0.5)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument(
        "--baseline_arch",
        action="store_true",
        help="Evaluate baseline architecture (Swin-L + depth decoder only, no text path)",
    )
    parser.add_argument("--pretrain", type=str, default=None, help="encoder pretrain (for model init)")

    # 기타
    parser.add_argument("--gpu_ids", type=str, default=None, help="comma-separated GPU ids")
    parser.add_argument("--num_threads", type=int, default=4, help="data loading workers")
    parser.add_argument("--cache_images", action="store_true")
    parser.add_argument("--do_random_rotate", action="store_true")
    parser.add_argument("--degree", type=float, default=2.5)

    # 출력 설정
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="optional path to save eval metrics as JSON (default: alongside checkpoint)",
    )

    return parser


def _build_vkitti2_eval_dataloader(args: argparse.Namespace) -> Any:
    """VKITTI2 eval dataloader 생성 (torch.utils.data.DataLoader wrapper)."""
    # relational_train.py의 VKITTI2 분기와 유사하게 args 필드를 세팅
    args.dataset = "vkitti2"
    args.mode = "online_eval"
    args.use_relational_loss = args.vkitti2_relations_dir is not None

    # VKITTI2-specific fields
    args.vkitti2_input_height = getattr(args, "input_height", 352)
    args.vkitti2_input_width = getattr(args, "input_width", 704)
    args.vkitti2_max_depth = getattr(args, "max_depth", 80.0)
    args.vkitti2_min_depth = getattr(args, "min_depth_eval", 0.1)

    return create_vkitti2_dataloader(args, mode="online_eval", use_ddp=False)


def _load_vkitti2_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """WorDepth 모델 + 체크포인트 로드."""
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
    # PyTorch 2.6+ 호환: weights_only=False를 명시, 실패 시 fallback
    try:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    state = ckpt["model"]
    # DataParallel/DDP prefix 정리
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=not baseline_arch)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    return model


def vkitti2_eval_loop(
    model: torch.nn.Module,
    dataloader_eval: Any,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, int]:
    """VKITTI2 eval 루프: eval_metrics 9개와 유효 샘플 수 리턴."""
    device = args.device
    eval_measures = torch.zeros(10, device=device)
    eval_model = model.module if hasattr(model, "module") else model

    for _, batch in enumerate(tqdm(dataloader_eval.data, desc="VKITTI2 Eval")):
        with torch.no_grad():
            image = batch["image"].to(device, non_blocking=True)
            gt_depth = batch["depth"]
            has_valid_depth = batch.get("has_valid_depth", torch.tensor(True))
            if isinstance(has_valid_depth, torch.Tensor):
                has_valid_depth = bool(has_valid_depth.item())
            elif isinstance(has_valid_depth, (list, tuple)):
                has_valid_depth = bool(has_valid_depth[0])
            if not has_valid_depth:
                continue

            baseline_arch = getattr(args, "baseline_arch", False)
            if baseline_arch:
                bsz = image.size(0)
                text_feature_list = torch.zeros(bsz, 1024, device=image.device, dtype=torch.float32)
            else:
                # VKITTI2 dataloader는 768D CLIP text_embedding을 직접 제공
                if "text_embedding" in batch:
                    text_emb = batch["text_embedding"].to(device, non_blocking=True)  # [B, 768]
                    bsz = text_emb.size(0)
                    text_feature_list = torch.zeros(bsz, 1024, device=image.device, dtype=torch.float32)
                    text_feature_list[:, :768] = text_emb
                else:
                    bsz = image.size(0)
                    text_feature_list = torch.zeros(bsz, 1024, device=image.device, dtype=torch.float32)

            pred_depth = eval_model(image, text_feature_list, sample_from_gaussian=False)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth_np = gt_depth.cpu().numpy().squeeze()

        # 크기 mismatch 시 resize
        if pred_depth.shape != gt_depth_np.shape:
            pred_depth = cv2.resize(
                pred_depth,
                (gt_depth_np.shape[1], gt_depth_np.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        pred_depth = np.clip(pred_depth, args.min_depth_eval, args.max_depth_eval)
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(
            gt_depth_np > args.min_depth_eval,
            gt_depth_np < args.max_depth_eval,
        )
        if valid_mask.sum() == 0:
            continue

        gt_valid = gt_depth_np[valid_mask]
        pred_valid = pred_depth[valid_mask].copy()

        measures = compute_errors(gt_valid, pred_valid)
        eval_measures[:9] += torch.tensor(measures, device=device)
        eval_measures[9] += 1

    cnt = int(eval_measures[9].item())
    eval_measures_cpu = (eval_measures / max(cnt, 1)).cpu()[:9] if cnt > 0 else torch.zeros(9)
    if cnt > 0:
        logger.info("VKITTI2 eval samples: %d", cnt)
        logger.info(
            "silog=%.4f abs_rel=%.4f log10=%.4f rms=%.4f sq_rel=%.4f log_rms=%.4f d1=%.4f d2=%.4f d3=%.4f",
            *[eval_measures_cpu[i].item() for i in range(9)],
        )
    else:
        logger.warning("No valid VKITTI2 eval samples.")

    return eval_measures_cpu, cnt


def run_vkitti2_eval_only() -> None:
    """Entry point: parse args, build model/dataloader, run VKITTI2 eval."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    argv = expand_argv_yaml(sys.argv[1:], repo_root)
    parser = _make_vkitti2_eval_parser()
    args = parser.parse_args(argv)

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if device.type == "cpu":
        logger.warning("CUDA not available; running VKITTI2 eval on CPU (slower).")

    model = _load_vkitti2_model(args, device)
    dataloader_eval = _build_vkitti2_eval_dataloader(args)

    metrics_tensor, num_samples = vkitti2_eval_loop(model, dataloader_eval, args)

    if num_samples > 0:
        out_json = args.output_json
        if not out_json:
            ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
            ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
            out_json = os.path.join(ckpt_dir, f"eval_results_vkitti2_{ckpt_stem}.json")

        results = {
            "checkpoint_path": args.checkpoint_path,
            "dataset": "vkitti2",
            "num_samples": num_samples,
            "vkitti2_data_path": args.vkitti2_data_path,
            "vkitti2_caption_cache": args.vkitti2_caption_cache,
            "vkitti2_relations_dir": args.vkitti2_relations_dir,
            "vkitti2_scenes": args.vkitti2_scenes,
            "vkitti2_conditions": args.vkitti2_conditions,
            "metrics": {name: metrics_tensor[i].item() for i, name in enumerate(eval_metrics)},
        }
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("VKITTI2 eval results saved to %s", out_json)


if __name__ == "__main__":
    run_vkitti2_eval_only()

