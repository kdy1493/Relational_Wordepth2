"""
Ordinal Metrics Evaluation: WHDR and ORD

기존 pixel-pair ordinal metric들을 측정하는 스크립트.
RSR(우리가 제안한 object-level metric)과 비교하기 위함.

Metrics:
1. ORD (Ordinal Error) - Chen et al. 2016
   - 랜덤 픽셀 쌍의 순서 정확도 측정
   
2. WHDR (Weighted Human Disagreement Rate) - DIW dataset 방식
   - 픽셀 쌍의 순서 예측 오류율 (threshold 기반)

사용 예시:
  python src/eval_ordinal_metrics.py @configs/arguments_eval_nyu_ordinal.yaml

  또는 직접 인자:
  python src/eval_ordinal_metrics.py \
    --checkpoint_path ./runs/model/model-best.pt \
    --data_path_eval ./data/nyu_v2_sync \
    --gt_path_eval ./data/nyu_v2_sync \
    --filenames_file_eval ./data_splits/nyudepthv2_test_split.txt \
    --num_pairs 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Ordinal Metrics Implementation
# ============================================================================

def compute_ord(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    num_pairs: int = 5000,
    valid_mask: Optional[np.ndarray] = None,
    threshold: float = 0.02,
) -> Dict[str, float]:
    """
    Compute ORD (Ordinal Error) - Chen et al. 2016
    
    랜덤 픽셀 쌍을 샘플링하고, GT와 예측의 순서가 일치하는지 측정.
    
    Args:
        pred_depth: 예측 depth map (H, W)
        gt_depth: GT depth map (H, W)
        num_pairs: 샘플링할 픽셀 쌍 수
        valid_mask: 유효한 픽셀 마스크 (None이면 gt_depth > 0)
        threshold: 순서 판단을 위한 상대적 threshold (|d1-d2|/max(d1,d2) > threshold)
    
    Returns:
        dict with:
            - ord_acc: 순서 정확도 (높을수록 좋음)
            - ord_err: 순서 오류율 = 1 - ord_acc (낮을수록 좋음)
            - num_valid_pairs: 실제 평가된 쌍 수
    """
    H, W = gt_depth.shape
    
    if valid_mask is None:
        valid_mask = gt_depth > 0
    
    # 유효한 픽셀 좌표 추출
    valid_coords = np.argwhere(valid_mask)  # (N, 2)
    num_valid = len(valid_coords)
    
    if num_valid < 2:
        return {"ord_acc": 0.0, "ord_err": 1.0, "num_valid_pairs": 0}
    
    # 랜덤 쌍 샘플링
    num_pairs = min(num_pairs, num_valid * (num_valid - 1) // 2)
    
    idx1 = np.random.randint(0, num_valid, size=num_pairs)
    idx2 = np.random.randint(0, num_valid, size=num_pairs)
    
    # 같은 픽셀 쌍 제거
    different_mask = idx1 != idx2
    idx1 = idx1[different_mask]
    idx2 = idx2[different_mask]
    
    coords1 = valid_coords[idx1]  # (N, 2)
    coords2 = valid_coords[idx2]  # (N, 2)
    
    # GT depth 값
    gt_d1 = gt_depth[coords1[:, 0], coords1[:, 1]]
    gt_d2 = gt_depth[coords2[:, 0], coords2[:, 1]]
    
    # Pred depth 값
    pred_d1 = pred_depth[coords1[:, 0], coords1[:, 1]]
    pred_d2 = pred_depth[coords2[:, 0], coords2[:, 1]]
    
    # 순서가 의미있는 쌍만 선택 (threshold 기반)
    gt_diff = gt_d1 - gt_d2
    gt_max = np.maximum(gt_d1, gt_d2)
    relative_diff = np.abs(gt_diff) / (gt_max + 1e-8)
    
    significant_mask = relative_diff > threshold
    
    if significant_mask.sum() == 0:
        return {"ord_acc": 0.0, "ord_err": 1.0, "num_valid_pairs": 0}
    
    gt_diff = gt_diff[significant_mask]
    pred_diff = (pred_d1 - pred_d2)[significant_mask]
    
    # 순서 일치 여부 (부호가 같으면 일치)
    correct = (gt_diff * pred_diff) > 0
    
    ord_acc = correct.mean()
    ord_err = 1.0 - ord_acc
    
    return {
        "ord_acc": float(ord_acc),
        "ord_err": float(ord_err),
        "num_valid_pairs": int(significant_mask.sum()),
    }


def compute_whdr(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    num_pairs: int = 5000,
    valid_mask: Optional[np.ndarray] = None,
    tau: float = 0.1,
) -> Dict[str, float]:
    """
    Compute WHDR (Weighted Human Disagreement Rate) - DIW dataset 방식
    
    WHDR = (틀린 쌍 수 + 0.5 * 모호한 쌍 수) / 전체 쌍 수
    
    판단 기준:
    - GT에서 d1 > d2 * (1+tau) → 예측도 d1 > d2여야 함
    - GT에서 d2 > d1 * (1+tau) → 예측도 d2 > d1여야 함
    - 그 외 (비슷한 depth) → equal로 판단
    
    Args:
        pred_depth: 예측 depth map (H, W)
        gt_depth: GT depth map (H, W)
        num_pairs: 샘플링할 픽셀 쌍 수
        valid_mask: 유효한 픽셀 마스크
        tau: 순서 판단 threshold (default 0.1 = 10%)
    
    Returns:
        dict with:
            - whdr: WHDR 값 (낮을수록 좋음, 0~1)
            - whdr_acc: 1 - WHDR (높을수록 좋음)
            - num_pairs: 평가된 쌍 수
            - breakdown: {correct, wrong, ambiguous} 개수
    """
    H, W = gt_depth.shape
    
    if valid_mask is None:
        valid_mask = gt_depth > 0
    
    valid_coords = np.argwhere(valid_mask)
    num_valid = len(valid_coords)
    
    if num_valid < 2:
        return {"whdr": 1.0, "whdr_acc": 0.0, "num_pairs": 0, "breakdown": {}}
    
    # 랜덤 쌍 샘플링
    num_pairs = min(num_pairs, num_valid * (num_valid - 1) // 2)
    
    idx1 = np.random.randint(0, num_valid, size=num_pairs)
    idx2 = np.random.randint(0, num_valid, size=num_pairs)
    
    different_mask = idx1 != idx2
    idx1 = idx1[different_mask]
    idx2 = idx2[different_mask]
    
    coords1 = valid_coords[idx1]
    coords2 = valid_coords[idx2]
    
    # Depth 값 추출
    gt_d1 = gt_depth[coords1[:, 0], coords1[:, 1]]
    gt_d2 = gt_depth[coords2[:, 0], coords2[:, 1]]
    pred_d1 = pred_depth[coords1[:, 0], coords1[:, 1]]
    pred_d2 = pred_depth[coords2[:, 0], coords2[:, 1]]
    
    # GT ordinal relation 결정
    # +1: d1 > d2 (d1이 더 멀다)
    # -1: d2 > d1 (d2가 더 멀다)
    #  0: equal (비슷하다)
    gt_relation = np.zeros(len(gt_d1), dtype=np.int32)
    gt_relation[gt_d1 > gt_d2 * (1 + tau)] = 1   # d1 farther
    gt_relation[gt_d2 > gt_d1 * (1 + tau)] = -1  # d2 farther
    
    # Pred ordinal relation 결정 (같은 기준)
    pred_relation = np.zeros(len(pred_d1), dtype=np.int32)
    pred_relation[pred_d1 > pred_d2 * (1 + tau)] = 1
    pred_relation[pred_d2 > pred_d1 * (1 + tau)] = -1
    
    # WHDR 계산
    # GT가 명확한 순서를 가질 때 (!=0), pred도 맞춰야 함
    gt_has_order = gt_relation != 0
    
    correct = (gt_relation == pred_relation) & gt_has_order
    wrong = (gt_relation != pred_relation) & gt_has_order
    
    # GT가 equal인 경우 (모호)
    gt_equal = gt_relation == 0
    
    n_correct = correct.sum()
    n_wrong = wrong.sum()
    n_ambiguous = gt_equal.sum()
    n_total = len(gt_relation)
    
    # WHDR = (wrong + 0.5 * ambiguous) / total
    # 또는 ordered pair만 볼 수도 있음
    if gt_has_order.sum() > 0:
        # Ordered pair 기준
        whdr_ordered = n_wrong / gt_has_order.sum()
        whdr_acc_ordered = n_correct / gt_has_order.sum()
    else:
        whdr_ordered = 0.0
        whdr_acc_ordered = 0.0
    
    # 전체 기준 (ambiguous 포함)
    whdr_full = (n_wrong + 0.5 * n_ambiguous) / n_total if n_total > 0 else 1.0
    
    return {
        "whdr": float(whdr_ordered),  # ordered pair 기준 (더 일반적)
        "whdr_acc": float(whdr_acc_ordered),
        "whdr_full": float(whdr_full),  # ambiguous 포함
        "num_pairs": int(n_total),
        "num_ordered_pairs": int(gt_has_order.sum()),
        "breakdown": {
            "correct": int(n_correct),
            "wrong": int(n_wrong),
            "ambiguous": int(n_ambiguous),
        },
    }


def compute_ordinal_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    num_pairs: int = 5000,
    valid_mask: Optional[np.ndarray] = None,
    ord_threshold: float = 0.02,
    whdr_tau: float = 0.1,
) -> Dict[str, float]:
    """
    ORD와 WHDR을 한 번에 계산.
    
    Args:
        pred_depth: 예측 depth (H, W)
        gt_depth: GT depth (H, W)
        num_pairs: 픽셀 쌍 샘플 수
        valid_mask: 유효 픽셀 마스크
        ord_threshold: ORD의 상대적 threshold
        whdr_tau: WHDR의 tau 값
    
    Returns:
        Combined metrics dict
    """
    ord_result = compute_ord(pred_depth, gt_depth, num_pairs, valid_mask, ord_threshold)
    whdr_result = compute_whdr(pred_depth, gt_depth, num_pairs, valid_mask, whdr_tau)
    
    return {
        "ord_acc": ord_result["ord_acc"],
        "ord_err": ord_result["ord_err"],
        "whdr": whdr_result["whdr"],
        "whdr_acc": whdr_result["whdr_acc"],
        "num_ord_pairs": ord_result["num_valid_pairs"],
        "num_whdr_pairs": whdr_result["num_ordered_pairs"],
    }


# ============================================================================
# Evaluation Script
# ============================================================================

def _text_feat_pt_path(text_feat_dir: str, rgb_key: str) -> Optional[str]:
    """Resolve .pt path under text_feat_dir."""
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


def _make_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    # Ensure we can import project-local utils when running from repo root
    utils_root = os.path.join(_REPO_ROOT, "src")
    if utils_root not in sys.path:
        sys.path.insert(0, utils_root)
    from utils import convert_arg_line_to_args
    
    parser = argparse.ArgumentParser(
        description="Ordinal metrics (ORD, WHDR) evaluation.",
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    
    # 필수 입력
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="nyu", choices=["nyu", "kitti"])
    parser.add_argument("--data_path_eval", type=str, required=True)
    parser.add_argument("--gt_path_eval", type=str, required=True)
    parser.add_argument("--filenames_file_eval", type=str, required=True)
    
    # 모델 설정
    parser.add_argument("--depth_scale", type=float, default=6553.5)
    parser.add_argument("--input_height", type=int, default=480)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--prior_mean", type=float, default=1.54)
    parser.add_argument("--min_depth_eval", type=float, default=1e-3)
    parser.add_argument("--max_depth_eval", type=float, default=10.0)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--weight_kld", type=float, default=1e-3)
    parser.add_argument("--alter_prob", type=float, default=0.5)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--baseline_arch", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default=None)
    
    # Ordinal metric 설정
    parser.add_argument("--num_pairs", type=int, default=5000, help="픽셀 쌍 샘플 수 (이미지당)")
    parser.add_argument("--ord_threshold", type=float, default=0.02, help="ORD threshold")
    parser.add_argument("--whdr_tau", type=float, default=0.1, help="WHDR tau")
    
    # Eval 설정 (eval.py와 일관되게)
    parser.add_argument("--do_kb_crop", action="store_true")
    parser.add_argument("--garg_crop", action="store_true")
    parser.add_argument("--eigen_crop", action="store_true")
    parser.add_argument("--post_process", action="store_true")
    parser.add_argument("--num_threads", type=int, default=4)
    
    # 출력
    parser.add_argument("--output_json", type=str, default=None, help="결과 저장 JSON 경로")
    
    return parser


def evaluate_ordinal_metrics(args: argparse.Namespace) -> Dict[str, float]:
    """
    전체 eval set에 대해 ordinal metrics 계산.
    """
    from networks.wordepth import WorDepth
    from dataloaders.dataloader import NewDataLoader
    
    # Device 설정
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드 (src/eval.py의 run_eval_only와 동일한 방식)
    logger.info(f"Loading model from {args.checkpoint_path}")
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
    # PyTorch 2.6+ 호환: weights_only=False 명시, 실패 시 fallback
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    # DataParallel / torch.compile 호환: prefix 제거
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=not baseline_arch)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    
    # DataLoader 생성 (eval.py와 동일하게 NewDataLoader 사용)
    dataloader = NewDataLoader(args, "online_eval")
    
    # Text feature 디렉토리
    text_feat_dir = os.path.join(_REPO_ROOT, "data", "text_feat", "nyu", "test")
    
    # 메트릭 누적
    all_ord_acc = []
    all_ord_err = []
    all_whdr = []
    all_whdr_acc = []
    
    np.random.seed(42)  # 재현성
    
    for batch in tqdm(dataloader.data, desc="Evaluating ordinal metrics"):
        with torch.no_grad():
            image = batch["image"].to(device)
            gt_depth = batch["depth"].cpu().numpy().squeeze()
            
            has_valid_depth = batch.get("has_valid_depth", True)
            if isinstance(has_valid_depth, torch.Tensor):
                has_valid_depth = has_valid_depth.item()
            if not has_valid_depth:
                continue
            
            # Text feature 로드
            if getattr(args, "baseline_arch", False):
                text_feature = torch.zeros(1, 1024, device=device, dtype=torch.float32)
            else:
                sample_key = batch["sample_path"][0].split(" ")[0][:-4]
                pt_path = _text_feat_pt_path(text_feat_dir, sample_key)
                if pt_path is None:
                    logger.warning(f"Text feature not found for {sample_key}")
                    continue
                text_feature = torch.load(pt_path, map_location=device)
            
            # 예측
            pred_depth = model(image, text_feature, sample_from_gaussian=False)
            pred_depth = pred_depth.cpu().numpy().squeeze()
        
        # Resize if needed
        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]))
        
        # Eigen crop
        if args.eigen_crop:
            h, w = gt_depth.shape
            pred_depth = pred_depth[45:471, 41:601]
            gt_depth = gt_depth[45:471, 41:601]
        
        # Valid mask
        valid_mask = (gt_depth > args.min_depth_eval) & (gt_depth < args.max_depth_eval)
        
        # Post process (flip augmentation)
        if args.post_process:
            # 간단한 post process: flip average
            pass  # 여기서는 생략
        
        # Ordinal metrics 계산
        metrics = compute_ordinal_metrics(
            pred_depth, gt_depth,
            num_pairs=args.num_pairs,
            valid_mask=valid_mask,
            ord_threshold=args.ord_threshold,
            whdr_tau=args.whdr_tau,
        )
        
        all_ord_acc.append(metrics["ord_acc"])
        all_ord_err.append(metrics["ord_err"])
        all_whdr.append(metrics["whdr"])
        all_whdr_acc.append(metrics["whdr_acc"])
    
    # 평균 계산
    results = {
        "ord_acc": float(np.mean(all_ord_acc)),
        "ord_err": float(np.mean(all_ord_err)),
        "whdr": float(np.mean(all_whdr)),
        "whdr_acc": float(np.mean(all_whdr_acc)),
        "num_images": len(all_ord_acc),
        "num_pairs_per_image": args.num_pairs,
    }
    
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = _make_parser()
    # Support YAML configs like eval.py / relational_eval_rsr.py via utils.expand_argv_yaml
    from utils import expand_argv_yaml
    repo_root = _REPO_ROOT
    argv = expand_argv_yaml(sys.argv[1:], repo_root)
    args = parser.parse_args(argv)
    
    logger.info("=" * 60)
    logger.info("Ordinal Metrics Evaluation (ORD, WHDR)")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Num pairs per image: {args.num_pairs}")
    logger.info(f"ORD threshold: {args.ord_threshold}")
    logger.info(f"WHDR tau: {args.whdr_tau}")
    
    results = evaluate_ordinal_metrics(args)
    
    logger.info("=" * 60)
    logger.info("Results:")
    logger.info(f"  ORD Accuracy:  {results['ord_acc']:.4f}")
    logger.info(f"  ORD Error:     {results['ord_err']:.4f}")
    logger.info(f"  WHDR:          {results['whdr']:.4f}")
    logger.info(f"  WHDR Accuracy: {results['whdr_acc']:.4f}")
    logger.info(f"  Num images:    {results['num_images']}")
    logger.info("=" * 60)
    
    # JSON 저장
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")
    
    return results


if __name__ == "__main__":
    main()
