"""
Relational RSR evaluation script for WorDepth.

기능:
- 주어진 WorDepth 체크포인트(베이스 / +Rel 어떤 것이든)에 대해
  NYU test split + relational annotation을 사용해
  Relation Satisfaction Rate(RSR)와 mean violation을 계산한다.

사용 예시 (NYU, 단일 GPU, repo root 기준):

  # YAML 설정 사용 (권장)
  python src/relational_eval_rsr.py @configs/arguments_eval_nyu_rsr.yaml

Relational 설정 (repr/margin 등)는 아래 인자로 제어:
- --rel_repr {"median","statistical"}
- --rel_statistical_alpha float
- --rel_margin float
- --rel_min_pixels int
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from tqdm import tqdm

from networks.wordepth import WorDepth
from networks.relational_depth_loss import RelationalDepthLoss
from dataloaders.nyu_relational_dataloader import create_nyu_relational_dataloader
from eval import _text_feat_pt_path  # 재사용: text feature 경로 해석


logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_rsr_parser() -> argparse.ArgumentParser:
    """Create argument parser for RSR evaluation only.

    YAML(@config)도 eval.py와 동일하게 지원할 수 있도록 설계한다.
    """
    from utils import convert_arg_line_to_args

    parser = argparse.ArgumentParser(
        description="Relational RSR evaluation for WorDepth (no training).",
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # 필수 입력
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to WorDepth checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="nyu", choices=["nyu", "kitti"])
    parser.add_argument("--data_path_eval", type=str, required=True, help="path to eval images")
    parser.add_argument("--gt_path_eval", type=str, required=True, help="path to eval ground truth")
    parser.add_argument(
        "--filenames_file_eval",
        type=str,
        required=True,
        help="path to eval filenames list (e.g., nyudepthv2_test_split.txt)",
    )
    parser.add_argument(
        "--relations_dir_eval",
        type=str,
        required=True,
        help="base directory containing relational annotations (masks + relations.json)",
    )

    # WorDepth / 데이터 설정 (eval.py와 유사)
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=6553.5,
        help="NYU: raw/depth_scale = m. Default 6553.5 for converted sync. Use 1000 for original mm.",
    )
    parser.add_argument("--input_height", type=int, default=480)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--prior_mean", type=float, default=1.54)
    parser.add_argument("--min_depth_eval", type=float, default=1e-3)
    parser.add_argument("--max_depth_eval", type=float, default=80.0)
    parser.add_argument("--pretrain", type=str, default=None, help="encoder pretrain (for model init)")
    parser.add_argument("--weight_kld", type=float, default=1e-3)
    parser.add_argument("--alter_prob", type=float, default=0.5)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default=None, help="comma-separated GPU ids")

    # DataLoader / 기타
    parser.add_argument("--num_threads", type=int, default=4, help="data loading workers")
    parser.add_argument("--use_dense_depth", action="store_true", help="use dense NYU depth like train config")
    parser.add_argument("--cache_images", action="store_true")
    parser.add_argument("--do_random_rotate", action="store_true")
    parser.add_argument("--degree", type=float, default=2.5)

    # Relational loss 설정 (RSR 정의에 직접 영향을 주는 파라미터)
    parser.add_argument(
        "--rel_repr",
        type=str,
        default="median",
        choices=["median", "statistical"],
        help="object representative depth mode for RSR computation",
    )
    parser.add_argument(
        "--rel_statistical_alpha",
        type=float,
        default=1.0,
        help="alpha for statistical representative (mu + alpha * sigma) when rel_repr='statistical'",
    )
    parser.add_argument(
        "--rel_margin",
        type=float,
        default=0.1,
        help="hinge margin m for ordering constraint in meters",
    )
    parser.add_argument(
        "--rel_min_pixels",
        type=int,
        default=20,
        help="minimum number of valid pixels per object to be considered in relations",
    )

    # 출력 제어
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="optional path to save RSR results as JSON (default: alongside checkpoint)",
    )
    parser.add_argument(
        "--viz_dir",
        type=str,
        default="",
        help="optional directory to save visualizations of violated relations (empty to disable)",
    )
    parser.add_argument(
        "--viz_topk",
        type=int,
        default=20,
        help="maximum number of most severe violated relations to visualize when viz_dir is set",
    )

    return parser


def _load_model_for_eval(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Load WorDepth model and checkpoint onto device."""
    model = WorDepth(
        pretrained=args.pretrain,
        max_depth=args.max_depth,
        prior_mean=args.prior_mean,
        img_size=(args.input_height, args.input_width),
        weight_kld=args.weight_kld,
        alter_prob=args.alter_prob,
        legacy=args.legacy,
    )
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    model_state: Dict[str, torch.Tensor] = ckpt["model"]
    # DataParallel/DDP 호환: "module." prefix 제거
    if any(key.startswith("module.") for key in model_state.keys()):
        model_state = {key.replace("module.", "", 1): value for key, value in model_state.items()}
    model.load_state_dict(model_state, strict=True)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    return model


def _build_relational_dataloader(args: argparse.Namespace) -> Any:
    """Create NYU relational dataloader for eval mode."""
    # create_nyu_relational_dataloader는 args의 여러 필드를 기대하므로 최소한의 값을 채워준다.
    args.mode = "online_eval"
    args.use_relational_loss = True
    # eval 전용 경로를 train 필드에 매핑 (함수 내부에서 mode에 따라 다시 읽음)
    args.data_path = getattr(args, "data_path_eval", args.data_path_eval)
    args.gt_path = getattr(args, "gt_path_eval", args.gt_path_eval)
    args.filenames_file = getattr(args, "filenames_file_eval", args.filenames_file_eval)
    args.relations_dir_train = getattr(args, "relations_dir_eval", args.relations_dir_eval)
    args.relations_dir_eval = args.relations_dir_eval

    # 기타 기본값 보정
    if not hasattr(args, "eigen_crop"):
        args.eigen_crop = False
    if not hasattr(args, "garg_crop"):
        args.garg_crop = False

    dataloader = create_nyu_relational_dataloader(args, mode="online_eval", use_ddp=False)
    return dataloader


def _preload_text_features_for_eval(args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    """Preload text features for eval split into CPU memory.

    filenames_file_eval와 dataset, test/train 구분을 사용해 text_feat 디렉터리를 찾는다.
    """
    text_cache: Dict[str, torch.Tensor] = {}

    with open(args.filenames_file_eval, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        logger.warning("No lines found in %s; text feature cache will be empty.", args.filenames_file_eval)
        return text_cache

    # 첫 번째 경로로부터 mode 추론 (eval.py와 동일한 로직)
    first_path = lines[0].split()[0]
    text_feat_mode = "test" if "/test/" in first_path or first_path.lstrip("/").startswith("test/") else "train"
    text_feat_dir = os.path.join(_REPO_ROOT, "data", "text_feat", args.dataset, text_feat_mode)

    logger.info("Preloading %d text features from %s", len(lines), text_feat_dir)
    for line in tqdm(lines, desc="Loading text feat (.pt) for eval"):
        parts = line.split()
        if not parts:
            continue
        rgb_key = parts[0][:-4]
        pt_path = _text_feat_pt_path(text_feat_dir, rgb_key)
        if pt_path is None:
            continue
        text_cache[rgb_key] = torch.load(pt_path, map_location="cpu")

    logger.info("Loaded %d text features into cache", len(text_cache))
    return text_cache


def compute_rsr_for_checkpoint(args: argparse.Namespace) -> Tuple[float, float, int]:
    """Compute RSR and mean violation for a single checkpoint.

    Returns:
        (rsr, mean_violation, num_relations)
    """
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if device.type == "cpu":
        logger.warning("CUDA not available; running RSR eval on CPU (slower).")

    model = _load_model_for_eval(args, device)
    dataloader_eval = _build_relational_dataloader(args)
    text_cache = _preload_text_features_for_eval(args)

    rel_loss_fn = RelationalDepthLoss(
        margin_rank=args.rel_margin,
        min_pixels=args.rel_min_pixels,
        min_valid_pixels=args.rel_min_pixels,
        repr_mode=args.rel_repr,
        valid_min_depth=args.min_depth_eval,
        valid_max_depth=args.max_depth_eval,
        statistical_alpha=args.rel_statistical_alpha,
        debug_relational=False,
    ).to(device)

    total_relations: float = 0.0
    total_satisfied: float = 0.0
    total_violation: float = 0.0

    # 시각화 설정
    viz_enabled: bool = bool(getattr(args, "viz_dir", ""))
    viz_topk: int = max(int(getattr(args, "viz_topk", 0)), 0)
    top_violations: List[Dict[str, Any]] = []

    eval_model = model.module if hasattr(model, "module") else model

    for batch in tqdm(dataloader_eval.data, desc="RSR Eval"):
        with torch.no_grad():
            image = batch["image"].to(device, non_blocking=True)

            # text features: cache에서 우선 조회, 없으면 디스크에서 on-demand 로딩
            text_list = []
            first_path = batch["sample_path"][0].split(" ")[0]
            text_feat_mode = "test" if "/test/" in first_path or first_path.lstrip("/").startswith("test/") else "train"
            text_feat_dir = os.path.join(_REPO_ROOT, "data", "text_feat", args.dataset, text_feat_mode)
            for sample_path in batch["sample_path"]:
                rgb_key = sample_path.split(" ")[0][:-4]
                feat = text_cache.get(rgb_key)
                if feat is None:
                    pt_path = _text_feat_pt_path(text_feat_dir, rgb_key)
                    if pt_path is None:
                        raise FileNotFoundError(f"Text feature not found for {rgb_key} under {text_feat_dir}")
                    feat = torch.load(pt_path, map_location="cpu")
                    text_cache[rgb_key] = feat
                text_list.append(feat.to(device))
            text_feature_list = torch.cat(text_list, dim=0)

            # WorDepth forward: inference 모드 (loss 계산 없음)
            pred_depth = eval_model(image, text_feature_list, sample_from_gaussian=False)

            masks_batch = batch.get("masks")
            relations_batch = batch.get("relations")
            if masks_batch is None or relations_batch is None:
                continue

            _ = rel_loss_fn(pred_depth, masks_batch, relations_batch)
            stats = rel_loss_fn.last_stats

            num_rel_batch = stats.get("num_relations", 0.0)
            if num_rel_batch <= 0:
                continue

            total_relations += num_rel_batch
            total_satisfied += stats.get("num_satisfied", 0.0)
            total_violation += stats.get("sum_violation", 0.0)

            # Collect hard violated relations for visualization (앞뒤 뒤집힌 사례)
            if viz_enabled and viz_topk > 0:
                batch_violations = _collect_batch_violations_for_viz(
                    depth_pred=pred_depth,
                    batch=batch,
                    args=args,
                )
                if batch_violations:
                    top_violations.extend(batch_violations)
                    # keep only top (viz_topk * 3) in memory for safety
                    top_violations.sort(key=lambda x: x["delta"], reverse=True)
                    if len(top_violations) > viz_topk * 3:
                        del top_violations[viz_topk:]

    if total_relations == 0:
        logger.warning("No valid relations found during RSR eval.")
        return 0.0, 0.0, 0

    rsr = total_satisfied / total_relations
    mean_violation = total_violation / total_relations

    # 최종 상위 K개 위반 관계 시각화
    if viz_enabled and viz_topk > 0 and top_violations:
        os.makedirs(args.viz_dir, exist_ok=True)
        _save_violation_visualizations(
            violations=top_violations,
            topk=viz_topk,
            out_dir=args.viz_dir,
        )

    return float(rsr), float(mean_violation), int(total_relations)


def _collect_batch_violations_for_viz(
    depth_pred: torch.Tensor,
    batch: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Collect severely flipped front/behind relations in a batch for visualization.

    A relation is considered "flipped" when, after normalization to 'front',
    the predicted representative depth violates the ordering strongly:
        d_A > d_B  (A should be in front of B, but is predicted deeper)

    We mirror the representative depth computation in RelationalDepthLoss to keep
    RSR and visualization semantics aligned.
    """
    device = depth_pred.device
    dtype = depth_pred.dtype
    B, _, H_d, W_d = depth_pred.shape

    masks_batch = batch.get("masks")
    relations_batch = batch.get("relations")
    depth_gt_batch = batch.get("depth")
    images_batch = batch.get("image")
    sample_paths = batch.get("sample_path")

    if masks_batch is None or relations_batch is None:
        return []

    valid_min_depth: float = float(getattr(args, "min_depth_eval", 1e-3))
    valid_max_depth: float = float(getattr(args, "max_depth_eval", 80.0))
    min_valid_pixels: int = int(getattr(args, "rel_min_pixels", 20))
    repr_mode: str = str(getattr(args, "rel_repr", "median")).lower()
    alpha: float = float(getattr(args, "rel_statistical_alpha", 1.0))

    results: List[Dict[str, Any]] = []

    for b in range(B):
        cur_depth = depth_pred[b, 0]  # (H,W)
        cur_masks = masks_batch[b]
        cur_rels = relations_batch[b]

        if cur_masks is None or cur_rels is None or len(cur_rels) == 0:
            continue

        if not torch.is_tensor(cur_masks):
            cur_masks = torch.as_tensor(cur_masks)
        if cur_masks.dim() == 2:
            cur_masks = cur_masks.unsqueeze(0)

        # Resize masks to depth resolution if needed
        if tuple(cur_masks.shape[-2:]) != (H_d, W_d):
            cur_masks = torch.nn.functional.interpolate(
                cur_masks.unsqueeze(1).float(),
                size=(H_d, W_d),
                mode="nearest",
            ).squeeze(1)
        cur_masks = cur_masks.to(device=device).float()

        N_obj = int(cur_masks.shape[0])
        if N_obj == 0:
            continue

        # Valid depth mask (same gating as loss)
        valid_depth = (cur_depth > valid_min_depth) & (cur_depth < valid_max_depth)

        # Representative depth per object
        obj_depths = torch.empty((N_obj,), device=device, dtype=dtype)
        obj_valid = torch.zeros((N_obj,), device=device, dtype=torch.bool)

        for k in range(N_obj):
            mk = cur_masks[k] > 0.5
            mk_valid = mk & valid_depth
            cnt = int(mk_valid.sum().item())
            if cnt < min_valid_pixels:
                obj_depths[k] = torch.nan
                continue

            vals = cur_depth[mk_valid].to(dtype)
            if repr_mode == "median":
                obj_depths[k] = vals.median()
            else:
                mu = vals.mean()
                sigma = vals.std(unbiased=False)
                obj_depths[k] = mu + alpha * sigma
            obj_valid[k] = True

        # Normalize relations: behind -> swap to front (same as loss)
        # We also keep any available class/name metadata if present.
        norm_rels: List[Tuple[int, int, float, str, str]] = []
        for rel in cur_rels:
            rel_type = str(rel.get("relation", "front")).lower()
            if rel_type not in {"front", "behind"}:
                continue
            s_idx = int(rel.get("subject_idx"))
            o_idx = int(rel.get("object_idx"))
            if rel_type == "behind":
                s_idx, o_idx = o_idx, s_idx
            if s_idx < 0 or s_idx >= N_obj or o_idx < 0 or o_idx >= N_obj:
                continue
            if (not obj_valid[s_idx]) or (not obj_valid[o_idx]):
                continue
            conf = float(rel.get("confidence", 1.0))
            # Try to extract human-readable class labels if available
            subj_label = (
                rel.get("subject_label")
                or rel.get("subject_name")
                or rel.get("subject_class")
                or rel.get("subject")
                or ""
            )
            obj_label = (
                rel.get("object_label")
                or rel.get("object_name")
                or rel.get("object_class")
                or rel.get("object")
                or ""
            )
            norm_rels.append((s_idx, o_idx, conf, str(subj_label), str(obj_label)))

        if not norm_rels:
            continue

        idx_A = torch.tensor([r[0] for r in norm_rels], device=device, dtype=torch.long)
        idx_B = torch.tensor([r[1] for r in norm_rels], device=device, dtype=torch.long)
        subj_labels = [r[3] for r in norm_rels]
        obj_labels = [r[4] for r in norm_rels]

        d_A = obj_depths[idx_A]
        d_B = obj_depths[idx_B]
        finite = torch.isfinite(d_A) & torch.isfinite(d_B)
        if not finite.any():
            continue

        d_A = d_A[finite]
        d_B = d_B[finite]
        idx_A = idx_A[finite]
        idx_B = idx_B[finite]
        subj_labels = [lbl for lbl, keep in zip(subj_labels, finite.tolist()) if keep]
        obj_labels = [lbl for lbl, keep in zip(obj_labels, finite.tolist()) if keep]

        # Flipped relations: A supposed to be front of B but predicted deeper (d_A > d_B)
        flipped_mask = d_A > d_B
        if not flipped_mask.any():
            continue

        d_A_flipped = d_A[flipped_mask]
        d_B_flipped = d_B[flipped_mask]
        idx_A_flipped = idx_A[flipped_mask]
        idx_B_flipped = idx_B[flipped_mask]
        subj_labels_flipped = [lbl for lbl, keep in zip(subj_labels, flipped_mask.tolist()) if keep]
        obj_labels_flipped = [lbl for lbl, keep in zip(obj_labels, flipped_mask.tolist()) if keep]

        # 한 배치에서 너무 많이 저장하지 않도록, delta 큰 순으로 상위 몇 개만 사용
        deltas = (d_A_flipped - d_B_flipped).detach().cpu()
        topk_local = min(len(deltas), 4)
        if topk_local <= 0:
            continue

        _, order = torch.topk(deltas, k=topk_local)

        # 준비: 시각화용 데이터 (CPU로 이동)
        image_b = images_batch[b].detach().cpu()  # (3,H,W)
        depth_gt_b = depth_gt_batch[b].detach().cpu() if depth_gt_batch is not None else None
        depth_pred_b = depth_pred[b].detach().cpu()  # (1,H,W)
        sample_path_b = (
            sample_paths[b] if isinstance(sample_paths, (list, tuple)) and len(sample_paths) > b else ""
        )

        for idx in order.tolist():
            obj_a = int(idx_A_flipped[idx].item())
            obj_b = int(idx_B_flipped[idx].item())
            delta_val = float(deltas[idx].item())
            d_a_val = float(d_A_flipped[idx].item())
            d_b_val = float(d_B_flipped[idx].item())

            mask_a = (cur_masks[obj_a] > 0.5).detach().cpu()
            mask_b = (cur_masks[obj_b] > 0.5).detach().cpu()

            # GT depth에서 대표 depth 값 계산 (가능하면)
            d_a_gt_val = float("nan")
            d_b_gt_val = float("nan")
            if depth_gt_b is not None:
                # depth_gt_b_tensor를 pred / masks와 동일한 device로 올려서 마스크 인덱싱 호환
                depth_gt_b_tensor = depth_gt_batch[b, 0].to(device)  # (H,W) on same device as valid_depth/cur_masks
                mk_a_valid_gt = (cur_masks[obj_a] > 0.5) & valid_depth  # valid_depth는 pred 기준이지만 범위 동일
                mk_b_valid_gt = (cur_masks[obj_b] > 0.5) & valid_depth

                if int(mk_a_valid_gt.sum().item()) >= min_valid_pixels:
                    vals_a_gt = depth_gt_b_tensor[mk_a_valid_gt].to(dtype)
                    if repr_mode == "median":
                        d_a_gt_val = float(vals_a_gt.median().item())
                    else:
                        mu_a = vals_a_gt.mean()
                        sigma_a = vals_a_gt.std(unbiased=False)
                        d_a_gt_val = float((mu_a + alpha * sigma_a).item())

                if int(mk_b_valid_gt.sum().item()) >= min_valid_pixels:
                    vals_b_gt = depth_gt_b_tensor[mk_b_valid_gt].to(dtype)
                    if repr_mode == "median":
                        d_b_gt_val = float(vals_b_gt.median().item())
                    else:
                        mu_b = vals_b_gt.mean()
                        sigma_b = vals_b_gt.std(unbiased=False)
                        d_b_gt_val = float((mu_b + alpha * sigma_b).item())

            results.append(
                {
                    "delta": delta_val,
                    "d_a": d_a_val,
                    "d_b": d_b_val,
                    "d_a_gt": d_a_gt_val,
                    "d_b_gt": d_b_gt_val,
                    "image": image_b,
                    "depth_gt": depth_gt_b,
                    "depth_pred": depth_pred_b,
                    "mask_a": mask_a,
                    "mask_b": mask_b,
                    "sample_path": sample_path_b,
                    "label_a": subj_labels_flipped[idx] if idx < len(subj_labels_flipped) else "",
                    "label_b": obj_labels_flipped[idx] if idx < len(obj_labels_flipped) else "",
                }
            )

    return results


def _save_violation_visualizations(
    violations: List[Dict[str, Any]],
    topk: int,
    out_dir: str,
) -> None:
    """Save visualizations for the top-K most severe flipped relations."""
    import numpy as np

    if not violations:
        return

    violations_sorted = sorted(violations, key=lambda x: x["delta"], reverse=True)[:topk]

    # ImageNet denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # depth colormap range
    vmin = 0.0
    vmax = 10.0

    for idx, v in enumerate(violations_sorted):
        img = v["image"]
        depth_gt = v.get("depth_gt")
        depth_pred = v.get("depth_pred")
        mask_a = v["mask_a"]
        mask_b = v["mask_b"]
        delta = v["delta"]
        sample_path = v.get("sample_path", "")
        d_a = float(v.get("d_a", float("nan")))
        d_b = float(v.get("d_b", float("nan")))
        d_a_gt = float(v.get("d_a_gt", float("nan")))
        d_b_gt = float(v.get("d_b_gt", float("nan")))
        label_a = v.get("label_a", "") or "A"
        label_b = v.get("label_b", "") or "B"

        # Denormalize RGB
        try:
            img_denorm = img * std + mean
        except Exception:
            img_denorm = img
        img_np = img_denorm.clamp(0.0, 1.0).permute(1, 2, 0).numpy()

        mask_a_np = mask_a.numpy().astype(bool)
        mask_b_np = mask_b.numpy().astype(bool)

        # --- Helper: apply semi-transparent mask overlay (RGB only) ---
        def _apply_overlay(base_img: np.ndarray, blend_alpha: float = 0.4) -> np.ndarray:
            out = base_img.copy()
            red = np.zeros_like(base_img)
            red[..., 0] = 1.0
            out[mask_a_np] = (1.0 - blend_alpha) * out[mask_a_np] + blend_alpha * red[mask_a_np]
            blue = np.zeros_like(base_img)
            blue[..., 2] = 1.0
            out[mask_b_np] = (1.0 - blend_alpha) * out[mask_b_np] + blend_alpha * blue[mask_b_np]
            return np.clip(out, 0.0, 1.0)

        ncols = 3 if depth_gt is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = list(axes.ravel())

        # ---- (1) RGB: 반투명 마스크 overlay + 아래에 객체 라벨 ----
        ax0 = axes[0]
        rgb_overlay = _apply_overlay(img_np, blend_alpha=0.35)
        ax0.imshow(rgb_overlay)
        ax0.contour(mask_a_np, colors="r", linewidths=1.0)
        ax0.contour(mask_b_np, colors="b", linewidths=1.0)
        ax0.set_title("RGB", fontsize=11)
        ax0.set_xlabel(
            f"Red: {label_a}  |  Blue: {label_b}",
            fontsize=9, color="black",
        )
        ax0.set_xticks([])
        ax0.set_yticks([])

        # ---- (2) GT depth: contour + 아래에 GT depth 값 ----
        col = 1
        if depth_gt is not None:
            ax1 = axes[1]
            ax1.imshow(depth_gt[0].numpy(), cmap=cm.magma, vmin=vmin, vmax=vmax)
            ax1.contour(mask_a_np, colors="r", linewidths=1.0)
            ax1.contour(mask_b_np, colors="b", linewidths=1.0)
            ax1.set_title("GT Depth", fontsize=11)
            ax1.set_xlabel(
                f"{label_a}(red): {d_a_gt:.2f}m  |  {label_b}(blue): {d_b_gt:.2f}m",
                fontsize=9, color="black",
            )
            ax1.set_xticks([])
            ax1.set_yticks([])
            col = 2

        # ---- (3) Pred depth: contour + 아래에 Pred depth 값 ----
        ax2 = axes[col]
        ax2.imshow(depth_pred[0].numpy(), cmap=cm.magma, vmin=vmin, vmax=vmax)
        ax2.contour(mask_a_np, colors="r", linewidths=1.0)
        ax2.contour(mask_b_np, colors="b", linewidths=1.0)
        ax2.set_title("Pred Depth", fontsize=11)
        ax2.set_xlabel(
            f"{label_a}(red): {d_a:.2f}m  |  {label_b}(blue): {d_b:.2f}m  →  FLIPPED (Δ={delta:.2f}m)",
            fontsize=9, color="red",
        )
        ax2.set_xticks([])
        ax2.set_yticks([])

        # ---- Title ----
        title = (
            f"GT: {label_a} is in FRONT of {label_b}\n"
            f"Pred: {label_a}={d_a:.2f}m > {label_b}={d_b:.2f}m → FLIPPED"
        )
        fig.suptitle(title, fontsize=10, y=1.02)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"flipped_relation_{idx:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Saved %d violation visualizations to %s", len(violations_sorted), out_dir)


def run_rsr_eval_only() -> None:
    """Entry point: parse args, run RSR eval, save/print results."""
    from utils import expand_argv_yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    argv = expand_argv_yaml(sys.argv[1:], repo_root)
    parser = _make_rsr_parser()
    args = parser.parse_args(argv)

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    rsr, mean_violation, num_rel = compute_rsr_for_checkpoint(args)

    logger.info(
        "RSR eval finished: RSR=%.4f, mean_violation=%.5f, num_relations=%d",
        rsr,
        mean_violation,
        num_rel,
    )

    # 결과 저장
    out_json = args.output_json
    if not out_json:
        # checkpoint와 같은 디렉터리에 저장
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        out_json = os.path.join(ckpt_dir, f"rsr_results_{ckpt_stem}.json")

    results: Dict[str, Any] = {
        "checkpoint_path": args.checkpoint_path,
        "dataset": args.dataset,
        "filenames_file_eval": args.filenames_file_eval,
        "relations_dir_eval": args.relations_dir_eval,
        "rel_repr": args.rel_repr,
        "rel_statistical_alpha": args.rel_statistical_alpha,
        "rel_margin": args.rel_margin,
        "rel_min_pixels": args.rel_min_pixels,
        "rsr": rsr,
        "mean_violation": mean_violation,
        "num_relations": num_rel,
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Saved RSR results to %s", out_json)


if __name__ == "__main__":
    run_rsr_eval_only()

