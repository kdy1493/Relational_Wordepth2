"""
WorDepth training script. 학습 전용 (eval은 src/eval.py 단독 실행).

실행 명령어 (from repo root):
  # 학습 (단일 GPU) — 설정: @config.txt 또는 configs/*.yaml
  python src/train.py configs/arguments_run_nyu_paper.yaml
  python src/train.py @configs/arguments_run_nyu_paper.txt
  # 학습 (다중 GPU, torchrun)
  torchrun --nproc_per_node=N src/train.py configs/arguments_run_nyu_paper.yaml --use_ddp

학습 후 eval까지 한 번에: scripts/run_train_then_eval.py 참고.
Supports: AMP, gradient accumulation, TensorBoard, EMA, tqdm, DDP.
"""

import argparse
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    block_print,
    enable_print,
    eval_metrics,
    convert_arg_line_to_args,
    expand_argv_yaml,
)
from networks.wordepth import WorDepth
from eval import online_eval, _text_feat_pt_path

try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    autocast = None
    GradScaler = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter  # type: ignore[no-redef]
    except ImportError:
        SummaryWriter = None  # type: ignore[misc, assignment]

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Repo root (train.py lives in src/). Use for text_feat/data paths so cwd does not matter.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WorDepth PyTorch implementation.",
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # Overall
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--model_name", type=str, default="WorDepth", help="model name")
    parser.add_argument("--pretrain", type=str, default=None, help="path of pretrained encoder")

    # Dataset
    parser.add_argument("--dataset", type=str, default="nyu", help="kitti or nyu")
    parser.add_argument("--data_path", type=str, required=True, help="path to the data")
    parser.add_argument("--gt_path", type=str, required=True, help="path to ground truth")
    parser.add_argument("--filenames_file", type=str, required=True, help="path to filenames text file")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=6553.5,
        help="NYU: raw/depth_scale = m. Default 6553.5 for converted sync. Use 1000 for original mm data.",
    )
    parser.add_argument("--input_height", type=int, default=480)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--max_depth", type=float, default=10.0, help="maximum depth in estimation")
    parser.add_argument("--prior_mean", type=float, default=1.54, help="prior mean of depth")

    # Log and save
    parser.add_argument("--log_directory", type=str, default="", help="directory for checkpoints and summaries")
    parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint to load")
    parser.add_argument("--log_freq", type=int, default=100, help="logging frequency in global steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="checkpoint save frequency in global steps")

    # Training
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--retrain", action="store_true", help="with checkpoint_path, restart from step 0")
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--end_learning_rate", type=float, default=-1.0)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="poly",
        choices=("poly", "cosine"),
        help="poly = (1 - step/total)^0.9; cosine = paper-style cosine decay",
    )
    parser.add_argument("--variance_focus", type=float, default=0.85)

    # Preprocessing
    parser.add_argument("--do_random_rotate", action="store_true")
    parser.add_argument("--degree", type=float, default=2.5, help="random rotation max degree")
    parser.add_argument("--do_kb_crop", action="store_true")
    parser.add_argument("--use_right", action="store_true", help="randomly use right images on KITTI")

    # Multi-GPU and precision
    parser.add_argument("--num_threads", type=int, default=1, help="data loading workers")
    parser.add_argument("--gpu_ids", type=str, default=None, help="comma-separated GPU ids")
    parser.add_argument("--use_amp", action="store_true", help="mixed precision (FP16)")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument(
        "--use_ddp",
        action="store_true",
        help="DDP; launch with: torchrun --nproc_per_node=N src/train.py @config --use_ddp",
    )
    parser.add_argument("--cache_images", action="store_true")

    # EMA
    parser.add_argument("--use_ema", action="store_true", help="exponential moving average of parameters")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")

    # Online eval
    parser.add_argument("--do_online_eval", action="store_true", help="eval every eval_freq steps")
    parser.add_argument("--data_path_eval", type=str, default=None)
    parser.add_argument("--gt_path_eval", type=str, default=None)
    parser.add_argument("--filenames_file_eval", type=str, default=None)
    parser.add_argument(
        "--filenames_file_test",
        type=str,
        default=None,
        help="optional test split; after training, run final eval once and log to test_metrics_final.json",
    )
    parser.add_argument("--min_depth_eval", type=float, default=1e-3)
    parser.add_argument("--max_depth_eval", type=float, default=80.0)
    parser.add_argument("--eigen_crop", action="store_true")
    parser.add_argument("--garg_crop", action="store_true")
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--eval_summary_directory", type=str, default="")
    parser.add_argument(
        "--save_best_metric_only",
        type=int,
        default=-1,
        help="only save best ckpt for this metric index (0-8); -1 = all",
    )

    # WorDepth
    parser.add_argument("--weight_kld", type=float, default=1e-3)
    parser.add_argument("--alter_prob", type=float, default=0.5)
    parser.add_argument(
        "--store_freq",
        type=int,
        default=3000,
        help="periodic checkpoint save every N steps (0=disable); enables resume from checkpoint",
    )
    parser.add_argument(
        "--store_freq_epochs",
        type=float,
        default=0.0,
        help="periodic checkpoint save every N epochs (overrides --store_freq when > 0)",
    )
    parser.add_argument("--legacy", action="store_true", help="keep skip connection in UNet")

    # Baseline mode (image-only, no text features from disk)
    parser.add_argument(
        "--baseline_mode",
        action="store_true",
        help="image-only baseline: do not load text features; feed zero text embeddings",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="use torch.compile(model) for faster forward/backward (PyTorch 2+); first epoch can be slower",
    )

    return parser


def parse_args() -> argparse.Namespace:
    argv = expand_argv_yaml(sys.argv[1:], _REPO_ROOT)
    args = _make_parser().parse_args(argv)
    return args


# -----------------------------------------------------------------------------
# Text feature cache
# -----------------------------------------------------------------------------

def preload_text_features(
    filenames_file: str,
    dataset: str,
    mode: str,
    is_main: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load all text features into memory. When is_main is False (DDP), tqdm is disabled. Uses repo root so cwd does not matter."""
    text_feat_dir = os.path.join(_REPO_ROOT, "data", "text_feat", dataset, mode)
    cache: Dict[str, torch.Tensor] = {}

    with open(filenames_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    if is_main:
        logger.info("Preloading %d text features from %s", len(lines), text_feat_dir)
        if not os.path.isdir(text_feat_dir):
            logger.warning("Text feature dir does not exist: %s", os.path.abspath(text_feat_dir))
    it: Any = tqdm(lines, desc="Loading .pt to memory") if is_main else lines

    for line in it:
        parts = line.split()
        if len(parts) < 1:
            continue
        rgb_path = parts[0][:-4]
        pt_path = _text_feat_pt_path(text_feat_dir, rgb_path)
        if pt_path is not None:
            cache[rgb_path] = torch.load(pt_path, map_location="cpu")

    if is_main:
        logger.info("Loaded %d text features (~%.1f MB est.)", len(cache), len(cache) * 5 / 1024)
        if len(cache) == 0 and lines:
            rel = lines[0].split()[0][:-4].lstrip("/")
            p1 = os.path.join(text_feat_dir, rel + ".pt")
            p2 = os.path.join(text_feat_dir, rel[len("train/"):] + ".pt") if rel.startswith("train/") else None
            logger.warning(
                "No .pt files found. Example: rgb_key=%s -> tried %s (exists=%s)%s",
                lines[0].split()[0][:-4],
                p1,
                os.path.isfile(p1),
                f", {p2} (exists={os.path.isfile(p2)})" if p2 else "",
            )
    return cache


# -----------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# -----------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow or not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_to_model(self, model: torch.nn.Module) -> None:
        """Temporarily replace model parameters with EMA values (e.g. for eval)."""
        # Handle DataParallel / DDP: model.module is the real module
        module = model.module if hasattr(model, "module") else model
        for name, param in module.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


# -----------------------------------------------------------------------------
# Metrics JSON
# -----------------------------------------------------------------------------

def _measures_to_dict(measures: torch.Tensor) -> Dict[str, float]:
    """Convert 9-element eval measures tensor to dict keyed by eval_metrics names."""
    return {name: float(measures[i].item()) for i, name in enumerate(eval_metrics)}


def append_metrics_json(metrics_path: str, step: int, measures: torch.Tensor) -> None:
    """Append one eval record to metrics history JSON (list of {step, metrics})."""
    record = {"step": step, "metrics": _measures_to_dict(measures)}
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            history = json.load(f)
    else:
        history = []
    history.append(record)
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)


def save_best_metrics_json(
    best_path: str,
    best_lower: torch.Tensor,
    best_higher: torch.Tensor,
    best_steps: np.ndarray,
) -> None:
    """Save best metric values and steps to best_metrics.json."""
    lower_names = eval_metrics[:6]
    higher_names = eval_metrics[6:9]
    data = {}
    for i, name in enumerate(lower_names):
        data[name] = {"best_value": float(best_lower[i].item()), "best_step": int(best_steps[i])}
    for i, name in enumerate(higher_names):
        data[name] = {"best_value": float(best_higher[i].item()), "best_step": int(best_steps[i + 6])}
    with open(best_path, "w") as f:
        json.dump(data, f, indent=2)


# -----------------------------------------------------------------------------
# Training worker (single process or DDP rank)
# -----------------------------------------------------------------------------

def main_worker(args: argparse.Namespace) -> None:
    use_ddp = getattr(args, "use_ddp", False)
    if use_ddp:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        is_main = torch.distributed.get_rank() == 0
    else:
        is_main = True

    # Output dir: if exists, use model_name2, model_name3, ...
    out_path = os.path.join(args.log_directory, args.model_name)
    if os.path.exists(out_path):
        n = 2
        while os.path.exists(os.path.join(args.log_directory, f"{args.model_name}{n}")):
            n += 1
        out_path = os.path.join(args.log_directory, f"{args.model_name}{n}")
        if is_main:
            logger.info("Output dir %s exists, using %s", os.path.join(args.log_directory, args.model_name), out_path)
    if is_main:
        os.makedirs(out_path, exist_ok=True)
        if len(sys.argv) >= 2:
            cfg = sys.argv[1].lstrip("@") if sys.argv[1].startswith("@") else sys.argv[1]
            if os.path.isfile(cfg):
                os.system('cp "' + cfg + '" "' + out_path + '/"')
        for label, src in [("train.py.backup", "src/train.py"), ("wordepth.py.backup", "src/networks/wordepth.py")]:
            if os.path.isfile(src):
                os.system('cp "' + src + '" "' + out_path + "/" + label + '"')

    # Model (baseline_arch=True => paper-style baseline: Swin-L + depth decoder only, no text path)
    baseline_mode = getattr(args, "baseline_mode", False)
    model = WorDepth(
        pretrained=args.pretrain,
        max_depth=args.max_depth,
        prior_mean=args.prior_mean,
        img_size=(args.input_height, args.input_width),
        weight_kld=args.weight_kld,
        alter_prob=args.alter_prob,
        legacy=args.legacy,
        baseline_arch=baseline_mode,
    )
    model.train()

    # Optional: torch.compile for faster step (PyTorch 2+); first run compiles and can be slower.
    # Use mode="default" to avoid CUDA graph overwrite errors with Swin (reduce-overhead can crash).
    if getattr(args, "compile", False):
        _compile = getattr(torch, "compile", None)
        if _compile is not None:
            model = _compile(model, mode="default")
            if is_main:
                logger.info("Model wrapped with torch.compile(mode='default')")
        elif is_main:
            logger.warning("--compile set but torch.compile not available (PyTorch 2+ required); ignoring")

    if is_main:
        nparams = sum(p.numel() for p in model.parameters())
        ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Parameters: total=%d trainable=%d", nparams, ntrain)

    if use_ddp:
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if is_main:
        logger.info("Model initialized")

    global_step = 0
    best_lower = torch.zeros(6).cpu() + 1e3
    best_higher = torch.zeros(3).cpu()
    best_steps = np.zeros(9, dtype=np.int32)

    optimizer = torch.optim.Adam(
        [{"params": model.module.parameters()}],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )

    use_amp = getattr(args, "use_amp", False) and autocast is not None and GradScaler is not None
    scaler = GradScaler() if use_amp else None
    accumulation_steps = max(1, getattr(args, "accumulation_steps", 1))

    ema: Optional[EMA] = None
    if getattr(args, "use_ema", False):
        ema = EMA(model.module, decay=getattr(args, "ema_decay", 0.999))
        if is_main:
            logger.info("EMA enabled (decay=%.4f)", ema.decay)

    # Load checkpoint (strict=False when baseline_arch so full-WorDepth ckpt can supply backbone/ref/outc only)
    model_just_loaded = False
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        if is_main:
            logger.info("Loading checkpoint: %s", args.checkpoint_path)
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        load_strict = not baseline_mode
        model.load_state_dict(ckpt["model"], strict=load_strict)
        if is_main and not load_strict:
            logger.info("Loaded checkpoint with strict=False (baseline_arch: only matching keys applied).")
        if not args.retrain:
            global_step = ckpt.get("global_step", 0)
            best_lower = ckpt.get("best_eval_measures_lower_better", best_lower).cpu()
            best_higher = ckpt.get("best_eval_measures_higher_better", best_higher).cpu()
            best_steps = ckpt.get("best_eval_steps", best_steps)
        if "ema" in ckpt and ema is not None:
            ema.load_state_dict(ckpt["ema"])
            if is_main:
                logger.info("Restored EMA state from checkpoint")
        if is_main:
            logger.info("Loaded checkpoint at global_step=%d", global_step)
        model_just_loaded = True
        del ckpt

    torch.backends.cudnn.benchmark = True

    # Data
    text_feat_cache: Optional[Dict[str, torch.Tensor]]
    if baseline_mode:
        text_feat_cache = None
        if is_main:
            logger.info("Baseline mode enabled: baseline_arch (Swin-L + depth decoder only), no text path.")
    else:
        text_feat_cache = preload_text_features(args.filenames_file, args.dataset, "train", is_main=is_main)
    dataloader = NewDataLoader(args, "train")
    dataloader_eval = None
    if getattr(args, "do_online_eval", False) and getattr(args, "filenames_file_eval", None) and os.path.isfile(args.filenames_file_eval):
        dataloader_eval = NewDataLoader(args, "online_eval")

    if is_main and getattr(args, "do_online_eval", False):
        logger.info("Online eval every %d steps; best models saved per metric.", args.eval_freq)
    if is_main and use_ddp:
        logger.info("DDP: rank 0 of %d (launch with torchrun --nproc_per_node=N ... --use_ddp).", torch.distributed.get_world_size())

    # TensorBoard and metrics paths (rank 0 only)
    summary_writer = None
    metrics_dir = out_path
    metrics_path = os.path.join(metrics_dir, "metrics.json")
    best_metrics_path = os.path.join(metrics_dir, "best_metrics.json")
    if is_main:
        summary_path = os.path.join(out_path, "summary")
        if args.do_online_eval and dataloader_eval and getattr(args, "eval_summary_directory", ""):
            summary_path = os.path.join(args.eval_summary_directory, os.path.basename(out_path))
        if SummaryWriter is not None:
            summary_writer = SummaryWriter(summary_path, flush_secs=30)

    end_lr = args.end_learning_rate if args.end_learning_rate != -1 else args.learning_rate
    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    save_best_only_idx = getattr(args, "save_best_metric_only", -1)

    # Store frequency in steps (allow overriding by epochs via store_freq_epochs, if provided)
    store_freq_steps = getattr(args, "store_freq", 0)
    store_freq_epochs = float(getattr(args, "store_freq_epochs", 0.0)) if hasattr(args, "store_freq_epochs") else 0.0
    if store_freq_epochs > 0 and steps_per_epoch > 0:
        store_freq_steps = int(store_freq_epochs * steps_per_epoch // accumulation_steps)
        if is_main:
            logger.info("store_freq_epochs=%.2f -> store_freq=%d steps", store_freq_epochs, store_freq_steps)

    # ---------- Training loop ----------
    while epoch < args.num_epochs:
        if use_ddp and getattr(dataloader, "train_sampler", None) is not None:
            dataloader.train_sampler.set_epoch(epoch)

        data_iter = tqdm(dataloader.data, total=len(dataloader.data), desc=f"Epoch {epoch}", dynamic_ncols=True) if is_main else dataloader.data

        for step, sample_batched in enumerate(data_iter):
            if step % accumulation_steps == 0:
                optimizer.zero_grad()

            image = sample_batched["image"].cuda(non_blocking=True)
            depth_gt = sample_batched["depth"].cuda(non_blocking=True)

            # Text features
            if baseline_mode:
                # Image-only baseline: feed zero text embeddings (no disk I/O, no language signal)
                batch_size = image.size(0)
                text_feature_list = torch.zeros(batch_size, 1024, device=image.device, dtype=torch.float32)
            else:
                text_list = []
                for i in range(len(sample_batched["sample_path"])):
                    rgb_key = sample_batched["sample_path"][i].split(" ")[0][:-4]
                    if text_feat_cache is not None and rgb_key in text_feat_cache:
                        text_list.append(text_feat_cache[rgb_key].to(image.device))
                    else:
                        text_feat_dir = os.path.join(
                            _REPO_ROOT,
                            "data",
                            "text_feat",
                            "nyu" if args.dataset == "nyu" else "kitti",
                            "train",
                        )
                        pt_path = _text_feat_pt_path(text_feat_dir, rgb_key)
                        if pt_path is None:
                            raise FileNotFoundError(f"Text feature not found for {rgb_key} under {text_feat_dir}")
                        feat = torch.load(pt_path, map_location=image.device)
                        if text_feat_cache is not None:
                            text_feat_cache[rgb_key] = feat.cpu()
                        text_list.append(feat.to(image.device))
                text_feature_list = torch.cat(text_list, dim=0)

            if use_amp and scaler is not None:
                with autocast():
                    _, loss = model(image, text_feature_list, depth_gt)
                loss = loss.mean() if loss.numel() > 1 else loss
                scaler.scale(loss / accumulation_steps).backward()
            else:
                _, loss = model(image, text_feature_list, depth_gt)
                loss = loss.mean() if loss.numel() > 1 else loss
                (loss / accumulation_steps).backward()

            if (step + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                global_step += 1
                if ema is not None:
                    ema.update(model)

                # LR schedule
                progress = global_step / max(num_total_steps, 1)
                if getattr(args, "lr_schedule", "poly") == "cosine":
                    lr = end_lr + (args.learning_rate - end_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                else:
                    lr = (args.learning_rate - end_lr) * (1 - progress) ** 0.9 + end_lr
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Log loss
                if global_step % args.log_freq == 0 and not model_just_loaded and summary_writer is not None:
                    summary_writer.add_scalar("training_loss", loss.item(), global_step)
                if is_main and global_step % args.log_freq == 0 and not model_just_loaded:
                    logger.info("epoch=%d step=%d loss=%.4f", epoch, global_step, loss.item())

                # Periodic checkpoint (for resume: model, step, and EMA if used)
                if is_main and store_freq_steps > 0 and global_step % store_freq_steps == 0:
                    periodic_ckpt = {"global_step": global_step, "model": model.state_dict()}
                    if ema is not None:
                        periodic_ckpt["ema"] = ema.state_dict()
                    torch.save(periodic_ckpt, os.path.join(out_path, f"model-{global_step}"))

                # Online eval
                if (
                    dataloader_eval is not None
                    and args.do_online_eval
                    and global_step % args.eval_freq == 0
                    and not model_just_loaded
                ):
                    model.eval()
                    if ema is not None:
                        ema.apply_to_model(model)
                    eval_measures = None
                    if is_main:
                        with torch.no_grad():
                            eval_measures, _ = online_eval(model, dataloader_eval, args, post_process=False)
                    if use_ddp:
                        torch.distributed.barrier()
                    if eval_measures is not None and is_main:
                        for i in range(9):
                            if summary_writer is not None:
                                summary_writer.add_scalar(eval_metrics[i], eval_measures[i].item(), global_step)
                            is_lower = i < 6
                            if is_lower and eval_measures[i] < best_lower[i]:
                                old_step, old_val = int(best_steps[i]), float(best_lower[i].item())
                                best_lower[i] = eval_measures[i].item()
                                best_steps[i] = global_step
                                if save_best_only_idx < 0 or i == save_best_only_idx:
                                    _save_best_ckpt(out_path, model, ema, global_step, best_lower, best_higher, best_steps, eval_metrics[i], eval_measures[i], old_step, old_val)
                            elif not is_lower and eval_measures[i] > best_higher[i - 6]:
                                old_step, old_val = int(best_steps[i]), float(best_higher[i - 6].item())
                                best_higher[i - 6] = eval_measures[i].item()
                                best_steps[i] = global_step
                                if save_best_only_idx < 0 or i == save_best_only_idx:
                                    _save_best_ckpt(out_path, model, ema, global_step, best_lower, best_higher, best_steps, eval_metrics[i], eval_measures[i], old_step, old_val)
                        if summary_writer is not None:
                            summary_writer.flush()
                        append_metrics_json(metrics_path, global_step, eval_measures)
                        save_best_metrics_json(best_metrics_path, best_lower, best_higher, best_steps)
                        logger.info("Saved metrics to %s and %s", metrics_path, best_metrics_path)
                    model.train()
                    if is_main:
                        block_print()
                        enable_print()

            model_just_loaded = False

        epoch += 1


def _save_best_ckpt(
    out_path: str,
    model: torch.nn.Module,
    ema: Optional[EMA],
    global_step: int,
    best_lower: torch.Tensor,
    best_higher: torch.Tensor,
    best_steps: np.ndarray,
    metric_name: str,
    measure: torch.Tensor,
    old_best_step: int,
    old_best_value: float,
) -> None:
    """Save best checkpoint for one metric; remove previous best file for this metric."""
    if old_best_step > 0:
        old_name = os.path.join(out_path, f"model-{old_best_step}-best_{metric_name}_{old_best_value:.5f}")
        if os.path.isfile(old_name):
            os.remove(old_name)
    ckpt = {
        "global_step": global_step,
        "model": model.state_dict(),
        "best_eval_measures_lower_better": best_lower,
        "best_eval_measures_higher_better": best_higher,
        "best_eval_steps": best_steps,
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    path = os.path.join(out_path, f"model-{global_step}-best_{metric_name}_{measure.item():.5f}")
    torch.save(ckpt, path)
    logger.info("New best %s=%.5f -> %s", metric_name, measure.item(), path)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

args = parse_args()
if args.dataset in ("kitti", "nyu", "nyu_matched"):
    from dataloaders.dataloader import NewDataLoader  # noqa: E402
else:
    NewDataLoader = None  # type: ignore[misc, assignment]


if __name__ == "__main__":
    if args.mode != "train":
        logger.warning("train.py is for training only. Eval은 src/eval.py 를 사용하세요.")
        sys.exit(0)
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.makedirs(args.log_directory or "./runs", exist_ok=True)
    torch.cuda.empty_cache()
    main_worker(args)
