"""
학습 후 eval까지 한 번에 실행. train 단독 → 완료 후 best_abs_rel 체크포인트로 eval 단독 실행.

실행 명령어 (from repo root):
  python scripts/run_train_then_eval.py
  python scripts/run_train_then_eval.py --train_config configs/arguments_run_nyu.txt --eval_config configs/arguments_eval_nyu.txt
  python scripts/run_train_then_eval.py --run_dir ./runs/nyu_train   # 이미 학습 끝난 디렉터리에서 eval만 실행
"""

import argparse
import glob
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_config_value(config_path: Path, key: str) -> str | None:
    """Config 파일에서 --key value 형태로 된 값을 읽음."""
    if not config_path.is_file():
        return None
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line.startswith(f"--{key}"):
                rest = line[len(f"--{key}"):].strip()
                if rest and not rest.startswith("-"):
                    return rest.split()[0]
                # 다음 토큰이 값일 수 있음 (같은 줄)
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == f"--{key}" and i + 1 < len(parts):
                        return parts[i + 1]
    return None


def _infer_run_dir(train_config: Path) -> Path | None:
    """train config에서 log_directory, model_name을 읽어 repo_root 기준 run 디렉터리 경로 반환."""
    log_dir = _parse_config_value(train_config, "log_directory")
    model_name = _parse_config_value(train_config, "model_name")
    if not log_dir or not model_name:
        return None
    # 항상 repo root 기준 (config 파일 위치와 무관)
    log_dir_clean = log_dir.strip("./").rstrip("/")
    return (_REPO_ROOT / log_dir_clean / model_name).resolve()


def _find_best_abs_rel_ckpt(run_dir: Path) -> Path | None:
    """run_dir 아래 model-*-best_abs_rel_* 파일 하나 반환 (확장자 없음)."""
    pattern = str(run_dir / "model-*-best_abs_rel_*")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # step 최대인 것 선택 (가장 최근 best)
    def step_from_path(p: str) -> int:
        m = re.search(r"model-(\d+)-best_abs_rel", p)
        return int(m.group(1)) if m else 0
    best = max(candidates, key=step_from_path)
    return Path(best)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train then eval with best_abs_rel checkpoint.")
    parser.add_argument(
        "--train_config",
        type=Path,
        default=_REPO_ROOT / "configs" / "arguments_run_nyu.txt",
        help="train config (@파일)",
    )
    parser.add_argument(
        "--eval_config",
        type=Path,
        default=_REPO_ROOT / "configs" / "arguments_eval_nyu.txt",
        help="eval config (@파일)",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="이미 학습이 끝난 run 디렉터리 (지정 시 train 건너뛰고 해당 디렉터리에서 ckpt 찾아 eval만 실행)",
    )
    parser.add_argument("--skip_train", action="store_true", help="train 건너뛰고 eval만 (run_dir 또는 train_config으로 run_dir 추론)")
    args = parser.parse_args()

    train_config = args.train_config if args.train_config.is_absolute() else _REPO_ROOT / args.train_config
    eval_config = args.eval_config if args.eval_config.is_absolute() else _REPO_ROOT / args.eval_config

    run_dir: Path | None = args.run_dir
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = _REPO_ROOT / run_dir

    # 1) Train (skip_train 또는 run_dir만 있으면 생략)
    if not args.skip_train and run_dir is None:
        logger.info("Running train: python src/train.py @%s", train_config)
        r = subprocess.run(
            [sys.executable, "src/train.py", f"@{train_config}"],
            cwd=str(_REPO_ROOT),
        )
        if r.returncode != 0:
            logger.error("Train failed with exit code %d", r.returncode)
            sys.exit(r.returncode)
        run_dir = _infer_run_dir(train_config)
        if run_dir is None:
            logger.error("Could not infer run_dir from train config; set --run_dir explicitly.")
            sys.exit(1)
    elif run_dir is None:
        run_dir = _infer_run_dir(train_config)

    if run_dir is None or not run_dir.is_dir():
        logger.error("Run dir not found: %s. Set --run_dir or run train first.", run_dir)
        sys.exit(1)

    # 2) Best abs_rel checkpoint 찾기
    ckpt = _find_best_abs_rel_ckpt(run_dir)
    if ckpt is None:
        logger.error("No model-*-best_abs_rel_* found in %s", run_dir)
        sys.exit(1)
    logger.info("Using checkpoint: %s", ckpt)

    # 3) Eval 실행 (checkpoint_path 오버라이드)
    try:
        eval_config_rel = eval_config.relative_to(_REPO_ROOT)
    except ValueError:
        eval_config_rel = eval_config
    cmd = [
        sys.executable, "src/eval.py",
        f"@{eval_config_rel}",
        "--checkpoint_path", str(ckpt),
    ]
    logger.info("Running eval: %s", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if r.returncode != 0:
        logger.error("Eval failed with exit code %d", r.returncode)
        sys.exit(r.returncode)
    logger.info("Done.")


if __name__ == "__main__":
    main()
