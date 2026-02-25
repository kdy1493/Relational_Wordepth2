#!/usr/bin/env python3
"""
Convert dense depth PNGs under data/nyu_v2_sync to 16-bit NYU v2 format
(raw/6553.5 = m). Overwrites in place. Optimized: numpy vectorized + multiprocessing.

Usage:
  python scripts/convert_data_dense_to_nyuv2.py              # all (train + test)
  python scripts/convert_data_dense_to_nyuv2.py --subset test   # test only
  python scripts/convert_data_dense_to_nyuv2.py --subset train  # train only
"""
import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DENSE_ROOT = REPO_ROOT / "data" / "nyu_v2_sync"

NYUV2_SCALE = 6553.5
MM_PER_M = 1000.0

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def convert_one(path: Path) -> bool:
    """Convert one PNG to NYU v2 16-bit in place. Returns True on success."""
    try:
        img = Image.open(path)
        arr = np.array(img, dtype=np.float64)
        raw_max = arr.max()
        if raw_max > 10000:
            depth_m = np.where(arr != 0, arr / NYUV2_SCALE, 0.0)
        else:
            depth_m = np.where(arr != 0, arr / MM_PER_M, 0.0)
        out = np.clip(np.round(depth_m * NYUV2_SCALE), 0, 65535).astype(np.uint16)
        Image.fromarray(out).save(path)
        return True
    except Exception:
        return False


def _init_worker():
    """Prevent child processes from logging to same handler."""
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dense depth PNGs to 16-bit NYU v2 (raw/6553.5=m)")
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=("train", "test"),
        help="Limit to train/ or test/ under data/nyu_v2_sync; default: all",
    )
    args = parser.parse_args()

    data_dense = DATA_DENSE_ROOT if args.subset is None else DATA_DENSE_ROOT / args.subset
    if not data_dense.exists():
        log.error("Not found: %s", data_dense)
        return
    files = sorted(data_dense.rglob("dense/sync_depth_dense_*.png"))
    n = len(files)
    if n == 0:
        log.warning("No dense PNGs under %s", data_dense)
        return
    log.info("Converting %s dense PNGs under %s (numpy + %s workers)", n, data_dense, mp.cpu_count())

    workers = max(1, mp.cpu_count() - 1)
    done = 0
    with mp.Pool(workers) as pool:
        for ok in pool.imap_unordered(convert_one, files, chunksize=64):
            done += 1
            if ok and done % 5000 == 0:
                log.info("  %s / %s", done, n)

    log.info("Done. %s files overwritten.", n)


if __name__ == "__main__":
    main()
