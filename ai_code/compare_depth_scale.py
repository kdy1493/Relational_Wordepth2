#!/usr/bin/env python3
"""
Compare raw value scale of depth maps across dataset roots:
  - /data/dongyub/nyu_v2 (training/depths, testing/depths)
  - /data/dongyub/sync  (scene/dense/sync_depth_dense_*.png)
  - matched_dataset: Relational_WorDepth/matched_dataset/nyu-depth-v2-matched (train|test/scene/depth_*.png)

Prints min, max, mean, and inferred meters for common scales (1000, 10000, 6553.5).
"""
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

NYUV2_SCALE = 6553.5
MM_PER_M = 1000.0


def collect_depth_files(path: Path, dataset_type: str, n_samples: int) -> list[Path]:
    """Collect up to n_samples depth PNG paths. dataset_type: 'nyu_v2' | 'sync' | 'matched'."""
    if not path.exists():
        return []
    if dataset_type == "nyu_v2":
        files = list((path / "training" / "depths").glob("*.png"))[:n_samples]
        if len(files) < n_samples:
            files += list((path / "testing" / "depths").glob("*.png"))[: max(0, n_samples - len(files))]
        return files
    if dataset_type == "sync":
        return sorted(path.rglob("dense/sync_depth_dense_*.png"))[:n_samples]
    if dataset_type == "matched":
        files = sorted((path / "train").rglob("depth_*.png"))[:n_samples]
        if len(files) < n_samples:
            files += sorted((path / "test").rglob("depth_*.png"))[: max(0, n_samples - len(files))]
        return files
    return []


def sample_depth_stats(path: Path, dataset_type: str, n_samples: int = 12) -> list[dict]:
    """Load depth PNGs and return list of {path, shape, dtype, min, max, mean, ...}."""
    files = collect_depth_files(path, dataset_type, n_samples)
    if not files:
        log.error("No depth PNGs under %s (type=%s)", path, dataset_type)
        return []
    out = []
    for f in files:
        try:
            img = Image.open(f)
            arr = np.array(img)
            if arr.ndim > 2:
                arr = arr[:, :, 0]
            valid = arr > 0
            nz = int(np.sum(valid))
            total = arr.size
            stats = {
                "path": str(f),
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": float(np.mean(arr[valid])) if nz else 0.0,
                "median": float(np.median(arr[valid])) if nz else 0.0,
                "nonzero_ratio": nz / total if total else 0.0,
            }
            out.append(stats)
        except Exception as e:
            log.warning("Skip %s: %s", f, e)
    return out


def print_section(name: str, stats_list: list[dict]) -> None:
    if not stats_list:
        return
    log.info("=== %s ===", name)
    mins = [s["min"] for s in stats_list]
    maxs = [s["max"] for s in stats_list]
    means = [s["mean"] for s in stats_list]
    log.info("  Files sampled: %s", len(stats_list))
    log.info("  Raw  min: %s  max: %s  mean(valid): %.1f", min(mins), max(maxs), np.mean(means))
    log.info("  Per-file min/max: first=%s/%s ... last=%s/%s", mins[0], maxs[0], mins[-1], maxs[-1])
    global_min, global_max = min(mins), max(maxs)
    log.info("  If raw/1000 (mm):     %.3f - %.3f m", global_min / MM_PER_M, global_max / MM_PER_M)
    log.info("  If raw/10000 (0.1mm):  %.3f - %.3f m", global_min / 10000, global_max / 10000)
    log.info("  If raw/6553.5 (NYU):   %.3f - %.3f m", global_min / NYUV2_SCALE, global_max / NYUV2_SCALE)
    log.info("  dtype: %s  shape: %s", stats_list[0]["dtype"], stats_list[0]["shape"])


def main() -> None:
    n_samples = 12
    sync_root = Path("/data/dongyub/sync")
    matched_root = Path("/home/dongyub/Relational_WorDepth/matched_dataset/nyu-depth-v2-matched")

    sync_stats = sample_depth_stats(sync_root, "sync", n_samples)
    matched_stats = sample_depth_stats(matched_root, "matched", n_samples)

    print_section("sync (data/dongyub/sync, dense/sync_depth_dense_*.png)", sync_stats)
    log.info("")
    print_section("matched_dataset (nyu-depth-v2-matched, train|test/scene/depth_*.png)", matched_stats)

    if sync_stats and matched_stats:
        log.info("")
        log.info("=== sync vs matched (raw scale) ===")
        s_min, s_max = min(s["min"] for s in sync_stats), max(s["max"] for s in sync_stats)
        m_min, m_max = min(s["min"] for s in matched_stats), max(s["max"] for s in matched_stats)
        log.info("  sync   raw: %s - %s", s_min, s_max)
        log.info("  matched raw: %s - %s", m_min, m_max)
        if s_max > 0:
            log.info("  matched/sync max ratio: %.2f", m_max / s_max)
        if m_max > 0:
            log.info("  sync/matched max ratio: %.2f", s_max / m_max)
        # Same scale if ratio ~1; if matched is mm and sync is same physical depth, similar raw
        log.info("  (ratio ~1 => similar raw scale; >>1 or <<1 => different scale or unit)")


if __name__ == "__main__":
    main()
