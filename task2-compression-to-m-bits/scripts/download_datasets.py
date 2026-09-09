#!/usr/bin/env python3
"""Download the event-based datasets used for task 2.

Datasets land in <repo>/data/<name>/, which is gitignored. Override the
location with --data-root or the SNN_DATA_ROOT environment variable; on
Mouad's machine <repo>/data is a symlink to the shared 03-data folder.

Usage:
    python scripts/download_datasets.py                    # N-MNIST + SHD (default)
    python scripts/download_datasets.py -d shd             # just one
    python scripts/download_datasets.py -d all             # adds DVS128 Gesture (~3 GB)
    python scripts/download_datasets.py --data-root /mnt/big/data

tonic writes its progress bars to stderr, so keep the streams separate if you
want a readable log:

    python scripts/download_datasets.py > download.log 2> progress.log
"""

import argparse
import os
import sys
import time
from pathlib import Path

import tonic.datasets as td

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("SNN_DATA_ROOT", REPO_ROOT / "data"))

# name -> (tonic class, human label, approximate download size)
DATASETS = {
    "nmnist": (td.NMNIST, "N-MNIST", "~1 GB"),
    "shd": (td.SHD, "Spiking Heidelberg Digits", "~400 MB"),
    "dvsgesture": (td.DVSGesture, "DVS128 Gesture", "~3 GB"),
}
DEFAULT = ["nmnist", "shd"]


DATA_ROOT = DEFAULT_DATA_ROOT  # rebound in main() from --data-root


def folder_size(path: Path) -> str:
    if not path.exists():
        return "0 B"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def fetch(name: str) -> bool:
    cls, label, size = DATASETS[name]
    target = DATA_ROOT / name
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label} ({size}) -> {target}", flush=True)

    for train in (True, False):
        split = "train" if train else "test"
        started = time.time()
        try:
            ds = cls(save_to=str(target), train=train)
            print(f"    {split:5s}: {len(ds)} samples  ({time.time() - started:.0f}s)", flush=True)
        except Exception as exc:
            print(f"    {split:5s}: FAILED  {type(exc).__name__}: {exc}", flush=True)
            return False

    print(f"    on disk: {folder_size(target)}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--datasets", nargs="+", default=DEFAULT,
                        choices=list(DATASETS) + ["all"],
                        help="datasets to download (default: nmnist shd)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"where to store datasets (default: {DEFAULT_DATA_ROOT})")
    args = parser.parse_args()

    global DATA_ROOT
    DATA_ROOT = args.data_root.resolve()

    wanted = list(DATASETS) if "all" in args.datasets else args.datasets
    print(f"data root: {DATA_ROOT}")

    failed = [name for name in wanted if not fetch(name)]

    print("\n=== summary")
    for name in wanted:
        mark = "FAILED" if name in failed else "ok"
        print(f"    {DATASETS[name][1]:28s} {mark:8s} {folder_size(DATA_ROOT / name)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
