#!/usr/bin/env python
"""Bin the raw N-MNIST recordings into the cached spike-train tensors."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_cache, default_data_root

p = argparse.ArgumentParser()
p.add_argument("--n-bins", type=int, default=16)
p.add_argument("--data-root", type=Path, default=None)
p.add_argument("--workers", type=int, default=12)
a = p.parse_args()
build_cache(a.data_root or default_data_root(), a.n_bins, a.workers)
