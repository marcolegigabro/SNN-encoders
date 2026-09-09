"""N-MNIST as fixed-length binary spike trains.

An N-MNIST recording is a list of events (x, y, t, p). We bin it onto a fixed
grid of T time bins over a fixed 320 ms window and binarise, which gives the
spike train

    x in {0,1}^(T, 2, 34, 34)

that task 2 asks us to compress. Binarising (rather than keeping event counts)
is deliberate: the subject's x(t) is a spike train, and a bin holding two events
is still a single spike as far as the downstream SNN is concerned.

Decoding the raw .bin files through tonic costs ~2 min for the 70k recordings,
so the binned dataset is cached once as bit-packed uint8 (277 MB train, 46 MB
test instead of 2.2 GB / 370 MB unpacked) and unpacked per batch.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SENSOR = (2, 34, 34)  # (polarity, y, x)
WINDOW_US = 320_000  # recordings run to ~308 ms; see 03-data/README.md


def default_data_root() -> Path:
    root = os.environ.get("SNN_DATA_ROOT")
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[2] / "data"


def _bin_one(args):
    """Bin one recording into a packed bit vector of length T*2*34*34."""
    path, n_bins = args
    from tonic.io import read_mnist_file

    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ev = read_mnist_file(path, dtype=dtype)

    bin_us = WINDOW_US // n_bins
    t = np.clip(ev["t"] // bin_us, 0, n_bins - 1)
    # tonic emits (time, polarity, height, width); ev['y'] indexes height.
    flat = ((t * 2 + ev["p"]) * 34 + ev["y"]) * 34 + ev["x"]

    frame = np.zeros(n_bins * 2 * 34 * 34, dtype=bool)
    frame[flat] = True
    return np.packbits(frame)


def build_cache(data_root: Path, n_bins: int, workers: int = 12) -> None:
    """Bin both splits and write <data_root>/nmnist-cache/{split}-T{n_bins}.npz."""
    import tonic

    out_dir = data_root / "nmnist-cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    for train in (True, False):
        split = "train" if train else "test"
        out = out_dir / f"{split}-T{n_bins}.npz"
        if out.exists():
            print(f"{out.name}: already cached, skipping")
            continue

        ds = tonic.datasets.NMNIST(save_to=str(data_root / "nmnist"), train=train)
        paths = list(ds.data)
        print(f"{split}: binning {len(paths)} recordings into T={n_bins} bins")

        with ProcessPoolExecutor(max_workers=workers) as pool:
            packed = list(
                pool.map(_bin_one, [(p, n_bins) for p in paths], chunksize=64)
            )

        packed = np.stack(packed)
        labels = np.asarray(ds.targets, dtype=np.int64)
        np.savez(out, packed=packed, labels=labels, n_bins=n_bins)

        occ = np.unpackbits(packed[:256]).mean()
        print(f"{split}: {packed.shape} uint8, occupancy {occ:.3%} -> {out}")


class NMNISTSpikes(Dataset):
    """Binned, binarised N-MNIST. Returns (spikes, label) with spikes float32
    of shape (T, 2, 34, 34) in {0, 1}."""

    def __init__(self, data_root: Path, n_bins: int, train: bool):
        split = "train" if train else "test"
        path = data_root / "nmnist-cache" / f"{split}-T{n_bins}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run scripts/build_cache.py --n-bins {n_bins} first."
            )
        blob = np.load(path)
        self.packed = blob["packed"]
        self.labels = blob["labels"]
        self.n_bins = n_bins
        self.shape = (n_bins, *SENSOR)
        self.n_cells = int(np.prod(self.shape))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        bits = np.unpackbits(self.packed[i])[: self.n_cells]
        x = torch.from_numpy(bits.reshape(self.shape).astype(np.float32))
        return x, int(self.labels[i])
