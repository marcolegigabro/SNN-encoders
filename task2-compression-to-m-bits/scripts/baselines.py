#!/usr/bin/env python
"""Non-learned m-bit baselines, so the SNN's numbers mean something.

Three reference points at the same bit budget:

* `silent`   predict no spikes at all. The floor: any model scoring worse than
             this on van Rossum is actively harmful.
* `mean`     spend 0 bits, always emit the dataset-average spike train
             thresholded to the right occupancy. Catches "the model only learned
             the average digit".
* `pca-sign` the classical linear analogue of our system: project the time-summed
             frame onto its top-m PCA directions, keep the sign of each
             coefficient (m bits), reconstruct linearly, and spread the result
             back over time using the dataset-average temporal profile. This is
             a real m-bit codec, just not a spiking or a learned one.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import NMNISTSpikes, default_data_root
from src.metrics import image_metrics, pick_threshold, spike_agreement, van_rossum


def stack(ds, n=None):
    n = n or len(ds)
    dl = DataLoader(ds, 256, num_workers=3)
    out, seen = [], 0
    for x, _ in dl:
        out.append(x)
        seen += x.shape[0]
        if seen >= n:
            break
    return torch.cat(out)[:n]


def report(name, pred, target, extra=None):
    row = {"baseline": name}
    row.update(spike_agreement(pred, target))
    row["van_rossum"] = van_rossum(pred, target)
    row.update(image_metrics(pred, target))
    row["pred_occupancy"] = pred.mean().item()
    if extra:
        row.update(extra)
    print(f"{name:>14} | F1 {row['f1']:.4f} | IoU {row['iou']:.4f} | "
          f"vR {row['van_rossum']:.3f} | frame_corr {row['frame_corr']:.4f} | "
          f"frame_nmse {row['frame_nmse']:.3f}", flush=True)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--n-bins", type=int, default=16)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--n-fit", type=int, default=8000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None)
    a = p.parse_args()
    root = a.data_root or default_data_root()

    train_x = stack(NMNISTSpikes(root, a.n_bins, train=True), a.n_fit)
    test_x = stack(NMNISTSpikes(root, a.n_bins, train=False), a.n_test)
    T = a.n_bins
    rows = []

    rows.append(report("silent", torch.zeros_like(test_x), test_x, {"n_bits": 0}))

    # dataset-average spike train, thresholded for max F1 on training data
    mean_train = train_x.mean(0, keepdim=True)
    th, _ = pick_threshold(mean_train.expand_as(train_x[:1024]), train_x[:1024])
    rows.append(report("mean", (mean_train > th).float().expand_as(test_x),
                       test_x, {"n_bits": 0}))

    # PCA of the time-summed frame + 1-bit-per-coefficient quantisation
    frames_tr = train_x.sum(dim=(1, 2)).flatten(1).numpy()
    frames_te = test_x.sum(dim=(1, 2)).flatten(1).numpy()
    mu = frames_tr.mean(0)
    # temporal/polarity profile used to spread a frame back over (T, 2)
    profile = (train_x.sum(dim=(3, 4)) / train_x.sum(dim=(1, 2, 3, 4))[:, None, None]
               ).mean(0)  # (T, 2)

    from sklearn.decomposition import PCA
    for m in a.bits:
        pca = PCA(n_components=min(m, frames_tr.shape[0], frames_tr.shape[1])).fit(frames_tr - mu)
        def decode(F):
            c = pca.transform(F - mu)
            # 1 bit per coefficient: keep the sign, restore a per-component scale
            q = np.sign(c) * np.abs(c).mean(0, keepdims=True)
            return (pca.inverse_transform(q) + mu).reshape(-1, 34, 34)

        rec_tr = torch.from_numpy(decode(frames_tr[:1024])).float()
        rec_te = torch.from_numpy(decode(frames_te)).float()
        # spread each frame over time with the average profile -> a spike-rate tensor
        def spread(rec):
            return (rec[:, None, None] * profile[None, :, :, None, None]).clamp(0, 1)
        th, _ = pick_threshold(spread(rec_tr), train_x[:1024])
        rows.append(report(f"pca-sign m={m}", (spread(rec_te) > th).float(),
                           test_x, {"n_bits": m}))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")
