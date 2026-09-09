#!/usr/bin/env python
"""Figures for a trained checkpoint: reconstructions, time profile, rate-distortion."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import NMNISTSpikes, default_data_root
from src.metrics import accumulated_frame
from src.model import SpikeCompressor
from src.train import Config

READOUTS = ("soft", "sample", "hard")


def readout(probs, mode, th, seed=0):
    """Turn the decoder's Bernoulli intensity into something displayable.

    `sample` is the honest one for task 2: it is an actual binary spike train.
    `soft` shows the intensity the decoder really emits, `hard` the max-F1
    thresholding. See `src.train.evaluate`.
    """
    if mode == "soft":
        return probs
    if mode == "hard":
        return (probs > th).float()
    g = torch.Generator(device=probs.device).manual_seed(seed)
    return torch.bernoulli(probs, generator=g)


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ck["cfg"])
    model = SpikeCompressor(cfg.n_bits, cfg.n_bins, cfg.width, cfg.hidden,
                            cfg.beta).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg, ck["threshold"], ck["stats"]


@torch.no_grad()
def figure_reconstructions(model, cfg, th, ds, out, device, n=10, seed=0):
    """One column per digit class: original vs reconstruction, summed over time."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(ds.labels)
    idx = [int(rng.choice(np.flatnonzero(labels == c))) for c in range(n)]
    x = torch.stack([ds[i][0] for i in idx]).to(device)
    logits, b = model(x)
    probs = torch.sigmoid(logits)

    rows = [("original", accumulated_frame(x).cpu().numpy())]
    for mode in READOUTS:
        rows.append((f"recon ({mode})",
                     accumulated_frame(readout(probs, mode, th)).cpu().numpy()))

    fig, axes = plt.subplots(len(rows), n, figsize=(1.35 * n, 1.5 * len(rows)))
    orig = rows[0][1]
    for j in range(n):
        for row, (name, imgs) in enumerate(rows):
            ax = axes[row, j]
            img = imgs[j]
            # each row on its own scale: the readouts differ by a global gain,
            # and the question here is shape, not brightness
            ax.imshow(img, cmap="inferno", vmin=0, vmax=max(img.max(), 1e-6))
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
        axes[0, j].set_title(str(labels[idx[j]]), fontsize=10)
    fig.suptitle(
        f"N-MNIST test set, spike trains summed over time and polarity\n"
        f"m = {cfg.n_bits} bits per recording "
        f"({cfg.n_bins * 2 * 34 * 34} bits in, {model.compression_ratio():.0f}x compression)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


@torch.no_grad()
def figure_time_slices(model, cfg, th, ds, out, device, sample=0):
    """Per-time-bin frames: does the reconstruction reproduce the saccades?"""
    x = ds[sample][0][None].to(device)
    logits, _ = model(x)
    pred = readout(torch.sigmoid(logits), "sample", th)
    o = x[0].sum(1).cpu().numpy()
    r = pred[0].sum(1).cpu().numpy()

    T = cfg.n_bins
    fig, axes = plt.subplots(2, T, figsize=(0.85 * T, 2.4))
    vmax = max(o.max(), 1)
    for t in range(T):
        axes[0, t].imshow(o[t], cmap="inferno", vmin=0, vmax=vmax)
        axes[1, t].imshow(r[t], cmap="inferno", vmin=0, vmax=vmax)
        for row in range(2):
            axes[row, t].set_xticks([]); axes[row, t].set_yticks([])
        axes[0, t].set_title(f"{t}", fontsize=7)
    axes[0, 0].set_ylabel("orig", fontsize=8)
    axes[1, 0].set_ylabel("recon", fontsize=8)
    fig.suptitle(f"Per-time-bin spike frames (Bernoulli readout), m = {cfg.n_bits} "
                 f"bits, 20 ms per bin, digit {ds[sample][1]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


@torch.no_grad()
def figure_rate_profile(model, cfg, th, ds, out, device, n=512):
    """Spikes per time bin, original vs reconstruction, averaged over samples."""
    xs = torch.stack([ds[i][0] for i in range(n)]).to(device)
    orig, rec = [], []
    for i in range(0, n, 128):
        x = xs[i:i + 128]
        logits, _ = model(x)
        pred = readout(torch.sigmoid(logits), "sample", th)
        orig.append(x.sum(dim=(2, 3, 4)).cpu())
        rec.append(pred.sum(dim=(2, 3, 4)).cpu())
    orig = torch.cat(orig).mean(0).numpy()
    rec = torch.cat(rec).mean(0).numpy()

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    t = np.arange(cfg.n_bins) * (320 / cfg.n_bins)
    ax.plot(t, orig, "o-", label="original", color="#1b3a6b")
    ax.plot(t, rec, "s--", label=f"reconstruction, sampled (m={cfg.n_bits})",
            color="#c1440e")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("spikes per bin")
    ax.set_title("Temporal spike profile: the three micro-saccades")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def figure_rate_distortion(sweep_dir, out):
    """Distortion against the bit budget m, over a directory of runs."""
    runs = []
    for d in sorted(Path(sweep_dir).glob("m*")):
        h = d / "history.json"
        if not h.exists():
            continue
        hist = json.loads(h.read_text())
        best = max(hist, key=lambda s: s["soft_frame_corr"])
        runs.append((json.loads((d / "config.json").read_text())["n_bits"], best))
    if not runs:
        print("no completed runs in sweep dir"); return
    runs.sort()
    m = [r[0] for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    # baselines measured in scripts/baselines.py, for context
    REF = {"frame_corr": [("dataset mean, 0 bits", 0.720), ("PCA-sign codec", 0.907)],
           "van_rossum": [("silent reconstruction", 1.000), ("dataset mean, 0 bits", 1.305)],
           "f1": [("dataset mean, 0 bits", 0.380)]}
    specs = [("frame_corr", "correlation of time-summed frame", True),
             ("van_rossum", "normalised van Rossum distance", False),
             ("f1", "spike F1", True)]
    for ax, (key, label, up) in zip(axes, specs):
        for tag, style in [("soft", "o-"), ("sample", "s--")]:
            ax.plot(m, [r[1][f"{tag}_{key}"] for r in runs], style,
                    label=f"{tag} readout")
        for name, val in REF.get(key, []):
            ax.axhline(val, ls=":", lw=1, color="grey")
            ax.annotate(name, (m[0], val), fontsize=7, color="grey",
                        va="bottom", ha="left")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("m (bits per recording)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(("higher is better" if up else "lower is better"), fontsize=9)
    fig.suptitle("Rate-distortion tradeoff, N-MNIST test set (block = 36992 bits)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--sweep-dir", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if a.ckpt:
        model, cfg, th, stats = load(a.ckpt, device)
        print("checkpoint stats:", json.dumps(stats, indent=2))
        ds = NMNISTSpikes(a.data_root or default_data_root(), cfg.n_bins, train=False)
        tag = f"m{cfg.n_bits}"
        figure_reconstructions(model, cfg, th, ds, a.out_dir / f"reconstructions-{tag}.png", device)
        figure_time_slices(model, cfg, th, ds, a.out_dir / f"time-slices-{tag}.png", device)
        figure_rate_profile(model, cfg, th, ds, a.out_dir / f"rate-profile-{tag}.png", device)
    if a.sweep_dir:
        figure_rate_distortion(a.sweep_dir, a.out_dir / "rate-distortion.png")
