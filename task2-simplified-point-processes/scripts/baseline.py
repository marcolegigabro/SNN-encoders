#!/usr/bin/env python
"""Rubin's binned-count baseline against the Poisson rate-distortion theory.

For every rate lambda T and every bin count, encodes the shared 10k-window
evaluation set with Rubin's scheme (Section VII), decodes with both placements,
and measures D (exact counting-function L1) and the rate twice: ideal
-log2 p, and the real range-coded bitstream. Also records the zero-rate
distortion (best fixed reconstruction) and the finite-window count cost.

Writes runs/baseline/baseline.json and figures/rd-theory-baseline.png.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import baseline, coding, theory
from src.distortion import counting_l1
from src.plotting import INK, INK2, MUTED, SLOTS, SURFACE, style
from src.sources import Events, eval_set, homogeneous_poisson

SERIES = {"midpoint": SLOTS[0], "median": SLOTS[1]}
LABEL = {"midpoint": "Binned counts, midpoint (Thm 5)",
         "median": "Binned counts, order-stat medians (Thm 4)"}


def run(rates, bins, n_test, T):
    out = {}
    for rate in rates:
        test = eval_set(rate, n_test, T=T)
        train = homogeneous_poisson(50_000, rate, T, generator=torch.Generator().manual_seed(99))
        zr = baseline.zero_rate_reconstruction(train)
        zr = Events(zr.times.expand(len(test), -1), zr.counts.expand(len(test)), T)
        points = []
        for n_bins in bins:
            sym = baseline.encode(test, n_bins)
            real = coding.poisson_stream(sym.numpy(), rate * T / n_bins)
            D_th, R_th = theory.rubin_binned_scheme(T / n_bins, rate)
            for placement in ("midpoint", "median"):
                r = baseline.evaluate(test, rate, n_bins, placement)
                r.update(real_bits_per_time=real["real_bits_per_window"] / T,
                         coder_overhead=real["overhead"],
                         D_theorem5=float(D_th), R_theorem5=float(R_th))
                points.append(r)
                print(f"lambdaT={rate * T:<4g} bins={n_bins:<4d} {placement:<8s} "
                      f"D={r['D']:.4f} (thm5 {float(D_th):.4f})  "
                      f"R={r['bits_per_time']:.2f} real={r['real_bits_per_time']:.2f} "
                      f"(thm5 {float(R_th):.2f}) bits/T", flush=True)
        out[f"{rate * T:g}"] = {
            "rate": rate, "T": T,
            "zero_rate_D": counting_l1(test, zr).mean().item(),
            "count_cost_bits_per_time": theory.count_cost(rate, T),
            "points": points,
        }
    return out


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    keys = list(results)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        res = results[key]
        rate, T = res["rate"], res["T"]
        style(ax)
        pts = res["points"]
        d_min = min(p["D"] for p in pts) / 1.5
        d_max = max(res["zero_rate_D"], math.e / 4) * 1.3
        r_max = 1.15 * max(p["real_bits_per_time"] for p in pts)

        D = np.logspace(np.log10(d_min), np.log10(d_max), 400)
        low, up = theory.rubin_lower_bound(D, rate), theory.rubin_upper_bound(D, rate)
        ax.fill_between(D, low, up, color=MUTED, alpha=0.18, linewidth=0)
        ax.plot(D, up, color=INK2, linewidth=1)
        ax.plot(D, low, color=INK2, linewidth=1)

        dT = np.logspace(np.log10(T / 512), np.log10(T), 200)
        D_th, R_th = theory.rubin_binned_scheme(dT, rate)
        ax.plot(D_th, R_th, color=SERIES["midpoint"], linewidth=2, solid_capstyle="round")

        for placement, color in SERIES.items():
            sel = [p for p in pts if p["placement"] == placement]
            ax.plot([p["D"] for p in sel], [p["real_bits_per_time"] for p in sel],
                    linestyle="none", marker="o", markersize=7, markerfacecolor=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)

        cc = res["count_cost_bits_per_time"]
        ax.axhline(cc, color=MUTED, linewidth=1)
        ax.text(d_min * 1.1, cc, f"  count cost H(N)/T = {cc:.1f}", color=INK2,
                fontsize=8, va="bottom")
        ax.plot(res["zero_rate_D"], 0, marker="D", markersize=7, color=INK,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, clip_on=False)

        ax.set_xscale("log")
        ax.set_xlim(d_min, d_max)
        ax.set_ylim(0, r_max)
        ax.set_title(f"λT = {key}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Distortion D, counting-function L1", color=INK2, fontsize=9)
    axes[0].set_ylabel("Rate (bits per unit time, real bitstream)", color=INK2, fontsize=9)

    handles = [
        Patch(facecolor=MUTED, alpha=0.35, edgecolor=INK2, label="Rubin bounds, T → ∞ (eq. 40)"),
        Line2D([], [], color=SERIES["midpoint"], linewidth=2, label="Theorem 5, analytic"),
        *[Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=c,
                 markeredgecolor=SURFACE, label=LABEL[k]) for k, c in SERIES.items()],
        Line2D([], [], linestyle="none", marker="D", markersize=7, color=INK,
               label="Best fixed reconstruction (0 bits)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Poisson source: Rubin's binned-count baseline vs rate-distortion theory",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rates", type=float, nargs="+", default=[2.0, 5.0, 10.0],
                   help="event rates lambda (events per unit time)")
    p.add_argument("--bins", type=int, nargs="+", default=[2 ** k for k in range(9)])
    p.add_argument("--n-test", type=int, default=10_000)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "baseline" / "baseline.json")
    p.add_argument("--fig", type=Path, default=ROOT / "figures" / "rd-theory-baseline.png")
    a = p.parse_args()

    results = run(a.rates, a.bins, a.n_test, a.T)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {a.out}")
    plot(results, a.fig)
