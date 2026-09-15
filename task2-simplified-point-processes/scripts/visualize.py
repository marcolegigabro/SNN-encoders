#!/usr/bin/env python
"""Learned rate-distortion curves against Rubin's baseline and the Poisson theory.

Left panel, for one lambda T: Rubin's bounds (T -> infinity), the binned-count
baseline with order-statistic median placement (the better of Rubin's two
decoders), the zero-rate point, and every learned sweep given with --sweep.

Right panel: distortion of each learned codec divided by the baseline's
distortion at the same rate. Between two of its operating points the baseline
is linearly interpolated in (rate, D), which is achievable by time-sharing the
two neighbouring schemes across windows, so a ratio below 1 is a real win and
not an artefact of the interpolation.

    python scripts/visualize.py --sweep SNN=runs/sweep-snn-l8a16-lt5 \
                                --sweep GRU=runs/sweep-ann-l8a16-lt5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import theory
from src.plotting import INK, INK2, MUTED, SLOTS, SURFACE, style


def load_baseline(path: Path, lambda_t: float):
    res = json.loads(path.read_text())[f"{lambda_t:g}"]
    pts = sorted((p for p in res["points"] if p["placement"] == "median"),
                 key=lambda p: p["real_bits_per_time"])
    R = np.array([0.0] + [p["real_bits_per_time"] for p in pts])
    D = np.array([res["zero_rate_D"]] + [p["D"] for p in pts])
    return res, R, D


def load_sweep(path: Path):
    rows = sorted(json.loads((path / "summary.json").read_text()), key=lambda s: s["rate_real"])
    return (np.array([s["rate_real"] for s in rows]), np.array([s["D"] for s in rows]),
            np.array([s["beta"] for s in rows]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="append", default=[], metavar="LABEL=DIR",
                   help="a sweep directory holding summary.json (repeatable, max 2)")
    p.add_argument("--lambda-t", type=float, default=5.0)
    p.add_argument("--baseline", type=Path, default=ROOT / "runs" / "baseline" / "baseline.json")
    p.add_argument("--fig", type=Path, default=None)
    a = p.parse_args()
    if not 1 <= len(a.sweep) <= 2:
        raise SystemExit("give one or two --sweep LABEL=DIR (slot 1 is the baseline)")

    res, R_base, D_base = load_baseline(a.baseline, a.lambda_t)
    rate = res["rate"]
    sweeps = []
    for spec in a.sweep:
        label, _, d = spec.partition("=")
        d = Path(d) if Path(d).is_absolute() else ROOT / d
        sweeps.append((label, *load_sweep(d)))

    print(f"lambda T = {a.lambda_t:g}; baseline at the same rate = time-sharing interpolation")
    for label, R, D, betas in sweeps:
        print(f"\n{label}\n{'beta':>7} {'rate':>7} {'D':>8} {'D base':>8} {'ratio':>7}")
        for r, d, b in zip(R, D, betas):
            db = float(np.interp(r, R_base, D_base))
            print(f"{b:>7g} {r:>7.2f} {d:>8.4f} {db:>8.4f} {d / db:>7.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [1.3, 1]})
    style(ax)
    style(axr)

    r_top = 1.2 * max(32.0, *(R.max() for _, R, _, _ in sweeps))
    shown_base = R_base <= r_top
    d_min = min(D_base[shown_base].min(), *(D.min() for _, _, D, _ in sweeps)) / 1.5
    d_max = res["zero_rate_D"] * 1.35

    Dg = np.logspace(np.log10(d_min), np.log10(d_max), 400)
    low, up = theory.rubin_lower_bound(Dg, rate), theory.rubin_upper_bound(Dg, rate)
    ax.fill_between(Dg, low, up, color=MUTED, alpha=0.18, linewidth=0)
    ax.plot(Dg, up, color=INK2, linewidth=1)
    ax.plot(Dg, low, color=INK2, linewidth=1)

    ax.plot(D_base, R_base, color=SLOTS[0], linewidth=2, marker="o", markersize=6,
            markerfacecolor=SLOTS[0], markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    cc = res["count_cost_bits_per_time"]
    ax.axhline(cc, color=MUTED, linewidth=1)
    ax.text(d_min * 1.1, cc, f"  count cost H(N)/T = {cc:.1f}", color=INK2, fontsize=8, va="bottom")
    # above the learned curves: a collapsed run sits almost exactly on this point
    ax.plot(res["zero_rate_D"], 0, marker="D", markersize=7, color=INK, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=6, clip_on=False)

    for (label, R, D, _), color in zip(sweeps, SLOTS[1:]):
        ax.plot(D, R, color=color, linewidth=2, marker="o", markersize=7, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5, clip_on=False)
        ax.annotate(label, (D[-1], R[-1]), xytext=(6, 4), textcoords="offset points",
                    color=INK2, fontsize=9)

        keep = R > 0.05                      # a collapsed run has no meaningful ratio
        ratio = D[keep] / np.interp(R[keep], R_base, D_base)
        axr.plot(R[keep], ratio, color=color, linewidth=2, marker="o", markersize=7,
                 markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
        axr.annotate(label, (R[keep][-1], ratio[-1]), xytext=(6, 0), textcoords="offset points",
                     color=INK2, fontsize=9, va="center")

    ax.set_xscale("log")
    ax.set_xlim(d_min, d_max)
    ax.set_ylim(0, r_top)
    ax.set_xlabel("Distortion D, counting-function L1", color=INK2, fontsize=9)
    ax.set_ylabel("Rate (bits per unit time, real bitstream)", color=INK2, fontsize=9)
    ax.set_title("Rate vs distortion", color=INK, fontsize=11, loc="left")

    axr.axhline(1.0, color=INK2, linewidth=1)
    axr.text(0.3, 1.0, " baseline", color=INK2, fontsize=8, va="bottom")
    axr.set_yscale("log")
    ticks = [0.5, 1, 2, 4, 8, 16, 32, 64]
    axr.set_yticks(ticks)
    axr.set_yticklabels([f"{t:g}×" for t in ticks])
    axr.minorticks_off()
    axr.set_xlim(0, 1.12 * max(R.max() for _, R, _, _ in sweeps))
    axr.set_xlabel("Rate (bits per unit time)", color=INK2, fontsize=9)
    axr.set_ylabel("D codec / D baseline at the same rate", color=INK2, fontsize=9)
    axr.set_title("Distortion relative to the baseline (lower is better)", color=INK,
                  fontsize=11, loc="left")

    handles = [
        Patch(facecolor=MUTED, alpha=0.35, edgecolor=INK2, label="Rubin bounds, T → ∞"),
        Line2D([], [], color=SLOTS[0], linewidth=2, marker="o", markersize=6,
               markeredgecolor=SURFACE, label="Rubin binned counts (median placement)"),
        *[Line2D([], [], color=c, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                 label=f"Learned codec, {label}") for (label, *_), c in zip(sweeps, SLOTS[1:])],
        Line2D([], [], linestyle="none", marker="D", markersize=7, color=INK,
               label="Best fixed reconstruction (0 bits)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Poisson source, λT = {a.lambda_t:g}: learned codecs vs Rubin's baseline",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))

    out = a.fig or ROOT / "figures" / f"rd-lambda{a.lambda_t:g}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
