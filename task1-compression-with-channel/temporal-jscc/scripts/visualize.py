#!/usr/bin/env python
"""Every figure: three rate-distortion planes, the capacities, and the code.

Plot conventions, fixed once here.

* Rate on x in bits per source bin, distortion on y. Down and to the left is
  better, and the computed R(D) is a floor: no point may lie below it.
* The bound is drawn in ink, not in a series colour, because it is not one
  system among several -- it is the axis the systems are measured against.
* Rate is bought two ways, and the two are drawn as different series with
  different markers: more code neurons at fixed jitter (circles) and less
  jitter at a fixed number of code neurons (squares). If the pricing of a
  channel use by C(sigma) is right, the two lie on one curve.
* Identity is never carried by colour alone: every series is direct-labelled as
  well as legended, and the two measured series differ in marker shape too.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Slots 1-3 of the reference categorical palette, which is the documented
# subset that validates on all pairs (scatter forms put every pair on screen at
# once, so the eight-slot order is not usable here).
C_K, C_SIGMA, C_BASE = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
SURFACE = "#fcfcfb"

TITLES = {
    "hamming": "Hamming distortion",
    "count_mse": "Squared error on the spike count",
    "van_rossum": "van Rossum distortion",
}
YLAB = {
    "hamming": "P(bit error) per bin",
    "count_mse": "mean sq. count error per neuron",
    "van_rossum": "mean sq. PSP error per bin",
}


def style(ax, xlabel, ylabel, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    # 26 points of pad, not 14: at 14 the title's own box lands on top of the
    # subtitle drawn at 1.02 and both become unreadable.
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=26 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=9, va="bottom")


def rd_figure(dname, theory, rows, baselines, out_dir, log_y=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=SURFACE)

    th_r = np.asarray(theory[dname]["rate_bits_per_bin"])
    th_d = np.asarray(theory[dname]["distortion"])
    ax.plot(th_r, th_d, color=INK, lw=2.0, zorder=4, label="R(D), computed bound")
    ax.fill_between(th_r, th_d, th_d.min(), color=INK, alpha=0.05, lw=0, zorder=1)
    if dname == "van_rossum" and "van_rossum_binary" in theory:
        b = theory["van_rossum_binary"]
        ax.plot(b["rate_bits_per_bin"], b["distortion"], color=INK, lw=1.5,
                ls=(0, (5, 2)), zorder=4,
                label="R(D) if the output must be spikes")

    pts = [r for r in rows if r["distortion"] == dname]
    ksw = sorted([r for r in pts if r["sigma"] == 1.0], key=lambda r: r["rate_bits_per_bin"])
    ssw = sorted([r for r in pts if r["sigma"] != 1.0], key=lambda r: r["rate_bits_per_bin"])
    for group, colour, marker, label in (
            (ksw, C_K, "o", "trained, rate from code size K"),
            (ssw, C_SIGMA, "s", "trained, rate from jitter $\\sigma$")):
        if not group:
            continue
        x = [r["rate_bits_per_bin"] for r in group]
        y = [r[dname] for r in group]
        ax.plot(x, y, color=colour, lw=2.0, marker=marker, ms=8, mew=2.0,
                mfc=SURFACE, mec=colour, zorder=5, label=label)

    zr = [b for b in baselines if b["name"] == "zero_rate"]
    if zr:
        ax.axhline(zr[0][dname], color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=3)
        ax.text(ax.get_xlim()[1], zr[0][dname], " zero rate ", color=INK2,
                fontsize=9, va="bottom", ha="right")
    ac = sorted([b for b in baselines if b["name"] == "analog_count"],
                key=lambda b: b["rate_bits_per_bin"])
    if ac:
        x = [b["rate_bits_per_bin"] for b in ac]
        y = [b[dname] for b in ac]
        ax.plot(x, y, color=C_BASE, lw=2.0, ls=(0, (1, 2)), marker="^", ms=8,
                mew=2.0, mfc=SURFACE, mec=C_BASE, zorder=5,
                label="uncoded analog count")
        ax.annotate("uncoded analog count", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(6, 4), color=C_BASE, fontsize=9, fontweight="bold")
    oc = [b for b in baselines if b["name"] == "oracle_count"]
    if oc:
        ax.plot([oc[0]["rate_bits_per_bin"]], [oc[0][dname]], marker="*", ms=13,
                color=INK2, zorder=6, ls="none", label="count sent noiselessly")

    if ksw:
        ax.annotate("K = %d" % ksw[-1]["n_code"], (ksw[-1]["rate_bits_per_bin"],
                    ksw[-1][dname]), textcoords="offset points", xytext=(0, -16),
                    color=C_K, fontsize=9, fontweight="bold", ha="center")
        ax.annotate("K = %d" % ksw[0]["n_code"], (ksw[0]["rate_bits_per_bin"],
                    ksw[0][dname]), textcoords="offset points", xytext=(6, 6),
                    color=C_K, fontsize=9, fontweight="bold")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(left=-0.01)
    style(ax, "rate  (bits per source bin)", YLAB[dname], TITLES[dname],
          "nothing can lie below the black curve; the shaded region is unreachable")
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    p = out_dir / f"rate-distortion-{dname}.png"
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def capacity_figure(theory, out_dir):
    cap = theory["capacity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), facecolor=SURFACE)
    s = np.asarray(cap["sigma"])
    for ax, key, colour, title in (
            (axes[0], "ttfs", C_K, "TTFS: one spike time through jitter"),
            (axes[1], "multispike", C_SIGMA, "Multi-spike: a %d-slot word, spikes displaced"
             % theory["n_bins"])):
        y = np.asarray(cap[key])
        ax.plot(s, y, color=colour, lw=2.0, marker="o", ms=8, mew=2.0,
                mfc=SURFACE, mec=colour, zorder=5)
        if key == "ttfs":
            asym = np.asarray(cap["ttfs_highsnr_asymptote"])
            m = asym > 0
            ax.plot(s[m], asym[m], color=INK, lw=1.5, ls=(0, (4, 3)), zorder=4)
            ax.text(0.03, 0.06, "dashed: high-resolution asymptote,\n"
                    "$\\log_2(T/\\sigma) - \\frac{1}{2}\\log_2(2\\pi e)$",
                    transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
        else:
            ax.axhline(theory["n_bins"], color=INK, lw=1.5, ls=(0, (4, 3)), zorder=4)
            ax.set_ylim(top=theory["n_bins"] * 1.14)
            ax.text(0.03, 0.965, "dashed: T bits, the noiseless ceiling",
                    transform=ax.transAxes, color=INK2, fontsize=9, va="top")
        ax.set_xscale("log")
        # label the jitter values actually computed: the default log locator
        # shows only the single decade tick, which reads as an unlabelled axis
        ax.set_xticks(list(s))
        ax.set_xticklabels([f"{v:g}" for v in s], fontsize=9)
        ax.set_xticks([], minor=True)
        style(ax, "jitter $\\sigma$  (bins)", "capacity  (bits per channel use)", title)
    fig.suptitle("What one channel use is worth", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "capacity.png"
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def code_figure(ckpt, out_dir, n=4096):
    """What the temporal code looks like: sent times, and received times.

    The point of the figure is the overlap. If the sent latencies are spread
    across the window and the jitter blurs them into each other, that blur is
    the rate loss the capacity number is accounting for.
    """
    import torch
    from src.model import JSCCSystem, SystemConfig
    from src.source import PoissonSource
    from src.train import Config

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**blob["cfg"])
    model = JSCCSystem(cfg.system())
    model.load_state_dict(blob["state"])
    model.eval()
    src = PoissonSource(cfg.n_neurons, cfg.n_bins, cfg.rate, cfg.source, seed=999)
    with torch.no_grad():
        x = src.sample(n)
        code = model.compressor(x)
        rx = model.transmit(code)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    bins = np.linspace(-2, cfg.n_bins + 2, 70)
    ax.hist(code.flatten().numpy(), bins=bins, color=C_K, alpha=0.85, lw=0,
            label="sent spike time", zorder=4)
    ax.hist(rx.flatten().numpy(), bins=bins, histtype="step", color=INK,
            lw=2.0, label="received spike time", zorder=5)
    ax.axvspan(0, cfg.n_bins, color=MUTED, alpha=0.08, lw=0, zorder=1)
    ax.text(0.02, 0.95, "window [0, T]", transform=ax.transAxes, color=INK2,
            fontsize=9, va="top")
    style(ax, "spike time (bins)", "count",
          f"The temporal code, K = {cfg.n_code}, $\\sigma$ = {cfg.sigma:g} bins",
          "the encoder places its latencies inside the window; jitter moves them out of it")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    p = out_dir / "code-times.png"
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", type=Path, default=Path("runs/theory.json"))
    ap.add_argument("--sweep", type=Path, default=Path("runs/sweep/summary.json"))
    ap.add_argument("--baselines", type=Path, default=Path("runs/baselines.json"))
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--grid", default="primary")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--log-y", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    theory = json.loads(a.theory.read_text())
    made = [capacity_figure(theory, a.out_dir)]

    if a.sweep.exists():
        rows = [r for r in json.loads(a.sweep.read_text())
                if f"/{a.grid}/" in r["out"]]
        base = json.loads(a.baselines.read_text())["rows"] if a.baselines.exists() else []
        for dname in ("hamming", "count_mse", "van_rossum"):
            made.append(rd_figure(dname, theory, rows, base, a.out_dir,
                                  log_y=a.log_y))
    if a.ckpt:
        made.append(code_figure(a.ckpt, a.out_dir))
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
