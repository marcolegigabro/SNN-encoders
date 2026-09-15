#!/usr/bin/env python
"""One-page overview of task-2-simplified: the problem, Rubin's baseline, the codec
architecture in detail, training, results and interpretability. Data panels are drawn from the saved runs.

Writes figures/overview.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import baseline, theory
from src.distortion import counting_l1
from src.features import FEATURES, LABELS
from src.plotting import INK, INK2, MUTED, SLOTS, SURFACE, style
from src.sources import eval_set
from src.train import Config, rate_weight

BOX_FACE = "#f5f4f0"
KIND = {"learned": SLOTS[0], "estimated": SLOTS[1], "fixed": MUTED}
RUNS = ROOT / "runs"


# ----------------------------------------------------------------------------- helpers

def box(ax, x, y, w, h, title, body, kind, title_size=10.5):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=BOX_FACE, edgecolor=KIND[kind], linewidth=2))
    ax.text(x + 0.9, y + h - 0.9, title, ha="left", va="top", fontsize=title_size,
            fontweight="bold", color=INK)
    if body:
        ax.text(x + 0.9, y + h - 3.0, body, ha="left", va="top", fontsize=8.6, color=INK2,
                linespacing=1.45)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, shrinkA=0, shrinkB=0))


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=12.5, color=INK, fontweight="bold", pad=10)


def note(ax, x, y, text, ha="left", va="top"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.8, color=INK2,
            linespacing=1.45, bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE,
                                        edgecolor="#e1e0d9"))


def load_baseline(key="5"):
    res = json.loads((RUNS / "baseline" / "baseline.json").read_text())[key]
    pts = sorted((p for p in res["points"] if p["placement"] == "median"),
                 key=lambda p: p["real_bits_per_time"])
    R = np.array([0.0] + [p["real_bits_per_time"] for p in pts])
    D = np.array([res["zero_rate_D"]] + [p["D"] for p in pts])
    return res, R, D


# ----------------------------------------------------------------------------- panels

def draw_problem(ax):
    style(ax)
    panel_title(ax, "1. Problem and distortion")
    test = eval_set(5.0, 10_000)
    w = int(torch.nonzero(test.counts == 5)[0])
    ev = test[w:w + 1]
    rec = baseline.decode(baseline.encode(ev, 2), 1.0, "median")
    D = counting_l1(ev, rec).item()
    t_true = ev.times[0, :int(ev.counts[0])].numpy()
    t_rec = rec.times[0, :int(rec.counts[0])].numpy()
    g = np.linspace(0, 1, 2001)
    N = np.searchsorted(t_true, g, side="right")
    Nh = np.searchsorted(t_rec, g, side="right")
    ax.fill_between(g, N, Nh, color=MUTED, alpha=0.3, linewidth=0, step="post")
    ax.step(g, N, where="post", color=INK, linewidth=1.8)
    ax.step(g, Nh, where="post", color=SLOTS[0], linewidth=2)
    ax.vlines(t_true, -0.9, -0.35, color=INK, linewidth=1.2)
    ax.vlines(t_rec, -1.6, -1.05, color=SLOTS[0], linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.8, 6.2)
    ax.set_xlabel("time t (ticks: individual events)", color=INK2, fontsize=9.5)
    ax.set_ylabel("events so far, N(t)", color=INK2, fontsize=9.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.8, label="original N(t)"),
                       Line2D([], [], color=SLOTS[0], lw=2, label="reconstruction N̂(t) (baseline, 2 bins)"),
                       Patch(facecolor=MUTED, alpha=0.3, label=f"area between them = D ({D:.3f} here)")],
              loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    note(ax, 0.98, 0.25,
         "D = (1/T) ∫ |N(t) − N̂(t)| dt   (Rubin 1974)\n"
         "event late or early: costs the time shift\n"
         "event missing or extra: costs until the end T\n"
         "source: Poisson, λT = 5 (about 5 events per window)\n"
         "rate R: bits per unit time",
         ha="right", va="bottom")


def draw_baseline(ax):
    style(ax)
    panel_title(ax, "2. Theory and baseline (λT = 5)")
    res, R, D = load_baseline()
    Dg = np.logspace(np.log10(0.003), np.log10(1.6), 300)
    ax.fill_between(Dg, theory.rubin_lower_bound(Dg, 5.0), theory.rubin_upper_bound(Dg, 5.0),
                    color=MUTED, alpha=0.2, linewidth=0)
    ax.plot(Dg, theory.rubin_upper_bound(Dg, 5.0), color=INK2, lw=1)
    ax.plot(Dg, theory.rubin_lower_bound(Dg, 5.0), color=INK2, lw=1)
    ax.plot(D, R, color=SLOTS[0], lw=2, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5)
    ax.plot(res["zero_rate_D"], 0, marker="D", markersize=7, color=INK, markeredgecolor=SURFACE,
            clip_on=False, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 1.6)
    ax.set_ylim(0, 38)
    ax.set_xlabel("distortion D", color=INK2, fontsize=9.5)
    ax.set_ylabel("rate (bits per unit time)", color=INK2, fontsize=9.5)
    ax.annotate("Rubin bounds on R(D), T → ∞\n(no codec can go below the band)", (0.0042, 17),
                color=INK2, fontsize=9, ha="left")
    ax.annotate("binned-count codec", (0.16, 15), color=INK2, fontsize=9)
    note(ax, 0.98, 0.97,
         "Rubin's binned-count codec\n"
         "1. count the events in n equal bins\n"
         "2. entropy code the counts (Poisson model)\n"
         "3. place each bin's events at its midpoint\n"
         "    or at order-statistic medians (better)\n"
         "n = 1, 2, 4, …, 256 traces the curve;\n"
         "matches Rubin's Theorem 5 exactly\n"
         "black diamond: best fixed reconstruction (0 bits)",
         ha="right", va="top")


def draw_architecture(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 41)
    ax.axis("off")
    panel_title(ax, "3. Codec architecture (one window: left to right on top, then back along the bottom)")

    # sender side
    box(ax, 0, 29, 10.5, 9, "Input window",
        "Poisson process\nλT = 5, T = 1\n~5 event times", "fixed")
    box(ax, 12.5, 29, 11.5, 9, "Binning",
        "200 equal bins\n(width 0.005)\n→ counts x₁ … x₂₀₀", "fixed")

    box(ax, 26, 18.5, 27.5, 21.5, "Encoder (compared: SNN or GRU)", "", "learned")
    box(ax, 27, 27.4, 25.5, 9.8, "SNN encoder, 9.3k parameters",
        "LIF layer 1: 64 neurons, driven by events (no bias)\n"
        "LIF layer 2: 64 neurons, recurrent\n"
        "   leak 0.97 at init, recurrent gradient × 0.5\n"
        "non-spiking leaky readout integrates the spikes\n"
        "BatchNorm → Linear → y ∈ ℝ⁸", "learned", title_size=9.8)
    box(ax, 27, 19.5, 25.5, 6.8, "GRU encoder, 9.4k parameters (matched)",
        "GRU with 56 units reads the same 200 bins\n"
        "final state → BatchNorm → Linear → y ∈ ℝ⁸", "learned", title_size=9.8)

    box(ax, 55.5, 28, 17.5, 10, "Quantizer (× 8)",
        "v = (A − 1) · sigmoid(y)\n"
        "z = round(v) ∈ {0, …, A − 1}\n"
        "A = 16 or 64 (8×16, 8×64)\n"
        "ordinal levels, straight-through", "fixed")
    box(ax, 75.5, 28, 24.5, 10, "Entropy coding",
        "prior: 8 × A frequency table,\n"
        "   moving average of symbol usage\n"
        "cost of a code: −log₂ p(z) bits\n"
        "range coder → bitstream (lossless)", "estimated")

    for x0, x1 in ((10.5, 12.5), (24, 26), (53.5, 55.5), (73, 75.5)):
        arrow(ax, x0, 33.5, x1, 33.5)
    arrow(ax, 87.75, 28, 87.75, 11)
    ax.text(86.8, 19.5, "receiver decodes\nthe 8 symbols z", ha="right", va="center", fontsize=8.8,
            color=INK2, linespacing=1.4)

    # receiver side
    box(ax, 75.5, 1, 24.5, 10, "Level embeddings",
        "learned table 8 × A × 32:\n"
        "one vector per (coordinate, level)\n"
        "lookup + concatenate → 256 numbers", "learned")
    box(ax, 58, 1, 15.5, 10, "MLP decoder",
        "256 → 1024 → 1024\n"
        "GELU activations", "learned")
    box(ax, 38.5, 1, 17.5, 10, "Two heads",
        "total count c = softplus(a) > 0\n"
        "shape p = softmax(1000 scores)\n"
        "   over 1000 time cells", "learned")
    box(ax, 19, 1, 17.5, 10, "Counting curve",
        "N̂(tₖ) = c · (p₁ + … + pₖ)\n"
        "starts at 0, ends at c,\n"
        "non-decreasing by construction", "fixed")
    box(ax, 0, 1, 17, 10, "Decoded events",
        "k-th event where N̂\n"
        "first reaches k − ½\n"
        "→ reconstructed point process", "fixed")
    for x0, x1 in ((75.5, 73.5), (58, 56), (38.5, 36.5), (19, 17)):
        arrow(ax, x0, 6, x1, 6)

    ax.text(0.5, 26.5,
            "Training signal\n"
            "loss = w_rate · R + β · D\n"
            "R: −log₂ p(z) under the prior\n"
            "D: mean |N − N̂| on 1000 time points\n"
            "gradients cross the rounding\n"
            "(straight-through)",
            ha="left", va="top", fontsize=9, color=INK2, linespacing=1.45)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BOX_FACE, edgecolor=KIND[k], linewidth=2, label=lab)
               for k, lab in (("learned", "learned by gradient"),
                              ("estimated", "estimated from symbol counts"),
                              ("fixed", "fixed operation"))]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.645, 0.48), frameon=False,
              fontsize=9, labelcolor=INK2, title="box outline", title_fontsize=9)


def draw_training(ax):
    style(ax)
    panel_title(ax, "4. Training")
    cfg = Config()
    steps = np.arange(0, cfg.steps + 1, 50)
    ax.plot(steps, [rate_weight(int(s), cfg) for s in steps], color=SLOTS[0], lw=2)
    ax.set_xlim(0, cfg.steps)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("training step", color=INK2, fontsize=9.5)
    ax.set_ylabel("rate weight w_rate", color=INK2, fontsize=9.5)
    for x, y, t in ((2000, 0.07, "distortion only"), (6000, 0.55, "ramp"),
                    (15000, 1.05, "full objective R + βD")):
        ax.text(x, y, t, color=INK2, fontsize=9, ha="center")
    note(ax, 0.98, 0.05,
         "loss = w_rate · R + β · D, one run per β = one curve point\n"
         "each step: 512 fresh Poisson windows\n"
         "AdamW, learning rate up to 2e-3 (one-cycle), 20,000 steps\n"
         "gradient clipping: encoder and the rest separately\n"
         "prior: moving average of symbol usage (not a gradient)\n"
         "SNN: surrogate spike gradient, torch.compile",
         ha="right", va="bottom")


def draw_results(ax):
    style(ax)
    panel_title(ax, "5. Results vs the baseline (first round, 8×16 latent)")
    _, R_base, D_base = load_baseline()
    from matplotlib.lines import Line2D
    handles = []
    for label, d, color in (("SNN", "sweep-snn-l8a16-lt5", SLOTS[1]),
                            ("GRU", "sweep-ann-l8a16-lt5", SLOTS[2])):
        rows = sorted(json.loads((RUNS / d / "summary.json").read_text()), key=lambda s: s["rate_real"])
        R = np.array([s["rate_real"] for s in rows])
        D = np.array([s["D"] for s in rows])
        keep = R > 0.05
        ratio = D[keep] / np.interp(R[keep], R_base, D_base)
        ax.plot(R[keep], ratio, color=color, lw=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(label, (R[keep][-1], ratio[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=9.5, va="center")
        handles.append(Line2D([], [], color=color, lw=2, marker="o", markeredgecolor=SURFACE,
                              label=f"{label} encoder"))
    ax.axhline(1.0, color=INK2, lw=1)
    ax.text(1.0, 1.04, "baseline", color=INK2, fontsize=9, va="bottom")
    ax.set_yscale("log")
    ticks = [0.5, 1, 2, 4, 8, 16]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0, 34)
    ax.set_xlabel("rate (bits per unit time)", color=INK2, fontsize=9.5)
    ax.set_ylabel("D codec / D baseline at the same rate", color=INK2, fontsize=9.5)
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    note(ax, 0.02, 0.97,
         "below 1 = the learned codec wins\n"
         "wins below ~6 bits (up to ~30% lower D),\n"
         "loses 7 to 11× near 29 bits\n\n"
         "re-run with all fixes (GRU):\n"
         "8×16 best D 0.086 at 26 bits (4.5×)\n"
         "8×64 best D 0.058 at 33 bits (7.7×)\n"
         "open issue: low-rate points collapse",
         ha="left", va="top")


def draw_interpretability(ax):
    style(ax)
    ax.grid(axis="y", visible=False)
    panel_title(ax, "6. What the latent encodes (SNN 8×16, β = 200, 17 bits)")
    runs = json.loads((RUNS / "sweep-snn-l8a16-lt5" / "analysis.json").read_text())
    r = next(x for x in runs if x["run"] == "beta200")
    best = []
    for j, f in enumerate(FEATURES):
        vals = [abs(r["spearman"][i][j]) for i in range(r["n_latents"])
                if r["symbol_entropy"][i] > 0.05 and np.isfinite(r["spearman"][i][j])]
        best.append((max(vals) if vals else 0.0, LABELS[f]))
    best.sort()
    y = np.arange(len(best))
    ax.barh(y, [v for v, _ in best], height=0.6, color=SLOTS[0])
    for yi, (v, _) in zip(y, best):
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [lab for _, lab in best], fontsize=9.5)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("strongest |Spearman ρ| with any active latent coordinate", color=INK2, fontsize=9.5)
    note(ax, 0.98, 0.03,
         "roles learned as the rate grows:\n"
         "1. event count, early events first\n"
         "2. time of the first event\n"
         "3. the rest; late events and\n"
         "    burstiness stay weakly encoded",
         ha="right", va="bottom")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 27), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.3, 1, 0.8])
    draw_problem(fig.add_subplot(gs[0, 0]))
    draw_baseline(fig.add_subplot(gs[0, 1]))
    draw_architecture(fig.add_subplot(gs[1, :]))
    draw_training(fig.add_subplot(gs[2, 0]))
    draw_results(fig.add_subplot(gs[2, 1]))
    draw_interpretability(fig.add_subplot(gs[3, :]))
    fig.suptitle("task-2-simplified: learned lossy compression of a Poisson point process with an SNN encoder",
                 color=INK, fontsize=16, x=0.01, ha="left", fontweight="bold")
    out = ROOT / "figures" / "overview.png"
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
