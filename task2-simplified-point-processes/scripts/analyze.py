#!/usr/bin/env python
"""Stage 4: what did the learned latent encode?

For every beta run of the given sweeps (final checkpoint `last.pt`), on the first
N windows of the shared evaluation set:

* symbol usage per coordinate and its entropy;
* Spearman rank correlation and mutual information (plug-in, feature binned
  into quantiles) between every coordinate and a set of window features
  (`src/features.py`): event count, count per half, first/last/mean event time,
  longest gap, inter-event-interval CV;
* coordinate importance: the increase in D when one coordinate is replaced by
  its most frequent symbol (the receiver decodes the ablated code);
* latent traversals and reconstructions for a low, a mid and a high rate run.

Writes <sweep>/analysis.json and figures/interp-*.png. Runs on CPU by default so
it can share the machine with a GPU sweep.

    python scripts/analyze.py --sweep SNN=runs/sweep-snn-l8a16-lt5 --sweep GRU=runs/sweep-ann-l8a16-lt5
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bottleneck import symbol_entropy_bits
from src.decoder import to_events
from src.distortion import counting_l1
from src.features import FEATURES, LABELS, window_features
from src.model import Codec
from src.plotting import GRID, INK, INK2, MUTED, SLOTS, SURFACE, style
from src.sources import eval_set

ACTIVE_BITS = 0.05
# sequential blue ramp (reference palette steps 250 -> 650) and the diverging pair
BLUE_RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281")
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")


# ----------------------------------------------------------------------------- analysis

def load_codec(run_dir: Path, device: str):
    ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = Codec(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def encode_all(model, test, device, batch=1000):
    syms, curves = [], []
    for i in range(0, len(test), batch):
        z, curve, _, _ = model(test[i:i + batch].to(device))
        syms.append(z.argmax(-1).cpu())
        curves.append(curve.cpu())
    return torch.cat(syms), torch.cat(curves)


@torch.no_grad()
def distortion_of_symbols(model, sym, test, T, device, batch=1000):
    curves = torch.cat([model.decode_symbols(sym[i:i + batch].to(device)).cpu()
                        for i in range(0, len(sym), batch)])
    return counting_l1(test, to_events(curves, T)).mean().item()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def mutual_info_bits(sym: np.ndarray, feat: np.ndarray, n_bins: int = 8) -> float:
    ok = np.isfinite(feat)
    s, f = sym[ok], feat[ok]
    edges = np.unique(np.quantile(f, np.linspace(0, 1, n_bins + 1)[1:-1]))
    fb = np.searchsorted(edges, f, side="right")
    joint = np.zeros((s.max() + 1, fb.max() + 1))
    np.add.at(joint, (s, fb), 1.0)
    p = joint / joint.sum()
    outer = p.sum(1, keepdims=True) @ p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / outer[nz])).sum())


def analyze_run(run_dir, test, feats, device):
    model, cfg = load_codec(run_dir, device)
    last = json.loads((run_dir / "history.json").read_text())[-1]
    sym, curves = encode_all(model, test, device)
    rec = to_events(curves, cfg.T)
    D = counting_l1(test, rec).mean().item()
    s_np = sym.numpy()
    L, A = cfg.n_latents, cfg.alphabet

    ablation = []
    for i in range(L):
        s2 = sym.clone()
        s2[:, i] = torch.bincount(sym[:, i], minlength=A).argmax()
        ablation.append(distortion_of_symbols(model, s2, test, cfg.T, device) - D)

    return {
        "run": run_dir.name, "beta": cfg.beta, "encoder": cfg.encoder, "alphabet": A,
        "n_latents": L, "rate": last["rate_real"], "D": D,
        "symbol_entropy": symbol_entropy_bits(sym, A).tolist(),
        "spearman": [[spearman(s_np[:, i].astype(float), feats[f]) for f in FEATURES] for i in range(L)],
        "mutual_info_bits": [[mutual_info_bits(s_np[:, i], feats[f]) for f in FEATURES] for i in range(L)],
        "ablation_dD": ablation,
    }, sym, rec


def pick_operating_points(runs):
    """Indices of a low, a mid and a high rate run among the non-collapsed ones."""
    live = [i for i, r in enumerate(runs) if r["rate"] > 0.05]
    return sorted({live[0], live[len(live) // 2], live[-1]})


# ----------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _title(r):
    return f"β={r['beta']:g}: {r['rate']:.1f} bits, D={r['D']:.3f}"


def fig_correlations(label, runs, sel, out):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, axes = plt.subplots(1, len(sel), figsize=(4.1 * len(sel) + 1.2, 4.6), facecolor=SURFACE,
                             layout="constrained")
    axes = np.atleast_1d(axes)
    im = None
    for ax, idx in zip(axes, sel):
        r = runs[idx]
        M = np.array(r["spearman"], dtype=float)
        ent = np.array(r["symbol_entropy"])
        L = M.shape[0]
        im = ax.imshow(np.nan_to_num(M), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(L):
            if ent[i] <= ACTIVE_BITS:
                ax.add_patch(Rectangle((-0.5, i - 0.5), len(FEATURES), 1, facecolor=GRID, edgecolor="none"))
                ax.text(len(FEATURES) / 2 - 0.5, i, "unused (constant)", ha="center", va="center",
                        fontsize=7, color=MUTED)
                continue
            for j in range(len(FEATURES)):
                if np.isfinite(M[i, j]) and abs(M[i, j]) >= 0.5:
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(M[i, j]) > 0.75 else INK)
        ax.set_xticks(range(len(FEATURES)), [LABELS[f] for f in FEATURES], rotation=45, ha="right",
                      fontsize=8, color=INK2)
        ax.set_yticks(range(L), [f"z{i}" for i in range(L)], fontsize=8, color=INK2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(_title(r), loc="left", fontsize=10, color=INK)
    assert im is not None, "no operating point to draw"
    cb = fig.colorbar(im, ax=list(axes), shrink=0.8)
    cb.set_label("Spearman rank correlation, coordinate vs feature", color=INK2, fontsize=9)
    cb.ax.tick_params(labelsize=8, labelcolor=INK2, color=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle(f"{label} encoder: what each latent coordinate tracks (|ρ| ≥ 0.5 labelled)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _steps(times_row, count, T):
    t = times_row[:count].numpy()
    return np.r_[0.0, t, T], np.r_[0, np.arange(1, count + 1), count]


def fig_reconstructions(label, runs, sel, recs, test, color, out):
    plt = _plt()
    from matplotlib.lines import Line2D
    T = test.T
    wanted = (2, 5, 9)
    windows = [int(torch.nonzero(test.counts == n)[0]) for n in wanted if (test.counts == n).any()]
    fig, axes = plt.subplots(len(windows), len(sel), figsize=(3.8 * len(sel), 2.3 * len(windows)),
                             facecolor=SURFACE, squeeze=False, sharex=True)
    for col, idx in enumerate(sel):
        r, rec = runs[idx], recs[idx]
        for row, w in enumerate(windows):
            ax = axes[row, col]
            style(ax)
            x, y = _steps(test.times[w], int(test.counts[w]), T)
            xr, yr = _steps(rec.times[w], int(rec.counts[w]), T)
            ax.step(x, y, where="post", color=INK, linewidth=1.5)
            ax.step(xr, yr, where="post", color=color, linewidth=2)
            top = max(y.max(), yr.max()) + 1
            ax.vlines(test.times[w, :int(test.counts[w])], -0.9, -0.3, color=INK, linewidth=1)
            ax.vlines(rec.times[w, :int(rec.counts[w])], -1.7, -1.1, color=color, linewidth=1.5)
            ax.set_ylim(-2, top)
            ax.set_xlim(0, T)
            if row == 0:
                ax.set_title(_title(r), loc="left", fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(f"N(t), window with {int(test.counts[w])} events", color=INK2, fontsize=8)
            if row == len(windows) - 1:
                ax.set_xlabel("t", color=INK2, fontsize=9)
    handles = [Line2D([], [], color=INK, linewidth=1.5, label="original counting function and events"),
               Line2D([], [], color=color, linewidth=2, label=f"decoded ({label} encoder)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{label} encoder: original vs reconstructed windows at three operating points",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


@torch.no_grad()
def fig_traversal_usage(label, run_dir, r, sym, device, color, out):
    plt = _plt()
    from matplotlib.colors import ListedColormap
    model, cfg = load_codec(run_dir, device)
    A, L, T = cfg.alphabet, cfg.n_latents, cfg.T
    order = np.argsort(r["ablation_dD"])[::-1]
    top = [int(i) for i in order[:2] if r["symbol_entropy"][int(i)] > ACTIVE_BITS]
    reference = torch.from_numpy(np.median(sym.numpy(), axis=0).round().astype(np.int64))

    fig = plt.figure(figsize=(13, 4.4), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(1, 2 + len(top), width_ratios=[1.3, 1] + [1.1] * len(top))

    ax = fig.add_subplot(gs[0, 0])
    freq = torch.nn.functional.one_hot(sym, A).float().mean(0).numpy()   # (L, A)
    im = ax.imshow(freq, aspect="auto", cmap=ListedColormap(("#f0efec",) + BLUE_RAMP), vmin=0, vmax=freq.max())
    ax.set_yticks(range(L), [f"z{i} ({h:.1f} b)" for i, h in enumerate(r["symbol_entropy"])],
                  fontsize=8, color=INK2)
    ax.set_xlabel("symbol level", color=INK2, fontsize=9)
    ax.tick_params(length=0, labelsize=8, labelcolor=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Symbol usage (entropy per coordinate)", loc="left", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("frequency", color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7, labelcolor=INK2)
    cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(L)
    ax.barh(y, r["ablation_dD"], height=0.6, color=color)
    for yi, v in zip(y, r["ablation_dD"]):
        ax.text(v, yi, f" +{v:.3f}" if v >= 0 else f" {v:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [f"z{i}" for i in range(L)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("increase in D when fixed to its most frequent symbol", color=INK2, fontsize=8)
    ax.set_title("Coordinate importance", loc="left", fontsize=10, color=INK)

    grid_t = (np.arange(cfg.n_grid) + 0.5) * T / cfg.n_grid
    for k, i in enumerate(top):
        ax = fig.add_subplot(gs[0, 2 + k])
        style(ax)
        used = np.unique(sym[:, i].numpy())
        levels = used[np.linspace(0, len(used) - 1, min(7, len(used))).round().astype(int)]
        codes = reference.repeat(len(levels), 1)
        codes[:, i] = torch.from_numpy(levels)
        curves = model.decode_symbols(codes.to(device)).cpu().numpy()
        ramp = [BLUE_RAMP[j] for j in np.linspace(0, len(BLUE_RAMP) - 1, len(levels)).round().astype(int)]
        for lv, c, cv in zip(levels, ramp, curves):
            ax.plot(grid_t, cv, color=c, linewidth=2)
        for lv, cv in ((levels[0], curves[0]), (levels[-1], curves[-1])):
            ax.annotate(f"z{i}={lv}", (grid_t[-1], cv[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK2, va="center")
        ax.set_xlim(0, T * 1.18)
        ax.set_xlabel("t", color=INK2, fontsize=9)
        ax.set_ylabel("decoded N̂(t)", color=INK2, fontsize=9)
        ax.set_title(f"z{i} traversal, others at median", loc="left", fontsize=10, color=INK)
    fig.suptitle(f"{label} encoder, {_title(r)}", color=INK, fontsize=12, x=0.01, ha="left")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_features_vs_rate(results, out):
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), facecolor=SURFACE, sharey=True, sharex=True)
    for ax, (j, f) in zip(axes.ravel(), enumerate(FEATURES)):
        style(ax)
        for runs, color in zip(results.values(), SLOTS[1:]):
            live = [r for r in runs if r["rate"] > 0.05]
            best = []
            for r in live:
                vals = [abs(r["spearman"][i][j]) for i in range(r["n_latents"])
                        if r["symbol_entropy"][i] > ACTIVE_BITS and np.isfinite(r["spearman"][i][j])]
                best.append(max(vals) if vals else 0.0)
            ax.plot([r["rate"] for r in live], best, color=color, linewidth=2, marker="o", markersize=6,
                    markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_ylim(0, 1.02)
        ax.set_title(LABELS[f], loc="left", fontsize=10, color=INK)
    for ax in axes[1]:
        ax.set_xlabel("rate (bits per unit time)", color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("best |ρ| over active coordinates", color=INK2, fontsize=9)
    handles = [Line2D([], [], color=c, linewidth=2, marker="o", markeredgecolor=SURFACE, label=f"{label} encoder")
               for label, c in zip(results, SLOTS[1:])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("How strongly some latent coordinate tracks each window feature, across operating points",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="append", required=True, metavar="LABEL=DIR",
                   help="sweep directory with beta*/last.pt (repeatable, max 2)")
    p.add_argument("--n-windows", type=int, default=5000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    a = p.parse_args()
    if len(a.sweep) > 2:
        raise SystemExit("at most two sweeps per call (two categorical slots after the baseline)")
    a.fig_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for (label, _, d), color in zip((s.partition("=") for s in a.sweep), SLOTS[1:]):
        sweep_dir = Path(d) if Path(d).is_absolute() else ROOT / d
        run_dirs = [r for r in sweep_dir.glob("beta*") if (r / "last.pt").exists()]
        cfg0 = SimpleNamespace(**json.loads((run_dirs[0] / "config.json").read_text()))
        test = eval_set(cfg0.rate, 10_000, T=cfg0.T)[:a.n_windows]
        feats = {k: v.numpy().astype(float) for k, v in window_features(test).items()}

        runs, syms, recs = [], [], []
        for run_dir in run_dirs:
            r, sym, rec = analyze_run(run_dir, test, feats, a.device)
            runs.append(r)
            syms.append(sym)
            recs.append(rec)
            print(f"[{label}] {run_dir.name:>11}: rate {r['rate']:5.2f}  D {r['D']:.4f}  "
                  f"active {sum(h > ACTIVE_BITS for h in r['symbol_entropy'])}/{r['n_latents']}", flush=True)
        order = np.argsort([r["rate"] for r in runs])
        runs, syms, recs = [runs[i] for i in order], [syms[i] for i in order], [recs[i] for i in order]
        run_dirs = [run_dirs[i] for i in order]
        (sweep_dir / "analysis.json").write_text(json.dumps(runs, indent=2))
        results[label] = runs

        tag = f"{label.lower()}-{sweep_dir.name}"
        sel = pick_operating_points(runs)
        fig_correlations(label, runs, sel, a.fig_dir / f"interp-correlations-{tag}.png")
        fig_reconstructions(label, runs, sel, recs, test, color, a.fig_dir / f"interp-reconstructions-{tag}.png")
        mid = sel[len(sel) // 2]
        fig_traversal_usage(label, run_dirs[mid], runs[mid], syms[mid], a.device, color,
                            a.fig_dir / f"interp-usage-traversal-{tag}.png")

        print(f"\n[{label}] strongest coordinate per feature (Spearman), by run:")
        for r in runs:
            if r["rate"] <= 0.05:
                continue
            parts = []
            for j, f in enumerate(FEATURES):
                col = [(abs(r["spearman"][i][j]), i) for i in range(r["n_latents"])
                       if r["symbol_entropy"][i] > ACTIVE_BITS and np.isfinite(r["spearman"][i][j])]
                if col:
                    _, i = max(col)
                    parts.append(f"{f}: z{i} {r['spearman'][i][j]:+.2f}")
            print(f"  {r['run']:>11} ({r['rate']:5.2f} bits): " + " | ".join(parts))

    fig_features_vs_rate(results, a.fig_dir / "interp-features-vs-rate.png")
    print(f"\nwrote analysis.json per sweep and figures to {a.fig_dir}")


if __name__ == "__main__":
    main()
