#!/usr/bin/env python
"""Reference systems that are not learned, at rates that are also computed.

A trained point sitting between the bound and nothing is hard to read, so the
plot needs company. Three references, all evaluated on the same source and the
same distortion definitions as the trained models.

**zero_rate** -- the best system that receives nothing at all. Its distortion is
the D-axis intercept of R(D), and any trained system that fails to beat it has
learned nothing, whatever its loss curve looked like.

**analog_count** -- uncoded analog joint source-channel coding, and the obvious
thing to do here. Source neuron n's spike count is already a number in [0, T],
so send it *as a spike time*: one channel use per source neuron, no coding at
all, and decode with the exact MMSE estimator against the Binomial prior. This
is the reference the learned system has to beat to justify itself, and for
squared error on a matched Gaussian channel uncoded transmission is famously
hard to beat.

**oracle_count** -- the same count-only strategy with a *noiseless* channel, at
rate H(count)/T bits per bin. It separates two failures that look alike: how
much the jitter costs, versus how much is lost by summarising a spike train by
its count in the first place. For Hamming and van Rossum this ceiling is low no
matter how many bits are spent, which is the sharpest statement in the whole
experiment that the three distortions ask for different codes.

Each reference is scored under two decoder rules -- emit nothing, or spread the
estimated count evenly over the window -- and the better one per distortion is
kept, since a real receiver would pick the better rule too.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binom, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import distortions as dist
from src import theory as th
from src.source import PoissonSource

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("runs/baselines.json"))
ap.add_argument("--n-neurons", type=int, default=16)
ap.add_argument("--n-bins", type=int, default=12)
ap.add_argument("--rate", type=float, default=0.15)
ap.add_argument("--source", default="iid")
ap.add_argument("--tau", type=float, default=3.0)
ap.add_argument("--sigmas", default="0.25,0.5,1.0,2.0,4.0")
ap.add_argument("--batch", type=int, default=20000)
a = ap.parse_args()

T, N, p = a.n_bins, a.n_neurons, a.rate
src = PoissonSource(N, T, p, a.source, seed=777)
x = src.sample(a.batch)                                  # (B, T, N)
counts = x.sum(1)                                        # (B, N)
ds = {n: dist.build(n, a.tau, T) for n in dist.NAMES}


silence = torch.zeros_like(x)
const_p = torch.full_like(x, p)


def even_placement(c_hat: torch.Tensor) -> torch.Tensor:
    """Spread round(c_hat) spikes as evenly as the window allows, (B,N)->(B,T,N).

    The maximum-entropy guess given only a count: with no timing information the
    best a decoder can do is avoid clustering, since any clustering it invents is
    wrong in a specific place rather than diffusely.
    """
    B, Nn = c_hat.shape
    k = c_hat.round().clamp(0, T).long()
    idx = torch.arange(T).view(1, T, 1)
    # slot t carries a spike iff floor(t * k / T) increments there
    kk = k.unsqueeze(1)
    out = ((idx * kk) // T != ((idx + 1) * kk) // T).float()
    return out


def score(recon_soft, recon_hard):
    """Distortion of a reference under the best of its decoder rules.

    Three rules: the soft estimate, the even placement, and *silence*. Silence
    belongs in the set because a receiver that has been told something useless
    for the distortion at hand would ignore it, and leaving that option out
    would report a count-based scheme as worse than nothing under Hamming when
    the true statement is that the count buys nothing under Hamming.
    """
    res = {}
    for n, d in ds.items():
        cand = [d.measure(silence, x), d.measure(const_p, x)]
        for r in (recon_soft, recon_hard):
            if r is not None:
                cand.append(d.measure(r, x))
        res[n] = min(cand)
    return res


rows = []
zr = {}
for n, d in ds.items():
    zr[n] = min(d.measure(silence, x), d.measure(const_p, x))
rows.append({"name": "zero_rate", "rate_bits_per_bin": 0.0,
             "sigma": None, "channel_uses": 0, **zr})

# ---- oracle count (noiseless channel, count only) --------------------------
c_probs = binom.pmf(np.arange(T + 1), T, p)
h_count = float(-(c_probs * np.log2(c_probs + 1e-300)).sum())
soft = (counts / T).unsqueeze(1).expand(-1, T, -1)       # count spread as a rate
hard = even_placement(counts)
rows.append({"name": "oracle_count", "rate_bits_per_bin": h_count / T,
             "sigma": None, "channel_uses": N,
             **score(soft, hard)})

# ---- analog count through the jitter channel -------------------------------
cache = Path("runs/capacity-cache.json")
grid = np.arange(T + 1)
prior = binom.pmf(grid, T, p)
for sigma in [float(s) for s in a.sigmas.split(",")]:
    t_rx = counts + sigma * torch.randn(counts.shape)     # uncoded: time = count
    lik = norm.pdf((t_rx.numpy()[..., None] - grid[None, None, :]) / sigma)
    post = lik * prior[None, None, :]
    post /= post.sum(-1, keepdims=True)
    c_hat = torch.tensor((post * grid[None, None, :]).sum(-1), dtype=torch.float32)
    soft = (c_hat / T).clamp(0, 1).unsqueeze(1).expand(-1, T, -1)
    hard = even_placement(c_hat)
    capacity = th.channel_capacity("ttfs", T, sigma, cache)
    rows.append({"name": "analog_count", "rate_bits_per_bin": capacity / T,
                 "sigma": sigma, "channel_uses": N,
                 "count_mse_direct": float(((c_hat - counts) ** 2).mean()),
                 **score(soft, hard)})

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"config": vars(a) | {"out": str(a.out)},
                             "rows": rows}, indent=1, default=str))
hdr = f"{'baseline':16s} {'sigma':>6s} {'bits/bin':>9s} {'hamming':>9s} {'count_mse':>10s} {'van_rossum':>11s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['name']:16s} {str(r['sigma'] or '-'):>6s} {r['rate_bits_per_bin']:9.4f} "
          f"{r['hamming']:9.4f} {r['count_mse']:10.4f} {r['van_rossum']:11.5f}")
print(f"\nwrote {a.out}")
