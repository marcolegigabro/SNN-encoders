"""Rubin's binned-count codec (Rubin 1974, Section VII): the non-neural baseline.

Encoder: split [0, T] into n_bins bins of width dT and send the event count of
every bin, entropy-coded with a Poisson(lambda dT) model.

Decoder, two placements of the n events received for a bin:

* `midpoint`  all n at the bin centre (Theorem 5). D = lambda dT / 4 exactly.
* `median`    the i-th at the median of the i-th order statistic of n uniforms
              on the bin, i.e. start + dT * median(Beta(i, n - i + 1))
              (Theorem 4, eq. 44). Optimal for this encoder; only differs from
              `midpoint` when bins are wide enough to hold several events.

The count of each window is transmitted exactly, so decoded realizations always
have the true number of events.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import betaincinv

from .distortion import counting_l1
from .sources import Events
from .theory import LN2

_MEDIAN_NMAX = 128


def _median_table(nmax: int) -> torch.Tensor:
    """tab[n, j] = median of the (j+1)-th order statistic of n uniforms on [0, 1]."""
    tab = np.zeros((nmax + 1, nmax))
    for n in range(1, nmax + 1):
        j = np.arange(n)
        tab[n, :n] = betaincinv(j + 1, n - j, 0.5)
    return torch.from_numpy(tab)


_MEDIANS = _median_table(_MEDIAN_NMAX)


def encode(ev: Events, n_bins: int) -> torch.Tensor:
    """(B, n_bins) long: the symbols the baseline transmits."""
    return ev.binned(n_bins).round().long()


def decode(bin_counts: torch.Tensor, T: float, placement: str = "midpoint") -> Events:
    B, n_bins = bin_counts.shape
    dT = T / n_bins
    counts = bin_counts.sum(1)
    L = max(int(counts.max()), 1)
    rank = torch.arange(L, device=bin_counts.device).expand(B, L).contiguous()

    csum = bin_counts.cumsum(1)
    k = torch.searchsorted(csum, rank, right=True).clamp(max=n_bins - 1)  # bin of each event
    before = torch.cat([torch.zeros_like(csum[:, :1]), csum[:, :-1]], 1).gather(1, k)
    j = (rank - before).clamp(min=0)          # rank of the event inside its bin
    n_k = bin_counts.gather(1, k)             # events in that bin

    if placement == "midpoint":
        frac = torch.full((B, L), 0.5, dtype=torch.float64)
    elif placement == "median":
        in_table = n_k <= _MEDIAN_NMAX
        lookup = _MEDIANS[n_k.clamp(max=_MEDIAN_NMAX), j.clamp(max=_MEDIAN_NMAX - 1)]
        frac = torch.where(in_table, lookup, (j + 1) / (n_k + 1.0))  # eq. 48 beyond the table
    else:
        raise ValueError(f"unknown placement {placement!r}")

    times = ((k.double() + frac) * dT).float()
    valid = rank < counts[:, None]
    times = torch.where(valid, times, torch.full_like(times, T))
    return Events(times, counts, T)


def ideal_bits(bin_counts: torch.Tensor, rate: float, T: float) -> torch.Tensor:
    """-log2 of the Poisson(rate * dT) probability of every bin, summed -> (B,)."""
    mean = rate * T / bin_counts.shape[1]
    n = bin_counts.double()
    log_p = n * math.log(mean) - mean - torch.lgamma(n + 1)
    return (-log_p.sum(1) / LN2).float()


def zero_rate_reconstruction(train: Events) -> Events:
    """The best fixed reconstruction under `counting_l1`, fitted on training windows.

    With zero bits the decoder must output one realization for every input.
    Since D is a sum of |W_i - W_hat_i| over padded positions, the optimum sets
    W_hat_i to the median of the (padded) i-th event time.
    """
    med = train.times.median(0).values
    return Events(med[None], (med < train.T).sum()[None], train.T)


def evaluate(test: Events, rate: float, n_bins: int, placement: str) -> dict:
    sym = encode(test, n_bins)
    rec = decode(sym, test.T, placement)
    d = counting_l1(test, rec)
    bits = ideal_bits(sym, rate, test.T)
    n = len(test)
    return {
        "n_bins": n_bins,
        "placement": placement,
        "D": d.mean().item(),
        "D_se": (d.std() / math.sqrt(n)).item(),
        "bits_per_time": bits.mean().item() / test.T,
        "bits_per_time_se": (bits.std() / math.sqrt(n)).item() / test.T,
    }
