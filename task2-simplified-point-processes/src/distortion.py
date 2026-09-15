"""Distortion measures between point-process realizations.

The headline measure is Rubin's (1974, eq. 28) magnitude error on the
counting function,

    D(X, X_hat) = (1/T) * integral_0^T |N(t) - N_hat(t)| dt.

By Rubin's Proposition 2 it equals (1/T) * sum_i |W_i - W_hat_i|, pairing the
i-th event of each realization and extending the shorter one with events at T.
The identity is easy to see: at any t both indicator sequences 1{W_i <= t} are
non-increasing in i, so their differences all share one sign and the absolute
value of the sum is the sum of absolute values; integrating |1{W_i <= t} -
1{W_hat_i <= t}| over [0, T] gives |W_i - W_hat_i|. With padding at T this
makes `counting_l1` exact, batched, and differentiable in the event times.

Secondary measures, reported but never trained on:

* van Rossum distance, the standard spike-train metric (exponential kernel).
* Victor-Purpura edit distance (insert/delete cost 1, shift cost q * |dt|).
* absolute count error.
"""

from __future__ import annotations

import torch

from .sources import Events


def _aligned(a: Events, b: Events) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(a.times.shape[1], b.times.shape[1])
    return a.padded_to(L).times, b.padded_to(L).times


def counting_l1(a: Events, b: Events) -> torch.Tensor:
    """Exact (1/T) * integral |N_a - N_b| dt, per realization -> (B,)."""
    ta, tb = _aligned(a, b)
    return (ta - tb).abs().sum(1) / a.T


def counting_curve(ev: Events, n_grid: int) -> torch.Tensor:
    """N(t) at the midpoints of n_grid equal cells of [0, T] -> (B, n_grid)."""
    grid = (torch.arange(n_grid, device=ev.times.device, dtype=ev.times.dtype) + 0.5)
    grid = (grid * ev.T / n_grid).expand(ev.times.shape[0], n_grid).contiguous()
    return torch.searchsorted(ev.times.contiguous(), grid, right=True).to(ev.times.dtype)


def counting_l1_grid(curve: torch.Tensor, curve_hat: torch.Tensor) -> torch.Tensor:
    """Midpoint-rule version of `counting_l1` on counting curves -> (B,).

    This is the training loss: `curve_hat` may be a soft, real-valued curve
    produced by the decoder. For integer curves from real events it converges
    to `counting_l1` as the grid is refined.
    """
    return (curve_hat - curve).abs().mean(1)


def van_rossum(a: Events, b: Events, tau: float) -> torch.Tensor:
    """van Rossum distance with kernel exp(-t/tau), closed form -> (B,).

    D^2 = 1/2 [sum_ij k(a_i - a_j) + sum_ij k(b_i - b_j) - 2 sum_ij k(a_i - b_j)],
    k(u) = exp(-|u| / tau). The filtered trains are integrated over the whole
    real line, so kernel tails past T are included.
    """
    def cross(x, mx, y, my):
        k = torch.exp(-(x[:, :, None] - y[:, None, :]).abs() / tau)
        return (k * mx[:, :, None] * my[:, None, :]).sum((1, 2))

    ma, mb = a.mask.to(a.times.dtype), b.mask.to(b.times.dtype)
    d2 = 0.5 * (cross(a.times, ma, a.times, ma) + cross(b.times, mb, b.times, mb)
                - 2 * cross(a.times, ma, b.times, mb))
    return d2.clamp(min=0).sqrt()


@torch.no_grad()
def victor_purpura(a: Events, b: Events, q: float) -> torch.Tensor:
    """Victor-Purpura spike-time distance, dynamic program batched over rows -> (B,).

    G[i][j] = min(G[i-1][j] + 1, G[i][j-1] + 1, G[i-1][j-1] + q |a_i - b_j|).
    Prefix costs never depend on later events, so the padded tail is harmless:
    the answer for each realization is read at G[n_a][n_b].
    """
    x, y = a.times.double(), b.times.double()
    B, La = x.shape
    Lb = y.shape[1]
    prev = torch.arange(Lb + 1, dtype=x.dtype).expand(B, Lb + 1).clone()
    out = prev.gather(1, b.counts[:, None]).squeeze(1)  # n_a = 0 case
    for i in range(1, La + 1):
        cur = torch.empty_like(prev)
        cur[:, 0] = i
        shift = q * (x[:, i - 1, None] - y).abs()
        for j in range(1, Lb + 1):
            cur[:, j] = torch.minimum(torch.minimum(prev[:, j], cur[:, j - 1]) + 1,
                                      prev[:, j - 1] + shift[:, j - 1])
        out = torch.where(a.counts == i, cur.gather(1, b.counts[:, None]).squeeze(1), out)
        prev = cur
    return out.to(a.times.dtype)


def count_error(a: Events, b: Events) -> torch.Tensor:
    return (a.counts - b.counts).abs().to(a.times.dtype)
