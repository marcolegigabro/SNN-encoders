"""Decoder: latent embedding -> counting curve N_hat(t) -> point process X_hat.

The decoder outputs the counting function directly, as a total count times a
cumulative distribution over the K cells of the time grid:

    N_hat(t_k) = total * sum_{j <= k} softmax(logits)_j.

This is non-decreasing by construction, trains well (every logit gets a useful
gradient, unlike per-cell softplus increments of size ~ lambda T / K), and is
what the training loss `distortion.counting_l1_grid` compares against.

`to_events` turns the soft curve into an actual point process by placing the
k-th event where the curve first reaches k - 1/2, so evaluation is always on
real decoded points with the exact continuous metric.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sources import Events


class CountingDecoder(nn.Module):
    def __init__(self, in_dim: int, n_grid: int = 1000, hidden: int = 256,
                 init_total: float = 5.0):
        super().__init__()
        self.n_grid = n_grid
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.shape_head = nn.Linear(hidden, n_grid)
        self.total_head = nn.Linear(hidden, 1)
        # near-zero rather than zero weights: the untrained decoder predicts a
        # uniform curve with the mean count, but gradients still reach the code
        nn.init.normal_(self.shape_head.weight, std=1e-2)
        nn.init.zeros_(self.shape_head.bias)
        nn.init.normal_(self.total_head.weight, std=1e-2)
        # softplus^{-1}(init_total), so the untrained decoder already predicts the mean count
        nn.init.constant_(self.total_head.bias, float(torch.log(torch.expm1(torch.tensor(init_total)))))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) -> (B, n_grid) soft counting curve."""
        x = self.body(h)
        total = F.softplus(self.total_head(x))
        return total * F.softmax(self.shape_head(x), dim=-1).cumsum(-1)


def to_events(curve: torch.Tensor, T: float) -> Events:
    """Events at the first grid cell where the curve reaches k - 1/2, k = 1, 2, ..."""
    B, K = curve.shape
    n_max = max(int(torch.floor(curve[:, -1].max() + 0.5)), 1)
    levels = (torch.arange(1, n_max + 1, device=curve.device, dtype=curve.dtype) - 0.5)
    idx = torch.searchsorted(curve.contiguous(), levels.expand(B, n_max).contiguous())
    valid = idx < K
    times = torch.where(valid, (idx.to(curve.dtype) + 0.5) * T / K,
                        torch.full_like(levels.expand(B, n_max), T))
    return Events(times, valid.sum(1), T)
