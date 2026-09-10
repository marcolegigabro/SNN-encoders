"""The two timing channels, matched one-for-one to the capacities in theory.py.

The rule followed here is that the channel simulated in the forward pass and
the channel whose capacity sets the rate axis have to be the *same* channel. So
each function below has a named counterpart in `theory.py`, and the docstrings
say which.

Gaussian jitter is the right channel model for temporal coding: what a timing
channel corrupts is *when* a spike arrives, not whether it arrives, and
independent Gaussian displacement is the standard model for that (and, unlike
the queueing delay in ../rate_snn.py, it is memoryless, which is what makes its
capacity computable).
"""

from __future__ import annotations

import torch

from .snn import StraightThrough


def jitter_times(times: torch.Tensor, sigma: float,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """TTFS channel: t' = t + N(0, sigma^2), one use per code neuron.

    Capacity counterpart: `theory.ttfs_channel_capacity`. The input is confined
    to [0, t_win] by `snn.latency_readout` (the peak constraint the capacity is
    computed under); the output is deliberately *not* clipped, because clipping
    it would be extra receiver-side processing that the capacity calculation
    does not assume, and would make the measured system look better than the
    channel allows.

    Differentiable as it stands: additive noise independent of the input is
    already the reparameterisation trick, so the gradient of the loss reaches
    the encoder's spike times exactly, with no surrogate anywhere. That is the
    main practical argument for the TTFS bottleneck over the multi-spike one.
    """
    if sigma <= 0:
        return times
    noise = torch.randn(times.shape, generator=generator, device=times.device,
                        dtype=times.dtype)
    return times + sigma * noise


def jitter_rebin(spikes: torch.Tensor, sigma: float, hard: bool = True,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Multi-spike channel: every spike is displaced and re-binned.

    `spikes` is (B, T, K). Each spike in slot t is emitted at t + N(0, sigma^2)
    and rounded back to a slot; spikes leaving [0, T-1] are lost, and two spikes
    landing in one slot become one. Capacity counterpart:
    `theory.multispike_channel_matrix`, which builds exactly this kernel.

    Not differentiable -- it moves spikes between integer slots -- so the result
    is wrapped in a straight-through estimator: the forward pass carries the true
    delayed, collided, truncated train, the backward pass treats the channel as
    transparent. Same choice as `delay_channel` in ../rate_snn.py, and the reason
    the TTFS variant is the primary system.
    """
    B, T, K = spikes.shape
    hard_in = (spikes > 0.5).float() if hard else spikes
    slots = torch.arange(T, device=spikes.device, dtype=spikes.dtype).view(1, T, 1)
    noise = torch.randn(spikes.shape, generator=generator, device=spikes.device,
                        dtype=spikes.dtype)
    dest = torch.round(slots + sigma * noise).long()             # (B, T, K)
    keep = (dest >= 0) & (dest < T) & (hard_in > 0.5)

    out = torch.zeros_like(hard_in)
    dest = dest.clamp(0, T - 1)
    out.scatter_add_(1, dest, keep.to(out.dtype))
    out = out.clamp(max=1.0)                                     # collisions merge
    return StraightThrough.apply(spikes, out)
