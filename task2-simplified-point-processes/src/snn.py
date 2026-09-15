"""Spiking primitives for a 1-D input stream, stepped one time bin at a time.

Copied from `task2-compression-to-m-bits/src/snn.py` (the repo keeps each task
self-contained) and trimmed to what a single-channel point-process input needs:
no convolutions and no temporal batch norm. The neuron model is unchanged,

    u_t = beta * u_{t-1} * (1 - s_{t-1}) + I_t,     s_t = H(u_t - theta),

with an arctan surrogate gradient and a detached reset (Neftci, Mostafa &
Zenke 2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ATanSpike(torch.autograd.Function):
    """Heaviside forward, arctan surrogate backward (bell of width ~1/alpha)."""

    @staticmethod
    def forward(ctx, u, alpha):
        ctx.save_for_backward(u)
        ctx.alpha = alpha
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        a = ctx.alpha
        sg = a / (2 * (1 + (torch.pi / 2 * a * u) ** 2))
        return grad_out * sg, None


def spike(u, alpha: float = 2.0):
    return ATanSpike.apply(u, alpha)


class LIFState:
    """Membrane potential + last spike of one LIF layer."""

    __slots__ = ("u", "s")

    def __init__(self, u=None, s=None):
        self.u, self.s = u, s


class LIFCell(nn.Module):
    """Leaky integrate-and-fire with learnable per-neuron leak and threshold."""

    def __init__(self, n: int, beta: float = 0.9, theta: float = 1.0, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta_logit = nn.Parameter(torch.logit(torch.full((n,), beta)))
        self.theta = nn.Parameter(torch.full((n,), theta))

    def forward(self, current, state: LIFState) -> tuple[torch.Tensor, LIFState]:
        beta = torch.sigmoid(self.beta_logit)
        if state.u is None:
            u, s = torch.zeros_like(current), torch.zeros_like(current)
        else:
            u, s = state.u, state.s
        u = beta * u * (1.0 - s.detach()) + current
        s = spike(u - self.theta, self.alpha)
        return s, LIFState(u, s)


class LIFReadout(nn.Module):
    """Non-firing integrator with a learnable leak per output channel.

    Per-channel leaks matter for timing. A channel with beta close to 1 sums its
    input over the whole window, so a hidden neuron that turns on at an event
    and stays on contributes in proportion to T - t_event, which is exactly the
    area under that event's step in the counting function. Smaller betas
    forget early activity and weight the end of the window instead.
    """

    def __init__(self, n: int, beta: float = 0.99):
        super().__init__()
        self.beta_logit = nn.Parameter(torch.logit(torch.full((n,), beta)))

    def forward(self, current, u):
        beta = torch.sigmoid(self.beta_logit)
        return current if u is None else beta * u + current


class SpikingLinear(nn.Module):
    """Linear synapse -> LIF, optionally with a recurrent spike connection."""

    def __init__(self, n_in: int, n_out: int, beta: float = 0.9, recurrent: bool = False,
                 bias: bool = True, rec_grad_scale: float = 0.0):
        super().__init__()
        self.rec_grad_scale = rec_grad_scale
        self.syn = nn.Linear(n_in, n_out, bias=bias)
        self.rec = nn.Linear(n_out, n_out, bias=False) if recurrent else None
        if self.rec is not None:
            nn.init.orthogonal_(self.rec.weight, gain=0.5)
        self.lif = LIFCell(n_out, beta=beta)

    def forward(self, x, state: LIFState):
        drive = self.syn(x)
        if self.rec is not None and state.s is not None:
            # The recurrent spikes drive the forward pass unchanged, but their
            # gradient is scaled by rec_grad_scale at every step (0 = detached).
            # Fully attached, the gradient compounded through W_rec times the
            # surrogate slope over 200 steps until its norm overflowed float32,
            # which froze training. Fully detached, the gradient only reaches
            # ~1/(1 - beta) steps back through the leak, too short to learn
            # timing across the window: an encoder-alone test (regress 8x64
            # event-time symbols) gave D 0.179 detached, 0.097 with scale 0.5,
            # and 0.090 with scale 0.5 plus leak 0.97 at init (GRU: 0.052).
            s = state.s
            g = self.rec_grad_scale
            s_rec = s.detach() if g == 0 else s * g + s.detach() * (1.0 - g)
            drive = drive + self.rec(s_rec)
        return self.lif(drive, state)
