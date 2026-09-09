"""Spiking primitives: surrogate-gradient LIF neurons and the layers built on them.

Everything here follows the standard direct-training recipe for SNNs (Neftci,
Mostafa & Zenke 2019): the neuron is the iterative LIF of Wu et al., the
Heaviside firing function is left intact on the forward pass, and its derivative
is replaced by a smooth surrogate on the backward pass.

    u_t = beta * u_{t-1} * (1 - s_{t-1}) + I_t          (soft state, hard reset)
    s_t = H(u_t - theta)

Layers are written as "cells": one call advances one time step and carries the
membrane potential in an explicit state object, so the same module can be driven
by an encoder loop or a decoder loop without any hidden global state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATanSpike(torch.autograd.Function):
    """Heaviside forward, arctan surrogate backward.

    The surrogate is the derivative of (1/pi) * arctan(pi * alpha * u / 2), i.e.
    a bell curve of width ~1/alpha centred on the threshold. Bounded gradient,
    non-zero everywhere, which is what keeps deep SNNs trainable.
    """

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


class StraightThroughBinary(torch.autograd.Function):
    """Hard threshold at 0 with a clipped straight-through gradient.

    Used for the m-bit bottleneck. The identity pass-through is clipped to
    |u| <= 1 (Hubara et al.'s binarised-network estimator): outside that range
    the bit is saturated and pushing it further should not be rewarded.
    """

    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        return grad_out * (u.abs() <= 1.0).to(u.dtype)


class LIFState:
    """Membrane potential + last spike of one LIF layer."""

    __slots__ = ("u", "s")

    def __init__(self, u=None, s=None):
        self.u, self.s = u, s


class LIFCell(nn.Module):
    """Leaky integrate-and-fire dynamics wrapped around an arbitrary synapse op.

    `beta` (leak) and `theta` (threshold) are learnable per channel: letting the
    network pick its own time constants matters a lot on N-MNIST, where the
    three saccades put energy at very different timescales.
    """

    def __init__(self, n_channels: int, beta: float = 0.9, theta: float = 1.0,
                 learn_dynamics: bool = True, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha
        # beta is stored through a sigmoid so it stays in (0, 1) under training.
        beta_logit = torch.logit(torch.full((n_channels,), beta))
        self.beta_logit = nn.Parameter(beta_logit, requires_grad=learn_dynamics)
        self.theta = nn.Parameter(torch.full((n_channels,), theta),
                                  requires_grad=learn_dynamics)

    def _shape(self, x):
        # broadcast per-channel parameters over (B, C) or (B, C, H, W)
        return (1, -1) + (1,) * (x.dim() - 2)

    def forward(self, current, state: LIFState) -> tuple[torch.Tensor, LIFState]:
        shape = self._shape(current)
        beta = torch.sigmoid(self.beta_logit).view(shape)
        theta = self.theta.view(shape)

        if state.u is None:
            u = torch.zeros_like(current)
            s = torch.zeros_like(current)
        else:
            u, s = state.u, state.s

        # The reset is detached: gradients do not flow back through the
        # (1 - s) factor. This is the second of SuperSpike's two approximations
        # (Neftci, Mostafa & Zenke 2019, around Eq. 11) -- dropping the reset
        # term empirically trains better, and it also stops the surrogate
        # gradient being applied twice to the same spike on each step.
        u = beta * u * (1.0 - s.detach()) + current
        s = spike(u - theta, self.alpha)
        return s, LIFState(u, s)


class LIFReadout(nn.Module):
    """Non-firing integrator. Its membrane potential is the network's real-valued
    output, as in the spike-to-image decoding of Kamata et al. (2021)."""

    def __init__(self, beta: float = 0.9, learn_dynamics: bool = True):
        super().__init__()
        self.beta_logit = nn.Parameter(torch.logit(torch.tensor(beta)),
                                       requires_grad=learn_dynamics)

    def forward(self, current, u):
        beta = torch.sigmoid(self.beta_logit)
        return current if u is None else beta * u + current


class TemporalBatchNorm(nn.Module):
    """BatchNorm whose affine parameters are indexed by time step.

    This module exists because of a measured failure. A plain BatchNorm applied
    independently at each time step renormalises every step to zero mean and
    unit variance, which erases any *global temporal envelope* in the signal.
    On N-MNIST that is fatal: the three micro-saccades are precisely such an
    envelope, and a decoder built from per-step BatchNorm emits a flat ~140
    spikes/bin against a true profile swinging between 15 and 270 (see
    figures/rate-profile-m128.png before this change). Every layer faithfully
    normalised the saccades away.

    The fix is to normalise without affine, then apply a learned gain and bias
    indexed by the time step. Normalisation still stabilises the membrane
    potentials, but gamma_t and beta_t hand the envelope back, and let the
    decoder *learn* a temporal profile rather than merely avoid destroying one.

    Same motivation as the tdBN of Zheng et al. (2021), which normalises jointly
    over batch and time. Indexing the affine parameters by time is the variant
    that fits a network stepped one bin at a time, including recurrent layers
    where the whole sequence is not available up front.
    """

    def __init__(self, n_channels: int, n_bins: int, spatial: bool = True):
        super().__init__()
        self.norm = (nn.BatchNorm2d if spatial else nn.BatchNorm1d)(
            n_channels, affine=False
        )
        shape = (n_bins, n_channels, 1, 1) if spatial else (n_bins, n_channels)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, x, t: int):
        return self.norm(x) * self.gamma[t] + self.beta[t]


class SpikingConv(nn.Module):
    """Conv -> temporal BatchNorm -> LIF, advanced one time step at a time."""

    def __init__(self, c_in, c_out, n_bins, stride=1, kernel=3, beta=0.9,
                 transpose=False, output_padding=0):
        super().__init__()
        pad = kernel // 2
        if transpose:
            self.syn = nn.ConvTranspose2d(c_in, c_out, kernel, stride, pad,
                                          output_padding=output_padding, bias=False)
        else:
            self.syn = nn.Conv2d(c_in, c_out, kernel, stride, pad, bias=False)
        self.bn = TemporalBatchNorm(c_out, n_bins, spatial=True)
        self.lif = LIFCell(c_out, beta=beta)

    def forward(self, x, state: LIFState, t: int):
        return self.lif(self.bn(self.syn(x), t), state)


class SpikingLinear(nn.Module):
    """Linear -> temporal BatchNorm -> LIF, one time step."""

    def __init__(self, n_in, n_out, n_bins, beta=0.9):
        super().__init__()
        self.syn = nn.Linear(n_in, n_out, bias=False)
        self.bn = TemporalBatchNorm(n_out, n_bins, spatial=False)
        self.lif = LIFCell(n_out, beta=beta)

    def forward(self, x, state: LIFState, t: int):
        return self.lif(self.bn(self.syn(x), t), state)


class RecurrentSpikingLinear(nn.Module):
    """Linear -> temporal BatchNorm -> LIF with an explicit recurrent spike
    connection.

    The decoder needs this. Its input (the m-bit code) is constant over time, so
    without recurrence the only source of temporal structure would be the LIF
    leak, which cannot produce the three-saccade profile of N-MNIST. The
    recurrent weights let the layer act as a learned spiking oscillator that the
    code modulates.
    """

    def __init__(self, n_in, n_out, n_bins, beta=0.9):
        super().__init__()
        self.syn = nn.Linear(n_in, n_out, bias=False)
        self.rec = nn.Linear(n_out, n_out, bias=False)
        nn.init.orthogonal_(self.rec.weight, gain=0.5)
        self.bn = TemporalBatchNorm(n_out, n_bins, spatial=False)
        self.lif = LIFCell(n_out, beta=beta)

    def forward(self, x, state: LIFState, t: int):
        drive = self.syn(x)
        if state.s is not None:
            drive = drive + self.rec(state.s)
        return self.lif(self.bn(drive, t), state)


def psp_filter(spikes: torch.Tensor, tau: float = 3.0) -> torch.Tensor:
    """Exponential post-synaptic potential filter along dim 1 (time).

    PSP_t = (1 - 1/tau) * PSP_{t-1} + (1/tau) * s_t

    This is the kernel behind the van Rossum spike-train distance and behind the
    MMD of Kamata et al.: comparing filtered trains instead of raw ones makes the
    loss tolerant to a bin or two of jitter rather than all-or-nothing.
    """
    decay = 1.0 - 1.0 / tau
    out = torch.empty_like(spikes)
    acc = torch.zeros_like(spikes[:, 0])
    for t in range(spikes.shape[1]):
        acc = decay * acc + spikes[:, t] / tau
        out[:, t] = acc
    return out
