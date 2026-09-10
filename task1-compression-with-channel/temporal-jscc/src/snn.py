"""Spiking primitives, the 1-D subset needed here.

Same recipe as `task2-compression-to-m-bits/src/snn.py` (iterative LIF of Wu et
al., Heaviside forward with an arctan surrogate backward, cells advanced one
step at a time carrying an explicit state) and deliberately a separate copy:
the repo README keeps the tasks independent, and this task has no spatial
topology to convolve over, so only the linear layers are wanted. The one thing
that is genuinely new here is `latency_readout`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ATanSpike(torch.autograd.Function):
    """Heaviside forward, arctan surrogate backward (Neftci, Mostafa & Zenke)."""

    @staticmethod
    def forward(ctx, u, alpha):
        ctx.save_for_backward(u)
        ctx.alpha = alpha
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        a = ctx.alpha
        return grad_out * a / (2 * (1 + (torch.pi / 2 * a * u) ** 2)), None


def spike(u, alpha: float = 2.0):
    return ATanSpike.apply(u, alpha)


class StraightThrough(torch.autograd.Function):
    """Forward: the true, non-differentiable output. Backward: transparent.

    Used for the multi-spike channel, which re-assigns spikes to integer time
    slots -- a discrete, data-dependent operation with no useful derivative.
    Same device as `delay_channel` in ../rate_snn.py.
    """

    @staticmethod
    def forward(ctx, x, y):
        return y

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


class LIFState:
    __slots__ = ("u", "s")

    def __init__(self, u=None, s=None):
        self.u, self.s = u, s


class LIFCell(nn.Module):
    """Leaky integrate-and-fire, learnable per-unit leak and threshold.

        u_t = beta * u_{t-1} * (1 - s_{t-1}) + I_t
        s_t = H(u_t - theta)

    The reset factor is detached: SuperSpike's second approximation, which also
    stops the surrogate being applied twice to the same spike.
    """

    def __init__(self, n_units: int, beta: float = 0.9, theta: float = 1.0,
                 learn_dynamics: bool = True, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta_logit = nn.Parameter(torch.logit(torch.full((n_units,), beta)),
                                       requires_grad=learn_dynamics)
        self.theta = nn.Parameter(torch.full((n_units,), theta),
                                  requires_grad=learn_dynamics)

    def forward(self, current, state: LIFState):
        beta = torch.sigmoid(self.beta_logit)
        if state.u is None:
            u = torch.zeros_like(current)
            s = torch.zeros_like(current)
        else:
            u, s = state.u, state.s
        u = beta * u * (1.0 - s.detach()) + current
        s = spike(u - self.theta, self.alpha)
        return s, LIFState(u, s)


class LIFReadout(nn.Module):
    """Non-firing integrator; its membrane potential is the real-valued output
    (the spike-to-value decoding of Kamata et al. 2021)."""

    def __init__(self, beta: float = 0.9, learn_dynamics: bool = True):
        super().__init__()
        self.beta_logit = nn.Parameter(torch.logit(torch.tensor(beta)),
                                       requires_grad=learn_dynamics)

    def forward(self, current, u):
        beta = torch.sigmoid(self.beta_logit)
        return current if u is None else beta * u + current


class TemporalBatchNorm(nn.Module):
    """BatchNorm without affine, followed by a gain and bias indexed by time
    step. Normalising each step independently *with* affine would renormalise
    away any global temporal envelope; indexing the affine by t stabilises the
    potentials while letting the network keep an envelope. Motivation and the
    measured failure it fixes are written up in task 2's `snn.py`."""

    def __init__(self, n_units: int, n_bins: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(n_units, affine=False)
        self.gamma = nn.Parameter(torch.ones(n_bins, n_units))
        self.beta = nn.Parameter(torch.zeros(n_bins, n_units))

    def forward(self, x, t: int):
        return self.norm(x) * self.gamma[t] + self.beta[t]


class SpikingLinear(nn.Module):
    def __init__(self, n_in, n_out, n_bins, beta=0.9):
        super().__init__()
        self.syn = nn.Linear(n_in, n_out, bias=False)
        self.bn = TemporalBatchNorm(n_out, n_bins)
        self.lif = LIFCell(n_out, beta=beta)

    def forward(self, x, state: LIFState, t: int):
        return self.lif(self.bn(self.syn(x), t), state)


class RecurrentSpikingLinear(nn.Module):
    """Spiking linear layer with an explicit recurrent spike connection.

    Both ends of this system need it. The encoder has to integrate evidence
    across the whole window before it commits to a spike time, and the decoder's
    input is (in the TTFS variant) a single arrival per code neuron, so all of
    its output timing has to be generated internally: the recurrent weights are
    a learned spiking oscillator that the received code modulates.
    """

    def __init__(self, n_in, n_out, n_bins, beta=0.9):
        super().__init__()
        self.syn = nn.Linear(n_in, n_out, bias=False)
        self.rec = nn.Linear(n_out, n_out, bias=False)
        nn.init.orthogonal_(self.rec.weight, gain=0.5)
        self.bn = TemporalBatchNorm(n_out, n_bins)
        self.lif = LIFCell(n_out, beta=beta)

    def forward(self, x, state: LIFState, t: int):
        drive = self.syn(x)
        if state.s is not None:
            drive = drive + self.rec(state.s)
        return self.lif(self.bn(drive, t), state)


def psp_filter(spikes: torch.Tensor, tau: float = 3.0) -> torch.Tensor:
    """Exponential PSP filter along dim 1 (time):

        PSP_t = (1 - 1/tau) * PSP_{t-1} + s_t / tau

    The kernel behind the van Rossum distance. `theory.psp_trace` runs the same
    recursion in numpy so the bound and the measurement use one definition.
    """
    decay = 1.0 - 1.0 / tau
    out = torch.empty_like(spikes)
    acc = torch.zeros_like(spikes[:, 0])
    for t in range(spikes.shape[1]):
        acc = decay * acc + spikes[:, t] / tau
        out[:, t] = acc
    return out


def latency_readout(u: torch.Tensor, t_win: float) -> torch.Tensor:
    """Membrane potential -> spike time in [0, t_win], monotonically decreasing.

    This is the encoder's temporal code. A readout neuron that integrates its
    drive and fires once fires *earlier* the harder it is driven, so latency is
    a decreasing function of accumulated potential; here that function is
    t = t_win * sigmoid(-u), which is smooth, bounded to the window by
    construction (the peak constraint the channel capacity is computed under),
    and never saturates its gradient to exactly zero.

    It is a monotone reparameterisation of the drive, not a threshold-crossing
    solve: unlike ../alpha_ttfs.py there is no exact spike time here. What is
    kept is what the rate calculation needs -- one real number per code neuron,
    confined to the window.
    """
    return t_win * torch.sigmoid(-u)
