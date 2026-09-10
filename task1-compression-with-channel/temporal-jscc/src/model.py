"""SNN compressor -> jitter channel -> SNN decompressor, trained end to end.

    x in {0,1}^(T,N)  --SNN-->  code  --jitter-->  code'  --SNN-->  x_hat

This is joint source-channel coding: there is no bit-string in the middle and no
error-correcting code. The encoder's output goes straight onto the channel, the
decoder sees only what came out, and the two are trained together against the
distortion. Nothing in the system knows the channel except through the gradient
that reaches back through it.

The code itself takes one of two forms, selected by `bottleneck`:

**ttfs** -- K code neurons, each firing exactly once, at a real-valued time in
[0, T]. The channel adds Gaussian jitter to each time. This is temporal coding
in the strict sense: all the information is in K spike *times*, and the rate is
K uses of a peak-constrained AWGN channel.

**multispike** -- K code neurons emitting a full T-slot spike train, each spike
independently displaced and re-binned by the channel. Closer to the rate-coded
setting of ../rate_snn.py, and it needs a straight-through estimator, which is
why it is the control rather than the primary system.

The decoder never receives a number labelled "the code". In the TTFS variant the
received times are turned into post-synaptic currents by
`arrival_current` -- literally the PSP each arriving spike would evoke -- so the
decoder is driven by spike arrivals, and the gradient still reaches the arrival
times because that kernel is smooth in them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .channel import jitter_rebin, jitter_times
from .snn import (
    LIFReadout,
    LIFState,
    RecurrentSpikingLinear,
    SpikingLinear,
    latency_readout,
)


@dataclass
class SystemConfig:
    n_neurons: int = 16          # N, source neurons
    n_bins: int = 12             # T, bins per block; also the TTFS window
    n_code: int = 8              # K, code neurons = channel uses per block
    hidden: int = 256
    bottleneck: str = "ttfs"     # "ttfs" | "multispike"
    sigma: float = 1.0           # channel jitter, in bins, for either variant
    beta: float = 0.9
    psp_tau_in: float = 2.0      # decoder input kernel time constant
    source_rate: float = 0.15    # only used to initialise the output bias

    @property
    def t_win(self) -> float:
        """The TTFS window, in bins. A code neuron may fire anywhere inside the
        block it is describing and nowhere else, so the window is T."""
        return float(self.n_bins)


def arrival_current(t_rx: torch.Tensor, n_bins: int, tau: float = 2.0,
                    eps: float = 0.5) -> torch.Tensor:
    """Received spike times (B, K) -> post-synaptic current (B, T, K).

    The current at bin t is the PSP of a spike that arrived at t_rx: an
    exponential decay with time constant `tau`, gated on so that a spike has no
    effect before it arrives. The gate is a sigmoid of width `eps` rather than a
    step, which is what keeps the current differentiable in the arrival time and
    therefore keeps the channel in the gradient path.
    """
    t = torch.arange(n_bins, device=t_rx.device, dtype=t_rx.dtype).view(1, n_bins, 1)
    s = t - t_rx.unsqueeze(1)
    return torch.sigmoid(s / eps) * torch.exp(-torch.relu(s) / tau)


class Compressor(nn.Module):
    """SNN reading the source block one bin at a time and emitting the code."""

    def __init__(self, cfg: SystemConfig):
        super().__init__()
        self.cfg = cfg
        self.l1 = SpikingLinear(cfg.n_neurons, cfg.hidden, cfg.n_bins, beta=cfg.beta)
        self.l2 = RecurrentSpikingLinear(cfg.hidden, cfg.hidden, cfg.n_bins, beta=cfg.beta)
        if cfg.bottleneck == "ttfs":
            # One latency per code neuron, so the head integrates the whole
            # block before committing: beta near 1, and a BatchNorm to keep the
            # latencies spread across the window instead of piling up at one
            # end, which is the same load-bearing role the code BatchNorm plays
            # in task 2's m-bit bottleneck.
            self.to_code = nn.Linear(cfg.hidden, cfg.n_code, bias=False)
            self.readout = LIFReadout(beta=0.95)
            self.code_norm = nn.BatchNorm1d(cfg.n_code)
        elif cfg.bottleneck == "multispike":
            self.code_layer = SpikingLinear(cfg.hidden, cfg.n_code, cfg.n_bins,
                                            beta=cfg.beta)
        else:
            raise ValueError(cfg.bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N) -> (B, K) spike times, or (B, T, K) spike trains."""
        st1, st2, st3 = LIFState(), LIFState(), LIFState()
        u, code_spikes = None, []
        for t in range(self.cfg.n_bins):
            s1, st1 = self.l1(x[:, t], st1, t)
            s2, st2 = self.l2(s1, st2, t)
            if self.cfg.bottleneck == "ttfs":
                u = self.readout(self.to_code(s2), u)
            else:
                s3, st3 = self.code_layer(s2, st3, t)
                code_spikes.append(s3)
        if self.cfg.bottleneck == "ttfs":
            return latency_readout(self.code_norm(u), self.cfg.t_win)
        return torch.stack(code_spikes, dim=1)


class Decompressor(nn.Module):
    """SNN driven by the received code, emitting one spike logit per (bin, neuron)."""

    def __init__(self, cfg: SystemConfig):
        super().__init__()
        self.cfg = cfg
        self.l1 = RecurrentSpikingLinear(cfg.n_code, cfg.hidden, cfg.n_bins, beta=cfg.beta)
        self.l2 = SpikingLinear(cfg.hidden, cfg.hidden, cfg.n_bins, beta=cfg.beta)
        self.out = nn.Linear(cfg.hidden, cfg.n_neurons, bias=False)
        self.out_readout = LIFReadout(beta=0.3)
        # Start at the source's marginal: with no information at all the correct
        # output is p everywhere, which is also the zero-rate optimum for two of
        # the three distortions. Training then only has to earn what the code
        # actually carries.
        p = min(max(cfg.source_rate, 1e-3), 1 - 1e-3)
        self.out_bias = nn.Parameter(torch.full((cfg.n_neurons,),
                                                float(torch.logit(torch.tensor(p)))))
        self.t_bias = nn.Parameter(torch.zeros(cfg.n_bins, 1))

    def forward(self, code_rx: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.bottleneck == "ttfs":
            drive = arrival_current(code_rx, cfg.n_bins, tau=cfg.psp_tau_in)
        else:
            drive = code_rx
        st1, st2 = LIFState(), LIFState()
        u, logits = None, []
        for t in range(cfg.n_bins):
            s1, st1 = self.l1(drive[:, t], st1, t)
            s2, st2 = self.l2(s1, st2, t)
            u = self.out_readout(self.out(s2), u)
            logits.append(u + self.out_bias + self.t_bias[t])
        return torch.stack(logits, dim=1)


class JSCCSystem(nn.Module):
    """Compressor, channel and decompressor as one differentiable object."""

    def __init__(self, cfg: SystemConfig):
        super().__init__()
        self.cfg = cfg
        self.compressor = Compressor(cfg)
        self.decompressor = Decompressor(cfg)

    def transmit(self, code: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
        s = self.cfg.sigma if sigma is None else sigma
        if self.cfg.bottleneck == "ttfs":
            return jitter_times(code, s)
        return jitter_rebin(code, s)

    def forward(self, x: torch.Tensor, sigma: float | None = None):
        code = self.compressor(x)
        code_rx = self.transmit(code, sigma)
        return self.decompressor(code_rx), code, code_rx

    @property
    def channel_uses(self) -> int:
        """Channel uses per source block.

        K either way, but the two variants' uses are not commensurable, which is
        exactly why the rate axis is K * C(sigma) and not K: one TTFS use is one
        real spike time through jitter, one multi-spike use is one T-slot word
        through displacement, and `theory` prices each in bits.
        """
        return self.cfg.n_code
