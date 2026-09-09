"""SNN encoder -> m-bit code -> SNN decoder, for task 2 of the subject sheet.

    x in {0,1}^(T,2,34,34)  --SNN encoder-->  b in {0,1}^m  --SNN decoder-->  x_hat

The whole pipeline is spiking. The only non-spiking objects are the two readout
membrane potentials (one at the bottleneck, one at the output), which is the
standard spike-to-value decoding of Kamata et al. (2021): a non-firing neuron
whose membrane potential is read directly.

Three design points are worth spelling out.

1. The bottleneck is a *hard* threshold, not a relaxation. b really is m bits at
   evaluation time, so the rate axis of the rate-distortion curve is exact and
   not a stand-in for one. Gradients pass through it with a clipped
   straight-through estimator. Read physically, the bottleneck is a population
   of m readout neurons each allowed to fire at most once over the recording:
   the code is the population's firing pattern.

2. A BatchNorm sits on the accumulated potential just before the threshold. It
   centres the pre-threshold values on 0, which (a) keeps the bits near a 50/50
   marginal, so m bits carry close to m bits of entropy rather than collapsing to
   a constant, and (b) keeps the potentials inside the straight-through window
   where gradients are non-zero. Without it the code saturates within an epoch.

3. The decoder's input is constant in time, so its temporal structure has to be
   generated internally. It gets three mechanisms for that: an explicitly
   recurrent spiking layer (a learned spiking oscillator the code modulates),
   time-indexed affine parameters in every batch norm, and a per-time-step bias
   on the output logit.

   The batch norms are the part that is easy to get wrong, and we did get it
   wrong first: with a plain per-step BatchNorm every layer renormalises each
   time step to the same scale, which erases the temporal envelope, and the
   decoder emits a flat ~140 spikes/bin against a true profile swinging between
   15 and 270. See `snn.TemporalBatchNorm`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .snn import (
    LIFCell,
    LIFReadout,
    LIFState,
    RecurrentSpikingLinear,
    SpikingConv,
    SpikingLinear,
    StraightThroughBinary,
)


class SpikingEncoder(nn.Module):
    """Convolutional SNN reading the spike train one time bin at a time and
    accumulating an m-dimensional readout potential over the whole recording."""

    def __init__(self, n_bits: int, n_bins: int, width: int = 32,
                 hidden: int = 256, beta: float = 0.9):
        super().__init__()
        w = width
        self.conv1 = SpikingConv(2, w, n_bins, stride=2, beta=beta)          # 34->17
        self.conv2 = SpikingConv(w, 2 * w, n_bins, stride=2, beta=beta)      # 17-> 9
        self.conv3 = SpikingConv(2 * w, 4 * w, n_bins, stride=2, beta=beta)  #  9-> 5
        self.flat_dim = 4 * w * 5 * 5
        self.fc = SpikingLinear(self.flat_dim, hidden, n_bins, beta=beta)
        self.to_code = nn.Linear(hidden, n_bits, bias=False)
        # beta near 1: the code should integrate the entire recording, not just
        # its tail.
        self.readout = LIFReadout(beta=0.95)
        self.code_norm = nn.BatchNorm1d(n_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 2, 34, 34) -> pre-threshold code potential (B, m)."""
        s1 = s2 = s3 = s4 = None
        st1, st2, st3, st4 = (LIFState() for _ in range(4))
        u = None
        for t in range(x.shape[1]):
            s1, st1 = self.conv1(x[:, t], st1, t)
            s2, st2 = self.conv2(s1, st2, t)
            s3, st3 = self.conv3(s2, st3, t)
            s4, st4 = self.fc(s3.flatten(1), st4, t)
            u = self.readout(self.to_code(s4), u)
        return self.code_norm(u)


class SpikingDecoder(nn.Module):
    """Deconvolutional SNN expanding the static m-bit code back into T frames."""

    def __init__(self, n_bits: int, n_bins: int, width: int = 32,
                 hidden: int = 256, beta: float = 0.9):
        super().__init__()
        w = width
        self.n_bins = n_bins
        self.width = w
        self.rec = RecurrentSpikingLinear(n_bits, hidden, n_bins, beta=beta)
        self.fc = SpikingLinear(hidden, 4 * w * 5 * 5, n_bins, beta=beta)
        self.up1 = SpikingConv(4 * w, 2 * w, n_bins, stride=2, transpose=True,
                               output_padding=0, beta=beta)          # 5 ->  9
        self.up2 = SpikingConv(2 * w, w, n_bins, stride=2, transpose=True,
                               output_padding=0, beta=beta)          # 9 -> 17
        self.out_syn = nn.ConvTranspose2d(w, 2, 3, 2, 1, output_padding=1,
                                          bias=False)                # 17 -> 34
        # short memory on the output neuron: mostly instantaneous, free to
        # smooth a little if that helps
        self.out_readout = LIFReadout(beta=0.3)
        self.out_bias = nn.Parameter(torch.full((2, 34, 34), -2.5))
        # per-time-step bias on the output logit. The temporal batch norms above
        # can now carry an envelope, but the firing *rate* per bin is set here
        # most directly: N-MNIST's spikes-per-bin swings by a factor of ~18
        # across the three saccades, and one scalar per bin expresses that
        # without spending any of the code's capacity on it.
        self.t_out_bias = nn.Parameter(torch.zeros(n_bins, 1, 1, 1))

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """b: (B, m) in {0,1} -> spike logits (B, T, 2, 34, 34)."""
        stR, st2, st3, st4 = (LIFState() for _ in range(4))
        u = None
        logits = []
        for t in range(self.n_bins):
            sR, stR = self.rec(b, stR, t)
            s2, st2 = self.fc(sR, st2, t)
            s3, st3 = self.up1(s2.view(-1, 4 * self.width, 5, 5), st3, t)
            s4, st4 = self.up2(s3, st4, t)
            u = self.out_readout(self.out_syn(s4), u)
            logits.append(u + self.out_bias + self.t_out_bias[t])
        return torch.stack(logits, dim=1)


class SpikeCompressor(nn.Module):
    """The full end-to-end system of task 2."""

    def __init__(self, n_bits: int = 128, n_bins: int = 16, width: int = 32,
                 hidden: int = 256, beta: float = 0.9):
        super().__init__()
        self.n_bits = n_bits
        self.n_bins = n_bins
        self.encoder = SpikingEncoder(n_bits, n_bins, width, hidden, beta)
        self.decoder = SpikingDecoder(n_bits, n_bins, width, hidden, beta)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Spike train -> hard m-bit code (differentiable through the STE)."""
        return StraightThroughBinary.apply(self.encoder(x))

    def forward(self, x: torch.Tensor):
        b = self.encode(x)
        return self.decoder(b), b

    @torch.no_grad()
    def compression_ratio(self) -> float:
        return self.n_bins * 2 * 34 * 34 / self.n_bits
