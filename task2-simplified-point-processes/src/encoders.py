"""Encoders: binned event stream (B, M) -> raw code vector (B, n_out).

Both encoders read the same input, one time bin per step, and emit a vector
whose meaning is set by the bottleneck: L * A logits for the one-hot quantizer,
or one value per coordinate for the scalar one. The bottleneck, prior, decoder,
loss and evaluation are shared, so only the computation in between differs
(Stage 5 of the plan).

* `SNNEncoder`  spiking: LIF layer -> recurrent LIF layer -> leaky non-firing
                readout integrated over the window.
* `ANNEncoder`  a GRU over the same bins, final hidden state -> code. Its
                width is chosen by `matched_ann` to match the SNN's parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .snn import LIFReadout, LIFState, SpikingLinear


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class SNNEncoder(nn.Module):
    def __init__(self, hidden: int = 64, n_out: int = 128, beta: float = 0.9,
                 rec_grad_scale: float = 0.0):
        super().__init__()
        # No bias on the input layer and a weight above threshold: layer 1 fires
        # because of events, not spontaneously. With a bias it was driven mostly
        # by its own constant current and every window produced the same code.
        self.l1 = SpikingLinear(1, hidden, beta=beta, bias=False)
        nn.init.normal_(self.l1.syn.weight, mean=1.5, std=0.5)
        self.l2 = SpikingLinear(hidden, hidden, beta=beta, recurrent=True,
                                rec_grad_scale=rec_grad_scale)
        self.readout = LIFReadout(hidden, beta=0.99)
        # Normalize the integrated *features*, then map linearly to the code.
        # Normalizing the logits themselves (per logit, across the batch) was
        # tried first: it forces every symbol to win about equally often, so the
        # rate stayed pinned at log2(A) bits per coordinate whatever beta was.
        self.norm = nn.BatchNorm1d(hidden)
        self.to_code = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor, return_spikes: bool = False):
        """x: (B, M) event counts per bin -> code (B, n_out) [, spike counts per layer]."""
        st1, st2 = LIFState(), LIFState()
        u = None
        n_spikes = [x.new_zeros(()), x.new_zeros(())]
        for t in range(x.shape[1]):
            s1, st1 = self.l1(x[:, t:t + 1], st1)
            s2, st2 = self.l2(s1, st2)
            u = self.readout(s2, u)
            if return_spikes:
                n_spikes[0] = n_spikes[0] + s1.detach().sum()
                n_spikes[1] = n_spikes[1] + s2.detach().sum()
        out = self.to_code(self.norm(u))
        return (out, n_spikes) if return_spikes else out


class ANNEncoder(nn.Module):
    def __init__(self, hidden: int = 64, n_out: int = 128):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.norm = nn.BatchNorm1d(hidden)
        self.to_code = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x.unsqueeze(-1))
        return self.to_code(self.norm(h[-1]))


def matched_ann(target: nn.Module, n_out: int) -> ANNEncoder:
    """The ANNEncoder whose parameter count is closest to `target`'s."""
    goal = n_params(target)
    best = min(range(4, 512), key=lambda h: abs(n_params(ANNEncoder(h, n_out)) - goal))
    return ANNEncoder(best, n_out)
