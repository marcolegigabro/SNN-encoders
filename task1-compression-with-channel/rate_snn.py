"""
rate_snn.py -- Rate-coded SNN compressor / decompressor with a discrete-time
queueing DELAY channel, as an alternative to the TTFS + exact-gradient
architecture in alpha_ttfs.py.

Why a different file / different neuron model
----------------------------------------------
In alpha_ttfs.py, each neuron spikes AT MOST ONCE and information lives in
WHEN it spikes -- this is what let us solve the exact spike time (Lambert W)
and its exact gradient in closed form. Rate coding is the opposite regime:
a neuron can spike any number of times over T steps, and information lives
in HOW MANY times it spikes (its rate). There is no simple closed-form
"exact gradient" for a multi-spike neuron in this project's timeframe, so
we go back to the discretized/BPTT simulation (snnTorch, T explicit steps,
surrogate gradient) used earlier in this project for the rate-coded
autoencoder, and train the reconstruction with a van Rossum distance rather
than an MSE on scalar spike times.

The delay channel
------------------
Literature check (see conversation): there is essentially no published
"delay channel" model specific to compressed rate-coded spike trains. The
literature review agent has flagged this as a genuinely open angle at the
same time as also identifying the closest existing framework to build on:
the discrete-time queueing channel of Bedekar & Azizoglu (1998) and
Prabhakar & Gallager (2003) (the discrete-time analogues of Anantharam &
Verdu's "Bits through queues", 1996), generalised here to BULK arrivals:
at every timestep, ALL N bottleneck neurons share the SAME channel of fixed
throughput `capacity` spikes/timestep (this matches the project's own
framing: the K compressed neurons all go through one shared communication
link). Spikes that cannot be served immediately queue (FIFO) and depart
later -- a genuine, capacity-limited DELAY, as opposed to the jitter
channel (independent per-spike noise) or a trivial fixed shift (which is
lossless and non-realistic, see earlier discussion). Spikes still queued
when the T-step window ends are lost -- a real, lossy channel.

This delay/requeueing operation is NOT differentiable (it moves spikes
between integer time slots depending on how loaded the shared queue is,
which is a discrete, data-dependent process) -- it is wrapped with a
straight-through estimator (same idea as snn.surrogate.straight_through_estimator
already used for the neurons themselves): forward pass uses the true,
delayed spike train; backward pass passes the loss gradient straight
through as if the channel were transparent.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ---------------------------------------------------------------------------
# Rate encoding / decoding
# ---------------------------------------------------------------------------

def rate_encode(pixels, T, p_max=0.8):
    """
    pixels: (batch, D) in [0, 16] -> spike train (batch, T, D), Bernoulli at
    each time step with a rate proportional to the pixel's intensity.
    """
    rate = (pixels / 16.0).clamp(0, 1) * p_max  # (batch, D)
    batch, D = pixels.shape
    rate_expanded = rate.unsqueeze(1).expand(batch, T, D)
    return torch.bernoulli(rate_expanded)


def decode_rate(spikes, p_max=0.8):
    """spikes: (batch, T, D) -> reconstructed intensity (batch, D) in [0,16]."""
    rate_hat = spikes.mean(dim=1) / p_max
    return (rate_hat * 16.0).clamp(0, 16)


# ---------------------------------------------------------------------------
# Distortion: van Rossum (taken from utils.py, included here so the file
# is self-contained)
# ---------------------------------------------------------------------------

def exponential_filter(spikes, tau):
    batch, T, D = spikes.shape
    decay = float(torch.exp(torch.tensor(-1.0 / tau)))
    running = torch.zeros(batch, D, device=spikes.device, dtype=spikes.dtype)
    outputs = []
    for t in range(T):
        running = decay * running + spikes[:, t, :]
        outputs.append(running)
    return torch.stack(outputs, dim=1)


def van_rossum_distance(x, x_hat, tau=5.0):
    trace_x = exponential_filter(x, tau)
    trace_xhat = exponential_filter(x_hat, tau)
    return torch.mean((trace_x - trace_xhat) ** 2)


# ---------------------------------------------------------------------------
# A single SNN layer (no hidden layer -- same convention as
# AlphaTTFSLayer: the compressor and decompressor are each ONE layer)
# ---------------------------------------------------------------------------

class SNNLayer(nn.Module):
    def __init__(self, d_in, d_out, beta=0.9, threshold=1.0, spike_grad=None):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.straight_through_estimator()
        self.fc = nn.Linear(d_in, d_out)
        self.neuron = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.d_out = d_out

    def forward(self, x):
        """x: (batch, T, d_in) -> spk: (batch, T, d_out). A neuron can
        spike multiple times over the T steps -- this is the key difference from
        AlphaTTFSLayer (at most one spike per neuron)."""
        batch, T, _ = x.shape
        mem = torch.zeros(batch, self.d_out, device=x.device)
        outs = []
        for t in range(T):
            cur = self.fc(x[:, t, :])
            spk, mem = self.neuron(cur, mem)
            outs.append(spk)
        return torch.stack(outs, dim=1)

class SNNStack(nn.Module):
    """
    Stacking several SNNLayer -- unlike the TTFS case, nothing
    special is needed: each SNNLayer consumes and produces a
    full spike train (batch, T, d), so we just chain them.
    """
    def __init__(self, layer_sizes, beta=0.9, threshold=1.0, spike_grad=None):
        super().__init__()
        assert len(layer_sizes) >= 2
        self.layers = nn.ModuleList([
            SNNLayer(layer_sizes[i], layer_sizes[i + 1], beta=beta,
                     threshold=threshold, spike_grad=spike_grad)
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# ---------------------------------------------------------------------------
# Delay channel: discrete bulk-arrival queue,
# shared by the N bottleneck neurons -- a discrete generalization of
# Bedekar-Azizoglu / Anantharam-Verdu.
# ---------------------------------------------------------------------------

def _bulk_queue_delay_numpy(spikes_np, capacity):
    """spikes_np: (batch, T, N) 0/1 -> same shape, delayed/dropped spikes."""
    batch, T, N = spikes_np.shape
    out = np.zeros_like(spikes_np)
    for b in range(batch):
        events = [(t, n) for t in range(T) for n in range(N) if spikes_np[b, t, n] > 0.5]
        events.sort(key=lambda e: e[0])
        queue = []
        ev_idx = 0
        for t in range(T):
            while ev_idx < len(events) and events[ev_idx][0] == t:
                queue.append(events[ev_idx][1])
                ev_idx += 1
            n_depart = min(capacity, len(queue))
            for _ in range(n_depart):
                n = queue.pop(0)
                out[b, t, n] = 1.0
        # whatever remains in `queue` at the end of the T window is lost
        # (finite-capacity channel: a genuinely lossy channel)
    return out


def bulk_queue_delay_channel(spikes, capacity=1):
    """
    Delay channel (shared queue, fixed capacity `capacity`
    spikes/time step). NOT differentiable (discrete reassignment of
    spikes to new time indices) -- use via `delay_channel`
    below, which adds the straight-through estimator.
    """
    spikes_np = spikes.detach().cpu().numpy()
    out_np = _bulk_queue_delay_numpy(spikes_np, capacity)
    return torch.tensor(out_np, dtype=spikes.dtype, device=spikes.device)


def delay_channel(spikes, capacity=1):
    """
    Trainable (straight-through) version of the delay channel: the forward pass
    uses the true, delayed/truncated spike train, the backward pass lets
    the loss gradient pass through as if the channel were transparent --
    exactly the same principle as the straight-through estimator already
    used for the neurons' own spike function.
    """
    delayed = bulk_queue_delay_channel(spikes, capacity=capacity)
    return spikes + (delayed - spikes).detach()
