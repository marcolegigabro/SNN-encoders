"""
alpha_ttfs.py -- Exact-gradient time-to-first-spike (TTFS) layer, ported
from google/ihmehimmeli (branch `autoencoder`, file tempcoding/tempcoder.cc,
function ActivateNeuronAlpha), which implements Comsa et al.'s
"Spiking Autoencoders With Temporal Coding" (Frontiers in Neuroscience, 2021).

Key difference vs a snnTorch/discretized simulation: there is NO time-step
loop and NO surrogate gradient. Each neuron's exact spike time is solved
analytically (via the Lambert W function, closed form for an alpha-function
post-synaptic potential f(s) = s * exp(-K*s)), and the gradient of that exact
spike time w.r.t. every presynaptic weight and time is also closed-form
(implicit function theorem on the threshold-crossing equation), matching the
formulas in ActivateNeuronAlpha exactly. This should give a much cleaner,
lower-variance gradient signal than a T-step surrogate-gradient simulation,
which is the most likely reason the original paper avoids the "bottleneck
collapse" we saw with our snnTorch version.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

NO_SPIKE_TIME_DEFAULT = 1000.0  # sentinel finite "very late" time (not inf, to keep MSE finite)
_MIN_LAMBERT_ARG = -1.0 / math.e


def lambert_w0(x: torch.Tensor, n_iter: int = 8) -> torch.Tensor:
    """
    Real principal branch W0(x) for x >= -1/e, via Halley's iteration.
    No gradient needed through this: our custom autograd.Function never
    backpropagates through lambert_w0 itself, it uses the closed-form
    derivatives from the paper instead (see AlphaActivation.backward).
    """
    x = x.clamp(min=_MIN_LAMBERT_ARG + 1e-12)
    # initial guess (standard piecewise approximation)
    w = torch.where(
        x < 1.0,
        x * (1.0 - x + 1.5 * x * x),  # good near 0
        torch.log(x.clamp(min=1e-12)) - torch.log(torch.log(x.clamp(min=1.0001)).clamp(min=1e-6)),
    )
    w = torch.where(x <= -0.3, torch.full_like(x, -0.5), w)  # near the branch point, safer start
    for _ in range(n_iter):
        ew = torch.exp(w)
        wew = w * ew
        f = wew - x
        denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0).clamp(min=1e-6)
        denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
        w = w - f / denom
    return w


class AlphaActivation(torch.autograd.Function):
    """
    Vectorized batched version of Tempcoder::ActivateNeuronAlpha.

    times   : (batch, N_in)  presynaptic spike times (shared across out neurons)
    weights : (N_out, N_in)  synaptic weights
    Returns : spike_time (batch, N_out), has_spike (batch, N_out) bool
    """

    @staticmethod
    def forward(ctx, times, weights, threshold, decay_rate, clip_derivative, no_spike_time):
        batch, n_in = times.shape
        n_out = weights.shape[0]
        K = decay_rate

        sorted_t, sort_idx = torch.sort(times, dim=-1)  # (batch, n_in)
        idx_exp = sort_idx.unsqueeze(1).expand(batch, n_out, n_in)
        w_exp_full = weights.unsqueeze(0).expand(batch, n_out, n_in)
        sorted_w = torch.gather(w_exp_full, 2, idx_exp)  # (batch, n_out, n_in)

        exp_t = torch.exp(K * sorted_t)  # (batch, n_in)
        exp_t_b = exp_t.unsqueeze(1).expand(batch, n_out, n_in)
        sorted_t_b = sorted_t.unsqueeze(1).expand(batch, n_out, n_in)

        w_exp = sorted_w * exp_t_b
        A_cum = torch.cumsum(w_exp, dim=-1)
        B_cum = torch.cumsum(w_exp * sorted_t_b, dim=-1)

        A_safe = torch.where(A_cum > 1e-12, A_cum, torch.full_like(A_cum, 1e-12))
        b_over_a = B_cum / A_safe
        lambert_arg = -K * threshold / A_safe * torch.exp(K * b_over_a)

        valid_domain = (A_cum > 0) & (lambert_arg >= _MIN_LAMBERT_ARG)
        W = lambert_w0(lambert_arg)
        cand_t = b_over_a - W / K

        next_t = torch.cat(
            [sorted_t_b[..., 1:], torch.full_like(sorted_t_b[..., :1], float("inf"))], dim=-1
        )
        causal_valid = valid_domain & (cand_t >= sorted_t_b - 1e-9) & (cand_t <= next_t + 1e-9)

        has_valid = causal_valid.any(dim=-1)
        first_idx = causal_valid.float().argmax(dim=-1)  # (batch, n_out)

        gather_idx = first_idx.unsqueeze(-1)
        spike_time = torch.gather(cand_t, 2, gather_idx).squeeze(-1)
        A_final = torch.gather(A_cum, 2, gather_idx).squeeze(-1)
        B_final = torch.gather(B_cum, 2, gather_idx).squeeze(-1)
        W_final = torch.gather(W, 2, gather_idx).squeeze(-1)

        spike_time = torch.where(has_valid, spike_time, torch.full_like(spike_time, no_spike_time))
        A_final = torch.where(has_valid, A_final, torch.ones_like(A_final))
        B_final = torch.where(has_valid, B_final, torch.zeros_like(B_final))
        W_final = torch.where(has_valid, W_final, torch.zeros_like(W_final))

        ctx.save_for_backward(times, weights, spike_time, A_final, B_final, W_final, has_valid)
        ctx.K = K
        ctx.clip_derivative = clip_derivative
        return spike_time, has_valid

    @staticmethod
    def backward(ctx, grad_spike_time, grad_has_valid):
        times, weights, spike_time, A_final, B_final, W_final, has_valid = ctx.saved_tensors
        K = ctx.K
        clip = ctx.clip_derivative
        batch, n_in = times.shape
        n_out = weights.shape[0]

        tp = times.unsqueeze(1).expand(batch, n_out, n_in)
        wp = weights.unsqueeze(0).expand(batch, n_out, n_in)
        e_K_tp = torch.exp(K * tp)

        A = A_final.unsqueeze(-1)
        Wf = W_final.unsqueeze(-1)
        st = spike_time.unsqueeze(-1)
        b_over_a = (B_final / A_final.clamp(min=1e-12)).unsqueeze(-1)

        denom = A * (1.0 + Wf)
        denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)

        dspike_dw = e_K_tp * (tp - b_over_a + Wf / K) / denom
        dspike_dt = wp * e_K_tp * (K * (tp - b_over_a) + Wf + 1.0) / denom

        causal_mask = tp <= st + 1e-9
        valid_mask = has_valid.unsqueeze(-1) & causal_mask
        dspike_dw = torch.where(valid_mask, dspike_dw, torch.zeros_like(dspike_dw))
        dspike_dt = torch.where(valid_mask, dspike_dt, torch.zeros_like(dspike_dt))

        if clip and clip > 0:
            dspike_dw = dspike_dw.clamp(-clip, clip)
            dspike_dt = dspike_dt.clamp(-clip, clip)

        grad_out = grad_spike_time.unsqueeze(-1)  # (batch, n_out, 1)
        grad_weights = (grad_out * dspike_dw).sum(dim=0)  # (n_out, n_in)
        grad_times = (grad_out * dspike_dt).sum(dim=1)  # (batch, n_in)

        return grad_times, grad_weights, None, None, None, None


class AlphaTTFSLayer(nn.Module):
    """
    One exact-gradient TTFS layer with Comsa-style learnable synchronization
    pulses appended to the presynaptic inputs (same convention as
    Tempcoder: weights_[layer][post][pre], last n_pulses columns = pulses).
    """

    def __init__(self, d_in, d_out, n_pulses=10, threshold=1.0, decay_rate=0.2,
                 pulse_range=(0.0, 16.0), nonpulse_weight_mean_multiplier=0.0,
                 pulse_weight_mean_multiplier=0.0, clip_derivative=500.0,
                 no_spike_time=NO_SPIKE_TIME_DEFAULT):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_pulses = n_pulses
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.clip_derivative = clip_derivative
        self.no_spike_time = no_spike_time

        fan_in_total = d_in + n_pulses
        sigma = math.sqrt(2.0 / (fan_in_total + d_out))
        w = torch.randn(d_out, fan_in_total) * sigma
        w[:, :d_in] += nonpulse_weight_mean_multiplier * sigma
        w[:, d_in:] += pulse_weight_mean_multiplier * sigma
        self.weight = nn.Parameter(w)

        lo, hi = pulse_range
        spacing = (hi - lo) / (n_pulses + 1)
        init_pulses = torch.tensor([lo + spacing * (i + 1) for i in range(n_pulses)])
        self.pulse_times = nn.Parameter(init_pulses)

    def forward(self, pre_times):
        batch = pre_times.shape[0]
        pulse_t = self.pulse_times.unsqueeze(0).expand(batch, self.n_pulses)
        times_full = torch.cat([pre_times, pulse_t], dim=-1)
        spike_time, has_spike = AlphaActivation.apply(
            times_full, self.weight, self.threshold, self.decay_rate,
            self.clip_derivative, self.no_spike_time,
        )
        return spike_time, has_spike

    def no_spike_penalty(self, has_spike):
        """
        Analog of the paper's "small positive penalty added to each input
        weight derivative when a neuron doesn't spike, to encourage
        spiking". Implemented as an explicit auxiliary loss term (instead of
        hand-injecting a gradient) whose gradient w.r.t. that neuron's
        weights is a constant negative value proportional to how often it
        fails to spike across the batch -- pushing Adam to increase those
        weights until the neuron starts firing.
        """
        frac_no_spike = (~has_spike).float().mean(dim=0)  # (d_out,)
        return -(frac_no_spike * self.weight.mean(dim=1)).sum()
