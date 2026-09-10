"""The three distortions, each as a training loss and as a measurement.

A rate-distortion curve is only meaningful if the distortion the system was
*trained* for is the distortion it is *scored* on and the distortion the *bound*
was computed for. So each entry below fixes all three at once, and the units are
stated because they differ:

| name        | measured on                          | unit                  |
| ----------- | ------------------------------------ | --------------------- |
| hamming     | thresholded reconstruction, per bin  | error probability     |
| count_mse   | expected spike count, per neuron     | spikes^2              |
| van_rossum  | filtered soft reconstruction, per bin| squared PSP amplitude |

Only Hamming is scored on a thresholded, genuinely binary reconstruction, and
its bound uses a binary reproduction alphabet to match. The other two are
squared-error distortions, whose minimiser is a conditional *mean* and not a
spike train, so both are scored on the real-valued output and both bounds are
computed over real-valued reproductions -- a grid on [0, T] for the count, and
optimised reproduction vectors for van Rossum (`theory.blahut_arimoto_rd_real`).

Measuring a squared-error distortion on a thresholded output was tried first and
is a mistake worth recording: it cost about a factor of two on van Rossum
(0.018 soft against 0.035 thresholded at the same trained model), because
rounding a posterior to the nearest spike train discards exactly the hedging
that squared error rewards. The thresholded number is still reported alongside,
against the binary-reproduction bound, for the case where the receiver really
must emit spikes.

On the training surrogates:

* Hamming is trained through binary cross-entropy, not through a soft error
  count. BCE is a proper loss, so its minimiser is the true posterior
  P(x=1 | received code), and thresholding a true posterior at 1/2 is exactly
  the Bayes rule for Hamming distortion. Optimising the surrogate therefore
  optimises the target, which is not true of an arbitrary soft relaxation.
* The other two are differentiable functions of the output probabilities as they
  stand, and are used unmodified.

A note on what "trained for Hamming" looks like at low rate: with a Bernoulli
source, predicting silence everywhere scores D = p, which is precisely the
zero-rate point of h(p) - h(D). A model that collapses to silence has not
failed, it has found the R = 0 solution, and the curve is the right place to see
whether it beat it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .snn import psp_filter


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

def hamming_distortion(pred_bits: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of bins where the reconstruction disagrees, per bin."""
    return float((pred_bits - target).abs().mean())


def spike_counts(x: torch.Tensor) -> torch.Tensor:
    """(B, T, N) -> (B, N): spikes per neuron over the window."""
    return x.sum(dim=1)


def count_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared error on the per-neuron spike count, averaged over neurons.

    `pred` may be soft (probabilities, giving the expected count) or hard.
    """
    return ((spike_counts(pred) - spike_counts(target)) ** 2).mean()


def van_rossum_sq(pred: torch.Tensor, target: torch.Tensor,
                  tau: float = 3.0) -> torch.Tensor:
    """Mean squared difference of PSP-filtered trains, per bin.

    The van Rossum distance itself is the square root of the sum over bins; the
    mean square is reported instead so the number is per-bin and comparable
    across T, and so it matches the distortion matrix in `theory.van_rossum_rd`
    element for element. Unnormalised, unlike task 2's `metrics.van_rossum`: a
    rate-distortion axis needs an absolute distortion, not one divided by the
    target's own energy.
    """
    return ((psp_filter(pred, tau) - psp_filter(target, tau)) ** 2).mean()


# ---------------------------------------------------------------------------
# training losses
# ---------------------------------------------------------------------------

class HammingLoss(torch.nn.Module):
    """Binary cross-entropy: the proper surrogate for Hamming (see module doc)."""

    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target)


class CountMSELoss(torch.nn.Module):
    def forward(self, logits, target):
        return count_mse(torch.sigmoid(logits), target)


class VanRossumLoss(torch.nn.Module):
    def __init__(self, tau: float = 3.0):
        super().__init__()
        self.tau = tau

    def forward(self, logits, target):
        return van_rossum_sq(torch.sigmoid(logits), target, self.tau)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

class Distortion:
    """One distortion: its loss, its measurement, and how to price its rate.

    `bits_per_symbol_divisor` converts the bound's rate, which `theory` returns
    per *source symbol* for that distortion, into bits per bin: Hamming's source
    symbol is one bin, the other two treat a whole neuron-window as one symbol.
    """

    def __init__(self, name, loss, measure, unit, symbol_bins, on_hard):
        self.name = name
        self.loss = loss
        self.measure = measure
        self.unit = unit
        self.symbol_bins = symbol_bins
        self.on_hard = on_hard


def build(name: str, tau: float = 3.0, n_bins: int = 12) -> Distortion:
    if name == "hamming":
        return Distortion("hamming", HammingLoss(),
                          lambda p, t: hamming_distortion(p, t),
                          "P(bit error)", 1, True)
    if name == "count_mse":
        return Distortion("count_mse", CountMSELoss(),
                          lambda p, t: float(count_mse(p, t)),
                          "spikes^2 per neuron", n_bins, False)
    if name == "van_rossum":
        return Distortion("van_rossum", VanRossumLoss(tau),
                          lambda p, t: float(van_rossum_sq(p, t, tau)),
                          "mean sq. PSP error", n_bins, False)
    raise ValueError(f"unknown distortion: {name}")


NAMES = ("hamming", "count_mse", "van_rossum")
