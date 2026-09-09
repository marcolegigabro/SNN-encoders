"""Distortion measures for reconstructed spike trains.

There is no single right distortion for a spike train, so we report three
families and let them disagree:

* Cell-wise agreement (precision / recall / F1 / IoU) on the binarised
  reconstruction. Strict: a spike one bin early counts as two errors.
* van Rossum distance, the standard spike-train metric: both trains are
  convolved with an exponential PSP kernel and compared in L2, so small timing
  jitter is penalised smoothly instead of catastrophically.
* Image-domain error on the time-summed frame. N-MNIST integrated over time is
  an ordinary grayscale digit, so this is the one a human can eyeball, and it is
  what "does the reconstruction still look like the digit" means. Correlation,
  not PSNR: see `image_metrics`.

A warning that shapes how these should be read. Measured on this data, a
*zero-bit* baseline that always emits the dataset-average spike train scores
F1 = 0.380, which is roughly what a trained m=128 model scores too. Cell-wise
agreement is dominated by "spikes land where digits generally have strokes" and
barely separates a real codec from a constant one. Rank models by `frame_corr`
and by the probes in `scripts/probe.py`, and quote F1 only next to the
baselines in `scripts/baselines.py`.

The code itself is also measured: a bit that is constant across the test set
carries no information, so the entropy of the code tells us how many of the m
bits are actually being spent.
"""

from __future__ import annotations

import torch

from .snn import psp_filter


def _flat(x):
    return x.flatten(1)


def spike_agreement(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Cell-wise precision/recall/F1/IoU on binary tensors, averaged per sample."""
    p, t = _flat(pred), _flat(target)
    tp = (p * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    fn = ((1 - p) * t).sum(1)
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "iou": iou.mean().item(),
    }


def van_rossum(pred: torch.Tensor, target: torch.Tensor, tau: float = 3.0) -> float:
    """Normalised van Rossum distance, ||PSP(x_hat) - PSP(x)|| / ||PSP(x)||.

    1.0 is what an all-silent reconstruction scores, so values below 1 mean the
    reconstruction beats predicting nothing.
    """
    fp, ft = psp_filter(pred, tau), psp_filter(target, tau)
    num = (fp - ft).flatten(1).norm(dim=1)
    den = ft.flatten(1).norm(dim=1) + 1e-8
    return (num / den).mean().item()


def accumulated_frame(x: torch.Tensor) -> torch.Tensor:
    """(B,T,2,H,W) -> (B,H,W): events summed over time and polarity, the
    grayscale digit an event camera integrates to."""
    return x.sum(dim=(1, 2))


def image_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Image-domain error on the time-summed frame.

    Two of these three need care on data this sparse:

    * `frame_corr` is the headline number. Pearson correlation between the two
      accumulated frames, per sample. It is invariant to the reconstruction's
      overall intensity, so it answers "is this the same digit, in the same
      place, with the same strokes" and nothing else. A silent or constant
      reconstruction scores 0 by construction, and the dataset-mean baseline
      scores only what the average digit happens to share with this one.
    * `frame_nmse` is MSE against the target, divided by the MSE a silent
      reconstruction would get. Below 1 means the codec beats predicting
      nothing; above 1 means it is worse than silence.
    * `psnr_db` is kept for comparability with the image-generation literature,
      but it is close to useless here and is not used to rank models: at 7%
      occupancy the all-zero reconstruction scores 11.8 dB, better than any
      real reconstruction, because peak-signal-to-noise rewards silence on
      sparse data.
    """
    a, b = accumulated_frame(pred).flatten(1), accumulated_frame(target).flatten(1)

    ac = a - a.mean(1, keepdim=True)
    bc = b - b.mean(1, keepdim=True)
    corr = (ac * bc).sum(1) / (ac.norm(dim=1) * bc.norm(dim=1) + 1e-8)

    mse = ((a - b) ** 2).mean(1)
    mse_silent = (b ** 2).mean(1) + 1e-8

    peak = b.max(dim=1).values.clamp(min=1.0)
    psnr = 10 * torch.log10(peak ** 2 / (mse + 1e-8))

    return {
        "frame_corr": corr.mean().item(),
        "frame_nmse": (mse / mse_silent).mean().item(),
        "psnr_db": psnr.mean().item(),
    }


def code_usage(codes: torch.Tensor) -> dict:
    """How much of the m-bit budget the encoder actually spends.

    `entropy_bits` is the sum of the marginal per-bit entropies: an upper bound
    on the code's true entropy that ignores inter-bit dependence, but it catches
    the failure mode that matters here, bits stuck at a constant value.
    """
    p = codes.float().mean(0).clamp(1e-6, 1 - 1e-6)
    bit_entropy = -(p * p.log2() + (1 - p) * (1 - p).log2())
    return {
        "code_entropy_bits": bit_entropy.sum().item(),
        "code_bits_used": (bit_entropy > 0.05).sum().item(),
        "code_mean_on": p.mean().item(),
    }


def pick_threshold(probs: torch.Tensor, target: torch.Tensor,
                   grid=None) -> tuple[float, float]:
    """Choose the probability threshold that maximises spike F1.

    The decoder emits Bernoulli probabilities; turning them into a spike train
    needs a threshold, and 0.5 is not it when spikes occupy only 7% of the
    tensor. Picked on training data, applied unchanged at test time.
    """
    if grid is None:
        grid = torch.linspace(0.05, 0.95, 37)
    best_f1, best_th = -1.0, 0.5
    for th in grid:
        f1 = spike_agreement((probs > th).float(), target)["f1"]
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    return best_th, best_f1
