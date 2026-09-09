"""Reconstruction loss for a sparse binary spike train.

At T=16 bins N-MNIST is ~7% ones, so a plain BCE is happy to predict silence
everywhere. Four terms, each fixing a different failure of the others:

* `bce`   binary cross-entropy. Dense, well-conditioned gradients on every
          cell, including the 93% that should stay silent. `pos_weight`
          defaults to 1: the dice term below already stops the model from
          collapsing to silence, and up-weighting positives on top of it just
          buys recall with precision (measured at pos_weight=4: recall 0.62,
          precision 0.28).
* `dice`  a soft F1. Scale-free and driven entirely by the spikes that exist, so
          it cannot be satisfied by predicting silence, and it balances
          precision against recall without needing the pos_weight tuned.
* `psp`   MSE between exponentially filtered trains, i.e. a soft van Rossum
          distance. Rewards getting the *timing envelope* right even when
          individual bins are off by one, which the two cell-wise terms above
          punish as hard errors.
* `frame` MSE on the time-summed frame. Anchors the spatial layout of the digit,
          which is the part a reader actually looks at.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .snn import psp_filter


class SpikeReconstructionLoss(torch.nn.Module):
    def __init__(self, pos_weight: float = 1.0, w_bce: float = 1.0,
                 w_dice: float = 1.0, w_psp: float = 20.0, w_frame: float = 0.3,
                 tau: float = 3.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.w_bce, self.w_dice = w_bce, w_dice
        self.w_psp, self.w_frame = w_psp, w_frame
        self.tau = tau

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple:
        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )

        p, t = probs.flatten(1), target.flatten(1)
        dice = 1.0 - (2 * (p * t).sum(1) + 1.0) / (p.sum(1) + t.sum(1) + 1.0)
        dice = dice.mean()

        psp = F.mse_loss(psp_filter(probs, self.tau), psp_filter(target, self.tau))

        frame = F.mse_loss(probs.sum(1), target.sum(1))

        total = (self.w_bce * bce + self.w_dice * dice
                 + self.w_psp * psp + self.w_frame * frame)
        parts = {"bce": bce.item(), "dice": dice.item(),
                 "psp": psp.item(), "frame": frame.item()}
        return total, parts


# Measured term magnitudes at a partly-trained m=128 model, used to set the
# default weights so no term is decorative:
#   bce 0.145 | dice 0.674 | psp 0.010 | frame 0.983
# giving contributions 0.145 / 0.674 / 0.200 / 0.295.
