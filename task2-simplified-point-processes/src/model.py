"""The learned codec: X -> encoder -> Z in {0..A-1}^L -> decoder -> N_hat(t).

    events --bin--> (B, M) --encoder--> raw code --quantizer--> z, h, bits
           h --CountingDecoder--> soft counting curve (B, K)

Two quantizers share everything else (`cfg.quantizer`):

* `onehot`  L x A logits, hard one-hot argmax with a softmax straight-through gradient
* `scalar`  L bounded values rounded to A ordinal levels (see `ScalarBottleneck`)

The prior over z lives here too, so one state dict holds everything the sender
and the receiver must share.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck import CategoricalBottleneck, FactorizedPrior, ScalarBottleneck
from .decoder import CountingDecoder
from .encoders import SNNEncoder, matched_ann
from .sources import Events


class Codec(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_input_bins = cfg.n_input_bins
        self.tau = cfg.tau
        self.quantizer = cfg.quantizer
        self.n_latents, self.alphabet = cfg.n_latents, cfg.alphabet

        if cfg.quantizer == "onehot":
            n_out = cfg.n_latents * cfg.alphabet
            self.bottleneck = CategoricalBottleneck(cfg.n_latents, cfg.alphabet, cfg.embed_dim)
        elif cfg.quantizer == "scalar":
            n_out = cfg.n_latents
            # getattr: checkpoints from before this option have no such field
            self.bottleneck = ScalarBottleneck(cfg.n_latents, cfg.alphabet,
                                               getattr(cfg, "level_embed_dim", 0))
        else:
            raise ValueError(f"unknown quantizer {cfg.quantizer!r}")

        # getattr defaults reproduce checkpoints from before these options; both only
        # affect initialisation and the backward pass, so evaluation is unchanged
        snn = SNNEncoder(cfg.hidden, n_out, getattr(cfg, "snn_beta", 0.9),
                         getattr(cfg, "rec_grad_scale", 0.0))
        if cfg.encoder == "snn":
            self.encoder = snn
        elif cfg.encoder == "ann":
            self.encoder = matched_ann(snn, n_out)
        else:
            raise ValueError(f"unknown encoder {cfg.encoder!r}")

        self.prior = FactorizedPrior(cfg.n_latents, cfg.alphabet)
        self.decoder = CountingDecoder(self.bottleneck.out_dim, cfg.n_grid, cfg.dec_hidden,
                                       init_total=cfg.rate * cfg.T)

    def forward(self, ev: Events, count_spikes: bool = False, encoder=None):
        """-> hard one-hot z (B, L, A), soft counting curve (B, K),
        differentiable bits (B,), spikes per batch or None.

        `encoder` overrides `self.encoder` for this call; training passes a
        torch.compile'd view of the same module (shared parameters), kept out of
        the module tree so the state dict is unchanged.
        """
        x = ev.binned(self.n_input_bins)
        spikes = None
        if count_spikes and isinstance(self.encoder, SNNEncoder):
            raw, per_layer = self.encoder(x, return_spikes=True)
            spikes = [float(s) for s in per_layer]
        else:
            # `is not None`, not `or`: truth-testing a compiled module calls len() and raises
            raw = (encoder if encoder is not None else self.encoder)(x)

        if self.quantizer == "onehot":
            z, h = self.bottleneck(raw.view(-1, self.n_latents, self.alphabet), self.tau)
            bits = self.prior.bits(z)
        else:
            z, h, bits = self.bottleneck(raw, self.prior)
        return z, self.decoder(h), bits, spikes

    def decode_symbols(self, sym: torch.Tensor) -> torch.Tensor:
        """Receiver side alone: (B, L) integer symbols -> soft counting curve (B, K).

        Used to probe the decoder with codes the encoder never produced
        (coordinate ablations, latent traversals).
        """
        if self.quantizer == "onehot":
            h = self.bottleneck.embed_code(F.one_hot(sym, self.alphabet).float())
        else:
            h = self.bottleneck.embed_symbols(sym)
        return self.decoder(h)
