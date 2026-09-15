"""Discrete latent Z in {0..A-1}^L and its learned probability model.

Quantization: the encoder emits logits (B, L, A); the forward pass takes the
hard one-hot argmax, the backward pass uses the softmax gradient (straight-
through). Z is therefore genuinely discrete at every step, train and test.

Rate: `FactorizedPrior` holds one categorical per coordinate, p(z) = prod_i
p_i(z_i). The ideal code length of a latent is -log2 p(z) bits; an arithmetic
coder driven by the same probabilities achieves it to within a small constant
per stream (`coding.py` measures that). Its gradient pushes the encoder toward
fewer, more predictable symbols.

The prior itself is *estimated*, not learned by gradient: for a fixed encoder
the rate term is minimized exactly by the empirical symbol frequencies, so the
prior tracks them with an exponential moving average. A gradient-trained prior
was tried first and lagged far behind the encoder: with every coordinate
collapsed to one constant symbol it still charged ~0.9 bits per coordinate.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def straight_through_onehot(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    soft = F.softmax(logits / tau, dim=-1)
    hard = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
    return hard + soft - soft.detach()


class CategoricalBottleneck(nn.Module):
    """One-hot quantizer plus a learned embedding per (coordinate, symbol)."""

    def __init__(self, n_latents: int = 8, alphabet: int = 16, embed_dim: int = 16):
        super().__init__()
        self.n_latents, self.alphabet, self.embed_dim = n_latents, alphabet, embed_dim
        self.embed = nn.Parameter(torch.randn(n_latents, alphabet, embed_dim) * 0.5)

    @property
    def out_dim(self) -> int:
        return self.n_latents * self.embed_dim

    def embed_code(self, z: torch.Tensor) -> torch.Tensor:
        """(B, L, A) one-hot -> (B, L * embed_dim) decoder input."""
        return torch.einsum("bla,lad->bld", z, self.embed).flatten(1)

    def forward(self, logits: torch.Tensor, tau: float = 1.0):
        z = straight_through_onehot(logits, tau)
        return z, self.embed_code(z)


class ScalarBottleneck(nn.Module):
    """Ordinal quantizer: one bounded scalar per coordinate, rounded to A levels.

        v = (A - 1) * sigmoid(y)   in [0, A-1]
        symbol = round(v)          (straight-through: forward rounds, backward is identity)

    Unlike the one-hot quantizer, neighbouring symbols are neighbouring values,
    so the encoder's gradient says which way to move. With one-hot argmax the 16
    symbols are unordered, and a GRU encoder trained on distortion alone only
    reached D ~ 0.65-1.6 in 1000 steps where an oracle code gave 0.15.

    The rate must stay differentiable in v, so during training the bit cost is
    interpolated linearly between the two levels around v; at a level it equals
    the exact -log2 p.

    Decoder input, `embed_dim`:

    * 0      the rounded values rescaled to [-1, 1] (one number per coordinate);
    * d > 0  a learned d-vector per (coordinate, level). The forward pass uses the
             exact vector of the rounded level; the encoder's gradient comes from
             the linear interpolation between the two neighbouring levels
             (straight-through, same trick as the rate), and the table itself is
             trained only through the levels actually used.

    Why the embedding: an MLP decoder struggles to turn a single number into
    sharp counting steps whose positions jump with every level. Given a perfect
    hand-built 8x64 event-time code, a decoder on raw numbers reached D 0.102
    (width 256) or 0.073 (width 1024) where decoding the symbols by hand gives
    0.038; with 32-d level embeddings it reached 0.055 (width 256) and 0.040
    (width 1024).
    """

    def __init__(self, n_latents: int = 8, alphabet: int = 16, embed_dim: int = 0):
        super().__init__()
        self.n_latents, self.alphabet, self.embed_dim = n_latents, alphabet, embed_dim
        if embed_dim:
            self.table = nn.Parameter(torch.randn(n_latents, alphabet, embed_dim) * 0.5)

    @property
    def out_dim(self) -> int:
        return self.n_latents * self.embed_dim if self.embed_dim else self.n_latents

    def embed_symbols(self, sym: torch.Tensor) -> torch.Tensor:
        """(B, L) integer symbols -> decoder input (B, out_dim), no encoder gradient."""
        if self.embed_dim:
            idx = torch.arange(self.n_latents, device=sym.device)
            return self.table[idx, sym].flatten(1)
        return sym.float() / (self.alphabet - 1) * 2 - 1

    def forward(self, y: torch.Tensor, prior: "FactorizedPrior"):
        """y: (B, L) -> hard one-hot z (B, L, A), decoder input (B, out_dim), bits (B,)."""
        A = self.alphabet
        v = (A - 1) * torch.sigmoid(y)
        sym = v.detach().round().long()
        z = F.one_hot(sym, A).to(v.dtype)

        nbits = -prior.probs().log2()                            # (L, A)
        lo = v.detach().floor().clamp(max=A - 2).long()          # (B, L)
        frac = v - lo.to(v.dtype)                                # gradient flows here
        idx = torch.arange(self.n_latents, device=y.device)
        bits = ((1 - frac) * nbits[idx, lo] + frac * nbits[idx, lo + 1]).sum(1)

        if self.embed_dim:
            hard = self.table[idx, sym]                                          # (B, L, d)
            w = frac.unsqueeze(-1)
            soft = (1 - w) * self.table[idx, lo].detach() + w * self.table[idx, lo + 1].detach()
            h = (hard + soft - soft.detach()).flatten(1)
        else:
            q = v + (v.round() - v).detach()
            h = q / (A - 1) * 2 - 1
        return z, h, bits


class FactorizedPrior(nn.Module):
    """p(z) = prod_i Categorical(z_i; freq_i), freq an EMA of symbol usage.

    `floor` keeps every symbol codable (the range coder needs p > 0) and bounds
    the cost of a symbol the encoder has stopped using to -log2(floor) bits.
    """

    def __init__(self, n_latents: int = 8, alphabet: int = 16,
                 momentum: float = 0.99, floor: float = 1e-6):
        super().__init__()
        self.momentum, self.floor = momentum, floor
        self.register_buffer("freq", torch.full((n_latents, alphabet), 1.0 / alphabet))

    @torch.no_grad()
    def update(self, z: torch.Tensor):
        """Fold a training batch of one-hot latents (B, L, A) into the estimate."""
        self.freq.mul_(self.momentum).add_(z.detach().mean(0), alpha=1.0 - self.momentum)

    def probs(self) -> torch.Tensor:
        p = self.freq.clamp(min=self.floor)
        return p / p.sum(-1, keepdim=True)

    def bits(self, z: torch.Tensor) -> torch.Tensor:
        """Ideal code length of one-hot latents (B, L, A) -> (B,) bits."""
        return -(z * self.probs().log()).sum((1, 2)) / math.log(2.0)


def symbols(z: torch.Tensor) -> torch.Tensor:
    """(B, L, A) one-hot -> (B, L) integer symbols."""
    return z.argmax(-1)


def symbol_entropy_bits(sym: torch.Tensor, alphabet: int) -> torch.Tensor:
    """Empirical entropy of each coordinate's symbol usage -> (L,) bits."""
    counts = F.one_hot(sym, alphabet).sum(0).double()
    p = counts / counts.sum(-1, keepdim=True)
    return -(p * torch.where(p > 0, p.log2(), torch.zeros_like(p))).sum(-1)
