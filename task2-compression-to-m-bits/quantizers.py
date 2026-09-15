"""
Quantizers for discretizing the SNN encoder's continuous output.

Two constructions, both operating on an n_dims-dimensional continuous vector
and matching the project's Z in {0,...,alphabet_size-1}^n_symbols convention
(default n_symbols=8, alphabet_size=16):

  - GridQuantizer:      independent per-coordinate uniform scalar quantizer.
                        Simplest option -- each coordinate snapped to its
                        nearest of `levels` grid points, ignoring
                        correlations across the n_dims coordinates.

  - E8LatticeQuantizer: vector quantizer onto (a scaled copy of) the E8
                        lattice, the densest known sphere packing in 8
                        dimensions. Decoding exploits correlations across
                        the 8 coordinates that GridQuantizer ignores, so at
                        the same rate it can reach lower distortion.

Both share the same interface so a pipeline can swap between them:

    z_q, idx = quantizer(z)              # hard decision, for eval/reconstruction
    soft = quantizer.soft_onehot(z, tau) # differentiable, for training the rate term

  z    : (B, n_dims) continuous encoder output
  z_q  : (B, n_dims) dequantized value; equals the hard quantized value on
         the forward pass, straight-through gradient on the backward pass
  idx  : (B, n_dims) long, one symbol index per coordinate in
         [0, quantizer.alphabet_size) -- feed F.one_hot(idx, alphabet_size)
         into an entropy model (e.g. FactorizedEntropyModel in
         new_pipeline.py) for the hard/eval rate
  soft : (B, n_dims, alphabet_size) soft per-coordinate assignment, used in
         place of the hard one-hot during training so the rate term has a
         gradient back into the encoder (same straight-through trick as the
         Gumbel-softmax path elsewhere in the pipeline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _straight_through(soft, hard):
    return soft + (hard - soft).detach()


# ----------------------------------------------------------------------
# Simple per-coordinate grid quantizer: Z in {0, ..., levels-1}^n_dims
# ----------------------------------------------------------------------

class GridQuantizer(nn.Module):
    """
    Uniform scalar quantizer: each of `n_dims` coordinates independently
    snapped to the nearest of `levels` evenly spaced points in
    [-value_range, value_range]. Reproduces the classical
    Z in {0, ..., levels-1}^n_dims code.
    """

    def __init__(self, n_dims=8, levels=16, value_range=1.0):
        super().__init__()
        self.n_dims = n_dims
        self.levels = levels
        self.alphabet_size = levels
        self.value_range = value_range
        self.step = 2 * value_range / (levels - 1)
        self.register_buffer("grid", torch.linspace(-value_range, value_range, levels))

    def forward(self, z):
        """z: (B, n_dims) -> z_q: (B, n_dims), idx: (B, n_dims) long in [0, levels)"""
        z_clamped = z.clamp(-self.value_range, self.value_range)
        idx = torch.round((z_clamped + self.value_range) / self.step).long()
        idx = idx.clamp(0, self.levels - 1)
        z_hard = self.grid[idx]
        z_q = _straight_through(z, z_hard)
        return z_q, idx

    def soft_onehot(self, z, tau=1.0):
        """(B, n_dims) -> (B, n_dims, levels) softmax over squared distance to each grid point."""
        dist2 = (z.unsqueeze(-1) - self.grid.view(1, 1, -1)) ** 2
        return F.softmax(-dist2 / tau, dim=-1)


# ----------------------------------------------------------------------
# E8 lattice vector quantizer
# ----------------------------------------------------------------------

def _decode_Dn(y):
    """
    Nearest point of D_n = {x in Z^n : sum(x_i) even} to y, batched over
    all leading dims (..., n). Standard construction (Conway & Sloane,
    "Fast Quantizing and Decoding Algorithms for Lattice Quantizers and
    Codes", 1982): round to the nearest integer point; if the coordinate
    sum has the wrong parity, flip the coordinate with the largest
    rounding error to the next-nearest integer.
    """
    x = torch.round(y)
    err = (y - x).abs()
    worst_idx = err.argmax(dim=-1)                        # (...,)
    parity_odd = (x.sum(dim=-1).long() % 2 != 0)           # (...,)

    delta = torch.sign(y - x)
    delta = torch.where(delta == 0, torch.ones_like(delta), delta)

    flat_shape = (-1, x.shape[-1])
    flat_x = x.reshape(*flat_shape).clone()
    flat_delta = delta.reshape(*flat_shape)
    flat_worst = worst_idx.reshape(-1)
    flat_fix = parity_odd.reshape(-1).float()

    rows = torch.arange(flat_x.shape[0], device=x.device)
    sel_delta = flat_delta[rows, flat_worst]
    flat_x[rows, flat_worst] += sel_delta * flat_fix

    return flat_x.view_as(x)


def _decode_E8(y):
    """
    Nearest point of E8 = D8 U (D8 + (1/2,...,1/2)) to y, batched (..., 8).
    Decode into each of the two cosets (as a D8 problem) and keep whichever
    candidate is closer.
    """
    half = 0.5
    y0 = _decode_Dn(y)
    y1 = _decode_Dn(y - half) + half
    d0 = ((y - y0) ** 2).sum(dim=-1, keepdim=True)
    d1 = ((y - y1) ** 2).sum(dim=-1, keepdim=True)
    return torch.where(d1 < d0, y1, y0)


class E8LatticeQuantizer(nn.Module):
    """
    Vector quantizer onto a scaled copy of the E8 lattice, the densest
    known packing in 8 dimensions. `n_dims` must be a multiple of 8; a
    longer vector is quantized as independent 8-dim blocks.

    `levels` sets the per-coordinate resolution the same way it does for
    GridQuantizer (an E8 point can land on an integer or a half-integer
    coordinate, so the entropy alphabet spans 2*levels-1 bins per
    coordinate at the same nominal step size).
    """

    def __init__(self, n_dims=8, levels=16, value_range=1.0):
        super().__init__()
        if n_dims % 8 != 0:
            raise ValueError("E8LatticeQuantizer requires n_dims to be a multiple of 8")
        self.n_dims = n_dims
        self.n_blocks = n_dims // 8
        self.levels = levels
        self.alphabet_size = 2 * levels - 1
        self.value_range = value_range
        self.scale = 2 * value_range / (levels - 1)   # lattice spacing in z-units

        half_span = self.alphabet_size // 2
        self.register_buffer(
            "half_grid", torch.arange(-half_span, half_span + 1) * 0.5 * self.scale)

    def forward(self, z):
        """z: (B, n_dims) -> z_q: (B, n_dims), idx: (B, n_dims) long in [0, alphabet_size)"""
        B = z.shape[0]
        z_blocks = z.view(B, self.n_blocks, 8) / self.scale
        q_blocks = _decode_E8(z_blocks)                      # in Z or Z+1/2
        z_hard = (q_blocks * self.scale).view(B, self.n_dims)
        z_q = _straight_through(z, z_hard)

        half_span = self.alphabet_size // 2
        idx = torch.round(q_blocks.view(B, self.n_dims) * 2).long() + half_span
        idx = idx.clamp(0, self.alphabet_size - 1)
        return z_q, idx

    def soft_onehot(self, z, tau=1.0):
        """(B, n_dims) -> (B, n_dims, alphabet_size), per-coordinate soft assignment
        over the half-integer-spaced candidate grid (an approximation that ignores
        the D8/E8 parity constraint across coordinates -- see module docstring)."""
        dist2 = (z.unsqueeze(-1) - self.half_grid.view(1, 1, -1)) ** 2
        return F.softmax(-dist2 / tau, dim=-1)


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("GridQuantizer:")
    gq = GridQuantizer(n_dims=8, levels=16)
    z = torch.randn(4, 8, requires_grad=True)
    z_q, idx = gq(z)
    assert z_q.shape == z.shape and idx.shape == z.shape
    assert idx.min() >= 0 and idx.max() < gq.levels
    soft = gq.soft_onehot(z)
    assert soft.shape == (4, 8, 16) and torch.allclose(soft.sum(-1), torch.ones(4, 8))
    z_q.sum().backward()
    assert z.grad is not None
    print(f"  idx range [{idx.min().item()}, {idx.max().item()}], gradient OK")

    print("D8/E8 decoding:")
    y = torch.randn(1000, 8) * 2
    d8 = _decode_Dn(y)
    parity = d8.sum(dim=-1).round().long() % 2
    assert (parity == 0).all(), "D8 decode produced odd-sum points"
    print("  D8 parity check passed on 1000 random points")

    e8 = _decode_E8(y)
    e8_parity_ok = (e8.sum(dim=-1) * 2).round().long() % 4 == 0  # covers both cosets
    assert e8_parity_ok.all(), "E8 decode produced a point outside E8"
    d_e8 = ((y - e8) ** 2).sum(-1)
    # Known invariant: E8's covering radius is 1 (min vector norm sqrt(2)), so
    # every point in R^8 must be within squared distance 1 of some E8 point.
    assert (d_e8 <= 1.0 + 1e-4).all(), f"decode exceeded E8's covering radius: max sq dist {d_e8.max():.4f}"
    print(f"  mean sq. error: {d_e8.mean():.4f} (covering-radius bound: 1.0, max observed {d_e8.max():.4f})")

    print("\nE8LatticeQuantizer:")
    eq = E8LatticeQuantizer(n_dims=8, levels=16)
    z2 = torch.randn(4, 8, requires_grad=True)
    z2_q, idx2 = eq(z2)
    assert idx2.min() >= 0 and idx2.max() < eq.alphabet_size
    soft2 = eq.soft_onehot(z2)
    assert soft2.shape == (4, 8, eq.alphabet_size)
    z2_q.sum().backward()
    assert z2.grad is not None
    print(f"  alphabet_size={eq.alphabet_size}, idx range [{idx2.min().item()}, {idx2.max().item()}], gradient OK")

    print("\nAll self-tests passed.")
