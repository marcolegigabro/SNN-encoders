"""
Lattice-Transform-Coding SNN Autoencoder  (task 2, LTC variant)
==============================================================

Same task as ``autoencoder.py`` -- compress a spike train x_{1:T} into a
compact code, reconstruct it with an SNN decoder -- but the scalar STE
bottleneck is replaced by the pipeline from

    Lei, Hassani & Saeedi Bidokhti, "Approaching Rate-Distortion Limits in
    Neural Compression with Lattice Transform Coding", ICLR 2025.

Pipeline
--------
    x_{1:T}  --[ SNN encoder (LIF stack) + analysis MLP g_a ]-->  y in R^{dy}
             --[ lattice quantizer  Q_Lambda  (blockwise, product lattice) ]--> y_hat
             --[ entropy model  p_yhat  ->  rate = E[-log2 p_yhat(y_hat)] ]
             --[ SNN decoder (autoregressive LIF, real-valued context) ]-->  x_hat_{1:T}

`y_hat` (a point on the chosen lattice) is "the code"; the entropy model's
cross-entropy is its description length in bits -- exactly how the LTC
paper reports rate (Sec. 5, "rates ... are given by the cross-entropy of
the density model"). An actual range/rANS coder would turn p_yhat into the
transmitted bitstream; it is not needed to train or to measure R-D.

Two bottleneck modes (``--quantizer``):

  entropy   variable-rate LTC (paper Sec. 4.1). Full infinite lattice +
            learned joint density p_y; loss = lambda * distortion + rate.
            Sweep lambda for the R-D curve. ``--lattice {Zn,Dn,Dn_star,A2,E8}``.
            ``--quantizer scalar`` is the NTC baseline: this exact path with
            the n=1 integer lattice Z (== per-dim scalar quantization).

  nested    fixed-rate LTC with nested-lattice quantization (paper Sec. 4.2).
            y_hat = Q_Lf(y) - Q_Lc(Q_Lf(y)),  Lc = Gamma * Lf (self-similar).
            No entropy coder: the rate is fixed at  dy * log2(Gamma)  bits.
            Sweep Gamma for the R-D curve.

Backprop through the non-differentiable quantizer: STE (paper Eq. 4) or
dithered / additive-uniform-noise over the Voronoi cell (paper Eq. 6),
selected with ``--quant-backward {ste,dither}``.

Shared SNN primitives (LIF cell, PSP distortion, spike sources, the
autoregressive decoder, the Bernoulli R(D) anchor) are imported from
``autoencoder.py`` -- see the README note about deliberate duplication.

    python ltc_autoencoder.py --help
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import (
    LIFCell,
    SNNDecoder,
    psp_mse,
    hamming_distortion,
    binary_entropy,
    bernoulli_rate_distortion,
    make_source,
    SYNTH_NAMES,
    REAL_NAMES,
)


# ----------------------------------------------------------------------
# 1. Lattice closest-point (CVP) routines
#    Each returns the closest point in the *canonical* lattice (the one
#    whose generator is `_lattice_generator` below); unit-volume scaling
#    is applied by LatticeCVP so different lattices are comparable.
# ----------------------------------------------------------------------

def _cvp_Zn(z, n):
    return torch.round(z)


def _cvp_Dn(z, n):
    """Closest point of D_n = {x in Z^n : sum(x) even}  (Conway & Sloane, Ch. 20)."""
    f = torch.round(z)
    diff = z - f
    odd = torch.remainder(f.sum(dim=-1), 2.0) > 0.5
    j = torch.argmax(diff.abs(), dim=-1, keepdim=True)
    step = torch.where(torch.gather(diff, -1, j) >= 0, 1.0, -1.0)
    g = f.clone()
    g.scatter_(-1, j, torch.gather(f, -1, j) + step)
    return torch.where(odd.unsqueeze(-1), g, f)


def _cvp_Dn_star(z, n):
    """D_n^* = Z^n  U  (Z^n + 1/2): pick the closer coset."""
    a = torch.round(z)
    b = torch.round(z - 0.5) + 0.5
    da = ((z - a) ** 2).sum(-1, keepdim=True)
    db = ((z - b) ** 2).sum(-1, keepdim=True)
    return torch.where(da <= db, a, b)


def _cvp_E8(z, n):
    """E8 = D8  U  (D8 + 1/2): pick the closer coset."""
    a = _cvp_Dn(z, 8)
    b = _cvp_Dn(z - 0.5, 8) + 0.5
    da = ((z - a) ** 2).sum(-1, keepdim=True)
    db = ((z - b) ** 2).sum(-1, keepdim=True)
    return torch.where(da <= db, a, b)


_A2_BASIS = [[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]]


def _cvp_A2(z, n):
    """Closest point of the 2-D hexagonal lattice (reduced 60-degree basis)."""
    basis = z.new_tensor(_A2_BASIS)                 # rows = basis vectors
    coords = z @ torch.linalg.inv(basis)           # point = coords @ basis
    base = torch.floor(coords)
    best = None
    best_d = None
    for du in (0.0, 1.0):
        for dv in (0.0, 1.0):
            cand = (base + z.new_tensor([du, dv])) @ basis
            d = ((z - cand) ** 2).sum(-1, keepdim=True)
            if best is None:
                best, best_d = cand, d
            else:
                take = d < best_d
                best = torch.where(take, cand, best)
                best_d = torch.where(take, d, best_d)
    return best


_CVP = {
    "Zn": _cvp_Zn,
    "Dn": _cvp_Dn,
    "Dn_star": _cvp_Dn_star,
    "A2": _cvp_A2,
    "E8": _cvp_E8,
}
_FIXED_DIM = {"A2": 2, "E8": 8}


def _lattice_generator(name: str, n: int) -> torch.Tensor:
    """Generator matrix (rows = basis vectors) of the canonical lattice."""
    if name == "Zn":
        return torch.eye(n)
    if name == "Dn":
        g = torch.zeros(n, n)
        for i in range(n - 1):
            g[i, i] = 1.0
            g[i, i + 1] = -1.0
        g[n - 1, n - 2] = 1.0
        g[n - 1, n - 1] = 1.0
        return g
    if name == "Dn_star":
        g = torch.eye(n)
        g[n - 1, :] = 0.5
        return g
    if name == "A2":
        return torch.tensor(_A2_BASIS)
    if name == "E8":
        g = torch.zeros(8, 8)
        g[0, 0] = 2.0
        for i in range(1, 7):
            g[i, i - 1] = -1.0
            g[i, i] = 1.0
        g[7, :] = 0.5
        return g
    raise ValueError(f"unknown lattice {name!r}")


class LatticeCVP(nn.Module):
    """
    Closest-point search + Voronoi-cell sampling for a lattice, rescaled so
    its fundamental cell has unit volume (paper assumes det(G G^T) = 1 so
    rates/distortions are comparable across lattices).
    """

    def __init__(self, name: str, n: int):
        super().__init__()
        if name in _FIXED_DIM and n != _FIXED_DIM[name]:
            raise ValueError(f"lattice {name} requires n={_FIXED_DIM[name]}, got n={n}")
        self.name = name
        self.n = n
        g = _lattice_generator(name, n)
        covol = float(abs(torch.det(g)))
        scale = covol ** (1.0 / n)
        self.register_buffer("gen_unit", g / scale)     # generates the unit-vol lattice
        self.register_buffer("scale", torch.tensor(scale))

    def _cvp_canon(self, z):
        return _CVP[self.name](z, self.n)

    def closest(self, y):
        """Closest point of the unit-volume lattice, in y-space. No gradient."""
        return self._cvp_canon(y * self.scale) / self.scale

    def sample_voronoi(self, shape, device):
        """Uniform samples over V(0) of the unit-volume lattice (Conway & Sloane 1984)."""
        s = torch.rand(*shape, self.n, device=device)
        u_tilde = s @ self.gen_unit.to(device)
        return u_tilde - self.closest(u_tilde)


# ----------------------------------------------------------------------
# 2. Bottlenecks
# ----------------------------------------------------------------------

class EntropyLatticeBottleneck(nn.Module):
    """
    Variable-rate LTC (paper Sec. 4.1). The latent y in R^{dy} is split
    into dy/n blocks, each quantized with the n-D lattice (a product
    lattice along the latent, as the paper does for image channels). A
    learned joint Gaussian p_y gives the rate via a Monte-Carlo estimate
    of the cell probability (paper Eqs. 5-6):

        p_yhat(yhat) = E_{u ~ Unif(V(0))}[ p_y(yhat + u) ]
        rate(bits)   = -log2 p_yhat(yhat)      (summed over the dy vector)
    """

    def __init__(self, lattice, n, latent_dim, backward="ste",
                 entropy="diag", mc_samples=256):
        super().__init__()
        if latent_dim % n != 0:
            raise ValueError(f"latent_dim {latent_dim} not divisible by lattice dim {n}")
        self.q = LatticeCVP(lattice, n)
        self.n = n
        self.blocks = latent_dim // n
        self.latent_dim = latent_dim
        self.backward = backward
        self.entropy = entropy
        self.mc_samples = mc_samples

        self.mu = nn.Parameter(torch.zeros(latent_dim))
        self.log_scale = nn.Parameter(torch.zeros(latent_dim))
        if entropy == "full":
            self.tril = nn.Parameter(torch.zeros(latent_dim, latent_dim))

    # --- joint density p_y -------------------------------------------------
    def _dist(self):
        diag = F.softplus(self.log_scale) + 1e-4
        if self.entropy == "full":
            L = torch.tril(self.tril, -1) + torch.diag(diag)
            return torch.distributions.MultivariateNormal(self.mu, scale_tril=L)
        return torch.distributions.Independent(
            torch.distributions.Normal(self.mu, diag), 1
        )

    def _dither(self, lead_shape, device):
        blk = self.q.sample_voronoi(tuple(lead_shape) + (self.blocks,), device)
        return blk.reshape(*lead_shape, self.latent_dim)

    def _rate_bits(self, point):
        """-log2 E_u[p_y(point + u)] via Monte-Carlo, per sample -> (B,)."""
        u = self._dither((self.mc_samples, point.shape[0]), point.device)
        logp = self._dist().log_prob(point.unsqueeze(0) + u)        # (mc, B)
        log_cell = torch.logsumexp(logp, dim=0) - math.log(self.mc_samples)
        return -log_cell / math.log(2.0)

    def forward(self, y):
        B = y.shape[0]
        q_hard = self.q.closest(y.view(B, self.blocks, self.n)).view(B, self.latent_dim)

        if self.backward == "dither" and self.training:
            point = y + self._dither((B,), y.device)               # continuous relaxation
            y_hat = point
        else:
            y_hat = y + (q_hard - y).detach()                      # STE (Eq. 4)
            point = y_hat

        rate_bits = self._rate_bits(point)
        return y_hat, rate_bits


class NestedLatticeBottleneck(nn.Module):
    """
    Fixed-rate LTC with self-similar nested lattices (paper Sec. 4.2).

        y_f    = Q_Lf(y)                     (fine lattice)
        y_hat  = y_f - Gamma * Q_Lf(y_f / Gamma)   (coset leader in V(Lc))

    Rate is fixed by construction: log2 |Lf / Lc| = n * log2(Gamma) bits
    per block, dy * log2(Gamma) total. No entropy model. Trained with STE
    (paper: "We use STE to train the BLTC models").
    """

    def __init__(self, lattice, n, latent_dim, nesting_ratio):
        super().__init__()
        if latent_dim % n != 0:
            raise ValueError(f"latent_dim {latent_dim} not divisible by lattice dim {n}")
        self.q = LatticeCVP(lattice, n)
        self.n = n
        self.blocks = latent_dim // n
        self.latent_dim = latent_dim
        self.gamma = float(nesting_ratio)
        self.total_bits = latent_dim * math.log2(self.gamma)

    def forward(self, y):
        B = y.shape[0]
        s = self.q.scale
        yb = y.view(B, self.blocks, self.n) * s                    # canonical space
        y_f = self.q._cvp_canon(yb)
        coarse = self.gamma * self.q._cvp_canon(y_f / self.gamma)
        q_hard = ((y_f - coarse) / s).view(B, self.latent_dim)

        y_hat = y + (q_hard - y).detach()                          # STE
        rate_bits = y.new_full((B,), self.total_bits)
        return y_hat, rate_bits

    def encode_indices(self, y):
        """Integer coset index per block (the transmittable symbols). No grad."""
        with torch.no_grad():
            B = y.shape[0]
            s = self.q.scale
            yb = y.view(B, self.blocks, self.n) * s
            y_f = self.q._cvp_canon(yb)
            resid = y_f - self.gamma * self.q._cvp_canon(y_f / self.gamma)
            # resid lies on Lf inside V(Lc); mod-Gamma of its lattice coords is the index
            coords = torch.linalg.solve(
                self.q.gen_unit.T.to(y) * s, resid.unsqueeze(-1)
            ).squeeze(-1)
            return torch.remainder(torch.round(coords), self.gamma).long()


# ----------------------------------------------------------------------
# 3. SNN encoder that emits a real latent (no threshold)
# ----------------------------------------------------------------------

class SNNLatentEncoder(nn.Module):
    """
    x_{1:T} (T, B, C_in) -> y in R^{latent_dim}

    Same LIF stack as autoencoder.SNNEncoder, but the readout membrane
    potential is *not* thresholded: it is averaged over T and (optionally)
    passed through an analysis MLP g_a (paper: 2 hidden layers, softplus),
    giving the continuous latent the lattice quantizer expects.
    """

    def __init__(self, c_in, hidden, latent_dim, tau_decay=0.5, v_th=1.0,
                 surrogate_width=1.0, ga_hidden=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.readout_tau = tau_decay
        self.fc1 = nn.Linear(c_in, hidden)
        self.fc2 = nn.Linear(hidden, latent_dim)
        self.lif1 = LIFCell(tau_decay, v_th, surrogate_width)
        if ga_hidden and ga_hidden > 0:
            self.g_a = nn.Sequential(
                nn.Linear(latent_dim, ga_hidden), nn.Softplus(),
                nn.Linear(ga_hidden, ga_hidden), nn.Softplus(),
                nn.Linear(ga_hidden, latent_dim),
            )
        else:
            self.g_a = None

    def forward(self, x):
        T, B, _ = x.shape
        u1, o1 = LIFCell.init_state((B, self.fc1.out_features), x.device)
        acc = torch.zeros(B, self.latent_dim, device=x.device)
        for t in range(T):
            u1, o1 = self.lif1(self.fc1(x[t]), u1, o1)
            acc = self.readout_tau * acc + self.fc2(o1)
        y = acc / T                                   # T-independent scale
        if self.g_a is not None:
            y = self.g_a(y)
        return y


class SNNLatticeAutoencoder(nn.Module):
    def __init__(self, cfg: "LTCConfig", feature_dim: int):
        super().__init__()
        self.cfg = cfg
        n = _FIXED_DIM.get(cfg.lattice, cfg.lattice_n)
        self.encoder = SNNLatentEncoder(
            feature_dim, cfg.hidden, cfg.latent_dim,
            cfg.tau_decay, cfg.v_th, cfg.surrogate_width, cfg.ga_hidden,
        )
        self.decoder = SNNDecoder(
            cfg.latent_dim, cfg.hidden, feature_dim, cfg.T,
            cfg.tau_decay, cfg.v_th, cfg.surrogate_width, cfg.inject,
        )
        if cfg.quantizer == "nested":
            self.bottleneck = NestedLatticeBottleneck(
                cfg.lattice, n, cfg.latent_dim, cfg.nesting_ratio
            )
        else:
            self.bottleneck = EntropyLatticeBottleneck(
                cfg.lattice, n, cfg.latent_dim,
                cfg.quant_backward, cfg.entropy, cfg.mc_samples,
            )

    def forward(self, x):
        y = self.encoder(x)
        y_hat, rate_bits = self.bottleneck(y)
        x_hat = self.decoder(y_hat)
        return x_hat, y, y_hat, rate_bits


# ----------------------------------------------------------------------
# 4. Config
# ----------------------------------------------------------------------

@dataclass
class LTCConfig:
    # --- source (consumed by autoencoder.make_source) ---
    sources: tuple = ("poisson_homog", "bursty_renewal")
    T: int = 32
    channels: int = 1
    batch_size: int = 64
    seed: int = 0
    data_root: str | None = None
    binarize: bool = True
    num_workers: int = 0
    cache_dir: str | None = None
    # --- SNN ---
    hidden: int = 64
    tau_decay: float = 0.5
    v_th: float = 1.0
    surrogate_width: float = 1.0
    inject: str = "constant"
    ga_hidden: int = 100
    # --- bottleneck ---
    quantizer: str = "entropy"          # entropy | nested | scalar
    lattice: str = "E8"                 # Zn | Dn | Dn_star | A2 | E8
    lattice_n: int = 8                  # lattice dim for the Zn/Dn/Dn_star family
    latent_dim: int = 8
    quant_backward: str = "ste"         # ste | dither
    entropy: str = "diag"              # diag | full
    mc_samples: int = 256
    nesting_ratio: float = 4.0
    # --- sweep / optim ---
    lambdas: tuple = (32.0, 64.0, 128.0, 256.0, 512.0)
    nesting_ratios: tuple = (2.0, 3.0, 4.0, 6.0, 9.0)
    steps: int = 200
    lr: float = 1e-3
    eval_batches: int = 4
    tau_syn: float = 4.0
    device: str = "cpu"
    log_every: int = 50


# ----------------------------------------------------------------------
# 5. Train one operating point
# ----------------------------------------------------------------------

def train_one_point(source, cfg: LTCConfig, lam: float | None = None,
                    gamma: float | None = None):
    torch.manual_seed(cfg.seed)
    point_cfg = cfg
    if gamma is not None:
        point_cfg = replace(cfg, nesting_ratio=gamma)

    model = SNNLatticeAutoencoder(point_cfg, source.feature_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    norm = source.feature_dim * cfg.T                       # bits -> bits per sample

    model.train()
    step = 0
    while step < cfg.steps:
        for x in source.train_batches(cfg.steps - step, cfg.device):
            x_hat, _, _, rate_bits = model(x)
            dist = psp_mse(x, x_hat, cfg.tau_syn)
            rate_bpp = rate_bits.mean() / norm
            if point_cfg.quantizer == "nested":
                loss = dist
            else:
                loss = lam * dist + rate_bpp

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if step % cfg.log_every == 0:
                tag = f"gamma={gamma}" if gamma is not None else f"lambda={lam}"
                print(f"[{source.name} | {tag}] step {step}/{cfg.steps}  "
                      f"loss={loss.item():.4f}  psp_mse={dist.item():.5f}  "
                      f"rate_bpp={rate_bpp.item():.4f}", flush=True)
            if step >= cfg.steps:
                break

    model.eval()
    with torch.no_grad():
        d_psp = d_ham = r_bpp = 0.0
        nb = 0
        for x in source.eval_batches(cfg.eval_batches, cfg.device):
            x_hat, _, _, rate_bits = model(x)
            d_psp += psp_mse(x, x_hat, cfg.tau_syn).item()
            d_ham += hamming_distortion(x, x_hat).item()
            r_bpp += (rate_bits.mean() / norm).item()
            nb += 1
    return {
        "psp_mse": d_psp / nb,
        "hamming": d_ham / nb,
        "rate_bpp": r_bpp / nb,
    }


# ----------------------------------------------------------------------
# 6. Rate-distortion sweep
# ----------------------------------------------------------------------

def run_ltc_sweep(cfg: LTCConfig):
    results = {}
    for name in cfg.sources:
        source = make_source(name, cfg)
        p = source.spike_prob(cfg.device)
        print(f"\n--- source={name} (N={source.feature_dim}, T={cfg.T}, "
              f"p_hat={p:.4f}, H(p)={binary_entropy(p):.4f} bits) ---", flush=True)

        points = []
        if cfg.quantizer == "nested":
            knobs = [("gamma", g, None, g) for g in cfg.nesting_ratios]
        else:
            knobs = [("lambda", lam, lam, None) for lam in cfg.lambdas]

        for label, value, lam, gamma in knobs:
            m = train_one_point(source, cfg, lam=lam, gamma=gamma)
            m["memoryless_bound_R"] = bernoulli_rate_distortion(p, m["hamming"])
            points.append((label, value, m))
            print(f"==> {name:16s} {label}={value:<7} "
                  f"rate_bpp={m['rate_bpp']:.4f}  PSP-MSE={m['psp_mse']:.5f}  "
                  f"Hamming={m['hamming']:.4f}  R(D)_memoryless={m['memoryless_bound_R']:.4f}",
                  flush=True)
        results[name] = {"spike_prob": p, "points": points}
    return results


# ----------------------------------------------------------------------
# 7. CLI
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    d = LTCConfig()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sources", nargs="+", default=list(d.sources),
                   help=f"synthetic {list(SYNTH_NAMES)} and/or real {list(REAL_NAMES)}")
    p.add_argument("--quantizer", choices=["entropy", "nested", "scalar"], default=d.quantizer)
    p.add_argument("--lattice", choices=list(_CVP), default=d.lattice)
    p.add_argument("--lattice-n", type=int, default=d.lattice_n,
                   help="lattice dim for Zn/Dn/Dn_star (A2->2, E8->8 forced)")
    p.add_argument("--latent-dim", type=int, default=d.latent_dim,
                   help="analysis-transform output dim dy (must be a multiple of the lattice dim)")
    p.add_argument("--quant-backward", choices=["ste", "dither"], default=d.quant_backward)
    p.add_argument("--entropy", choices=["diag", "full"], default=d.entropy,
                   help="joint density p_y: diagonal or full-covariance Gaussian")
    p.add_argument("--mc-samples", type=int, default=d.mc_samples,
                   help="Monte-Carlo samples for the cell-probability rate estimate")
    p.add_argument("--lambdas", type=float, nargs="+", default=list(d.lambdas),
                   help="R-D tradeoff multipliers to sweep (entropy/scalar mode)")
    p.add_argument("--nesting-ratios", type=float, nargs="+", default=list(d.nesting_ratios),
                   help="nested-lattice ratios Gamma to sweep (nested mode)")
    p.add_argument("--ga-hidden", type=int, default=d.ga_hidden, help="analysis MLP width (0 disables)")
    p.add_argument("--T", type=int, default=d.T)
    p.add_argument("--channels", type=int, default=d.channels)
    p.add_argument("--hidden", type=int, default=d.hidden)
    p.add_argument("--inject", choices=["constant", "first"], default=d.inject)
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--eval-batches", type=int, default=d.eval_batches)
    p.add_argument("--tau-decay", type=float, default=d.tau_decay)
    p.add_argument("--tau-syn", type=float, default=d.tau_syn)
    p.add_argument("--v-th", type=float, default=d.v_th)
    p.add_argument("--surrogate-width", type=float, default=d.surrogate_width)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--data-root", default=None)
    p.add_argument("--no-binarize", action="store_true")
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--cache-dir", default=None)
    return p


def config_from_args(args) -> LTCConfig:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lattice, lattice_n = args.lattice, args.lattice_n
    if args.quantizer == "scalar":
        lattice, lattice_n = "Zn", 1                       # NTC baseline
    if lattice in _FIXED_DIM:
        lattice_n = _FIXED_DIM[lattice]
    if args.latent_dim % lattice_n != 0:
        raise SystemExit(
            f"--latent-dim {args.latent_dim} must be a multiple of lattice dim {lattice_n}"
        )
    return LTCConfig(
        sources=tuple(args.sources),
        T=args.T, channels=args.channels, batch_size=args.batch_size, seed=args.seed,
        data_root=args.data_root, binarize=not args.no_binarize,
        num_workers=args.num_workers, cache_dir=args.cache_dir,
        hidden=args.hidden, tau_decay=args.tau_decay, v_th=args.v_th,
        surrogate_width=args.surrogate_width, inject=args.inject, ga_hidden=args.ga_hidden,
        quantizer=args.quantizer, lattice=lattice, lattice_n=lattice_n,
        latent_dim=args.latent_dim, quant_backward=args.quant_backward,
        entropy=args.entropy, mc_samples=args.mc_samples,
        lambdas=tuple(args.lambdas), nesting_ratios=tuple(args.nesting_ratios),
        steps=args.steps, lr=args.lr, eval_batches=args.eval_batches,
        tau_syn=args.tau_syn, device=device, log_every=args.log_every,
    )


def main(argv: Sequence[str] | None = None) -> int:
    cfg = config_from_args(build_arg_parser().parse_args(argv))
    n = _FIXED_DIM.get(cfg.lattice, cfg.lattice_n)
    print(f"Running on {cfg.device} | quantizer={cfg.quantizer} | lattice={cfg.lattice}(n={n}) "
          f"| latent_dim={cfg.latent_dim} | backward={cfg.quant_backward} | "
          f"sources={list(cfg.sources)} | T={cfg.T} | steps={cfg.steps}", flush=True)

    results = run_ltc_sweep(cfg)

    print("\n=== Summary: rate-distortion trade-off (LTC) ===")
    for name, res in results.items():
        p = res["spike_prob"]
        print(f"\nSource: {name}  (p_hat={p:.4f}, memoryless H(p)={binary_entropy(p):.4f} bits)")
        for label, value, m in res["points"]:
            print(f"  {label}={value:<7} rate_bpp={m['rate_bpp']:.4f}  "
                  f"PSP-MSE={m['psp_mse']:.5f}  Hamming={m['hamming']:.4f}  "
                  f"R(D)_memoryless={m['memoryless_bound_R']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
