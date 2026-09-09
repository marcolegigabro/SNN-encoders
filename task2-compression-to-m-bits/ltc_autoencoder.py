"""
Lattice-Transform-Coding SNN Autoencoder  (task 2, LTC variant)
==============================================================

Same task as ``autoencoder.py`` -- compress a spike train x_{1:T} into a
compact code, reconstruct it with an SNN decoder -- but the scalar STE
bottleneck is replaced by the pipeline from

    Lei, Hassani & Saeedi Bidokhti, "Approaching Rate-Distortion Limits in
    Neural Compression with Lattice Transform Coding", ICLR 2025.

This module is deliberately **self-contained**: it shares no code with
``autoencoder.py`` (which is being reworked onto snnTorch on its own
schedule). The SNN primitives here are a small torch-only reimplementation;
porting them to snnTorch later, to match ``autoencoder.py``, is a mechanical
change that does not touch the lattice/entropy code.

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

    python ltc_autoencoder.py --help
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]


# ======================================================================
# PART A -- self-contained SNN primitives, distortion, spike sources
# ======================================================================

# ----------------------------------------------------------------------
# A1. Surrogate-gradient spiking primitives
# ----------------------------------------------------------------------

class SurrogateSpike(torch.autograd.Function):
    """
    Heaviside firing with a rectangular surrogate gradient (Wu et al. 2019
    / FSVAE):  forward o = H(u - v_th),  backward do/du = (1/a) 1[|u-v_th| < a/2].
    v_th and a are kept as python floats so backward never mixes a CPU
    scalar tensor with a CUDA u.
    """

    @staticmethod
    def forward(ctx, u, v_th, a):
        ctx.save_for_backward(u)
        ctx.v_th = float(v_th)
        ctx.a = float(a)
        return (u >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        surrogate = (1.0 / ctx.a) * (torch.abs(u - ctx.v_th) < (ctx.a / 2)).float()
        return grad_output * surrogate, None, None


spike_fn = SurrogateSpike.apply


class LIFCell(nn.Module):
    """Iterative leaky integrate-and-fire layer (Wu et al. 2019)."""

    def __init__(self, tau_decay=0.5, v_th=1.0, surrogate_width=1.0):
        super().__init__()
        self.tau_decay = tau_decay
        self.v_th = v_th
        self.a = surrogate_width

    def forward(self, x_t, u_prev, o_prev):
        u_t = self.tau_decay * u_prev * (1.0 - o_prev) + x_t
        o_t = spike_fn(u_t, self.v_th, self.a)
        return u_t, o_t

    @staticmethod
    def init_state(shape, device):
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


# ----------------------------------------------------------------------
# A2. Distortion metrics + information-theory anchors
# ----------------------------------------------------------------------

def psp_filter(spike_train, tau_syn=4.0):
    """Recursive post-synaptic-potential filter (FSVAE / MMD-GLM)."""
    T = spike_train.shape[0]
    decay = 1.0 - 1.0 / tau_syn
    gain = 1.0 / tau_syn
    psp = torch.zeros_like(spike_train)
    running = torch.zeros_like(spike_train[0])
    for t in range(T):
        running = decay * running + gain * spike_train[t]
        psp[t] = running
    return psp


def psp_mse(x, x_hat, tau_syn=4.0):
    return F.mse_loss(psp_filter(x_hat, tau_syn), psp_filter(x, tau_syn))


def hamming_distortion(x, x_hat):
    """Fraction of binary samples that differ (raw bit-error rate)."""
    return (x_hat.round().clamp(0.0, 1.0) - x).abs().mean()


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def bernoulli_rate_distortion(p: float, D: float) -> float:
    """R(D) = H(p) - H(D) for a memoryless Bernoulli(p) source, Hamming distortion."""
    if D >= min(p, 1.0 - p):
        return 0.0
    return max(binary_entropy(p) - binary_entropy(D), 0.0)


# ----------------------------------------------------------------------
# A3. Synthetic spike-source generators
# ----------------------------------------------------------------------

def gen_homogeneous_poisson(T, B, C, rate, device="cpu"):
    return torch.bernoulli(torch.full((T, B, C), rate, device=device))


def gen_inhomogeneous_poisson(T, B, C, base_rate=0.05, amp=0.15, period=None, device="cpu"):
    if period is None:
        period = T
    t_idx = torch.arange(T, device=device).float()
    rate_t = base_rate + amp * (0.5 * (1 + torch.sin(2 * math.pi * t_idx / period)))
    rate_t = rate_t.clamp(0.0, 1.0).view(T, 1, 1).expand(T, B, C)
    return torch.bernoulli(rate_t)


def gen_bursty_renewal(T, B, C, burst_prob=0.03, refractory=4, device="cpu"):
    x = torch.zeros(T, B, C, device=device)
    refractory_left = torch.zeros(B, C, device=device)
    for t in range(T):
        can_fire = (refractory_left <= 0).float()
        fire = torch.bernoulli(torch.full((B, C), burst_prob, device=device)) * can_fire
        x[t] = fire
        refractory_left = torch.where(
            fire.bool(),
            torch.full_like(refractory_left, refractory),
            (refractory_left - 1).clamp(min=0),
        )
    return x


def gen_periodic(T, B, C, period=10, jitter=0.0, device="cpu"):
    x = torch.zeros(T, B, C, device=device)
    for start in range(0, T, period):
        if jitter > 0:
            offsets = torch.randint(
                low=max(-jitter, -start), high=jitter + 1, size=(B, C), device=device
            )
        else:
            offsets = torch.zeros(B, C, dtype=torch.long, device=device)
        idx = (start + offsets).clamp(0, T - 1)
        x.scatter_(0, idx.unsqueeze(0), 1.0)
    return x


SYNTH_GENERATORS = {
    "poisson_homog": lambda T, B, C, device: gen_homogeneous_poisson(T, B, C, rate=0.1, device=device),
    "poisson_inhomog": lambda T, B, C, device: gen_inhomogeneous_poisson(T, B, C, device=device),
    "bursty_renewal": lambda T, B, C, device: gen_bursty_renewal(T, B, C, device=device),
    "periodic": lambda T, B, C, device: gen_periodic(T, B, C, period=8, jitter=1, device=device),
}


# ----------------------------------------------------------------------
# A4. Source interface: synthetic generators + real tonic datasets
# ----------------------------------------------------------------------

class Source:
    """
    name, feature_dim (flattened N), T;
    train_batches / eval_batches(n, device) -> Iterator[(T, B, N) float in {0,1}];
    spike_prob(device) -> empirical mean spike probability.
    """

    name: str
    feature_dim: int
    T: int

    def train_batches(self, n_batches: int, device) -> Iterator[torch.Tensor]:
        raise NotImplementedError

    def eval_batches(self, n_batches: int, device) -> Iterator[torch.Tensor]:
        raise NotImplementedError

    def spike_prob(self, device="cpu", n_batches: int = 8) -> float:
        total, count = 0.0, 0
        for x in self.eval_batches(n_batches, device):
            total += x.mean().item()
            count += 1
        return total / max(count, 1)


class SyntheticSource(Source):
    def __init__(self, name, T, channels, batch_size, seed=0):
        self.name = name
        self.T = T
        self.channels = channels
        self.feature_dim = channels
        self.batch_size = batch_size
        self._gen = SYNTH_GENERATORS[name]
        self._seed = seed

    def train_batches(self, n_batches, device):
        for _ in range(n_batches):
            yield self._gen(self.T, self.batch_size, self.channels, device)

    def eval_batches(self, n_batches, device):
        state = torch.random.get_rng_state()
        torch.manual_seed(self._seed + 4242)
        batches = [self._gen(self.T, self.batch_size, self.channels, device)
                   for _ in range(n_batches)]
        torch.random.set_rng_state(state)
        return iter(batches)


_TONIC_DATASETS = {"nmnist": "NMNIST", "shd": "SHD", "dvsgesture": "DVSGesture"}


def resolve_data_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("SNN_DATA_ROOT")
    if env:
        return Path(env).resolve()
    return (REPO_ROOT / "data").resolve()


def _time_first_collate(batch, T, binarize):
    """(B, T, *) samples -> (T, B, N) binary, flattened + transposed once here."""
    frames = []
    for item in batch:
        frame = item[0] if isinstance(item, (tuple, list)) else item
        frames.append(torch.as_tensor(frame, dtype=torch.float32).reshape(T, -1))
    x = torch.stack(frames, dim=1)
    if binarize:
        x = (x > 0).float()
    return x


class RealSource(Source):
    """An event dataset (N-MNIST / SHD / DVS-Gesture) binned to T frames with tonic."""

    def __init__(self, name, T, batch_size, data_root=None, binarize=True,
                 num_workers=0, cache_dir=None):
        try:
            import tonic
            import tonic.transforms as tonic_transforms
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise ModuleNotFoundError(
                "real spike sources need tonic + h5py -- `uv pip install tonic h5py`"
            ) from exc

        cls = getattr(tonic.datasets, _TONIC_DATASETS[name])
        root = str(resolve_data_root(data_root) / name)
        transform = tonic_transforms.ToFrame(sensor_size=cls.sensor_size, n_time_bins=T)

        train_ds = cls(save_to=root, train=True, transform=transform)
        test_ds = cls(save_to=root, train=False, transform=transform)

        if cache_dir:
            from tonic import DiskCachedDataset
            base = Path(cache_dir) / name
            train_ds = DiskCachedDataset(train_ds, cache_path=str(base / "train"))
            test_ds = DiskCachedDataset(test_ds, cache_path=str(base / "test"))

        self.name = name
        self.T = T
        sample = torch.as_tensor(train_ds[0][0], dtype=torch.float32).reshape(T, -1)
        self.feature_dim = int(sample.shape[1])

        def collate(b):
            return _time_first_collate(b, T, binarize)

        self._train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
            collate_fn=collate, num_workers=num_workers,
        )
        self._test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, drop_last=True,
            collate_fn=collate, num_workers=num_workers,
        )

    @staticmethod
    def _take(loader, n_batches, device):
        it = iter(loader)
        for _ in range(n_batches):
            try:
                x = next(it)
            except StopIteration:
                it = iter(loader)
                x = next(it)
            yield x.to(device)

    def train_batches(self, n_batches, device):
        return self._take(self._train_loader, n_batches, device)

    def eval_batches(self, n_batches, device):
        return self._take(self._test_loader, n_batches, device)


SYNTH_NAMES = tuple(SYNTH_GENERATORS)
REAL_NAMES = tuple(_TONIC_DATASETS)


def make_source(name: str, cfg: "LTCConfig") -> Source:
    if name in SYNTH_GENERATORS:
        return SyntheticSource(name, cfg.T, cfg.channels, cfg.batch_size, seed=cfg.seed)
    if name in _TONIC_DATASETS:
        return RealSource(name, cfg.T, cfg.batch_size, data_root=cfg.data_root,
                          binarize=cfg.binarize, num_workers=cfg.num_workers,
                          cache_dir=cfg.cache_dir)
    raise ValueError(
        f"unknown source {name!r}; synthetic {list(SYNTH_NAMES)} or real {list(REAL_NAMES)}"
    )


# ----------------------------------------------------------------------
# A5. Autoregressive SNN decoder
# ----------------------------------------------------------------------

class SNNDecoder(nn.Module):
    """
    code b -> x_hat_{1:T}. Autoregressive LIF stack; the code is injected as
    a constant context every step ("constant") or only at t=0 ("first"),
    concatenated with the previously generated output spike.
    """

    def __init__(self, m, hidden, c_out, T, tau_decay=0.5, v_th=1.0,
                 surrogate_width=1.0, inject="constant"):
        super().__init__()
        if inject not in ("constant", "first"):
            raise ValueError(f"inject must be 'constant' or 'first', got {inject!r}")
        self.inject = inject
        self.T = T
        self.c_out = c_out
        self.m = m
        self.fc_in = nn.Linear(m + c_out, hidden)
        self.fc_out = nn.Linear(hidden, c_out)
        self.lif1 = LIFCell(tau_decay, v_th, surrogate_width)
        self.lif2 = LIFCell(tau_decay, v_th, surrogate_width)

    def forward(self, b):
        B, _ = b.shape
        device = b.device
        u1, o1 = LIFCell.init_state((B, self.fc_in.out_features), device)
        u2, o2 = LIFCell.init_state((B, self.c_out), device)
        x_prev = torch.zeros(B, self.c_out, device=device)
        zeros_code = torch.zeros_like(b)

        outputs = []
        for t in range(self.T):
            code = b if (self.inject == "constant" or t == 0) else zeros_code
            u1, o1 = self.lif1(self.fc_in(torch.cat([code, x_prev], dim=-1)), u1, o1)
            u2, o2 = self.lif2(self.fc_out(o1), u2, o2)
            outputs.append(o2)
            x_prev = o2

        return torch.stack(outputs, dim=0)


# ======================================================================
# PART B -- Lattice Transform Coding
# ======================================================================

# ----------------------------------------------------------------------
# B1. Lattice closest-point (CVP) routines
#     Each returns the closest point in the *canonical* lattice; unit-volume
#     rescaling is applied by LatticeCVP so lattices are comparable.
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
    basis = z.new_tensor(_A2_BASIS)
    coords = z @ torch.linalg.inv(basis)
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
    Closest-point search + Voronoi-cell sampling, rescaled so the
    fundamental cell has unit volume (paper assumes det(G G^T) = 1).
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
        self.register_buffer("gen_unit", g / scale)
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
# B2. Bottlenecks
# ----------------------------------------------------------------------

class EntropyLatticeBottleneck(nn.Module):
    """
    Variable-rate LTC (paper Sec. 4.1). The latent y in R^{dy} is split
    into dy/n blocks, each quantized with the n-D lattice (a product
    lattice along the latent). A learned joint Gaussian p_y gives the rate
    via a Monte-Carlo estimate of the cell probability (paper Eqs. 5-6):

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
        logp = self._dist().log_prob(point.unsqueeze(0) + u)
        log_cell = torch.logsumexp(logp, dim=0) - math.log(self.mc_samples)
        return -log_cell / math.log(2.0)

    def forward(self, y):
        B = y.shape[0]
        q_hard = self.q.closest(y.view(B, self.blocks, self.n)).view(B, self.latent_dim)

        if self.backward == "dither" and self.training:
            point = y + self._dither((B,), y.device)
            y_hat = point
        else:
            y_hat = y + (q_hard - y).detach()          # STE (Eq. 4)
            point = y_hat

        return y_hat, self._rate_bits(point)


class NestedLatticeBottleneck(nn.Module):
    """
    Fixed-rate LTC with self-similar nested lattices (paper Sec. 4.2).

        y_f    = Q_Lf(y)
        y_hat  = y_f - Gamma * Q_Lf(y_f / Gamma)     (coset leader in V(Lc))

    Rate is fixed: n * log2(Gamma) bits per block, dy * log2(Gamma) total.
    No entropy model. Trained with STE.
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
        yb = y.view(B, self.blocks, self.n) * s
        y_f = self.q._cvp_canon(yb)
        coarse = self.gamma * self.q._cvp_canon(y_f / self.gamma)
        q_hard = ((y_f - coarse) / s).view(B, self.latent_dim)

        y_hat = y + (q_hard - y).detach()              # STE
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
            coords = torch.linalg.solve(
                self.q.gen_unit.T.to(y) * s, resid.unsqueeze(-1)
            ).squeeze(-1)
            return torch.remainder(torch.round(coords), self.gamma).long()


# ----------------------------------------------------------------------
# B3. SNN encoder emitting a real latent, and the full autoencoder
# ----------------------------------------------------------------------

class SNNLatentEncoder(nn.Module):
    """
    x_{1:T} (T, B, C_in) -> y in R^{latent_dim}.  LIF stack; the readout
    membrane potential is averaged over T (not thresholded) and optionally
    passed through an analysis MLP g_a (paper: 2 hidden layers, softplus).
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
        y = acc / T
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


# ======================================================================
# PART C -- config, training, sweep, CLI
# ======================================================================

@dataclass
class LTCConfig:
    # --- source ---
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
    lattice_n: int = 8
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


def train_one_point(source, cfg: LTCConfig, lam: float | None = None,
                    gamma: float | None = None):
    torch.manual_seed(cfg.seed)
    point_cfg = cfg if gamma is None else replace(cfg, nesting_ratio=gamma)

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
            loss = dist if point_cfg.quantizer == "nested" else lam * dist + rate_bpp

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
    return {"psp_mse": d_psp / nb, "hamming": d_ham / nb, "rate_bpp": r_bpp / nb}


def run_ltc_sweep(cfg: LTCConfig):
    results = {}
    for name in cfg.sources:
        source = make_source(name, cfg)
        p = source.spike_prob(cfg.device)
        print(f"\n--- source={name} (N={source.feature_dim}, T={cfg.T}, "
              f"p_hat={p:.4f}, H(p)={binary_entropy(p):.4f} bits) ---", flush=True)

        if cfg.quantizer == "nested":
            knobs = [("gamma", g, None, g) for g in cfg.nesting_ratios]
        else:
            knobs = [("lambda", lam, lam, None) for lam in cfg.lambdas]

        points = []
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
                   help="analysis-transform output dim dy (multiple of the lattice dim)")
    p.add_argument("--quant-backward", choices=["ste", "dither"], default=d.quant_backward)
    p.add_argument("--entropy", choices=["diag", "full"], default=d.entropy,
                   help="joint density p_y: diagonal or full-covariance Gaussian")
    p.add_argument("--mc-samples", type=int, default=d.mc_samples)
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
