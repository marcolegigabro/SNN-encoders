"""The synthetic spike source: Poisson spike trains, N neurons x T bins.

Two generators, and the difference between them is the whole reason for having
both.

`iid` is a homogeneous Poisson process in discrete time, i.e. every bin of every
neuron an independent Bernoulli(p). It is memoryless, so its rate-distortion
function is exactly computable (closed form for Hamming, Blahut-Arimoto for the
other two, see `theory.py`) and there is *no* structure for the encoder to
exploit beyond the marginal statistics. That makes the measured gap to R(D) an
absolute statement about the codec rather than a statement about the dataset.

`latent` is an inhomogeneous Poisson process: a rate envelope drawn per trial
and shared across neurons, times a fixed per-neuron gain. Now consecutive bins
and different neurons are correlated, so a good encoder can spend its bits on
the envelope instead of on individual spikes. Its true R(D) is not tractable at
this block length, but the i.i.d. curve at the same marginal rate is a valid
*upper* bound on it (conditioning reduces entropy), so a system that drops below
the i.i.d. curve on this source is exploiting correlation, not beating theory.

Both return float tensors of shape (B, T, N) in {0, 1}: time on dim 1, which is
the axis `psp_filter` runs along.
"""

from __future__ import annotations

import torch


class PoissonSource:
    """Infinite stream of synthetic spike-train blocks.

    There is no train/test split to speak of: the source is a generator, so
    every batch is fresh and the test set is simply a batch drawn with a fixed
    seed. That removes overfitting from the experiment entirely, which is what
    we want when the quantity of interest is a distance to a bound.
    """

    def __init__(self, n_neurons: int = 16, n_bins: int = 12, rate: float = 0.15,
                 kind: str = "iid", envelope_depth: float = 0.9,
                 gain_spread: float = 0.4, device="cpu", seed: int | None = None):
        self.n_neurons = n_neurons
        self.n_bins = n_bins
        self.rate = rate
        self.kind = kind
        self.envelope_depth = envelope_depth
        self.device = device
        self.gen = torch.Generator(device="cpu")
        if seed is not None:
            self.gen.manual_seed(seed)
        # Per-neuron gains are a property of the *source*, fixed once, not
        # redrawn per trial: they are the neurons' tuning, and the encoder is
        # allowed to learn them.
        g = torch.exp(gain_spread * torch.randn(n_neurons, generator=self.gen))
        self.gain = (g / g.mean()).to(device)

    def rates(self, batch: int) -> torch.Tensor:
        """Per-(trial, bin, neuron) firing probability, (B, T, N)."""
        T, N = self.n_bins, self.n_neurons
        if self.kind == "iid":
            return torch.full((batch, T, N), self.rate, device=self.device)
        if self.kind == "latent":
            # one sinusoidal envelope per trial, random phase and frequency,
            # mean-preserving so the marginal rate stays `self.rate`
            phase = 2 * torch.pi * torch.rand(batch, 1, generator=self.gen)
            freq = torch.randint(1, 3, (batch, 1), generator=self.gen).float()
            t = torch.arange(T).float().unsqueeze(0) / T
            env = 1.0 + self.envelope_depth * torch.sin(2 * torch.pi * freq * t + phase)
            env = env.to(self.device).unsqueeze(-1)                      # (B, T, 1)
            lam = self.rate * env * self.gain.view(1, 1, N)
            return lam.clamp(0.0, 1.0)
        raise ValueError(f"unknown source kind: {self.kind}")

    def sample(self, batch: int) -> torch.Tensor:
        lam = self.rates(batch)
        u = torch.rand(lam.shape, generator=self.gen).to(self.device)
        return (u < lam).float()

    @property
    def block_bits(self) -> int:
        """Source symbols per block, i.e. the denominator of every per-bin rate."""
        return self.n_bins * self.n_neurons

    def marginal_rate(self, n_batch: int = 64, batch: int = 512) -> float:
        """Measured mean firing probability. For `latent` this is what the
        i.i.d. reference curves must be computed at for the comparison to mean
        anything, and it is measured rather than assumed because the clamp in
        `rates` can bite at large envelope depth or gain spread."""
        tot = 0.0
        for _ in range(n_batch):
            tot += float(self.rates(batch).mean())
        return tot / n_batch
