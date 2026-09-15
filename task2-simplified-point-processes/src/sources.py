"""Point-process sources on a finite window [0, T].

A batch of realizations is stored as a padded tensor of sorted event times.
Padding uses the value T itself rather than a sentinel. An "event" at T adds
nothing to the counting function on [0, T), so the distortions in
`distortion.py` can treat padded and real events identically: this is exactly
Rubin's convention in Proposition 2 of his 1974 paper, where the shorter of two
sequences is extended with events at the end of the window.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Events:
    """A batch of point-process realizations on [0, T]."""

    times: torch.Tensor   # (B, L) sorted event times, padded with T
    counts: torch.Tensor  # (B,) number of real events per realization
    T: float

    @property
    def mask(self) -> torch.Tensor:
        """(B, L) True on real events, False on padding."""
        L = self.times.shape[1]
        return torch.arange(L, device=self.times.device) < self.counts[:, None]

    def binned(self, n_bins: int) -> torch.Tensor:
        """(B, n_bins) float event counts per bin: the SNN's input spike tensor."""
        idx = (self.times / self.T * n_bins).long().clamp(0, n_bins - 1)
        out = torch.zeros(self.times.shape[0], n_bins, device=self.times.device)
        return out.scatter_add_(1, idx, self.mask.to(out.dtype))

    def padded_to(self, length: int) -> "Events":
        """Same realizations, padded (with T) to at least `length` columns."""
        extra = length - self.times.shape[1]
        if extra <= 0:
            return self
        pad = self.times.new_full((self.times.shape[0], extra), self.T)
        return Events(torch.cat([self.times, pad], 1), self.counts, self.T)

    def to(self, device) -> "Events":
        return Events(self.times.to(device), self.counts.to(device), self.T)

    def __getitem__(self, idx) -> "Events":
        return Events(self.times[idx], self.counts[idx], self.T)

    def __len__(self) -> int:
        return self.times.shape[0]


def from_times(times: torch.Tensor, T: float) -> Events:
    """Wrap a (B, L) tensor of times already padded with T (or larger)."""
    times = times.clamp(max=T).sort(1).values
    return Events(times, (times < T).sum(1), T)


def homogeneous_poisson(batch: int, rate: float, T: float = 1.0,
                        generator: torch.Generator | None = None,
                        device="cpu") -> Events:
    """N ~ Poisson(rate * T), then N i.i.d. uniform times, sorted.

    This is the order-statistics characterization of the Poisson process that
    Rubin's analysis rests on: given N(T) = n, the event times are distributed
    as the order statistics of n uniforms on [0, T].
    """
    mean = torch.full((batch,), rate * T, device=device)
    counts = torch.poisson(mean, generator=generator).long()
    L = max(int(counts.max()), 1)
    u = torch.rand(batch, L, generator=generator, device=device) * T
    mask = torch.arange(L, device=device) < counts[:, None]
    times = torch.where(mask, u, torch.full_like(u, T)).sort(1).values
    return Events(times, counts, T)


def eval_set(rate: float, n: int = 10_000, T: float = 1.0, seed: int = 1234) -> Events:
    """The fixed evaluation set shared by every codec at a given rate."""
    g = torch.Generator().manual_seed(seed)
    return homogeneous_poisson(n, rate, T, generator=g)
