"""Information-theoretic reference curves for the homogeneous Poisson source.

All rates are returned in **bits per unit time** (Rubin works in nats/s).
Distortion D is Rubin's counting-function magnitude error, `distortion.counting_l1`.

* `rubin_lower_bound`, `rubin_upper_bound`: Rubin 1974, eq. 40,
      lambda ln(1/(2D)) <= R(D) <= lambda ln(e/(4D))   nats/s.
  These are T -> infinity statements. Coding the count N(T) costs
  H_p(lambda T)/T, which vanishes in that limit (Rubin eq. 10) but not for a
  short window, see `count_cost`.
* `rubin_binned_scheme`: Rubin Theorem 5, the practical scheme that sends the
  count of every bin of width dT and places each bin's events at its midpoint,
      D = lambda dT / 4,   R = H_p(lambda dT) / dT.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

LN2 = math.log(2.0)


def poisson_entropy_bits(mean) -> np.ndarray:
    """Entropy of Poisson(mean) in bits, by direct summation of the pmf."""
    mean = np.asarray(mean, dtype=float)
    flat = np.atleast_1d(mean).ravel()
    kmax = int(np.ceil(flat.max() + 20 * np.sqrt(flat.max()) + 30))
    k = np.arange(kmax + 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = k * np.log(flat) - flat - gammaln(k + 1)
    # mean = 0 is the point mass at 0
    logp = np.where(flat > 0, logp, np.where(k == 0, 0.0, -np.inf))
    p = np.exp(logp)
    h = -(p * np.where(p > 0, logp, 0.0)).sum(0) / LN2
    return h.reshape(mean.shape)


def rubin_lower_bound(D, rate: float) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    return np.clip(rate * np.log(1.0 / (2.0 * D)) / LN2, 0.0, None)


def rubin_upper_bound(D, rate: float) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    return np.clip(rate * np.log(math.e / (4.0 * D)) / LN2, 0.0, None)


def rubin_binned_scheme(dT, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """(D, R) of Rubin's Theorem 5 scheme for bin widths dT."""
    dT = np.asarray(dT, dtype=float)
    return rate * dT / 4.0, poisson_entropy_bits(rate * dT) / dT


def count_cost(rate: float, T: float) -> float:
    """Bits per unit time to send N(T) losslessly: H_p(lambda T) / T."""
    return float(poisson_entropy_bits(rate * T)) / T
