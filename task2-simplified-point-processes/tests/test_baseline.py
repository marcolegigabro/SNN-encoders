import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import baseline, theory
from src.distortion import counting_l1
from src.sources import Events, homogeneous_poisson

RATE = 5.0


@pytest.fixture(scope="module")
def events():
    g = torch.Generator().manual_seed(7)
    return homogeneous_poisson(100_000, RATE, generator=g)


@pytest.mark.parametrize("n_bins", [1, 4, 16, 64])
def test_midpoint_matches_theorem5(events, n_bins):
    r = baseline.evaluate(events, RATE, n_bins, "midpoint")
    D_th, R_th = theory.rubin_binned_scheme(1.0 / n_bins, RATE)
    assert abs(r["D"] - D_th) < 4 * r["D_se"]
    assert abs(r["bits_per_time"] - R_th) < 4 * r["bits_per_time_se"]


@pytest.mark.parametrize("n_bins", [1, 4, 16])
def test_median_placement_never_worse(events, n_bins):
    mid = baseline.evaluate(events, RATE, n_bins, "midpoint")["D"]
    med = baseline.evaluate(events, RATE, n_bins, "median")["D"]
    assert med <= mid + 1e-9
    if n_bins == 1:
        assert med < 0.8 * mid


@pytest.mark.parametrize("placement", ["midpoint", "median"])
def test_decode_preserves_counts_and_order(events, placement):
    sub = events[:2000]
    rec = baseline.decode(baseline.encode(sub, 8), sub.T, placement)
    assert torch.equal(rec.counts, sub.counts)
    assert torch.all(rec.times[:, 1:] >= rec.times[:, :-1])
    assert torch.equal(baseline.encode(rec, 8), baseline.encode(sub, 8))


def test_zero_rate_reconstruction_is_a_minimum(events):
    rec = baseline.zero_rate_reconstruction(events)
    def D(r):
        return counting_l1(events, Events(r.times.expand(len(events), -1),
                                          r.counts.expand(len(events)), r.T)).mean()
    best = D(rec)
    for shift in (-0.02, 0.02):
        moved = Events((rec.times + shift).clamp(0, 1), rec.counts, rec.T)
        assert D(moved) >= best
    empty = Events(torch.ones(1, 1), torch.zeros(1, dtype=torch.long), 1.0)
    assert D(empty) > best


def test_poisson_entropy():
    assert theory.poisson_entropy_bits(0.0) == 0.0
    for m in (0.1, 1.0, 5.0, 50.0):
        assert math.isclose(theory.poisson_entropy_bits(m), poisson.entropy(m) / math.log(2),
                            rel_tol=1e-9)


def test_theory_ordering():
    dT = np.logspace(-3, 0, 40)
    D, R_bar = theory.rubin_binned_scheme(dT, RATE)
    low, up = theory.rubin_lower_bound(D, RATE), theory.rubin_upper_bound(D, RATE)
    assert np.all(low <= up)
    assert np.all(R_bar >= up - 1e-9)  # Rubin eq. 53: the scheme sits above the upper bound
