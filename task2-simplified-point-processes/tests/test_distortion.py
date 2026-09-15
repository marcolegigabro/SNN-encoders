import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distortion import (count_error, counting_curve, counting_l1, counting_l1_grid,
                            van_rossum, victor_purpura)
from src.sources import from_times, homogeneous_poisson


def ev(*rows, T=1.0):
    L = max(max(len(r) for r in rows), 1)
    t = torch.full((len(rows), L), T)
    for i, r in enumerate(rows):
        t[i, :len(r)] = torch.tensor(r, dtype=torch.float32)
    return from_times(t, T)


def poisson_pair(n=300, rate=5.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return homogeneous_poisson(n, rate, generator=g), homogeneous_poisson(n, rate, generator=g)


def test_counting_l1_hand_cases():
    a = ev([0.2], [0.2, 0.6], [])
    b = ev([0.5], [0.5], [0.3])
    # [0.2,0.6] vs [0.5]: |N-N_hat| = 1 on [0.2,0.5) and on [0.6,1]
    assert torch.allclose(counting_l1(a, b), torch.tensor([0.3, 0.7, 0.7]))


def test_counting_l1_normalized_by_T():
    a, b = ev([0.5], T=2.0), ev([], T=2.0)
    assert torch.allclose(counting_l1(a, b), torch.tensor([1.5 / 2.0]))


def test_counting_l1_equals_integral():
    """Rubin Prop. 2: the sorted-pairing sum equals the counting-function integral."""
    a, b = poisson_pair()
    K = 20_000
    grid = counting_l1_grid(counting_curve(a, K), counting_curve(b, K))
    assert (counting_l1(a, b) - grid).abs().max() < 5e-3


def test_counting_l1_metric_properties():
    a, b = poisson_pair()
    assert torch.equal(counting_l1(a, a), torch.zeros(len(a)))
    assert torch.allclose(counting_l1(a, b), counting_l1(b, a))


def test_binned_counts():
    x = ev([0.0, 0.05, 0.999], [])
    out = x.binned(10)
    assert out[0, 0] == 2 and out[0, 9] == 1 and out[0].sum() == 3
    assert out[1].sum() == 0


def test_victor_purpura_cases():
    a = ev([0.5], [0.5], [], [0.1, 0.9])
    b = ev([0.5], [0.6], [0.4], [0.9])
    assert torch.allclose(victor_purpura(a, b, q=1.0), torch.tensor([0.0, 0.1, 1.0, 1.0]))
    # a shift costing more than delete + insert is never taken
    assert torch.allclose(victor_purpura(ev([0.5]), ev([0.6]), q=100.0), torch.tensor([2.0]))


def test_van_rossum_cases():
    a, b = ev([0.3, 0.7]), ev([])
    assert torch.allclose(van_rossum(a, a, tau=0.1), torch.zeros(1), atol=1e-6)
    single = van_rossum(ev([0.5]), ev([]), tau=0.1)
    assert torch.allclose(single, torch.tensor([math.sqrt(0.5)]))
    assert van_rossum(a, b, tau=0.1) > single


def test_count_error():
    assert torch.equal(count_error(ev([0.1, 0.2], []), ev([0.5], [0.3])),
                       torch.tensor([1.0, 1.0]))
