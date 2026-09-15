import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import FEATURES, window_features
from src.sources import from_times


def ev(*rows, T=1.0):
    L = max(max(len(r) for r in rows), 1)
    t = torch.full((len(rows), L), T)
    for i, r in enumerate(rows):
        t[i, :len(r)] = torch.tensor(r, dtype=torch.float32)
    return from_times(t, T)


def test_hand_computed_window():
    f = window_features(ev([0.1, 0.4, 0.9]))
    assert f["count"][0] == 3
    assert f["count_first_half"][0] == 2 and f["count_second_half"][0] == 1
    assert math.isclose(f["first_time"][0].item(), 0.1, abs_tol=1e-6)
    assert math.isclose(f["last_time"][0].item(), 0.9, abs_tol=1e-6)
    assert math.isclose(f["mean_time"][0].item(), 1.4 / 3, abs_tol=1e-6)
    assert math.isclose(f["max_gap"][0].item(), 0.5, abs_tol=1e-6)        # gaps 0.1, 0.3, 0.5, 0.1
    assert math.isclose(f["isi_cv"][0].item(), 0.25, abs_tol=1e-5)        # intervals 0.3, 0.5


def test_degenerate_windows():
    f = window_features(ev([], [0.2, 0.3]))
    assert f["count"][0] == 0 and f["first_time"][0] == 1.0 and f["last_time"][0] == 0.0
    assert f["mean_time"][0] == 0.5 and f["max_gap"][0] == 1.0
    assert math.isnan(f["isi_cv"][0].item()) and math.isnan(f["isi_cv"][1].item())
    assert math.isclose(f["max_gap"][1].item(), 0.7, abs_tol=1e-6)
    assert set(f) == set(FEATURES)
