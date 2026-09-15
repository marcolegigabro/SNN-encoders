"""Per-window summary features of a point-process realization, for interpretability.

Each feature is a (B,) float tensor. Conventions for degenerate windows:
first event time = T and last event time = 0 when the window is empty, mean
event time = T/2 when empty, and the inter-event-interval CV is NaN unless the
window holds at least 3 events (2 intervals).
"""

from __future__ import annotations

import torch

from .sources import Events

FEATURES = ("count", "count_first_half", "count_second_half", "first_time", "last_time",
            "mean_time", "max_gap", "isi_cv")
LABELS = {"count": "N", "count_first_half": "N, first half", "count_second_half": "N, second half",
          "first_time": "first event", "last_time": "last event", "mean_time": "mean event time",
          "max_gap": "longest gap", "isi_cv": "ISI CV (burstiness)"}


def window_features(ev: Events) -> dict[str, torch.Tensor]:
    t, m, T = ev.times, ev.mask, ev.T
    n = ev.counts.to(t.dtype)
    empty = ev.counts == 0

    in_first_half = ((t < T / 2) & m).sum(1).to(t.dtype)
    first = torch.where(empty, torch.full_like(n, T), t[:, 0])
    last = t.gather(1, (ev.counts - 1).clamp(min=0)[:, None]).squeeze(1)
    last = torch.where(empty, torch.zeros_like(n), last)
    mean_time = torch.where(empty, torch.full_like(n, T / 2), (t * m).sum(1) / n.clamp(min=1))

    # gaps including both window edges; padding sits at T, so it only adds zero gaps
    edges = torch.cat([torch.zeros_like(t[:, :1]), t, torch.full_like(t[:, :1], T)], 1)
    max_gap = edges.diff(dim=1).max(1).values

    isi = t.diff(dim=1)
    isi_mask = m[:, 1:].to(t.dtype)          # both endpoints are real events
    k = isi_mask.sum(1).clamp(min=1)
    mu = (isi * isi_mask).sum(1) / k
    var = ((isi - mu[:, None]) ** 2 * isi_mask).sum(1) / k
    cv = torch.where(ev.counts >= 3, var.sqrt() / mu.clamp(min=1e-9), torch.full_like(n, float("nan")))

    return {"count": n, "count_first_half": in_first_half, "count_second_half": n - in_first_half,
            "first_time": first, "last_time": last, "mean_time": mean_time,
            "max_gap": max_gap, "isi_cv": cv}
