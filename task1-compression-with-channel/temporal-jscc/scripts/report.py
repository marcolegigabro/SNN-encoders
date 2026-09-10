#!/usr/bin/env python
"""The results table: every trained point next to the OPTA bound at its own rate.

The figures show the same thing, but the number that answers "how close to
theory" is the ratio in the last column, and it deserves to be printed rather
than eyeballed off a plot. A ratio below 1 is impossible and means a bug --
either in the rate accounting or in the distortion definition -- so this table
is also the check that the two halves of the experiment are consistent.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import theory as th

ap = argparse.ArgumentParser()
ap.add_argument("--theory", type=Path, default=Path("runs/theory.json"))
ap.add_argument("--sweep", type=Path, default=Path("runs/sweep/summary.json"))
ap.add_argument("--baselines", type=Path, default=Path("runs/baselines.json"))
ap.add_argument("--grid", default="primary")
ap.add_argument("--markdown", action="store_true")
a = ap.parse_args()

theory = json.loads(a.theory.read_text())
rows = [r for r in json.loads(a.sweep.read_text()) if f"/{a.grid}/" in r["out"]]
base = json.loads(a.baselines.read_text())["rows"] if a.baselines.exists() else []
zero = next((b for b in base if b["name"] == "zero_rate"), None)

for dname in ("hamming", "count_mse", "van_rossum"):
    curve = theory[dname]
    R = np.asarray(curve["rate_bits_per_bin"])
    D = np.asarray(curve["distortion"])
    pts = sorted([r for r in rows if r["distortion"] == dname],
                 key=lambda r: r["rate_bits_per_bin"])
    if not pts:
        continue
    print(f"\n### {dname}  [{curve['unit']}]   bound: {curve['source']}")
    head = ["K", "sigma", "C(sigma)", "rate spent", "measured D", "OPTA D",
            "rate needed", "rate excess", "vs zero rate"]
    fmt = "| " + " | ".join(f"{h:>11s}" for h in head) + " |"
    print(fmt)
    print("|" + "|".join(["-" * 13] * len(head)) + "|")
    for r in pts:
        spent = r["rate_bits_per_bin"]
        opta = th.opta_distortion(spent, R, D)
        meas = r[dname]
        needed = float(th.rate_needed(meas, R, D))
        gain = (zero[dname] / meas) if zero and meas > 0 else float("nan")
        cells = [f"{r['n_code']:d}", f"{r['sigma']:g}",
                 f"{r['capacity_bits_per_use']:.3f}",
                 f"{spent:.4f}", f"{meas:.5f}", f"{opta:.5f}",
                 f"{needed:.4f}",
                 f"{spent / needed:.1f}x" if needed > 1e-9 else "inf",
                 f"{gain:.2f}x" if gain == gain else "-"]
        print("| " + " | ".join(f"{c:>11s}" for c in cells) + " |")
