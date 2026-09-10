#!/usr/bin/env python
"""Compute every reference curve the experiment is judged against.

Writes runs/theory.json: the three R(D) curves in bits per source bin, both
channels' capacities over a jitter grid, and the self-checks that say whether
the numerics can be trusted. Nothing here depends on a trained model, and it is
the file to look at first when a measured point looks impossible.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import theory as th

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("runs/theory.json"))
ap.add_argument("--n-bins", type=int, default=12)
ap.add_argument("--rate", type=float, default=0.15)
ap.add_argument("--tau", type=float, default=3.0)
ap.add_argument("--sigmas", default="0.125,0.25,0.5,1.0,2.0,4.0,8.0")
a = ap.parse_args()

T, p = a.n_bins, a.rate
out = {"n_bins": T, "source_rate": p, "tau": a.tau, "checks": {}}
t_start = time.time()

# --- Hamming: closed form, on a grid dense near D = 0 where the curve is steep
D = np.concatenate([np.linspace(1e-4, min(p, 1 - p), 400)])
out["hamming"] = {"rate_bits_per_bin": th.hamming_rd(p, D).tolist(),
                  "distortion": D.tolist(),
                  "unit": "P(bit error)",
                  "source": "closed form h(p) - h(D)"}

# Blahut-Arimoto on the same problem, as a check on the solver that is then
# reused for the two distortions with no closed form to check against.
px = np.array([1 - p, p])
R_ba, D_ba = th.blahut_arimoto_rd(px, np.array([[0.0, 1.0], [1.0, 0.0]]))
out["checks"]["hamming_ba_vs_closed_form_max_abs_err"] = float(
    np.abs(R_ba - th.hamming_rd(p, D_ba)).max())

# --- count MSE: R(D) of the Binomial(T, p) count, per neuron -> per bin
t0 = time.time()
R, Dc = th.count_mse_rd(T, p)
out["count_mse"] = {"rate_bits_per_bin": (R / T).tolist(),
                    "distortion": Dc.tolist(),
                    "unit": "spikes^2 per neuron",
                    "source": f"Blahut-Arimoto, Binomial({T}, {p}) count, "
                              "real-valued reproduction grid"}
from scipy.stats import binom
q = binom.pmf(np.arange(T + 1), T, p)
out["checks"]["count_mse_zero_rate_vs_variance"] = [float(Dc.max()), float(T * p * (1 - p))]
out["checks"]["count_mse_max_rate_vs_count_entropy"] = [
    float(R.max()), float(-(q * np.log2(q + 1e-300)).sum())]
print(f"count_mse R(D): {len(R)} points in {time.time() - t0:.1f}s")

# --- van Rossum: R(D) of the whole length-T word, per neuron-window -> per bin
#
# Both reproduction alphabets, because both are the right answer to a different
# question. `van_rossum` (real-valued reproductions, optimised) is what the
# trained system is compared against, since its output is a probability per bin
# and squared error is minimised by posterior means. `van_rossum_binary`
# (reproductions restricted to spike trains) is the bound for a receiver that
# must emit spikes, and lies above the other by construction.
for key, mode in (("van_rossum", "real"), ("van_rossum_binary", "binary")):
    t0 = time.time()
    R, Dv = th.van_rossum_rd(T, p, tau=a.tau, reproduction=mode)
    out[key] = {"rate_bits_per_bin": (R / T).tolist(),
                "distortion": Dv.tolist(),
                "unit": "mean sq. PSP error",
                "source": f"Blahut-Arimoto over all 2^{T} words, "
                          f"{mode}-valued reproduction alphabet"}
    out["checks"][f"{key}_max_rate_vs_word_entropy"] = [
        float(R.max()), float(T * th.binary_entropy(np.array([p]))[0])]
    print(f"{key} R(D): {len(R)} points in {time.time() - t0:.1f}s")
# Both curves are upper bounds on the same R(D), and the real-reproduction run
# starts from the binary alphabet, so its estimate must be no looser: at matched
# rate its distortion has to sit at or below the binary one. A negative margin
# here means the reproduction-point optimisation is under-resolved, which is
# exactly how a 512-point subsample failed.
rr = np.asarray(out["van_rossum"]["rate_bits_per_bin"])
dr = np.asarray(out["van_rossum"]["distortion"])
rb = np.asarray(out["van_rossum_binary"]["rate_bits_per_bin"])
db = np.asarray(out["van_rossum_binary"]["distortion"])
out["checks"]["van_rossum_real_no_looser_than_binary_min_margin"] = float(
    (db - np.interp(rb, rr[::-1], dr[::-1])).min())

# --- capacities
sigmas = [float(s) for s in a.sigmas.split(",")]
cap = {"sigma": sigmas, "ttfs": [], "ttfs_highsnr_asymptote": [], "multispike": []}
cache = Path("runs/capacity-cache.json")
for s in sigmas:
    t0 = time.time()
    cap["ttfs"].append(th.channel_capacity("ttfs", T, s, cache))
    cap["ttfs_highsnr_asymptote"].append(float(th.ttfs_capacity_highsnr(float(T), s)))
    cap["multispike"].append(th.channel_capacity("multispike", T, s, cache))
    print(f"sigma={s:5.3f}  C_ttfs={cap['ttfs'][-1]:.4f}  "
          f"C_multispike={cap['multispike'][-1]:.4f}  ({time.time() - t0:.0f}s)")
cap["note"] = ("bits per channel use; one TTFS use is one spike time through "
               "jitter, one multispike use is one T-slot word through "
               "displacement, so the two are not comparable per use")
# A noiseless multi-spike channel must carry exactly T bits per window; if this
# is not T the exact transition-matrix construction is wrong.
out["checks"]["multispike_noiseless_capacity_vs_T"] = [
    float(th.channel_capacity("multispike", T, 1e-3, cache)), float(T)]
out["capacity"] = cap

out["total_seconds"] = time.time() - t_start
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=1))
print(f"\nchecks: {json.dumps(out['checks'], indent=1)}")
print(f"wrote {a.out}  ({out['total_seconds']:.0f}s)")
