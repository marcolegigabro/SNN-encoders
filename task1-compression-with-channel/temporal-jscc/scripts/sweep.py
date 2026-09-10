#!/usr/bin/env python
"""One model per (rate point, distortion) -> the rate-distortion curves.

The rate is swept two ways on purpose. Adding code neurons buys rate by using
the channel more often; shrinking the jitter buys rate by making each use worth
more. Both land on the same axis once each use is priced at C(sigma), and if the
two sweeps trace the same curve then that pricing is doing its job -- which is
the one assumption in the whole rate axis worth checking empirically.

Runs are one process each, single-threaded: these networks are small enough that
one thread per model and many models at once is about four times faster than the
reverse (measured: 46 ms/step at one thread, 60 ms at four).
"""
import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

K_SWEEP = [1, 2, 4, 8, 16, 32, 64]      # at the reference jitter
SIGMA_SWEEP = [0.25, 0.5, 2.0, 4.0]     # at the reference code size
SIGMA_REF = 1.0
K_REF = 8

GRIDS = {
    "primary": dict(bottleneck="ttfs", source="iid"),
    "multispike": dict(bottleneck="multispike", source="iid"),
    "latent": dict(bottleneck="ttfs", source="latent"),
}


# The control grids (multi-spike bottleneck, correlated source) exist to say
# whether a conclusion drawn on the primary grid survives a change of
# bottleneck or of source, and that question is answered by the shape of the
# curve, not by its resolution. So they get a coarse subset of the same points,
# never different points: the two grids have to be readable on one axis.
COARSE_K = [1, 4, 16, 64]
COARSE_SIGMA = [0.25, 2.0]


def rate_points(coarse: bool = False):
    ks = COARSE_K if coarse else K_SWEEP
    ss = COARSE_SIGMA if coarse else SIGMA_SWEEP
    return [(k, SIGMA_REF) for k in ks] + [(K_REF, s) for s in ss]


def _run(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from src.train import Config, train

    cfg_kw, out = job
    cfg = Config(**cfg_kw)
    out = Path(out)
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text()) | {"cached": True}
    res = train(cfg, out, cache_path=Path("runs/capacity-cache.json"),
                verbose=False)
    return res | {"cached": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/sweep"))
    ap.add_argument("--grid", default="primary", choices=[*GRIDS, "all"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--distortions", default="hamming,count_mse,van_rossum")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--coarse", action="store_true",
                    help="the reduced point set, for the control grids")
    a = ap.parse_args()

    from src.train import Config
    grids = list(GRIDS) if a.grid == "all" else [a.grid]

    jobs = []
    for gname in grids:
        for dname in a.distortions.split(","):
            for k, sigma in rate_points(a.coarse):
                cfg = Config(steps=a.steps, lr=a.lr, distortion=dname,
                             n_code=k, sigma=sigma, **GRIDS[gname])
                tag = f"{gname}/{dname}/K{k:03d}-s{sigma:g}"
                jobs.append((asdict(cfg), str(a.out / tag)))

    # Warm the capacity cache serially: the multi-spike capacity is a
    # 2^T x 2^T Blahut-Arimoto and having eight workers race to compute the
    # same one, then overwrite each other's cache file, is pure waste.
    from src import theory as th
    cache = Path("runs/capacity-cache.json")
    for gname in grids:
        bn = GRIDS[gname]["bottleneck"]
        for k, sigma in rate_points(a.coarse):
            th.channel_capacity(bn, Config().n_bins, sigma, cache)
    print(f"capacity cache: {len(json.loads(cache.read_text()))} entries")

    print(f"{len(jobs)} runs on {a.workers} workers")
    from multiprocessing import Pool
    done = 0
    rows = []
    with Pool(a.workers) as pool:
        for (cfg_kw, out), res in zip(jobs, pool.imap(_run, jobs)):
            done += 1
            rows.append({"out": out, **cfg_kw, **res})
            print(f"[{done}/{len(jobs)}] {out}  rate {res['rate_bits_per_bin']:.4f}"
                  f"  {cfg_kw['distortion']} {res['primary']:.5f}"
                  f"{'  (cached)' if res['cached'] else ''}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out / 'summary.json'}")


if __name__ == "__main__":
    main()
