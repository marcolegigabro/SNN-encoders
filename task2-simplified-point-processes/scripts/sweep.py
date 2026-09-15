#!/usr/bin/env python
"""Train one codec per beta to trace a learned rate-distortion curve.

Every run shares the architecture, schedule, seed and test set; only beta moves.
A run counts as done when its history reaches the configured number of steps,
so an interrupted sweep resumes where it stopped (a partial run is retrained).

The operating point reported for each beta is the **final** evaluation, i.e.
`last.pt`, trained on the full objective. `best.pt` is not used here.

Writes <out>/beta<b>/ per run and <out>/summary.json.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# keep torch.compile's cache with the project (a reboot wipes /tmp)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache" / "torchinductor"))
sys.path.insert(0, str(ROOT))
from src.train import Config, train

KEYS = ("rate_real", "rate_ideal", "rate_symbol_entropy", "D", "D_se", "D_soft",
        "count_error", "active_coords", "spikes_per_window", "skipped_encoder_steps")

p = argparse.ArgumentParser()
p.add_argument("--betas", type=float, nargs="+",
               default=[5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])
p.add_argument("--out", type=Path, required=True)
for f in fields(Config):
    if f.name != "beta":
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
a = p.parse_args()

a.out.mkdir(parents=True, exist_ok=True)
summary = []
for beta in a.betas:
    cfg = Config(beta=beta, **{f.name: getattr(a, f.name) for f in fields(Config) if f.name != "beta"})
    run_dir = a.out / f"beta{beta:g}"
    hist_path = run_dir / "history.json"
    done = hist_path.exists() and json.loads(hist_path.read_text())[-1]["step"] == cfg.steps
    if done:
        print(f"== beta={beta:g}: already done, skipping", flush=True)
    else:
        print(f"\n{'=' * 78}\n== beta = {beta:g}\n{'=' * 78}", flush=True)
        t0 = time.time()
        train(cfg, run_dir)
        print(f"== beta={beta:g} done in {(time.time() - t0) / 60:.1f} min", flush=True)

    last = json.loads(hist_path.read_text())[-1]
    summary.append({"beta": beta, "encoder": cfg.encoder, "quantizer": cfg.quantizer,
                    "rate_lambda": cfg.rate, "T": cfg.T, "steps": cfg.steps,
                    **{k: last.get(k) for k in KEYS}})
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))

print(f"\n{'beta':>7} {'rate real':>10} {'D':>8} {'+-se':>7} {'D soft':>8} {'active':>7}")
for s in summary:
    print(f"{s['beta']:>7g} {s['rate_real']:>10.2f} {s['D']:>8.4f} {s['D_se']:>7.4f} "
          f"{s['D_soft']:>8.4f} {s['active_coords']:>5}/{a.n_latents}")
