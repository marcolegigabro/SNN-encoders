#!/usr/bin/env python
"""Train one compressor per bit budget m, to trace the rate-distortion curve.

Task 2 asks for the tradeoff between the number of bits m and reconstruction
quality, so the sweep is the actual deliverable rather than a hyperparameter
search: every run shares the same architecture, schedule and seed, and only
`n_bits` moves.
"""
import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import Config, train

p = argparse.ArgumentParser()
p.add_argument("--bits", type=int, nargs="+",
               default=[16, 32, 64, 128, 256, 512, 1024])
p.add_argument("--out", type=Path, required=True)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--train-subset", type=int, default=0)
p.add_argument("--workers", type=int, default=6)
p.add_argument("--data-root", type=Path, default=None)
a = p.parse_args()

a.out.mkdir(parents=True, exist_ok=True)
summary = []
for m in a.bits:
    run_dir = a.out / f"m{m:05d}"
    if (run_dir / "config.json").exists():
        print(f"== m={m}: already done, skipping", flush=True)
    else:
        print(f"\n{'=' * 70}\n== m = {m} bits\n{'=' * 70}", flush=True)
        t0 = time.time()
        cfg = Config(n_bits=m, epochs=a.epochs, train_subset=a.train_subset,
                     workers=a.workers)
        train(cfg, run_dir, a.data_root)
        print(f"== m={m} done in {(time.time() - t0) / 60:.1f} min", flush=True)

    hist = json.loads((run_dir / "history.json").read_text())
    best = max(hist, key=lambda s: s["soft_frame_corr"])
    summary.append({"n_bits": m, "ratio": 16 * 2 * 34 * 34 / m, **best})
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))

hdr = (f"\n{'m':>6} {'ratio':>7} {'corr(soft)':>11} {'corr(samp)':>11} "
       f"{'vR(soft)':>9} {'vR(samp)':>9} {'F1(samp)':>9} {'bits used':>11}")
print(hdr)
for s in summary:
    print(f"{s['n_bits']:>6} {s['ratio']:>6.0f}x {s['soft_frame_corr']:>11.4f} "
          f"{s['sample_frame_corr']:>11.4f} {s['soft_van_rossum']:>9.3f} "
          f"{s['sample_van_rossum']:>9.3f} {s['sample_f1']:>9.4f} "
          f"{s['code_bits_used']:>5}/{s['n_bits']}")
