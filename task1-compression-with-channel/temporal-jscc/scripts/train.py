#!/usr/bin/env python
"""Train one point of the rate-distortion curve."""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import Config, train

p = argparse.ArgumentParser()
for f in fields(Config):
    p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default),
                   default=f.default)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
res = train(cfg, a.out, cache_path=Path("runs/capacity-cache.json"))
print(f"rate {res['rate_bits_per_bin']:.4f} bits/bin  "
      f"{cfg.distortion} = {res['primary']:.5f}")
