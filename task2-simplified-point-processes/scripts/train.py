#!/usr/bin/env python
"""Train one point-process codec at one rate-distortion tradeoff beta."""
import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# torch.compile's cache defaults to /tmp, which a reboot wipes (costing ~8 min
# of recompilation for the SNN encoder); keep it with the project instead
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache" / "torchinductor"))

sys.path.insert(0, str(ROOT))
from src.train import Config, train

p = argparse.ArgumentParser()
for f in fields(Config):
    p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
train(cfg, a.out)
