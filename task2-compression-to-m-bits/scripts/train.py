#!/usr/bin/env python
"""Train one m-bit spike-train compressor on N-MNIST."""
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
p.add_argument("--data-root", type=Path, default=None)
a = p.parse_args()

cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
train(cfg, a.out, a.data_root)
