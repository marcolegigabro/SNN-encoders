import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import baseline, coding
from src.sources import homogeneous_poisson


def random_latents(n=10_000, L=8, A=16, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(A, 0.7), size=L)
    sym = np.stack([rng.choice(A, size=n, p=probs[l]) for l in range(L)], 1)
    return sym, probs


def test_latent_roundtrip():
    sym, probs = random_latents(n=2000)
    table = np.broadcast_to(probs, (*sym.shape, probs.shape[-1]))
    compressed = coding.encode_categorical(sym, table)
    assert np.array_equal(coding.decode_categorical(compressed, table), sym.ravel())


def test_latent_real_bits_close_to_ideal():
    sym, probs = random_latents()
    r = coding.latent_stream(sym, probs)
    assert r["real_bits"] >= r["ideal_bits"] - 64
    assert r["real_bits"] <= 1.01 * r["ideal_bits"] + 64


def test_poisson_stream_close_to_ideal():
    g = torch.Generator().manual_seed(3)
    ev = homogeneous_poisson(10_000, 5.0, generator=g)
    counts = baseline.encode(ev, 16).numpy()
    r = coding.poisson_stream(counts, 5.0 / 16)
    assert abs(r["real_bits"] - r["ideal_bits"]) <= 0.01 * r["ideal_bits"] + 64
    ideal_torch = baseline.ideal_bits(torch.from_numpy(counts), 5.0, 1.0).double().sum().item()
    assert abs(ideal_torch - r["ideal_bits"]) < 1e-3 * r["ideal_bits"]
