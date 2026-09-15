"""Real entropy coding: turn symbols + probability tables into an actual bitstream.

Training and the rate axis use the ideal code length -log2 p. This module checks
that the ideal length is achievable, with constriction's range coder
(Bamler 2022). Every evaluation set is coded as **one stream**: a range coder
pays a constant flush overhead per stream (a few 32-bit words), which would
swamp the ~10-30 bits of a single window if each window were coded on its own.

`num_bits` is rounded up to whole 32-bit words, so real/ideal includes that.
"""

from __future__ import annotations

import constriction
import numpy as np
from scipy.stats import poisson


def _family():
    return constriction.stream.model.Categorical(perfect=False)


def encode_categorical(symbols: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Encode symbols[i] under categorical probs[i] (rows of an (N, A) table)."""
    enc = constriction.stream.queue.RangeEncoder()
    probs = np.ascontiguousarray(probs, dtype=np.float32).reshape(-1, probs.shape[-1])
    enc.encode(np.ascontiguousarray(symbols, dtype=np.int32).ravel(), _family(), probs)
    return enc.get_compressed()


def decode_categorical(compressed: np.ndarray, probs: np.ndarray) -> np.ndarray:
    dec = constriction.stream.queue.RangeDecoder(compressed)
    probs = np.ascontiguousarray(probs, dtype=np.float32).reshape(-1, probs.shape[-1])
    return dec.decode(_family(), probs)


def _report(compressed: np.ndarray, ideal_bits: float, n_windows: int) -> dict:
    real = 32 * int(compressed.size)
    return {
        "real_bits": real,
        "ideal_bits": float(ideal_bits),
        "real_bits_per_window": real / n_windows,
        "ideal_bits_per_window": float(ideal_bits) / n_windows,
        "overhead": real / max(float(ideal_bits), 1e-12) - 1.0,
    }


def latent_stream(symbols: np.ndarray, probs: np.ndarray) -> dict:
    """Code (B, L) latent symbols with a factorized prior of shape (L, A)."""
    B, L = symbols.shape
    probs = probs / probs.sum(-1, keepdims=True)
    table = np.broadcast_to(probs, (B, L, probs.shape[-1]))
    compressed = encode_categorical(symbols, table)
    ideal = -np.log2(probs[np.arange(L), symbols]).sum()
    return _report(compressed, ideal, B)


def poisson_table(mean: float, n_symbols: int) -> np.ndarray:
    """Poisson(mean) pmf on {0..n_symbols-1}, tail mass folded into the last symbol."""
    p = poisson.pmf(np.arange(n_symbols), mean)
    p[-1] += poisson.sf(n_symbols - 1, mean)
    return p


def poisson_stream(bin_counts: np.ndarray, mean: float) -> dict:
    """Code (B, n_bins) counts, each under Poisson(mean): Rubin's baseline bitstream."""
    B = bin_counts.shape[0]
    n_symbols = int(max(bin_counts.max(), poisson.ppf(1 - 1e-12, mean))) + 1
    p = poisson_table(mean, n_symbols)
    flat = bin_counts.ravel()
    compressed = encode_categorical(flat, np.broadcast_to(p, (flat.size, n_symbols)))
    ideal = -poisson.logpmf(flat, mean).sum() / np.log(2.0)
    return _report(compressed, ideal, B)
