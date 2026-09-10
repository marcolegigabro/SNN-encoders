"""Information-theoretic reference curves for the synthetic-Poisson experiment.

The point of this file is that every number the trained system is compared
against is *computed*, not quoted. Three objects live here.

1. **R(D) of the source**, one per distortion. Hamming has a closed form on a
   memoryless binary source; the other two do not, and are obtained with
   Blahut-Arimoto on the exact source distribution.

2. **C(sigma) of the channel**, one per bottleneck variant. The TTFS bottleneck
   sends one real spike time per code neuron through additive Gaussian jitter,
   with the time confined to the window: that is a *peak-amplitude-constrained*
   AWGN channel, whose capacity has no closed form either (the 1/2 log(1+SNR)
   of an average-power constraint does not apply and over-states it at low
   jitter), so it too is Blahut-Arimoto. The multi-spike bottleneck sends a
   T-slot binary word whose spikes are displaced and re-binned; its transition
   matrix is built exactly, not sampled.

3. **OPTA**, the only bound that a joint source-channel system can be held to:
   with rho channel uses per source symbol, no scheme can achieve a distortion
   better than the D solving R(D) = rho * C. Both halves are computed above, so
   the bound is exact up to the discretisation of the continuous channel.

Everything is reported in **bits per source bin**, so a rate axis can be shared
between the closed form, the numerics and the measured systems.

Why R(D) and not H(source): at these block lengths R(D) is not achievable, it
is an outer bound. A gap between the trained system and the curve is therefore
expected and is the quantity of interest, and the curve is a bound in the
correct direction -- nothing can sit below it.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Blahut-Arimoto, both directions
# ---------------------------------------------------------------------------

def _rd_pareto_envelope(rates, dists):
    """Reduce a cloud of (rate, distortion) pairs to the tightest curve they
    support: the Pareto front, then its lower convex hull.

    Both steps are sound rather than cosmetic. Every test channel P, converged
    or not, satisfies R(D(P)) <= I(P), so each computed pair sits on or above
    the true R(D) and dropping a dominated pair can only tighten the estimate.
    And R(D) is convex, so the lower convex hull of achievable pairs is itself
    achievable (time-share two test channels). This matters because at |s| near
    zero the alternating minimisation barely moves off its initial point and
    would otherwise leave visibly loose pairs on the curve.
    """
    pts = sorted(zip(np.asarray(dists, float), np.asarray(rates, float)))
    front, best = [], np.inf
    for d, r in pts:                      # increasing D: keep strict rate drops
        if r < best - 1e-15:
            front.append((d, r))
            best = r
    hull = []
    for d, r in front:                    # lower convex hull, increasing D
        while len(hull) >= 2:
            (d0, r0), (d1, r1) = hull[-2], hull[-1]
            if (r1 - r0) * (d - d0) >= (r - r0) * (d1 - d0):
                hull.pop()
            else:
                break
        hull.append((d, r))
    d_out, r_out = np.asarray([h[0] for h in hull]), np.asarray([h[1] for h in hull])
    return r_out, d_out


def blahut_arimoto_rd(px, dist, s_values=None, n_iter=400, tol=1e-10,
                      device="cpu", dtype=torch.float64):
    """R(D) of a discrete memoryless source, in bits per source symbol.

    `px` is the source pmf over n_x letters, `dist[i, j]` the distortion of
    reproducing letter i as reproduction letter j. Returns (rates, distortions)
    traced out by the Lagrange multiplier s <= 0: s -> -inf drives D to its
    minimum, s = 0 gives the zero-rate point.

    The algorithm is the standard alternating minimisation (Blahut 1972): with
    s fixed, minimise over the reproduction marginal q and the test channel
    P(xhat|x) in turn. Convergence is to the exact R(D) because the objective
    is convex in each argument.
    """
    px = torch.as_tensor(px, dtype=dtype, device=device)
    d = torch.as_tensor(dist, dtype=dtype, device=device)
    px = px / px.sum()

    # The useful range of the multiplier scales as 1 / (distortion scale), which
    # differs by two orders of magnitude between these three distortions, so the
    # grid is normalised by the zero-rate distortion rather than hard-coded.
    d_char = max(float((px @ torch.as_tensor(dist, dtype=dtype, device=device)).min()), 1e-12)
    if s_values is None:
        s_values = -np.concatenate([np.logspace(2.5, -2.0, 48) / d_char, [0.0]])

    # Per-row shift of the distortion: it multiplies exp(s*d) by a constant
    # along each row, which cancels in the row normalisation of P, so P and
    # hence R and D are untouched. Keeps exp() inside range at large |s|.
    d_shift = d - d.min(dim=1, keepdim=True).values

    rates, dists = [], []
    for s in s_values:
        if s == 0.0:
            # Zero rate: the reproduction cannot depend on the source, so the
            # best it can do is the single letter minimising average distortion.
            dists.append(float((px @ d).min()))
            rates.append(0.0)
            continue

        A = torch.exp(s * d_shift)
        q = torch.full_like(px[:1].expand(d.shape[1]), 1.0 / d.shape[1]).clone()
        for _ in range(n_iter):
            P = q.unsqueeze(0) * A
            P = P / P.sum(dim=1, keepdim=True).clamp_min(1e-300)
            q_new = px @ P
            if float((q_new - q).abs().sum()) < tol:
                q = q_new
                break
            q = q_new

        P = q.unsqueeze(0) * A
        P = P / P.sum(dim=1, keepdim=True).clamp_min(1e-300)
        q = px @ P
        ratio = torch.where(P > 0, P / q.clamp_min(1e-300), torch.ones_like(P))
        mi = (px.unsqueeze(1) * P * torch.log(ratio.clamp_min(1e-300))).sum()
        rates.append(float(mi) / np.log(2.0))
        dists.append(float((px.unsqueeze(1) * P * d).sum()))

    return _rd_pareto_envelope(rates, dists)


def blahut_arimoto_capacity(W, n_iter=3000, tol=1e-12, dtype=np.float64):
    """Capacity of a discrete memoryless channel in bits per channel use.

    `W[i, j]` = P(output j | input i), rows summing to 1. Arimoto's (1972)
    iteration on the input distribution; the returned `r` is capacity-achieving.
    """
    W = np.asarray(W, dtype=dtype)
    W = W / W.sum(axis=1, keepdims=True)
    n_in = W.shape[0]
    r = np.full(n_in, 1.0 / n_in, dtype=dtype)

    logW = np.where(W > 0, np.log(np.maximum(W, 1e-300)), 0.0)
    for _ in range(n_iter):
        q = r @ W
        dx = (W * (logW - np.log(np.maximum(q, 1e-300))[None, :])).sum(axis=1)
        r_new = r * np.exp(dx - dx.max())
        total = r_new.sum()
        if total <= 0:
            break
        r_new /= total
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new

    q = r @ W
    dx = (W * (logW - np.log(np.maximum(q, 1e-300))[None, :])).sum(axis=1)
    return float((r * dx).sum() / np.log(2.0)), r


# ---------------------------------------------------------------------------
# The source, and its three R(D) curves
# ---------------------------------------------------------------------------

def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    out[m] = -(p[m] * np.log2(p[m]) + (1 - p[m]) * np.log2(1 - p[m]))
    return out


def hamming_rd(p, D):
    """Closed form R(D) = h(p) - h(D) for a Bernoulli(p) source under Hamming
    distortion, in bits per bin. Shannon 1959; exact, no numerics involved.

    Zero above D = min(p, 1-p): beyond that the constant reproduction already
    achieves the distortion for free.
    """
    D = np.asarray(D, dtype=float)
    r = binary_entropy(np.full_like(D, float(p))) - binary_entropy(D)
    return np.maximum(np.where(D < min(p, 1 - p), r, 0.0), 0.0)


def binary_words(T):
    """All 2^T binary words of length T as a (2^T, T) float array, bit t of
    index i in column t (little-endian, so index = sum_t bit_t << t)."""
    idx = np.arange(2 ** T, dtype=np.int64)
    return ((idx[:, None] >> np.arange(T)[None, :]) & 1).astype(np.float64)


def bernoulli_word_pmf(T, p):
    """Exact pmf of an i.i.d. Bernoulli(p) word of length T over the 2^T words."""
    w = binary_words(T)
    k = w.sum(axis=1)
    return (p ** k) * ((1 - p) ** (T - k))


def psp_trace(words, tau):
    """Exponentially filtered trains, same recursion as src/snn.psp_filter, so
    the distortion the bound is computed on is bit-for-bit the one measured."""
    decay = 1.0 - 1.0 / tau
    out = np.empty_like(words)
    acc = np.zeros(words.shape[0])
    for t in range(words.shape[1]):
        acc = decay * acc + words[:, t] / tau
        out[:, t] = acc
    return out


def blahut_arimoto_rd_real(px, feats, n_repro=512, s_values=None, n_iter=200,
                           tol=1e-10, device="cpu", dtype=torch.float64,
                           seed=0):
    """R(D) for a squared-error distortion with *real-valued* reproductions.

        d(x, y) = ||f(x) - y||^2 / dim

    Blahut-Arimoto fixes the reproduction alphabet in advance, which forces a
    choice for a distortion like this one, whose optimal reproductions are
    conditional means and therefore not source letters at all. So the
    reproduction points are optimised too: each sweep alternates the usual test
    channel update with the centroid condition
    y_j <- sum_x p(x) P(j|x) f(x) / sum_x p(x) P(j|x), which is the
    entropy-constrained vector quantiser fixed point (Chou, Lookabaugh & Gray
    1989). Every iterate is still a valid test channel, so every returned pair
    is achievable and the curve remains an upper bound on R(D) -- a tighter one
    than any fixed alphabet gives.

    Needed because two of the three distortions here are squared-error type, and
    scoring a squared-error distortion on a thresholded reconstruction throws
    away most of what the system knows: the minimiser is the posterior mean, not
    the most likely spike train.
    """
    px = torch.as_tensor(px, dtype=dtype, device=device)
    px = px / px.sum()
    f = torch.as_tensor(feats, dtype=dtype, device=device)
    n_x, dim = f.shape

    g = torch.Generator(device="cpu").manual_seed(seed)
    start = torch.multinomial(px.cpu().clamp_min(1e-300), min(n_repro, n_x),
                              replacement=False, generator=g)
    y0 = f[start.to(device)].clone()

    d0 = torch.cdist(f, y0) ** 2 / dim
    d_char = max(float((px @ d0).min()), 1e-12)
    if s_values is None:
        s_values = -np.concatenate([np.logspace(2.5, -2.0, 40) / d_char, [0.0]])

    rates, dists = [], []
    for s in s_values:
        y = y0.clone()
        q = torch.full((y.shape[0],), 1.0 / y.shape[0], dtype=dtype, device=device)
        for _ in range(n_iter):
            d = torch.cdist(f, y) ** 2 / dim
            A = torch.exp(s * (d - d.min(dim=1, keepdim=True).values))
            P = q.unsqueeze(0) * A
            P = P / P.sum(dim=1, keepdim=True).clamp_min(1e-300)
            q_new = px @ P
            w = px.unsqueeze(1) * P                       # joint p(x, j)
            mass = w.sum(dim=0)
            y_new = (w.T @ f) / mass.unsqueeze(1).clamp_min(1e-300)
            y = torch.where(mass.unsqueeze(1) > 1e-14, y_new, y)
            moved = float((q_new - q).abs().sum())
            q = q_new
            if moved < tol:
                break
        d = torch.cdist(f, y) ** 2 / dim
        A = torch.exp(s * (d - d.min(dim=1, keepdim=True).values))
        P = q.unsqueeze(0) * A
        P = P / P.sum(dim=1, keepdim=True).clamp_min(1e-300)
        q = px @ P
        ratio = torch.where(P > 0, P / q.clamp_min(1e-300), torch.ones_like(P))
        mi = (px.unsqueeze(1) * P * torch.log(ratio.clamp_min(1e-300))).sum()
        rates.append(float(mi) / np.log(2.0))
        dists.append(float((px.unsqueeze(1) * P * d).sum()))

    return _rd_pareto_envelope(rates, dists)


def van_rossum_rd(T, p, tau=3.0, s_values=None, device="cpu",
                  reproduction="real", n_repro=512):
    """R(D) for the van Rossum distortion, bits per neuron-window.

    The distortion is not single-letter -- filtering couples the T bins -- so
    the source symbol has to be the whole length-T word, and the R(D) of that
    2^T-letter source is computed exactly by Blahut-Arimoto.

    Two reproduction alphabets, and which one is right depends on what the
    system is allowed to output.

    `reproduction="real"` (default) lets the decoder emit a real-valued trace,
    which is the correct pairing for a squared-error distortion: its minimiser
    is a posterior mean. This is the curve the trained system, whose output is a
    probability per bin, is compared against.

    `reproduction="binary"` restricts reproductions to the 2^T spike trains. It
    is the right bound for a system forced to emit spikes, and being a
    restriction it lies above the real-valued curve.
    """
    w = binary_words(T)
    f = psp_trace(w, tau)
    ft = torch.as_tensor(f, dtype=torch.float64, device=device)
    px = bernoulli_word_pmf(T, p)
    if reproduction == "real":
        return blahut_arimoto_rd_real(px, ft, n_repro=n_repro,
                                      s_values=s_values, device=device)
    # mean squared difference per bin, so the units match the measured metric
    d = torch.cdist(ft, ft, p=2.0) ** 2 / T
    return blahut_arimoto_rd(px, d, s_values=s_values, device=device)


def count_mse_rd(T, p, n_repro=257, s_values=None):
    """R(D) for squared error on the per-neuron spike count, bits per neuron.

    The distortion (count(x) - c_hat)^2 depends on the source word only through
    its count, so I(X;Xhat) >= I(count;Xhat) with equality when the test channel
    is driven by the count alone: the R(D) of the length-T word under this
    distortion *equals* the R(D) of its Binomial(T, p) count. That reduction is
    what makes this curve cheap and exact.

    The reproduction alphabet is a grid on [0, T] rather than the integers:
    squared error is minimised by conditional means, which are not integers, and
    the measured system is likewise scored on its real-valued expected count.
    """
    from scipy.stats import binom
    counts = np.arange(T + 1)
    px = binom.pmf(counts, T, p)
    repro = np.linspace(0.0, float(T), n_repro)
    d = (counts[:, None] - repro[None, :]) ** 2
    return blahut_arimoto_rd(px, d, s_values=s_values)


# ---------------------------------------------------------------------------
# The two channels, and their capacities
# ---------------------------------------------------------------------------

def ttfs_channel_capacity(t_win, sigma, n_in=193, n_out=None, pad=6.0):
    """Capacity of one TTFS spike time through Gaussian jitter, bits per spike.

    Input: a spike time confined to [0, t_win] -- a peak constraint, since a
    code neuron cannot fire before the window opens or after it closes.
    Output: that time plus N(0, sigma^2), unrestricted.

    Discretising the input restricts the encoder and discretising the output
    coarsens the receiver, so both approximations push the returned number
    *down*: this is a lower estimate of the true capacity, and refining the
    grids can only raise it. `pad` sets how far past the window the output grid
    reaches, in units of sigma.
    """
    x = np.linspace(0.0, t_win, n_in)
    if n_out is None:
        # keep the output bin comfortably finer than the noise it resolves
        span = t_win + 2 * pad * sigma
        n_out = int(min(40001, max(1025, span / (sigma / 12.0))))
    y = np.linspace(-pad * sigma, t_win + pad * sigma, n_out)
    W = norm.pdf((y[None, :] - x[:, None]) / sigma)
    W /= W.sum(axis=1, keepdims=True)
    return blahut_arimoto_capacity(W)[0]


def ttfs_capacity_highsnr(t_win, sigma):
    """log2(t_win / sigma) - 0.5*log2(2*pi*e), the high-resolution asymptote of
    the peak-constrained AWGN capacity (uniform input, so h(X) = log t_win, and
    the noise entropy is that of a Gaussian). Used only to check the numerics."""
    return np.log2(t_win / sigma) - 0.5 * np.log2(2 * np.pi * np.e)


def slot_displacement_matrix(T, sigma):
    """P(destination slot | origin slot) for one jittered spike, plus the
    probability the spike leaves the window and is lost.

    A spike nominally in slot t is emitted at time t + N(0, sigma^2) and
    re-binned by rounding, so it lands in slot d with the Gaussian mass of
    [d-1/2, d+1/2). Mass outside [-1/2, T-1/2) is a genuine erasure: the window
    has closed.
    """
    t = np.arange(T)[:, None]
    d = np.arange(T)[None, :]
    hi = norm.cdf((d + 0.5 - t) / sigma)
    lo = norm.cdf((d - 0.5 - t) / sigma)
    P = hi - lo
    return P, 1.0 - P.sum(axis=1)


def multispike_channel_matrix(T, sigma, device="cpu", dtype=torch.float64):
    """Exact transition matrix of the multi-spike bottleneck channel.

    Input and output are both length-T binary words (2^T of each). Every spike
    of the input word is displaced independently by `slot_displacement_matrix`
    and the output word is the OR of where they land, so two spikes colliding in
    one slot are indistinguishable from one -- a real, lossy channel.

    Built exactly rather than by Monte Carlo. Sampling would leave most output
    words unobserved for the heavier input words, which makes the empirical
    channel look cleaner than it is and *inflates* the capacity, i.e. it would
    corrupt the bound in the dangerous direction. The exact construction folds
    the spikes in one at a time, carrying the full distribution over the 2^T
    output masks: for each origin slot t and destination d, mass at mask m moves
    to mask m | (1 << d).
    """
    P, p_lost = slot_displacement_matrix(T, sigma)
    n = 2 ** T
    words = ((torch.arange(n, device=device) >> torch.arange(T, device=device)[:, None]) & 1)
    words = words.T  # (n, T)

    W = torch.zeros(n, n, dtype=dtype, device=device)
    W[:, 0] = 1.0  # no spikes placed yet: all mass on the empty output word
    masks = torch.arange(n, device=device)
    for t in range(T):
        has = words[:, t].bool()
        if not bool(has.any()):
            continue
        src = W[has]                                     # (n_has, n)
        nxt = torch.zeros_like(src)
        nxt.index_add_(1, masks, src * float(p_lost[t]))  # spike lost, mask unchanged
        for d in range(T):
            pd = float(P[t, d])
            if pd <= 0.0:
                continue
            nxt.index_add_(1, masks | (1 << d), src * pd)
        W[has] = nxt
    return W


def multispike_channel_capacity(T, sigma, device="cpu"):
    """Capacity of the multi-spike bottleneck channel, bits per neuron-window."""
    W = multispike_channel_matrix(T, sigma, device=device)
    return blahut_arimoto_capacity(W.cpu().numpy())[0]


# ---------------------------------------------------------------------------
# OPTA
# ---------------------------------------------------------------------------

def rate_needed(distortion, rates, dists):
    """The rate the bound says is required to reach a given distortion.

    The other reading of the same curve, and the useful one at high rate: a
    ratio of distortions blows up as the bound approaches zero, while
    "this system spends N times the rate theory needs for the quality it
    delivers" stays finite and interpretable everywhere. Returns 0 for a
    distortion at or above the zero-rate point, since nothing need be sent.
    """
    r = np.asarray(rates, dtype=float)
    d = np.asarray(dists, dtype=float)
    order = np.argsort(d)
    d, r = d[order], r[order]
    return np.interp(distortion, d, r, left=r[0], right=0.0)


def opta_distortion(rate_bits_per_symbol, rates, dists):
    """Invert a computed R(D): the smallest distortion whose rate requirement
    fits in the given budget. Both arrays must be in the same units per symbol.

    This is the whole content of the separation-theorem bound for a joint
    source-channel system: with a channel able to carry `rate_bits_per_symbol`
    bits per source symbol, no encoder-decoder pair, joint or separate, can go
    below this distortion.
    """
    r = np.asarray(rates, dtype=float)
    d = np.asarray(dists, dtype=float)
    order = np.argsort(r)
    r, d = r[order], d[order]
    budget = np.atleast_1d(np.asarray(rate_bits_per_symbol, dtype=float))
    out = np.interp(budget, r, d, left=d[0], right=d[-1])
    out = np.where(budget >= r[-1], d[-1], out)
    return out if np.ndim(rate_bits_per_symbol) else float(out[0])


# ---------------------------------------------------------------------------
# cached capacity lookup
# ---------------------------------------------------------------------------

def channel_capacity(bottleneck: str, n_bins: int, sigma: float,
                     cache_path=None) -> float:
    """C(sigma) in bits per channel use, for whichever channel the system uses.

    Cached on disk: the multi-spike capacity needs a 2^T x 2^T Blahut-Arimoto
    and the sweep asks for the same handful of (T, sigma) pairs dozens of times.
    """
    import json
    from pathlib import Path

    key = f"{bottleneck}|T={n_bins}|sigma={sigma:g}"
    cache = {}
    p = Path(cache_path) if cache_path else None
    if p is not None and p.exists():
        cache = json.loads(p.read_text())
        if key in cache:
            return float(cache[key])

    if bottleneck == "ttfs":
        c = ttfs_channel_capacity(float(n_bins), sigma)
    elif bottleneck == "multispike":
        c = multispike_channel_capacity(n_bins, sigma)
    else:
        raise ValueError(bottleneck)

    if p is not None:
        cache[key] = c
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return c
