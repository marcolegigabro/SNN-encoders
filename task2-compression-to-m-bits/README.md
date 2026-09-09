# Task 2 — end-to-end learned spike-train compression to an m-bit vector

> Implement an end-to-end system without a channel. The encoder outputs a
> fixed-length m-bit vector representing the input spike train. The decoder,
> implemented as an SNN, reconstructs the spike train from this compact
> representation. Investigate the tradeoff between the number of bits m and the
> reconstruction quality.

Source: N-MNIST (Orchard et al. 2015), the real event-camera recording of MNIST.

```
x in {0,1}^(16,2,34,34)  --SNN encoder-->  b in {0,1}^m  --SNN decoder-->  x_hat
        36992 bits                            m bits
```

## The system

**Spike source.** Each N-MNIST recording is binned onto 16 bins of 20 ms over a
fixed 320 ms window and binarised, giving a 36992-bit block at 7.4% occupancy
(~2000 spikes per recording). Binarising rather than counting is deliberate: the
subject's `x(t)` is a spike train, and a bin holding two events is still one
spike to the downstream SNN.

**Encoder.** A convolutional SNN of surrogate-gradient LIF neurons
(34→17→9→5, 32/64/128 channels, then a spiking FC layer), stepped once per time
bin. A non-firing readout layer integrates its membrane potential across all 16
bins into an m-dimensional vector.

**Bottleneck.** The readout potential is batch-normalised and hard-thresholded
at 0. This is a real threshold, not a relaxation: `b` is genuinely m bits at
evaluation, so the rate axis of the rate-distortion curve is exact.
Gradients cross it with a clipped straight-through estimator. Physically it is a
population of m readout neurons each allowed to fire at most once per recording,
and the code is the population's firing pattern.

The BatchNorm before the threshold is load-bearing. It centres the pre-threshold
values on 0, which keeps the bit marginals near 50/50 (so m bits carry close to
m bits of entropy instead of collapsing to constants) and keeps potentials inside
the straight-through window where gradients are non-zero.

**Decoder.** A deconvolutional SNN (5→9→17→34) driven by the *static* code
injected as a constant current at every time step. Its temporal structure is
therefore generated internally, by two mechanisms: an explicitly recurrent
spiking layer acting as a learned oscillator the code modulates, and a learned
per-time-step bias current. N-MNIST's three micro-saccades are a strongly
time-varying envelope and neither mechanism reproduces it alone. The output
neuron is again non-firing; its membrane potential is the Bernoulli logit of
each spike cell.

**Loss.** Four terms, each covering a failure of the others — pos-weighted BCE,
a soft-F1 (dice) term that cannot be satisfied by predicting silence, a
PSP-filtered MSE (a soft van Rossum distance, tolerant of one-bin jitter), and
an MSE on the time-summed frame. Weights are set from the measured magnitudes of
the terms so none is decorative; see `src/losses.py`.

## Layout

```
src/
  data.py      N-MNIST -> cached bit-packed binary spike trains
  snn.py       surrogate-gradient LIF neurons, spiking conv/linear/recurrent layers
  model.py     encoder, m-bit bottleneck, decoder
  losses.py    the four-term spike reconstruction loss
  metrics.py   spike agreement, van Rossum, image-domain, code usage
  train.py     training loop and evaluation
scripts/
  build_cache.py  bin the raw recordings once (~2 min, 12 workers)
  train.py        train one model at one m
  sweep.py        one model per m -> the rate-distortion curve
  baselines.py    silent / dataset-mean / PCA-sign reference points
  probe.py        semantic evaluation (code probe + reconstruction probe)
  visualize.py    all figures
```

## Reproducing

```bash
source ../../.venv/bin/activate
python scripts/build_cache.py --n-bins 16
python scripts/train.py --out runs/m128 --n-bits 128 --epochs 30
python scripts/visualize.py --ckpt runs/m128/best.pt --out-dir figures
python scripts/sweep.py --out runs/sweep --epochs 20
python scripts/visualize.py --sweep-dir runs/sweep --out-dir figures
```

## Results

All numbers on the 10000-sample N-MNIST test set, 20 epochs per budget, one
architecture and schedule shared across the sweep (`runs/sweep`).

### Rate-distortion

| m | ratio | corr (soft) | corr (sampled) | vR (soft) | vR (sampled) | F1 (sampled) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2312x | 0.904 | 0.866 | 0.639 | 0.768 | 0.484 |
| 32 | 1156x | 0.917 | 0.877 | 0.608 | 0.723 | 0.514 |
| 64 | 578x | 0.930 | 0.896 | 0.582 | 0.701 | 0.552 |
| 128 | 289x | 0.930 | 0.895 | 0.581 | 0.696 | 0.549 |
| 256 | 144x | 0.931 | 0.900 | 0.575 | 0.688 | 0.569 |
| 512 | 72x | 0.935 | 0.904 | 0.556 | 0.668 | 0.588 |
| 1024 | 36x | 0.935 | 0.905 | 0.560 | 0.670 | 0.584 |

Reference points at the same task (`scripts/baselines.py`):

| baseline | bits | corr | vR | F1 |
| --- | ---: | ---: | ---: | ---: |
| silent | 0 | 0.000 | 1.000 | 0.000 |
| dataset mean | 0 | 0.720 | 1.305 | 0.380 |
| PCA-sign, m=128 | 128 | 0.907 | 1.482 | 0.379 |
| PCA-sign, m=512 | 512 | 0.906 | 1.318 | 0.382 |

Every budget beats every baseline on every metric. **m = 16 bits, at 2312x
compression, already beats the m = 512 PCA-sign codec.**

**The curve saturates hard past m = 64.** Going from 64 to 1024 bits, a 16-fold
increase in rate, buys +0.009 correlation and +0.03 F1. Past that point the
binding constraint is not the bit budget.

### What the distortion metrics miss

Saturation is a statement about distortion, not about information. Feeding
reconstructions to a CNN trained only on *original* N-MNIST frames
(`scripts/probe.py`, the protocol of Skatchkovsky et al. 2021) tells a different
story:

| m | linear probe on the code | CNN on reconstructions | retention |
| ---: | ---: | ---: | ---: |
| 16 | 80.1% | 69.9% | 72.5% |
| 128 | 91.1% | 88.2% | 91.4% |
| 1024 | 95.2% | 90.8% | 94.1% |

*(the same CNN scores 96.5% on the originals)*

So between m = 128 and m = 1024 the distortion metrics move by under 0.01 while
recoverable digit identity moves by 3 points. The extra bits are buying
semantic content that spike-cell agreement is too blunt to see. This is the
distinction that matters for the task 3 detection extension: rate needed for
reconstruction and rate needed for inference are not the same number.

### Two measurement traps

Both of these cost real time, so they are worth recording.

**Cell-wise F1 and PSNR barely work on data this sparse.** At 7% occupancy the
zero-bit dataset-mean baseline scores F1 = 0.380, roughly what a trained codec
scores, because F1 is dominated by "spikes land where digits generally have
strokes". PSNR is worse than useless: the all-silent reconstruction scores
11.8 dB, beating every real reconstruction. Rank on `frame_corr` and the probes.

**Soft F1 and sampled F1 are the same number, by identity.** For a probability
map p and target t, the soft F1 is `2*sum(p*t) / (sum(p) + sum(t))`, which is
exactly the soft Dice coefficient; and since `E[sum(x*t)] = sum(p*t)` and
`E[sum(x)] = sum(p)` for `x ~ Bernoulli(p)`, sampling preserves it in
expectation. The two F1 curves in `figures/rate-distortion.png` therefore lie on
top of each other and no amount of tuning separates them. van Rossum and
`frame_corr`, being L2-type, do see the variance sampling adds.

## Metrics

No single distortion is right for a spike train, so three families are reported
and allowed to disagree:

| Metric | What it says |
| --- | --- |
| precision / recall / F1 / IoU | cell-wise agreement; strict, a spike one bin early is two errors |
| normalised van Rossum | PSP-filtered L2; 1.0 = a silent reconstruction, so <1 beats predicting nothing |
| PSNR on the time-summed frame | the digit a human eyeballs |
| code entropy / bits used | how much of the m-bit budget is actually spent |
| code probe / reconstruction probe accuracy | how much digit *identity* survives |

