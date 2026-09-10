# Temporal coding through a jitter channel, against the information-theoretic bound

> Part of task 1 (compression *with* a channel). The architecture is task 2's --
> SNN compressor, bottleneck, SNN decompressor -- with three things changed: the
> source is synthetic instead of recorded, the bottleneck is a *temporal* code
> instead of an m-bit vector, and there is a channel in the middle that the two
> networks are trained through jointly.

```
x in {0,1}^(T,N)  --SNN-->  K spike times  --Gaussian jitter-->  --SNN-->  x_hat
   Poisson source          the temporal code       the channel        reconstruction
```

The question this folder answers is not "does it reconstruct". It is **how far
from optimal is it**, and for that every reference number is computed rather
than quoted: `src/theory.py` produces the source's rate-distortion function and
the channel's capacity, and `scripts/theory.py` runs the self-checks that say
whether those numerics can be trusted.

## Why a synthetic source

N-MNIST answers "how good is the reconstruction" but cannot answer "how close to
the bound", because the rate-distortion function of a set of event-camera
recordings is not a computable object. A Poisson source is: it is memoryless, its
statistics are exactly known, and its R(D) is a closed form for one of the three
distortions and an exact Blahut-Arimoto computation for the other two.

Two sources are provided (`src/source.py`).

* **iid** -- homogeneous Poisson in discrete time, every bin an independent
  Bernoulli(p). The primary source. Nothing for the encoder to exploit beyond the
  marginal, so the measured gap to R(D) is a statement about the codec alone.
* **latent** -- inhomogeneous Poisson, a rate envelope drawn per trial and shared
  across neurons. There is now structure to compress, and the i.i.d. curve at the
  same marginal rate becomes an *upper* bound on this source's R(D): dropping
  below it is evidence of exploited correlation, not of beating theory.

## The rate axis

This is the part that has to be got right, because "the rate" of a spiking
system with an analog channel in it is not self-evident.

The bottleneck sends **K spike times**, and a spike time is one use of a channel:
its input is a real number confined to the window [0, T] (a code neuron cannot
fire before the block opens or after it closes) and its output is that number
plus N(0, sigma^2). That is a *peak-amplitude-constrained* AWGN channel. So

```
R  =  K * C(sigma)   bits per block   =   K * C(sigma) / (N*T)   bits per source bin
```

with `C(sigma)` computed by Blahut-Arimoto on the discretised channel, not by
1/2 log(1 + SNR), which assumes an average-power constraint and overstates this
channel at low jitter. Rate is then bought two independent ways -- more code
neurons, or less jitter -- and if pricing a use at C(sigma) is right, the two
sweeps have to land on the same curve. `figures/rate-distortion-*.png` is where
that either holds or does not.

The bound the system is held to is **OPTA**: with `rho = K/(N*T)` channel uses
per source symbol, no system, joint or separate, can achieve a distortion below
the D solving `R(D) = rho * C(sigma)`. Separation is not assumed anywhere in the
system -- there is no bit-string in the middle and no error-correcting code, the
encoder's output goes straight onto the channel -- but the *bound* holds
regardless, which is what makes it usable here.

## The three distortions

The subject asks for three, and they are kept genuinely different: each is a
training loss, a measurement and a bound, all three consistent.

| distortion | measured on | unit | bound |
| --- | --- | --- | --- |
| Hamming | thresholded reconstruction, per bin | P(bit error) | closed form, `h(p) - h(D)` |
| count MSE | expected spike count, per neuron | spikes^2 | Blahut-Arimoto on the Binomial count |
| van Rossum | thresholded reconstruction, per bin | squared PSP error | Blahut-Arimoto over all 2^T words |

The count MSE is scored on the *expected* count and bounded over a real-valued
reproduction grid, because squared error is minimised by conditional means and
those are not integers. The other two are scored on genuinely binary
reconstructions, matching the binary reproduction alphabet their bounds use.

Hamming is trained through binary cross-entropy rather than a soft error count.
That is not a convenience: BCE is a proper loss, so its minimiser is the true
posterior P(x=1 | received code), and thresholding a true posterior at 1/2 is the
Bayes rule for Hamming. The surrogate and the target have the same optimum.

## The two bottlenecks

**ttfs** (primary). K code neurons, one spike each, at a real time in [0, T].
Gaussian jitter on each time. Fully differentiable: additive input-independent
noise *is* the reparameterisation trick, so the loss gradient reaches the
encoder's spike times with no surrogate anywhere in the channel.

**multispike** (control). K code neurons emitting a full T-slot train, every
spike independently displaced and re-binned, collisions merged, spikes leaving
the window lost. Closer to the rate-coded setting of `../rate_snn.py`, and it
needs a straight-through estimator, which is why it is the control.

Their channel uses are not comparable one-for-one -- one is a real number, the
other a T-slot word -- which is exactly why the rate axis prices each in bits.
The multi-spike transition matrix is built **exactly** (folding one spike at a
time through the 2^T output masks) rather than sampled: a Monte-Carlo channel
matrix leaves most output words unobserved, which makes the channel look cleaner
than it is and *inflates* capacity, corrupting the bound in the dangerous
direction.

## Layout

```
src/
  theory.py       R(D), capacities, OPTA -- every reference number
  source.py       the two Poisson sources
  snn.py          LIF cells, PSP filter, the latency readout
  channel.py      Gaussian jitter on times; displacement and re-binning on trains
  model.py        compressor -> channel -> decompressor
  distortions.py  the three distortions as losses and as measurements
  train.py        one point of one curve
scripts/
  theory.py       precompute the bounds and their self-checks -> runs/theory.json
  train.py        one model
  sweep.py        the grid of (rate point, distortion) -> the curves
  baselines.py    zero-rate, uncoded analog count, noiseless count
  visualize.py    all figures
```

## Reproducing

```bash
python3 scripts/theory.py                          # bounds + self-checks, ~5 min
python3 scripts/baselines.py                       # reference systems
python3 scripts/sweep.py --grid primary  --workers 8
python3 scripts/sweep.py --grid multispike --workers 8
python3 scripts/sweep.py --grid latent     --workers 8
python3 scripts/visualize.py --ckpt runs/sweep/primary/hamming/K032-s1/model.pt
```

No dataset and no download: the source is a generator, so every batch is fresh,
the evaluation stream is separately seeded, and overfitting is not a possible
confound. Runs are single-threaded processes, many at once -- measured at 46
ms/step on one thread against 60 ms on four, so the parallelism belongs across
models, not inside them.
