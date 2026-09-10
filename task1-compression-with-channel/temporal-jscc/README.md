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

## The training budget, and why it is not a detail

The first pilot ran at lr = 3e-4, the value task 2 uses, and produced a
rate-distortion curve that was **flat at the zero-rate point** for Hamming: 0.150
at every rate, exactly what emitting silence scores. It would have been easy to
write that up as "learned coding cannot beat the trivial system on a memoryless
source". It was a step size.

The diagnostic that settled it, all at one point (K = 64, sigma = 1, Hamming,
where the rate is 0.65 bits/bin against the source's h(p) = 0.61, so the rate is
*ample* and any shortfall is optimisation). The column that matters is how many
of the source's 0.6098 bits per bin the code actually delivers, measured as
h(p) minus the converged cross-entropy:

| variant | bits delivered | Hamming |
| --- | ---: | ---: |
| lr 3e-4, 2000 steps (the pilot's setting) | 0.047 | 0.1507 |
| lr 1e-3, 2000 steps | 0.078 | 0.1486 |
| **noiseless channel**, lr 1e-3, 2000 steps | 0.091 | 0.1480 |
| hidden 512, lr 1e-3, 2000 steps | 0.095 | 0.1470 |
| lr 3e-3, 2000 steps | 0.109 | 0.1464 |
| lr 3e-3, 8000 steps | 0.177 | 0.1474 |
| lr 3e-3, hidden 512, 8000 steps | 0.205 | 0.1387 |
| lr 1e-2, 8000 steps | 0.262 | **0.1247** |
| lr 3e-2, 8000 steps | 0.291 | 0.1364 |
| lr 1e-2, 20000 steps | 0.325 | 0.1283 |
| lr 3e-2, 20000 steps | **0.327** | 0.1334 |

Deleting the channel entirely buys 0.013 bits. Raising the step size from 3e-4 to
1e-2 buys 0.215. The channel was never the binding constraint and neither was
the architecture; **at lr 3e-4 the system had barely started training**, and the
flat curve was measuring the optimiser.

Two things follow for anyone reading a curve out of this folder. Every point on
it has to be trained to the same budget, or the curve measures the budget
instead of the rate. And the run at lr 1e-2 was still improving at 8000 steps
(cross-entropy 0.2384 -> 0.2367 over the last eighth) while the 20000-step runs
had flattened (0.1988 -> 0.1968), so 8000 steps is short of convergence and
20000 is near it.

**The last four rows disagree with each other, and the disagreement is the point.**
Past 8000 steps the cross-entropy keeps falling -- 0.348 bits to 0.283 -- while
the Hamming distortion gets *worse*, 0.1247 to 0.1334. That is a real limit on
the argument made in `src/distortions.py`, that cross-entropy is the proper
surrogate for Hamming. The argument is asymptotic: a *perfect* posterior
thresholded at 1/2 is the Bayes rule, but between two imperfect models the one
with the better average log-loss need not have the better error rate, and here
it does not. Read with the caveat that these are single seeds, so the ordering
among the last four is within plausible run-to-run spread while the gap to the
first row is not. Recorded rather than tidied away, because the surrogate
argument is used to justify the Hamming curve and this is the size of the crack
in it.

Chosen for the sweep: **lr 1e-2, 8000 steps**, the cheapest setting on the
plateau and the best of the four on the distortion the Hamming curve actually
reports. 20000 steps for the primary grid alone is the upgrade to make if there
is time; it should help the two squared-error distortions, which are trained
directly on the quantity they are scored on and so have no surrogate gap to
worry about.

## Reproducing

```bash
python3 scripts/theory.py                    # bounds + self-checks, ~7 min, once
python3 scripts/baselines.py                 # reference systems, seconds
python3 scripts/sweep.py --grid primary    --lr 1e-2 --steps 8000 --workers 8
python3 scripts/sweep.py --grid multispike --lr 1e-2 --steps 8000 --workers 8 --coarse
python3 scripts/sweep.py --grid latent     --lr 1e-2 --steps 8000 --workers 8 --coarse
python3 scripts/report.py    --grid primary          # the tables
python3 scripts/visualize.py --grid primary \
        --ckpt runs/sweep/primary/hamming/K064-s1/model.pt
```

`scripts/theory.py` caches every capacity it computes in
`runs/capacity-cache.json`, and the sweep warms that cache serially before
forking, so the 2^T x 2^T Blahut-Arimoto behind the multi-spike capacity is paid
once rather than raced for by eight workers. `sweep.py` skips any point whose
`result.json` already exists, so an interrupted sweep resumes.

The primary grid is 11 rate points x 3 distortions = 33 models; the two control
grids use the coarse 6-point subset, 18 each. At lr 1e-2 and 8000 steps one
model is about 9 minutes of one core.

No dataset and no download: the source is a generator, so every batch is fresh,
the evaluation stream is separately seeded, and overfitting is not a possible
confound. Runs are single-threaded processes, many at once -- measured at 46
ms/step on one thread against 60 ms on four, so the parallelism belongs across
models, not inside them. This machine has 4 performance and 6 efficiency cores,
so eight workers is roughly six cores' worth of throughput, not eight.

## Numerical checks

`runs/theory.json` carries a `checks` block, and it is the first thing to read
if a measured point looks impossible. As computed at T = 12, p = 0.15:

| check | expected | got |
| --- | --- | --- |
| Blahut-Arimoto Hamming vs closed form `h(p) - h(D)` | 0 | 2.2e-16 |
| count-MSE R(D) at zero rate vs `Var[Binomial]` | 1.5300 | 1.5304 |
| count-MSE R(D) as D -> 0 vs `H(count)` | 2.290482 | 2.290482 |
| van Rossum R(D) as D -> 0 vs `T*h(p)` | 7.318084 | 7.318084 |
| multi-spike capacity at sigma -> 0 vs `T` bits | 12 | 12.000000 |
| van Rossum real-reproduction curve no looser than binary | >= 0 | -3.2e-5 |

The last one does not land where it should and the residual is understood rather
than dismissed. Both van Rossum curves are upper bounds on the same R(D), and
the real-reproduction run starts from the binary alphabet, so its estimate
cannot be genuinely looser. The comparison itself is what leaves the residual:
the two curves are sampled on different Lagrange-multiplier grids, so one is
linearly interpolated onto the other's rate points, and linear interpolation of
a convex curve lies *above* it -- biasing this very margin negative. At -3.2e-5
against a distortion scale of 0.024 it is 0.13% of the axis. To confirm rather
than infer, recompute both curves on a shared multiplier grid and compare
pointwise.

An earlier version of this check failed at -6.6e-4, and that one was real: the
reproduction points were initialised as 512 samples from the source, and 512
centroids cannot represent 4096 words near D = 0, so the "tighter" curve was
loose exactly where it mattered. `blahut_arimoto_rd_real` now starts from the
full alphabet.
