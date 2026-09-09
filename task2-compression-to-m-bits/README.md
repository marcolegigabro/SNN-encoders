# Task 2: end-to-end learned compression to m bits (no channel)

> Implement a similar end-to-end system, but without the channel. The encoder
> should output a fixed-length m-bit vector (or a short bit sequence) that
> represents the input spike train. The decoder, implemented as an SNN,
> reconstructs the spike train from this compact representation. Investigate
> the tradeoff between the number of bits m and the reconstruction quality for
> different spike sources.

## Contents

| Path | What it is |
| --- | --- |
| `autoencoder.py` | First pass: LIF cells with surrogate gradients, STE m-bit bottleneck, autoregressive SNN decoder, PSP-smoothed distortion, four synthetic sources, R-D sweep |
| `scripts/download_datasets.py` | Fetches N-MNIST and SHD into `../data/` |

## Shapes

An SNN receives one frame per step, in a loop, carrying membrane state across
steps. Time is not a tensor axis the network sees at once.

| Stage | Shape |
| --- | --- |
| binned sample, N-MNIST | `(T, 2, 34, 34)` = (time, polarity, y, x) |
| binned sample, SHD | `(T, 1, 700)`, polarity degenerate, squeeze it |
| batched, time-first | `(T, B, features)` |
| **one forward step** | `(B, features)` |
| the bottleneck | `(B, m)`, no time axis |

Dataloaders give batch-first `(B, T, ...)`, every SNN framework wants
time-first `(T, B, ...)`. Do the transpose once, in the collate function.

## Open design questions

- **Collapsing time into m bits.** Accumulate membrane potential over all
  steps and threshold at the end (what `autoencoder.py` currently does), take
  the final state only, or use spike counts of m output neurons.
- **Expanding m bits back over time.** Clamp the code as constant input at
  every step (current choice), inject once at t=0, or learn an unrolling layer.
- **Distortion measure.** PSP-smoothed MSE tolerates jitter but is not
  comparable to theory; Hamming is comparable to `R(D) = H(p) - H(D)` but
  punishes a one-step-late spike as hard as a missing one. Probably report
  both.
- **Block size.** Rate should be normalised as `m / (N*T)` bits per binary
  sample, and `T` is a preprocessing choice we make. Binning more coarsely
  "improves" the rate for free, so `T` has to be stated explicitly.

## Theory anchor

For a memoryless Bernoulli(p) source with Hamming distortion the
rate-distortion function is closed form, `R(D) = H(p) - H(D)`. Worth plotting
the learned system against it: on structured sources the encoder should beat
the memoryless bound by exploiting correlations, and that gap is the result.
