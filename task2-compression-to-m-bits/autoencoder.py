"""
Fixed-Rate SNN Spike-Train Autoencoder
========================================

Task: compress a spike train x_{1:T} (no channel) into a fixed-length
m-bit code b in {0,1}^m using an SNN encoder, then reconstruct x_hat_{1:T}
with an SNN decoder. Study the rate(m) - distortion trade-off for
different spike sources (homogeneous Poisson, inhomogeneous Poisson,
bursty/renewal, periodic).

Design choices (mirroring FSVAE / LTC ideas discussed):
  - LIF neurons come from snnTorch (snn.Leaky) with a rectangular surrogate
    gradient (Wu et al. 2019 / Zheng et al. 2021, used in FSVAE), rather than
    a hand-rolled autograd Function.
  - The m-bit bottleneck is produced by a dedicated readout layer of m LIF
    neurons whose *last-timestep* membrane potential is thresholded and
    trained with a Straight-Through Estimator (STE), exactly like the
    non-differentiable lattice quantizer in LTC.
  - The decoder is autoregressive (like FSVAE's decoder/prior), unrolling
    the fixed m-bit code into T timesteps by feeding it as a constant
    "context" input at every step, similar to Direct Input Encoding.
  - Distortion is measured with a PSP-smoothed MSE (post-synaptic-potential
    convolution), which is the same trick FSVAE/MMD-GLM use to compare
    spike trains in a way that tolerates small temporal jitter.

No entropy/rate term is needed in the loss: m is fixed by construction,
so "rate" is a hyperparameter you sweep, not something you regularize.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


# ----------------------------------------------------------------------
# 1. Surrogate-gradient spiking primitives
# ----------------------------------------------------------------------

def rectangular_surrogate(width=1.0):
    """
    Boxcar surrogate gradient, the approximation used in FSVAE / Wu et al. 2019:

        forward:  o = H(u - V_th)
        backward: do/du = (1/a) * 1[|u - V_th| < a/2]

    snnTorch hands the surrogate the already-shifted membrane (u - V_th), so
    the indicator is simply on |input_|.
    """

    def grad_fn(input_, grad_input, spikes):
        return grad_input * (input_.abs() < (width / 2)).float() / width

    return surrogate.custom_surrogate(grad_fn)


def make_lif(beta, v_th, surrogate_width=1.0):
    """
    Spiking LIF layer:  u_t = beta * u_{t-1} * (1 - o_{t-1}) + x_t

    reset_mechanism="zero" reproduces the reset-to-zero behaviour of the
    original hand-rolled cell. One difference worth knowing: snnTorch zeroes
    the state *after* integrating the incoming current, so on a timestep that
    follows a spike the input is discarded, whereas the hand-rolled cell zeroed
    only the decayed membrane and kept the input.
    """
    return snn.Leaky(
        beta=beta,
        threshold=v_th,
        spike_grad=rectangular_surrogate(surrogate_width),
        reset_mechanism="zero",
        init_hidden=False,
    )


def make_integrator(beta):
    """
    Non-firing leaky integrator:  u_t = beta * u_{t-1} + x_t

    reset_mechanism="none" means the membrane is never reset, so the emitted
    spikes can be ignored and only the membrane potential read out. This is
    FSVAE's spike-to-image trick (Eq. 19), used here for the encoder readout.
    """
    return snn.Leaky(beta=beta, reset_mechanism="none", init_hidden=False)


# ----------------------------------------------------------------------
# 2. Post-synaptic-potential (PSP) smoothing, for the distortion loss
# ----------------------------------------------------------------------

def psp_filter(spike_train, tau_syn=4.0):
    """
    spike_train: (T, B, ...) binary tensor
    Returns PSP(z_{<=t}) for every t, same recursive filter used in
    FSVAE / MMD-GLM:
        PSP(z_{<=t}) = (1 - 1/tau_syn) * PSP(z_{<=t-1}) + (1/tau_syn) * z_t
    """
    T = spike_train.shape[0]
    decay = 1.0 - 1.0 / tau_syn
    gain = 1.0 / tau_syn
    psp = torch.zeros_like(spike_train)
    running = torch.zeros_like(spike_train[0])
    for t in range(T):
        running = decay * running + gain * spike_train[t]
        psp[t] = running
    return psp


def psp_mse(x, x_hat, tau_syn=4.0):
    """Distortion metric: MSE between PSP-smoothed spike trains."""
    px = psp_filter(x, tau_syn)
    pxh = psp_filter(x_hat, tau_syn)
    return F.mse_loss(pxh, px)


# ----------------------------------------------------------------------
# 3. Encoder: spike train -> fixed m-bit code (STE bottleneck)
# ----------------------------------------------------------------------

class SNNEncoder(nn.Module):
    """
    x_{1:T} (T, B, C_in) -> b in {0,1}^m

    A stack of LIF layers processes the spike train timestep by timestep.
    The final LIF layer has exactly m neurons; their membrane potential is
    accumulated over T steps (like FSVAE's spike-to-image decoding via
    membrane potential), then thresholded once at the end to produce a
    single m-bit vector for the whole sequence (fixed-length code).
    """

    def __init__(self, c_in, hidden, m, tau_decay=0.5, v_th=1.0):
        super().__init__()
        self.fc1 = nn.Linear(c_in, hidden)
        self.fc2 = nn.Linear(hidden, m)
        self.lif1 = make_lif(tau_decay, v_th)
        self.readout = make_integrator(tau_decay)
        self.m = m

    def forward(self, x):
        # x: (T, B, C_in) binary spike train
        T = x.shape[0]

        mem1 = self.lif1.init_leaky()
        mem_out = self.readout.init_leaky()

        for t in range(T):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            # accumulate membrane potential of a non-firing readout neuron,
            # same trick as FSVAE's spike-to-image decoding (Eq. 19)
            _, mem_out = self.readout(cur2, mem_out)

        logits = mem_out  # real-valued "evidence" per bit, after T steps
        probs = torch.sigmoid(logits)
        # STE: forward pass uses a hard threshold at 0.5, backward pass
        # flows through the sigmoid as if it were the identity
        hard = (probs >= 0.5).float()
        b = probs + (hard - probs).detach()
        return b, probs  # b: (B, m) fixed-length binary code


# ----------------------------------------------------------------------
# 4. Decoder: fixed m-bit code -> reconstructed spike train
# ----------------------------------------------------------------------

class SNNDecoder(nn.Module):
    """
    b in {0,1}^m -> x_hat_{1:T} (T, B, C_out)

    Autoregressive SNN decoder (same spirit as FSVAE's decoder/prior):
    the m-bit code is injected as a constant context input at every
    timestep (Direct-Input-Encoding style), combined with the previously
    generated output spike, and a LIF stack produces the next output spike.
    """

    def __init__(self, m, hidden, c_out, T, tau_decay=0.5, v_th=1.0):
        super().__init__()
        self.T = T
        self.c_out = c_out
        self.fc_in = nn.Linear(m + c_out, hidden)
        self.fc_out = nn.Linear(hidden, c_out)
        self.lif1 = make_lif(tau_decay, v_th)
        self.lif2 = make_lif(tau_decay, v_th)

    def forward(self, b):
        B, _ = b.shape
        device = b.device

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        x_prev = torch.zeros(B, self.c_out, device=device)

        outputs = []
        for t in range(self.T):
            inp = torch.cat([b, x_prev], dim=-1)
            cur1 = self.fc_in(inp)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc_out(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            outputs.append(spk2)
            x_prev = spk2

        return torch.stack(outputs, dim=0)  # (T, B, c_out)


# ----------------------------------------------------------------------
# 5. Full autoencoder
# ----------------------------------------------------------------------

class SNNFixedRateAutoencoder(nn.Module):
    def __init__(self, c_in, c_out, T, m, hidden=64, tau_decay=0.5, v_th=1.0):
        super().__init__()
        self.encoder = SNNEncoder(c_in, hidden, m, tau_decay, v_th)
        self.decoder = SNNDecoder(m, hidden, c_out, T, tau_decay, v_th)

    def forward(self, x):
        b, probs = self.encoder(x)
        x_hat = self.decoder(b)
        return x_hat, b, probs


# ----------------------------------------------------------------------
# 6. Synthetic spike-source generators
# ----------------------------------------------------------------------

def gen_homogeneous_poisson(T, B, C, rate, device="cpu"):
    """Bernoulli-per-bin approx of a homogeneous Poisson process."""
    p = torch.full((T, B, C), rate, device=device)
    return torch.bernoulli(p)


def gen_inhomogeneous_poisson(T, B, C, base_rate=0.05, amp=0.15, period=None, device="cpu"):
    """Time-varying firing rate lambda(t) = base_rate + amp * sin(2*pi*t/period)."""
    if period is None:
        period = T
    t_idx = torch.arange(T, device=device).float()
    rate_t = base_rate + amp * (0.5 * (1 + torch.sin(2 * math.pi * t_idx / period)))
    rate_t = rate_t.clamp(0.0, 1.0).view(T, 1, 1).expand(T, B, C)
    return torch.bernoulli(rate_t)


def gen_bursty_renewal(T, B, C, burst_prob=0.03, refractory=4, device="cpu"):
    """
    Simple renewal process with a hard refractory period: after a spike,
    the neuron cannot fire again for `refractory` steps. Produces bursty /
    temporally-correlated trains compared to i.i.d. Poisson.
    """
    x = torch.zeros(T, B, C, device=device)
    refractory_left = torch.zeros(B, C, device=device)
    for t in range(T):
        can_fire = (refractory_left <= 0).float()
        fire = torch.bernoulli(torch.full((B, C), burst_prob, device=device)) * can_fire
        x[t] = fire
        refractory_left = torch.where(
            fire.bool(),
            torch.full_like(refractory_left, refractory),
            (refractory_left - 1).clamp(min=0),
        )
    return x


def gen_periodic(T, B, C, period=10, jitter=0.0, device="cpu"):
    """Deterministic periodic spike train (optionally with small timing jitter)."""
    x = torch.zeros(T, B, C, device=device)
    for start in range(0, T, period):
        if jitter > 0:
            offsets = torch.randint(
                low=max(-jitter, -start), high=jitter + 1, size=(B, C), device=device
            )
        else:
            offsets = torch.zeros(B, C, dtype=torch.long, device=device)
        idx = (start + offsets).clamp(0, T - 1)
        for b_i in range(B):
            for c_i in range(C):
                x[idx[b_i, c_i], b_i, c_i] = 1.0
    return x


SOURCES = {
    "poisson_homog": lambda T, B, C, device: gen_homogeneous_poisson(T, B, C, rate=0.1, device=device),
    "poisson_inhomog": lambda T, B, C, device: gen_inhomogeneous_poisson(T, B, C, device=device),
    "bursty_renewal": lambda T, B, C, device: gen_bursty_renewal(T, B, C, device=device),
    "periodic": lambda T, B, C, device: gen_periodic(T, B, C, period=8, jitter=1, device=device),
}


# ----------------------------------------------------------------------
# 7. Training loop for a single (source, m) pair
# ----------------------------------------------------------------------

def train_one_setting(source_name, m, T=32, C=1, hidden=64, epochs=200,
                       batch_size=64, lr=1e-3, device="cpu"):
    torch.manual_seed(0)
    model = SNNFixedRateAutoencoder(c_in=C, c_out=C, T=T, m=m, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen_fn = SOURCES[source_name]

    for epoch in range(epochs):
        x = gen_fn(T, batch_size, C, device)  # (T, B, C)
        x_hat, b, probs = model(x)
        loss = psp_mse(x, x_hat, tau_syn=4.0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 50 == 0:
            print(f"[{source_name} | m={m}] epoch {epoch+1}/{epochs} "
                  f"psp_mse={loss.item():.5f}")

    # final evaluation on a fresh batch
    with torch.no_grad():
        x_eval = gen_fn(T, 256, C, device)
        x_hat_eval, _, _ = model(x_eval)
        final_loss = psp_mse(x_eval, x_hat_eval, tau_syn=4.0).item()
        bit_err = (x_hat_eval.round() - x_eval).abs().mean().item()

    return model, final_loss, bit_err


# ----------------------------------------------------------------------
# 8. Rate-distortion sweep
# ----------------------------------------------------------------------

def run_rate_distortion_sweep(sources=("poisson_homog", "periodic"),
                               m_values=(4, 8, 16, 32),
                               T=32, epochs=150, device="cpu"):
    results = {}
    for src in sources:
        results[src] = []
        for m in m_values:
            _, dist, bit_err = train_one_setting(
                src, m, T=T, epochs=epochs, device=device
            )
            results[src].append((m, dist, bit_err))
            print(f"==> source={src:16s} m={m:3d}  "
                  f"PSP-MSE={dist:.5f}  raw-bit-err={bit_err:.4f}")
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    results = run_rate_distortion_sweep(
        sources=("poisson_homog", "poisson_inhomog", "bursty_renewal", "periodic"),
        m_values=(4, 8, 16, 32, 64),
        T=32,
        epochs=150,
        device=device,
    )

    print("\n=== Summary: rate-distortion trade-off ===")
    for src, points in results.items():
        print(f"\nSource: {src}")
        for m, dist, bit_err in points:
            print(f"  m={m:3d}  PSP-MSE={dist:.5f}  raw-bit-err={bit_err:.4f}")