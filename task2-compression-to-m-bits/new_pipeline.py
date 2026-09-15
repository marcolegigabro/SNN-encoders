"""
Complete Rate-Distortion Pipeline for Poisson Point-Process Compression
==========================================================================

Implements exactly the pipeline from the project description:

    X -> SNN encoder -> Z -> entropy coding -> Z_hat -> decoder -> X_hat

    Z in {0,...,15}^8          (8 discrete symbols, alphabet size 16)
    L = R(Z) + beta * D(X, X_hat)

    Z is produced by one of three interchangeable quantizer_type options
    (see quantizers.py and RDPipeline):
      - "gumbel": learned categorical logits + Gumbel-softmax (default,
                  original behavior).
      - "grid":   GridQuantizer, an independent per-coordinate uniform
                  scalar quantizer -- the literal {0,...,15}^8 grid.
      - "e8":     E8LatticeQuantizer, a vector quantizer onto the E8
                  lattice (densest known packing in 8 dimensions), which
                  can trade rate for distortion better than "grid" by
                  exploiting correlations across the 8 coordinates.

Sweeping beta traces the learned (R, D) operating points, which we then
compare against the theoretical rate-distortion function of a Bernoulli(p)
source under Hamming distortion:

    R(D) = Hb(p) - Hb(D),   0 <= D <= p

(derived from the Poisson -> Bernoulli-per-bin approximation discussed
earlier: X_t ~ Bernoulli(p) i.i.d., p = lambda * dt).

No channel is modeled: entropy coding is lossless (Z_hat = Z exactly),
so the only "loss" in the system comes from encoder -> Z compression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ltc_autoencoder import LIFCell, RealSource
from quantizers import GridQuantizer, E8LatticeQuantizer


# ----------------------------------------------------------------------
# 1. SNN Encoder -> categorical logits (n_symbols, alphabet_size)
# ----------------------------------------------------------------------

class SNNEncoderCategorical(nn.Module):
    """
    x_{1:T} (T, B, C_in) -> logits (B, n_symbols, alphabet_size)

    Same accumulation trick as SNNEncoder: a non-firing readout neuron's
    membrane potential is integrated over T steps, then reshaped into
    n_symbols independent categorical distributions over alphabet_size.
    """

    def __init__(self, c_in, hidden, n_symbols, alphabet_size,
                 tau_decay=0.5, v_th=1.0):
        super().__init__()
        self.fc1 = nn.Linear(c_in, hidden)
        self.fc_out = nn.Linear(hidden, n_symbols * alphabet_size)
        self.lif1 = LIFCell(tau_decay, v_th)
        self.lif_out_tau = tau_decay
        self.n_symbols = n_symbols
        self.alphabet_size = alphabet_size

    def forward(self, x):
        T, B, _ = x.shape
        device = x.device

        u1, o1 = LIFCell.init_state((B, self.fc1.out_features), device)
        u_out = torch.zeros(B, self.n_symbols * self.alphabet_size, device=device)

        for t in range(T):
            cur1 = self.fc1(x[t])
            u1, o1 = self.lif1(cur1, u1, o1)
            cur2 = self.fc_out(o1)
            u_out = self.lif_out_tau * u_out + cur2

        logits = u_out.view(B, self.n_symbols, self.alphabet_size)
        return logits


class SNNEncoderContinuous(nn.Module):
    """
    x_{1:T} (T, B, C_in) -> continuous z (B, n_dims) in (-1, 1)^n_dims.

    Same LIF accumulation trick as SNNEncoderCategorical, but the readout
    is squashed with tanh instead of reshaped into per-symbol categorical
    logits -- the natural encoder output for a quantizer (GridQuantizer or
    E8LatticeQuantizer, see quantizers.py) that expects a continuous
    vector to snap onto a fixed codebook.
    """

    def __init__(self, c_in, hidden, n_dims, tau_decay=0.5, v_th=1.0):
        super().__init__()
        self.fc1 = nn.Linear(c_in, hidden)
        self.fc_out = nn.Linear(hidden, n_dims)
        self.lif1 = LIFCell(tau_decay, v_th)
        self.lif_out_tau = tau_decay
        self.n_dims = n_dims

    def forward(self, x):
        T, B, _ = x.shape
        device = x.device

        u1, o1 = LIFCell.init_state((B, self.fc1.out_features), device)
        u_out = torch.zeros(B, self.n_dims, device=device)

        for t in range(T):
            cur1 = self.fc1(x[t])
            u1, o1 = self.lif1(cur1, u1, o1)
            cur2 = self.fc_out(o1)
            u_out = self.lif_out_tau * u_out + cur2

        return torch.tanh(u_out)


# ----------------------------------------------------------------------
# 2. Categorical Z via Gumbel-Softmax straight-through
# ----------------------------------------------------------------------

def gumbel_softmax_st(logits, tau=1.0, hard=True):
    """
    logits: (B, n_symbols, alphabet_size)
    Returns:
        z_soft: (B, n_symbols, alphabet_size) relaxed one-hot (for the
                 entropy model's expected log-prob during training)
        z_hard_idx: (B, n_symbols) integer symbol indices (the actual Z)
        z_st: (B, n_symbols, alphabet_size) straight-through one-hot,
              forward = hard one-hot, backward = soft gradient
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = (logits + gumbel_noise) / tau
    z_soft = F.softmax(y, dim=-1)

    if hard:
        idx = z_soft.argmax(dim=-1)                       # (B, n_symbols)
        z_hard = F.one_hot(idx, logits.shape[-1]).float()
        z_st = z_soft + (z_hard - z_soft).detach()
        return z_soft, idx, z_st
    else:
        idx = z_soft.argmax(dim=-1)
        return z_soft, idx, z_soft


# ----------------------------------------------------------------------
# 3. Factorized entropy model: R(Z)
# ----------------------------------------------------------------------

class FactorizedEntropyModel(nn.Module):
    """
    p(Z) = prod_i p_i(z_i), independent learned categorical per symbol.
    """

    def __init__(self, n_symbols, alphabet_size):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_symbols, alphabet_size))

    def rate_bits(self, z_soft_or_onehot):
        """
        z_soft_or_onehot: (B, n_symbols, alphabet_size), either a hard
        one-hot or the straight-through relaxation. Using the expected
        log-prob under this (near-)one-hot works for both cases and
        keeps the rate term differentiable end-to-end.
        """
        log_p = F.log_softmax(self.logits, dim=-1)          # (n_symbols, A)
        log_p = log_p.unsqueeze(0)                           # (1, n_symbols, A)
        log_p_z = (z_soft_or_onehot * log_p).sum(dim=-1)     # (B, n_symbols)
        nats = -log_p_z.sum(dim=-1)                          # (B,)
        bits = nats / math.log(2.0)
        return bits


# ----------------------------------------------------------------------
# 4. Categorical SNN Decoder: Z_hat -> X_hat probabilities
# ----------------------------------------------------------------------

class CategoricalSNNDecoderProb(nn.Module):
    """
    Same structure as CategoricalSNNDecoder, but outputs a per-bin
    firing PROBABILITY (via a non-firing readout + sigmoid) instead of
    a hard spike, so we can compute a differentiable Hamming-distortion
    surrogate directly against it.
    """

    def __init__(self, alphabet_size, n_symbols, embed_dim, hidden,
                 c_out, T, tau_decay=0.5, v_th=1.0):
        super().__init__()
        self.T = T
        self.c_out = c_out
        self.embedding = nn.Embedding(alphabet_size, embed_dim)
        context_dim = n_symbols * embed_dim

        self.fc_in = nn.Linear(context_dim + c_out, hidden)
        self.fc_out = nn.Linear(hidden, c_out)
        self.lif1 = LIFCell(tau_decay, v_th)
        self.lif_out_tau = tau_decay  # non-firing output readout

    def forward(self, z_onehot):
        """z_onehot: (B, n_symbols, alphabet_size) -- soft or hard one-hot."""
        B = z_onehot.shape[0]
        device = z_onehot.device

        # embedding via matmul with one-hot (works for soft one-hot too)
        emb_table = self.embedding.weight  # (alphabet_size, embed_dim)
        context = torch.einsum('bna,ad->bnd', z_onehot, emb_table)
        context = context.flatten(start_dim=1)  # (B, n_symbols*embed_dim)

        u1, o1 = LIFCell.init_state((B, self.fc_in.out_features), device)
        u_out = torch.zeros(B, self.c_out, device=device)
        x_prev = torch.zeros(B, self.c_out, device=device)

        probs = []
        for t in range(self.T):
            inp = torch.cat([context, x_prev], dim=-1)
            cur1 = self.fc_in(inp)
            u1, o1 = self.lif1(cur1, u1, o1)
            cur2 = self.fc_out(o1)
            u_out = self.lif_out_tau * u_out + cur2
            p_t = torch.sigmoid(cur2)          # per-bin firing probability
            probs.append(p_t)
            # feed a sampled/hard spike forward for autoregression
            x_prev = (p_t >= 0.5).float()

        return torch.stack(probs, dim=0)  # (T, B, c_out)


class ContinuousSNNDecoderProb(nn.Module):
    """
    Same role as CategoricalSNNDecoderProb, but the context comes from a
    continuous quantized vector z_q (B, n_dims) -- the output of
    GridQuantizer or E8LatticeQuantizer -- via a linear projection instead
    of an embedding-table lookup on a one-hot code.
    """

    def __init__(self, n_dims, hidden, c_out, T, tau_decay=0.5, v_th=1.0):
        super().__init__()
        self.T = T
        self.c_out = c_out
        self.fc_context = nn.Linear(n_dims, hidden)
        self.fc_in = nn.Linear(hidden + c_out, hidden)
        self.fc_out = nn.Linear(hidden, c_out)
        self.lif1 = LIFCell(tau_decay, v_th)
        self.lif_out_tau = tau_decay

    def forward(self, z_q):
        """z_q: (B, n_dims) -- dequantized continuous code."""
        B = z_q.shape[0]
        device = z_q.device
        context = self.fc_context(z_q)

        u1, o1 = LIFCell.init_state((B, self.fc_in.out_features), device)
        u_out = torch.zeros(B, self.c_out, device=device)
        x_prev = torch.zeros(B, self.c_out, device=device)

        probs = []
        for t in range(self.T):
            inp = torch.cat([context, x_prev], dim=-1)
            cur1 = self.fc_in(inp)
            u1, o1 = self.lif1(cur1, u1, o1)
            cur2 = self.fc_out(o1)
            u_out = self.lif_out_tau * u_out + cur2
            p_t = torch.sigmoid(cur2)
            probs.append(p_t)
            x_prev = (p_t >= 0.5).float()

        return torch.stack(probs, dim=0)  # (T, B, c_out)


# ----------------------------------------------------------------------
# 5. Full model + Hamming distortion surrogate
# ----------------------------------------------------------------------

class RDPipeline(nn.Module):
    """
    quantizer_type selects how the continuous encoder state becomes the
    discrete code Z:
      - "gumbel": learned categorical logits + Gumbel-softmax straight-
                  through (original behavior).
      - "grid":   continuous encoder output + GridQuantizer, an
                  independent per-coordinate uniform scalar quantizer,
                  Z in {0,...,alphabet_size-1}^n_symbols.
      - "e8":     continuous encoder output + E8LatticeQuantizer, a vector
                  quantizer onto the E8 lattice that exploits correlations
                  across the n_symbols coordinates a grid quantizer ignores.
    All three still produce a rate estimate via the same
    FactorizedEntropyModel and a Hamming-distortion-comparable X_hat, so
    their (R, D) points are directly comparable on the same plot.
    """

    def __init__(self, c_in, c_out, T, n_symbols=8, alphabet_size=16,
                 hidden=64, embed_dim=8, tau_decay=0.5, v_th=1.0,
                 quantizer_type="gumbel"):
        super().__init__()
        self.T = T
        self.quantizer_type = quantizer_type

        if quantizer_type == "gumbel":
            self.encoder = SNNEncoderCategorical(
                c_in, hidden, n_symbols, alphabet_size, tau_decay, v_th)
            self.entropy_model = FactorizedEntropyModel(n_symbols, alphabet_size)
            self.decoder = CategoricalSNNDecoderProb(
                alphabet_size, n_symbols, embed_dim, hidden, c_out, T,
                tau_decay, v_th)
        elif quantizer_type in ("grid", "e8"):
            self.encoder = SNNEncoderContinuous(c_in, hidden, n_symbols, tau_decay, v_th)
            if quantizer_type == "grid":
                self.quantizer = GridQuantizer(n_dims=n_symbols, levels=alphabet_size)
            else:
                self.quantizer = E8LatticeQuantizer(n_dims=n_symbols, levels=alphabet_size)
            self.entropy_model = FactorizedEntropyModel(n_symbols, self.quantizer.alphabet_size)
            self.decoder = ContinuousSNNDecoderProb(n_symbols, hidden, c_out, T, tau_decay, v_th)
        else:
            raise ValueError(f"unknown quantizer_type {quantizer_type!r}")

    def forward(self, x, tau_gumbel=1.0, hard=True):
        if self.quantizer_type == "gumbel":
            logits = self.encoder(x)                              # (B, n, A)
            z_soft, z_idx, z_st = gumbel_softmax_st(logits, tau_gumbel, hard)
            rate_bits = self.entropy_model.rate_bits(z_st)         # (B,)
            x_hat_probs = self.decoder(z_st)                       # (T, B, C_out)
            return x_hat_probs, rate_bits, z_idx

        z = self.encoder(x)                                       # (B, n_symbols)
        z_q, idx = self.quantizer(z)
        if hard:
            z_soft = self.quantizer.soft_onehot(z, tau=tau_gumbel)
            z_hard = F.one_hot(idx, self.quantizer.alphabet_size).float()
            z_st = z_soft + (z_hard - z_soft).detach()
        else:
            z_st = self.quantizer.soft_onehot(z, tau=tau_gumbel)
        rate_bits = self.entropy_model.rate_bits(z_st)             # (B,)
        x_hat_probs = self.decoder(z_q)                            # (T, B, C_out)
        return x_hat_probs, rate_bits, idx


def hamming_distortion(x, x_hat_probs):
    """
    Differentiable surrogate of expected Hamming distortion per bin:
        E[|X - X_hat|] = X(1-p) + (1-X)p = X + p - 2*X*p
    x, x_hat_probs: (T, B, C)
    Returns scalar: mean distortion per bin.
    """
    per_bin = x + x_hat_probs - 2 * x * x_hat_probs
    return per_bin.mean()


# ----------------------------------------------------------------------
# 6. Theoretical R(D) for Bernoulli(p) source, Hamming distortion
# ----------------------------------------------------------------------

def binary_entropy(p, eps=1e-12):
    p = min(max(p, eps), 1 - eps)
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def theoretical_RD_curve(p, n_points=200):
    """
    R(D) = Hb(p) - Hb(D), 0 <= D <= min(p, 1-p)
    Returns (D_values, R_values) per-bin rate in bits/bin.
    """
    D_max = min(p, 1 - p)
    Ds = [D_max * i / (n_points - 1) for i in range(n_points)]
    Hb_p = binary_entropy(p)
    Rs = [max(Hb_p - binary_entropy(D), 0.0) for D in Ds]
    return Ds, Rs


# ----------------------------------------------------------------------
# 7. Training loop for one beta
# ----------------------------------------------------------------------

def train_one_beta(beta, source, n_symbols=8, alphabet_size=16,
                    epochs=300, lr=1e-3, device="cpu", eval_batches=8,
                    quantizer_type="gumbel"):
    """source: a Source (see ltc_autoencoder.py) yielding (T, B, C) spike
    batches, e.g. RealSource("nmnist", ...). Its T and feature_dim fix the
    pipeline's time horizon and channel count.
    quantizer_type: "gumbel" (learned categorical), "grid" (uniform scalar
    quantizer), or "e8" (E8 lattice vector quantizer) -- see RDPipeline."""
    torch.manual_seed(0)
    T, C = source.T, source.feature_dim
    model = RDPipeline(c_in=C, c_out=C, T=T, n_symbols=n_symbols,
                        alphabet_size=alphabet_size,
                        quantizer_type=quantizer_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tau_start, tau_end = 1.5, 0.3  # softmax/Gumbel temperature annealing

    train_iter = source.train_batches(epochs, device)
    for epoch in range(epochs):
        tau = tau_start + (tau_end - tau_start) * (epoch / max(epochs - 1, 1))
        x = next(train_iter)

        x_hat_probs, rate_bits, z_idx = model(x, tau_gumbel=tau, hard=True)
        rate_per_bin = rate_bits.mean() / T
        distortion = hamming_distortion(x, x_hat_probs)

        loss = rate_per_bin + beta * distortion

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 100 == 0:
            print(f"[beta={beta}] epoch {epoch+1}/{epochs} "
                  f"R={rate_per_bin.item():.4f} bits/bin  D={distortion.item():.4f}")

    # final evaluation (hard Z, no Gumbel noise -> use low temperature)
    with torch.no_grad():
        x_eval = torch.cat(list(source.eval_batches(eval_batches, device)), dim=1)
        x_hat_probs, rate_bits, z_idx = model(x_eval, tau_gumbel=0.1, hard=True)
        R_final = (rate_bits.mean() / T).item()
        D_final = hamming_distortion(x_eval, x_hat_probs).item()

    return model, R_final, D_final


# ----------------------------------------------------------------------
# 8. Full sweep over beta + comparison plot
# ----------------------------------------------------------------------

def run_rd_sweep(source, betas=(0.5, 1, 2, 5, 10, 20, 50),
                  epochs=300, device="cpu", quantizer_type="gumbel"):
    """source fixes T and the channel count; its empirical spike probability
    (measured, not assumed) anchors the theoretical Bernoulli R(D) curve used
    as a reference bound. quantizer_type: see RDPipeline."""
    p = source.spike_prob(device)
    print(f"source={source.name}  T={source.T}  C={source.feature_dim}  "
          f"quantizer={quantizer_type}  empirical spike prob p={p:.4f}")

    learned_points = []
    for beta in betas:
        _, R, D = train_one_beta(beta, source, epochs=epochs, device=device,
                                  quantizer_type=quantizer_type)
        learned_points.append((R, D))
        print(f"==> beta={beta:6.2f}  R={R:.4f} bits/bin  D={D:.4f}")

    theo_D, theo_R = theoretical_RD_curve(p)

    return learned_points, (theo_D, theo_R), p


def plot_rd_curve(learned_points, theoretical_curve, p, save_path):
    import matplotlib.pyplot as plt

    theo_D, theo_R = theoretical_curve
    learned_R = [pt[0] for pt in learned_points]
    learned_D = [pt[1] for pt in learned_points]

    plt.figure(figsize=(6, 5))
    plt.plot(theo_D, theo_R, 'k-', linewidth=2, label=r"$R(D) = H_b(p) - H_b(D)$")
    plt.scatter(learned_D, learned_R, color='tab:red', zorder=5,
                label="Learned SNN codec (sweep over beta)")
    plt.xlabel("Distortion D (per-bin Hamming)")
    plt.ylabel("Rate R (bits per bin)")
    plt.title(f"Rate-distortion: learned codec vs. theoretical bound (p={p})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 16  # matches the project's 320ms / 16-bin N-MNIST binning

    source = RealSource("nmnist", T=T, batch_size=64)

    learned_points, theoretical_curve, p = run_rd_sweep(
        source, betas=(0.5, 1, 2, 5, 10, 20, 50), epochs=300, device=device
    )

    plot_rd_curve(learned_points, theoretical_curve, p,
                  save_path="rd_curve_nmnist.png")