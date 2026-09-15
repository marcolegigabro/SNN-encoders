"""Training and evaluation of the learned point-process codec.

Objective (the spec's L = R(Z) + beta D):

    loss = -log2 p(z) / T  +  beta * counting_l1_grid(N, N_hat_soft)

Training windows are drawn fresh from the source at every step, so there is no
training set to overfit. Evaluation always uses the shared fixed test set of
`sources.eval_set` and reports what a receiver would actually get:

* `D`          exact counting-function L1 of the *decoded points* (not the soft curve)
* `rate_real`  size of the range-coded bitstream of all test latents, per unit time
* `rate_ideal` mean -log2 p(z) / T under the learned prior
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from . import coding
from .bottleneck import symbol_entropy_bits
from .decoder import to_events
from .distortion import count_error, counting_curve, counting_l1, counting_l1_grid
from .model import Codec
from .sources import eval_set, homogeneous_poisson


@dataclass
class Config:
    # source
    rate: float = 5.0            # events per unit time
    T: float = 1.0
    # model
    encoder: str = "snn"         # snn | ann (parameter-matched)
    n_input_bins: int = 200
    hidden: int = 64
    # SNN encoder: hidden LIF leak at init and gradient scale through the recurrent
    # spikes (0 = detached). 0.97 / 0.5 halved the SNN's timing error in an
    # encoder-alone test (runs/diag/snn-memory-check.log)
    snn_beta: float = 0.97
    rec_grad_scale: float = 0.5
    n_latents: int = 8
    alphabet: int = 16
    embed_dim: int = 16
    dec_hidden: int = 1024       # 256 capped D near 0.10 even with a perfect 8x64 code
    n_grid: int = 1000
    # scalar (ordinal levels, see bottleneck.ScalarBottleneck) | onehot. Scalar is
    # the default: same GRU, beta=50, 6000 steps gave D 0.23 at 9.4 bits against
    # D 0.51 at 5.0 bits for one-hot (Rubin's baseline: ~0.21 at 9.4 bits).
    quantizer: str = "scalar"
    tau: float = 1.0             # softmax temperature of the one-hot straight-through estimator
    # scalar quantizer only: 0 feeds the decoder raw level values, d > 0 a learned
    # d-vector per (coordinate, level); see bottleneck.ScalarBottleneck
    level_embed_dim: int = 32
    # objective / optimisation
    beta: float = 50.0           # rate-distortion tradeoff
    # Rate warm-up, as fractions of `steps`: the rate term is off until
    # `rate_warmup_start`, then ramps linearly to full weight at `rate_warmup_end`.
    # Without it the code collapses before the decoder learns to read it: the
    # EMA prior rewards the most frequent symbols from step 1, and a GRU codec at
    # beta=50 ended at the zero-rate distortion (D = 1.163, same as beta=1).
    rate_warmup_start: float = 0.2
    rate_warmup_end: float = 0.5
    steps: int = 20000           # both encoders still improved well past 6000 (runs/diag/encoder-capacity-grid.log)
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 0.0
    eval_every: int = 500
    n_eval: int = 10_000
    seed: int = 0
    device: str = "cuda"
    # torch.compile the SNN encoder for training: ~6x faster steps (402 -> 65 ms
    # at B=512, M=200 on an RTX 4060) for a one-off compile cached on disk
    compile: int = 1


def check_finite(model: Codec, tensor: torch.Tensor, where: str):
    """Fail loudly, naming the parameters that went non-finite, instead of
    crashing later on a NaN in an unrelated place."""
    if torch.isfinite(tensor).all():
        return
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    bad += [n for n, b in model.named_buffers() if b.is_floating_point() and not torch.isfinite(b).all()]
    raise FloatingPointError(f"non-finite values in {where}; non-finite params/buffers: {bad[:12]}")


def rate_weight(step: int, cfg: Config) -> float:
    start, end = cfg.rate_warmup_start * cfg.steps, cfg.rate_warmup_end * cfg.steps
    if step <= start:
        return 0.0
    if step >= end or end <= start:
        return 1.0
    return (step - start) / (end - start)


def rd_loss(model: Codec, ev, cfg: Config, encoder=None, w_rate: float = 1.0):
    z, curve, bits, _ = model(ev, encoder=encoder)
    rate = bits.mean() / cfg.T
    dist = counting_l1_grid(counting_curve(ev, cfg.n_grid), curve).mean()
    return w_rate * rate + cfg.beta * dist, rate, dist, z


@torch.no_grad()
def evaluate(model: Codec, test, cfg: Config, batch: int = 1000) -> dict:
    model.eval()
    syms, d_exact, d_soft, bits, n_err = [], [], [], [], []
    spikes = None
    for i in range(0, len(test), batch):
        ev = test[i:i + batch].to(cfg.device)
        z, curve, _, s = model(ev, count_spikes=True)
        check_finite(model, curve, "evaluation curve")
        rec = to_events(curve, cfg.T)
        d_exact.append(counting_l1(ev, rec).cpu())
        d_soft.append(counting_l1_grid(counting_curve(ev, cfg.n_grid), curve).cpu())
        n_err.append(count_error(ev, rec).cpu())
        bits.append(model.prior.bits(z).cpu())
        syms.append(z.argmax(-1).cpu())
        if s is not None:
            spikes = s if spikes is None else [a + b for a, b in zip(spikes, s)]
    model.train()

    n = len(test)
    d_exact, sym = torch.cat(d_exact), torch.cat(syms)
    real = coding.latent_stream(sym.numpy(), model.prior.probs().double().cpu().numpy())
    ent = symbol_entropy_bits(sym, cfg.alphabet)
    return {
        "D": d_exact.mean().item(),
        "D_se": (d_exact.std() / math.sqrt(n)).item(),
        "D_soft": torch.cat(d_soft).mean().item(),
        "count_error": torch.cat(n_err).mean().item(),
        "rate_ideal": torch.cat(bits).mean().item() / cfg.T,
        "rate_real": real["real_bits_per_window"] / cfg.T,
        "coder_overhead": real["overhead"],
        # what a perfectly matched factorized prior would cost
        "rate_symbol_entropy": ent.sum().item() / cfg.T,
        "symbol_entropy": ent.tolist(),
        "active_coords": int((ent > 0.05).sum()),
        "spikes_per_window": None if spikes is None else [v / n for v in spikes],
    }


def train(cfg: Config, out_dir: Path):
    torch.manual_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    model = Codec(cfg).to(cfg.device)
    test = eval_set(cfg.rate, cfg.n_eval, cfg.T)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps,
                                                pct_start=0.1)
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 1)
    fast_encoder = (torch.compile(model.encoder)
                    if cfg.compile and cfg.encoder == "snn" else None)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"encoder={cfg.encoder} ({n_enc} params) | quantizer={cfg.quantizer} | "
          f"lambda T={cfg.rate * cfg.T:g} | "
          f"beta={cfg.beta:g} | latent {cfg.n_latents}x{cfg.alphabet} "
          f"(<= {cfg.n_latents * math.log2(cfg.alphabet):.0f} bits)", flush=True)

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]
    history, best = [], float("inf")
    run, n_run, t0 = {"loss": 0.0, "rate": 0.0, "dist": 0.0, "enc_grad_norm": 0.0}, 0, time.time()
    n_skipped = 0
    for step in range(1, cfg.steps + 1):
        ev = homogeneous_poisson(cfg.batch_size, cfg.rate, cfg.T, generator=gen, device=cfg.device)
        w_rate = rate_weight(step, cfg)
        loss, rate, dist, z = rd_loss(model, ev, cfg, fast_encoder, w_rate)
        check_finite(model, loss, f"training loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Clip the encoder and the rest of the codec separately, and drop a
        # non-finite encoder gradient instead of letting it into a global norm.
        # With one global clip, backprop through 200 SNN steps gave encoder
        # gradients of ~1e2-1e3 (later inf) that scaled every other gradient to
        # ~0 (then exactly 0: clip factor 5/inf), and the whole codec froze.
        enc_norm = torch.nn.utils.clip_grad_norm_(enc_params, 5.0)
        if torch.isfinite(enc_norm):
            run["enc_grad_norm"] += enc_norm.item()
        else:
            n_skipped += 1
            for p in enc_params:
                p.grad = None               # AdamW leaves parameters without a gradient untouched
        torch.nn.utils.clip_grad_norm_(other_params, 5.0)
        opt.step()
        sched.step()
        model.prior.update(z)
        for k, v in (("loss", loss), ("rate", rate), ("dist", dist)):
            run[k] += v.item()
        n_run += 1

        if step % cfg.eval_every == 0 or step == cfg.steps:
            stats = evaluate(model, test, cfg)
            stats.update(step=step, secs=time.time() - t0, rate_weight=w_rate,
                         skipped_encoder_steps=n_skipped,
                         **{f"train_{k}": v / n_run for k, v in run.items()})
            stats["objective"] = stats["rate_real"] + cfg.beta * stats["D"]
            run, n_run = dict.fromkeys(run, 0.0), 0
            history.append(stats)
            spk = stats["spikes_per_window"]
            print(f"step {step:5d} | w_rate {w_rate:.2f} | train loss {stats['train_loss']:8.3f} | "
                  f"rate ideal {stats['rate_ideal']:5.2f} real {stats['rate_real']:5.2f} bits/T | "
                  f"D {stats['D']:.4f} (soft {stats['D_soft']:.4f}) | "
                  f"active {stats['active_coords']}/{cfg.n_latents} | "
                  f"enc |g| {stats['train_enc_grad_norm']:.1e} skipped {n_skipped} | "
                  f"{'' if spk is None else f'{sum(spk):.0f} spk/win | '}"
                  f"{stats['secs']:.0f}s", flush=True)

            ckpt = {"model": model.state_dict(), "cfg": asdict(cfg), "stats": stats}
            torch.save(ckpt, out_dir / "last.pt")
            # only checkpoints trained on the full objective are eligible: during the
            # warm-up the rate is unpenalised and its measured value is not meaningful
            if step >= cfg.rate_warmup_end * cfg.steps and stats["objective"] < best:
                best = stats["objective"]
                torch.save(ckpt, out_dir / "best.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, history
